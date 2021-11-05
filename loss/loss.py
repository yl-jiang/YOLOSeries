import torch
import torch.nn as nn
import numpy as np
from utils import xyxy2xywhn
from utils import xywh2xyxy
from utils import gpu_CIoU


import torch
import torch.nn as nn
import numpy as np
from utils import xyxy2xywhn
from utils import xywh2xyxy
from utils import gpu_CIoU


class YOLOV5Loss:

    def __init__(self, anchors, hyp, stage_num=3):
        """
        3种loss的重要程度依次为：confidence > classification > regression 

        :param anchors: tensor / (3, 3, 2)
        :param loss_hyp: dict
        :param data_hyp: dict
        :param train_hyp: dict
        :param stage_num: int
        """
        self.anchors = anchors  # anchors shape = (3, 3, 2)
        self.hyp = hyp
        self.device = hyp['device']
        cls_pos_weight = torch.tensor(hyp["cls_pos_weight"], device=self.device)
        cof_pos_weight = torch.tensor(hyp['cof_pos_weight'], device=self.device)
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight, reduction='none').to(self.device)
        self.bce_cof = nn.BCEWithLogitsLoss(pos_weight=cof_pos_weight, reduction='none').to(self.device)
        self.input_img_size = hyp['input_img_size']
        self.balances = [4., 1., 0.4] if stage_num == 3 else [4., 1., 0.4, 0.1]

    def __call__(self, stage_preds, targets_batch):
        """
        通过对比preds和targets，找到与pred对应的target。
        注意：每一个batch的targets中的bbox_num都相同(参见cocodataset.py中fixed_imgsize_collector函数)。

        :param stage_preds: (out_small, out_mid, out_large) / [(bn, 255, 80, 80),(bn, 255, 40, 40),(bn, 255, 20, 20)]
        :param targets_batch: 最后一个维度上的值，表示当前batch中该target对应的img index
        :param targets_batch: tensor / (bn, bbox_num, 6) -> [xmin, ymin, xmax, ymax, cls, img_id]
        """
        assert isinstance(stage_preds, (list, tuple))
        assert isinstance(targets_batch, torch.Tensor), f"targets's type should be torch.Tensor but we got {type(targets_batch)}"
        assert stage_preds[0].size(0) == targets_batch.size(0), f"the length of predictions and targets should be the same, " \
            "but len(predictions)={preds_batch[0].size(0)} and len(targets)={targets_batch.size(0)}"

        batch_size = targets_batch.size(0)
        bbox_num = targets_batch.size(1)
        anchor_num = self.anchors.shape[1]
        targets = targets_batch.clone().detach()
        targets[..., :4] = xyxy2xywhn(targets[..., :4], self.input_img_size)
        # (bn, bbox_num, 6) -> (anchor_num, bn, bbox_num, 6) / 每个obj都与3个anchor进行匹配
        targets = targets.repeat(anchor_num, 1, 1, 1).contiguous()
        # anchor_ids: (anchor_num, 1)
        anchor_ids = torch.arange(anchor_num, device=self.device, dtype=torch.float32).reshape(-1, 1).contiguous()
        # anchor_ids: (anchor_num, 1) -> (anchor_num, bn, bbox_num, 1)
        anchor_ids = anchor_ids[:, None, None, :].repeat(1, batch_size, bbox_num, 1).contiguous()
        # (anchor_num, bn, bbox_num, 6) -> (anchor_num, bn, bbox_num, 7) / 最后一维多出来的一个元素标记匹配的anchor
        targets = torch.cat([targets, anchor_ids], dim=-1).contiguous()
        assert torch.sum(targets[0][..., -1]) == 0., f"please check the data format of anchor_ids"

        cls_loss, obj_loss, cof_loss = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
        balances = [4., 1., 0.4] if len(stage_preds) == 3 else [4., 1., 0.4, 0.1]
        tot_tar_num = 0
        s = 3 / len(stage_preds)
        for i in range(len(stage_preds)):
            bn, fm_h, fm_w = torch.tensor(stage_preds[i].shape)[[0, 2, 3]]
            ds_scale = self.input_img_size[1] / fm_w  # downsample scale
            # anchor: (3, 2)
            anchor = self.anchors[i]
            anchor_stage = anchor / ds_scale
            anchor_num = anchor_stage.shape[0]
            # preds: (bn, 255, h, w) -> (bn, 3, 85, h, w) -> (bn, 3, h, w, 85)
            preds = stage_preds[i].reshape(bn, anchor_num, -1, fm_h, fm_w).permute(0, 1, 3, 4, 2).contiguous()
            assert preds.size(-1) == self.hyp['num_class'] + 5
            # match anchor(正样本匹配) / box, cls, img_idx, anchor_idx, grid_y, grid_x
            tar_box, tar_cls, img_idx, anc_idx, gy, gx = self.match(targets, anchor_stage, (fm_w, fm_h))

            cur_tar_num = tar_box.shape[0]
            tot_tar_num += cur_tar_num
            # cur_preds: (N, 85) / [pred_x, pred_y, pred_w, pred_h, confidence, c1, c2, c3, ..., c80]
            cur_preds = preds[img_idx, anc_idx, gy, gx]

            # Classification
            # 只有正样本才参与分类损失的计算
            if cur_preds[:, 5:].size(1) > 1:  # if only one class then we don't compute class loss
                # t_cls: (N, 80)
                t_cls = torch.zeros_like(cur_preds[:, 5:]).to(self.device)
                t_cls[torch.arange(tar_cls.size(0)), tar_cls] = self.hyp['class_smooth_factor']

                if self.hyp['use_focal_loss']:
                    cls_factor = self.focal_loss_factor(cur_preds[:, 5:], t_cls)
                else:
                    cls_factor = torch.ones_like(t_cls)
                
                cls_loss += (self.bce_cls(cur_preds[:, 5:], t_cls) * cls_factor).mean()

            # Confidence and Regression
            # 只有正样本才参与回归损失的计算
            t_cof = torch.zeros_like(preds[..., 4]).to(self.device)
            if cur_tar_num > 0:
                # sigmoid(-5) ≈ 0; sigmoid(0) = 0.5; sigmoid(5) ≈ 1
                # sigmoid(-5) * 2 - 0.5 = -0.5; sigmoid(0) * 2 - 0.5 = 0.5; sigmoid(5) * 2 - 0.5 = 0.5
                pred_xy = cur_preds[:, :2].sigmoid() * 2. - 0.5
                # (N, 2) & (N, 2) -> (N, 2)
                pred_wh = (cur_preds[:, 2:4].sigmoid() * 2.) ** 2 * anchor_stage[anc_idx]
                # pred_box: (N, 4)
                pred_box = torch.cat((pred_xy, pred_wh), dim=1).to(self.device)
                # because pred_box and tar_box's format is xywh, before compute iou loss we should turn it to xyxy format
                pred_box, tar_box = xywh2xyxy(pred_box), xywh2xyxy(tar_box)
                # iou: (N,)
                iou = gpu_CIoU(pred_box, tar_box)
                obj_loss += (1. - iou).mean()
                # t_cof: (bn, 3, h, w) / 所有grid均参与confidence loss的计算
                t_cof[img_idx, anc_idx, gy, gx] = iou.detach().clamp(0).type_as(t_cof)

            if self.hyp['use_focal_loss']:
                cof_factor = self.focal_loss_factor(preds[..., 4], t_cof)
            else:
                cof_factor = torch.ones_like(t_cof)

            # 所有样本均参与置信度损失的计算 / 在3中loss中confidence loss是最为重要的
            cof_loss_tmp = (self.bce_cof(preds[..., 4], t_cof) * cof_factor).mean() * balances[i]
            balances[i] = balances[i] * 0.9999 + 0.0001 / cof_loss_tmp.detach().item()
            cof_loss += cof_loss_tmp

        self.balances = [x/self.balances[1] for x in self.balances]
        obj_loss *= self.hyp['obj_loss_scale'] * s
        cof_loss *= self.hyp['cof_loss_scale'] * s * (1. if len(stage_preds) == 3 else 1.4)
        cls_loss *= self.hyp['cls_loss_scale'] * s
        tot_loss = (obj_loss + cof_loss + cls_loss) * batch_size
        return tot_loss, obj_loss.detach().item(), cof_loss.detach().item(), cls_loss.detach().item(), tot_tar_num

    def match(self, targets, anchor_stage, fm_shape):
        """
        正样本分配策略。

        并不是传入的所有target都可以参与最终loss的计算，只有那些与anchor的width/height ratio满足一定条件的targe才有资格；
        :param targets: (num_anchor=3, batch_num, bbox_num, 7) / [norm_x, norm_y, norm_w, norm_h, cls, img_id, anchor_id]
        :param anchor_stage: (3, 2) / (w, h)
        :param fm_shape: [w, h] / stage feature map shape
        :return:
        """

        g = torch.ones_like(targets)
        # [fm_w, fm_h, fm_w, fm_h, 1, 1, 1]
        g[..., :4] *= torch.tensor(fm_shape, device=g.device, dtype=g.dtype)[[0, 1, 0, 1]]
        # (stage_x, stage_y, stage_w, stage_h, cls, img_id, anchor_id)
        t_stage = targets * g

        # fliter by anchor ratio
        # t_stage_whs: (3, bn, bbox_num, 2)
        t_stage_whs = t_stage[..., [2, 3]]
        batch_size, bbox_num = t_stage.size(1), t_stage.size(2)
        # anchor_wh: (3, 2) -> (3, bn, bbox_num, 2)
        anchor_stage_whs = anchor_stage[:, None, None, :].repeat(1, batch_size, bbox_num, 1).contiguous()
        # ratio: (3, bn, bbox_num, 2)
        ratio = t_stage_whs / anchor_stage_whs + 1e-16
        # match_index: (3, bn, bbox_num) 为target选取符合条件的anchor
        ar_mask = torch.max(ratio, 1/ratio).max(dim=-1)[0] < self.hyp['anchor_match_thr']  # anchor ratio mask
        # targets: (3, bn, bbox_num, 7) -> (X, 7)
        t_stage = t_stage[ar_mask]

        # augmentation by grid
        # t_stage_xys: (X, 2) / (x, y)
        grid_xys = t_stage[..., [0, 1]]
        grid_xys_offset = torch.tensor(fm_shape, device=self.device).float()[None, :] - grid_xys
        assert len(grid_xys_offset[grid_xys_offset < 0.]) == 0, "grid_xys必须位于当前feature map内"
        # 舍弃那些中心点x和y同时靠近图片边界的object；舍弃那些中心点x和y同时过于远离所在grid左上角的object（参数grid_ratio控制）
        grid_thr = 0.5
        # offset: (5, 2) / (x_offset, y_offset)
        offset = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float() * grid_thr
        # transpose: row index, cloumn index -> column index, row index
        grid_mask_i, grid_mask_j = ((grid_xys % 1.0 < grid_thr) & (grid_xys > 1.)).T
        # transpose: row index, cloumn index -> column index, row index
        grid_mask_l, grid_mask_m = ((grid_xys_offset % 1.0 < grid_thr) & (grid_xys_offset > 1.)).T
        # grid_mask: (X,) -> (5, X)
        grid_mask = torch.stack((torch.ones_like(grid_mask_i), grid_mask_i, grid_mask_j, grid_mask_l, grid_mask_m), dim=0)
        # t_stage: (X, 7) -> (5, X, 7) & (5, X) -> (N, 7)
        t_stage = t_stage.repeat(5, 1, 1).contiguous()
        t_stage = t_stage[grid_mask]
        # obj_xy_offset: (1, X, 2) & (5, 1, 2) -> (5, X, 2) & (5, X) -> (N, 2)
        grid_xys_expand = torch.zeros_like(grid_xys)[None] + offset[:, None, :]
        grid_xys_expand = grid_xys_expand[grid_mask]
        # tar_grid_xys在对应特征图尺寸中的xy
        tar_grid_xys = t_stage[:, [0, 1]]  # (N, 2)
        # 放宽obj预测的中心坐标精度的限制，在真实grid_center_xy范围内浮动一格，均认为是预测正确；tar_grid_coors表示obj所在grid的xy坐标
        tar_grid_coors = (tar_grid_xys - grid_xys_expand).long()  # (N, 2)
        # tar_grid_off:相对于所在grid的偏移量
        tar_grid_off = tar_grid_xys - tar_grid_coors
        tar_grid_whs = t_stage[:, [2, 3]]  # (N, 2)
        # tar_box: (N, 2) & (N, 2) -> (N, 4) / (x, y, w, h)
        tar_box = torch.cat((tar_grid_off, tar_grid_whs), dim=-1)  # (grid_off_x, grid_off_y, w_stage, h_stage)
        # tar_cls: (N, )
        tar_cls = t_stage[:, 4]
        # tar_img_idx: (N, )
        tar_img_idx = t_stage[:, 5]  # 一个batch中的img id
        # tar_anc_idx： （N,)
        tar_anc_idx = t_stage[:, 6]  # anchor id
        # tar_grid_i: (N, ) / row index; tar_grid_j: (N, ) / cloumn index
        tar_grid_x, tar_grid_y = tar_grid_coors.T
        tar_grid_x = torch.clamp(tar_grid_x, 0, fm_shape[0]-1)
        tar_grid_y = torch.clamp(tar_grid_y, 0, fm_shape[1]-1)

        del g, offset, t_stage
        return tar_box, tar_cls.long(), tar_img_idx.long(), tar_anc_idx.long(), tar_grid_y.long(), tar_grid_x.long()

    def focal_loss_factor(self, pred, target):
        prob = torch.sigmoid(pred)
        # target * prob：将正样本预测正确的概率； (1.0 - target) * (1.0 - prob)：将负样本预测正确的概率
        acc_scale = target * prob + (1.0 - target) * (1.0 - prob)
        # 对那些预测错误程度越大的预测加大惩罚力度
        gamma = self.hyp.get('focal_loss_gamma', 1.5)
        gamma_factor = (1.0 - acc_scale) ** gamma
        # 当alpha值小于0.5时，意味着更加关注将负类样本预测错误的情况
        alpha = self.hyp.get('focal_loss_alpha', 0.25)
        alpha_factor = target * alpha + (1.0 - target) * (1.0 - alpha)
        factor = gamma_factor * alpha_factor

        return factor



class YOLOXLoss:

    def __init__(self, num_stages=3, num_anchor=1, img_sz=[224, 224]) -> None:
        self.grids = [torch.zeros(1) for _ in range(num_stages)]
        self.num_anchor = num_anchor
        self.img_sz = img_sz

    def __call__(self, tars, preds):
        """
        假设输入训练的图片尺寸为224x224x3

        Args:
            preds: 字典；{'pred_s': (N, 85, 28, 28), 'pred_m': (N, 85, 14, 14), 'pred_l': (N, 85, 7, 7)}
            tars: tensor; (N, bbox_num, 6)
        """
        batch_size = tars.size(0)
        dtype = tars.type()
        for i, k in enumerate(preds.keys()):
            h, w = preds[k].shape[-2:]
            stride = self.img_sz[0] / h  # 该stage下采样尺度
            grid = self.grids[i]
            pred = preds[i]
            if grid[i].shape[2:4] != preds[k].shape[2:4]:
                grid = self._make_grid(h, w, dtype)
                self.grids[i] = grid
            # (N, 85, h, w) -> (N, 1, 85, h, w)
            pred = pred.view(batch_size, self.num_anchor, -1, h, w).contigugous()
            # (N, 1, 85, h, w) -> (N, 1, h, w, 85)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            # (N, 1, h, w, 85) -> (N, h*w, 85)
            pred = pred.reshape(batch_size, self.num_anchor * h * w, -1).contiguous()
            # (1, 1, h, w, 2) -> (1, h*w, 2)
            grid = grid.view(1, -1, 2).contiguous()
            pred[..., :2] = (pred[..., :2] + grid) * stride
            pred[..., 2:4] = torch.exp(pred[..., 2:4]) * stride



    def calculate_each(self, tar, pred, gird):
        """
        Args:
            pred: 某个stage的预测输出：(N, 85, 28, 28)或(N, 85, 14, 14)或(N, 85, 7, 7)
        """

    def _make_grid(h, w, dtype):
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w))
        # 排列成(x, y)的形式，是因为模型输出的预测结果的排列是[x, y, w, h, cof, cls1, cls2, ...]
        grid = torch.stack((xs, ys), dim=2).view(1, 1, h, w, 2).contiguous().type(dtype)
        return grid
