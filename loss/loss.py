from math import cos
import torch
import torch.nn as nn
import numpy as np
from utils import xyxy2xywhn, xyxy2xywh, xywh2xyxy
from utils import gpu_CIoU
from utils import gpu_iou, gpu_DIoU, gpu_Giou
import torch.nn.functional as F
from torch.cuda.amp import autocast


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

        cls_loss, reg_loss, cof_loss = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
        balances = [4., 1., 0.4] if len(stage_preds) == 3 else [4., 1., 0.4, 0.1]
        tot_tar_num = 0
        s = 3 / len(stage_preds)
        for i in range(len(stage_preds)):  # each stage
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
                reg_loss += (1.0 - iou).mean()
                # t_cof: (bn, 3, h, w) / 所有grid均参与confidence loss的计算
                t_cof[img_idx, anc_idx, gy, gx] = iou.detach().clamp(0).type_as(t_cof)

            if self.hyp['use_focal_loss']:
                cof_factor = self.focal_loss_factor(preds[..., 4], t_cof)
            else:
                cof_factor = torch.ones_like(t_cof)

            # 所有样本均参与置信度损失的计算 / 在3中loss中confidence loss是最为重要的
            cof_loss_tmp = (self.bce_cof(preds[..., 4], t_cof) * cof_factor).mean()
            cof_loss_tmp *= balances[i]
            balances[i] = balances[i] * 0.9999 + 0.0001 / cof_loss_tmp.detach().item()
            cof_loss += cof_loss_tmp

        self.balances = [x/self.balances[1] for x in self.balances]
        reg_loss *= self.hyp['reg_loss_scale'] * s
        cof_loss *= self.hyp['cof_loss_scale'] * s * (1. if len(stage_preds) == 3 else 1.4)
        cls_loss *= self.hyp['cls_loss_scale'] * s
        tot_loss = (reg_loss + cof_loss + cls_loss) * batch_size
        return tot_loss, reg_loss.detach().item(), cof_loss.detach().item(), cls_loss.detach().item(), tot_tar_num

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
        """
        Args:
            pred: (N, 80)
            target: (N, 80)
        Return:
            focal loss factor: (N, 80)
        """
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

    def __init__(self, hyp) -> None:
        self.hyp = hyp
        self.num_anchors = hyp['num_anchors']
        self.num_stage = hyp.get('num_stage', 3)
        self.grids = {}
        self.img_sz = hyp['input_img_size']
        self.num_class = hyp['num_class']
        self.use_l1 = hyp.get('use_l1', True)
        self.reg_loss_scale = hyp.get('reg_loss_scale', 0.5)
        self.cls_loss_scale = hyp.get('cls_loss_scale', 1.0)
        self.l1_loss_scale = hyp.get('l1_loss_scale', 1.0)
        self.cof_loss_scale = hyp.get('cof_loss_scale', 1.0)
        self.device = hyp['device']
        cls_pos_weight = torch.tensor(hyp.get("cls_pos_weight", 1.), device=self.device)
        cof_pos_weight = torch.tensor(hyp.get('cof_pos_weight', 1.), device=self.device)
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight, reduction='none').to(self.device)
        self.bce_cof = nn.BCEWithLogitsLoss(pos_weight=cof_pos_weight, reduction='none').to(self.device)
        self.l1_loss = nn.L1Loss(reduction='none')
        self.cls_smoothness = hyp['class_smooth_factor']

    def __call__(self, tars, preds):
        """
        假设输入训练的图片尺寸为224x224x3

        Args:
            preds: 字典;{'pred_s': (N, num_anchors, 85, 28, 28), 'pred_m': (N, num_anchors, 85, 14, 14), 'pred_l': (N, num_anchors, 85, 7, 7)} / [x, y, w, h, cof, cls1, cls2, ...]
            tars: tensor; (N, bbox_num, 6) / [xmin, ymin, xmax, ymax, class_id, img_id]
        """
        batch_size = tars.size(0)
        dtype = tars.type()
        tars[..., :4] = xyxy2xywh(tars[..., :4])

        tot_num_fg, tot_num_gt = 0, 0
        tot_cls_loss, tot_reg_loss, tot_cof_loss, tot_l1_reg_loss = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
        for k in preds.keys():  # each stage
            h, w = preds[k].shape[-2:]
            stride = self.img_sz[0] / h  # 该stage下采样尺度 
            grid = self.grids.get(f"stride_{stride}", torch.zeros(1))
            pred = preds[k]
            if grid.shape[2:4] != preds[k].shape[2:4]:
                grid = self._make_grid(h, w, dtype)
                self.grids[f"stride_{stride}"] = grid

            # (N, num_anchors, 85, h, w) -> (N, num_anchors, h, w, 85)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            # (N, num_anchors, h, w, 85) -> (N, num_anchors*h*w, 85)
            pred = pred.reshape(batch_size, self.num_anchors * h * w, -1)
            # (1, 1, h, w, 2) -> (1, num_anchors*h*w, 2)
            grid = grid.repeat(1, self.num_anchors, 1, 1, 1).reshape(1, -1, 2)
            
            stage_out_dict = self.calculate_loss_of_each_stage(tars, pred, grid[0], stride)
            tot_num_fg += stage_out_dict['num_fg']
            tot_num_gt += stage_out_dict['num_gt']
            tot_cls_loss += stage_out_dict['cls_loss'] 
            tot_reg_loss += stage_out_dict['reg_loss'] 
            tot_cof_loss += stage_out_dict['cof_loss'] 
            tot_l1_reg_loss += stage_out_dict['l1_reg_loss']
            del stage_out_dict
        
        if self.num_class == 1:
            tot_cls_loss = tot_cof_loss.new_tensor(0.0)

        tot_reg_loss *= self.reg_loss_scale
        tot_cls_loss *= self.cls_loss_scale
        tot_cof_loss *= self.cof_loss_scale
        tot_l1_reg_loss *= self.l1_loss_scale
        tot_loss = (tot_reg_loss + tot_cls_loss + tot_cof_loss + tot_l1_reg_loss) * batch_size

        loss_dict = {'tot_loss': tot_loss, 
                    'reg_loss': tot_reg_loss, 
                    'cls_loss': tot_cls_loss, 
                    'cof_loss': tot_cof_loss, 
                    'l1_reg_loss': tot_l1_reg_loss, 
                    'num_fg': tot_num_fg, 
                    'num_gt': tot_num_gt}

        return loss_dict

    def calculate_loss_of_each_stage(self, tars, preds, grid, stride):
        """
        Args:
            preds: 某个stage的预测输出 (N, num_anchors*h*w, 85)：(N, num_anchors*28*28, 85)或(N, num_anchors*14*14, 85)或(N, num_anchors*7*7, 85) / [x, y, w, h, cof, cls1, cls2, ...]
            tars: tensor; (N, num_bbox, 6) / [x_ctr, y_ctr, w, h, class_id, img_id]
            stride: a scalar / 下采样尺度 / 取值：8 or 16 or 32
            grid: (num_anchors*h*w, 2)
        Return:

        """
        batch_tar_cls, batch_tar_box, batch_tar_cof, fg, batch_tar_box_l1 = [], [], [], [], []
        tot_num_gt, tot_num_fg = 0, 0
        # (N, num_anchors*h*w, 4)
        origin_pred_box = preds[..., :4].clone()
        # restore predictions to input scale / (N, num_anchors*h*w, 85)
        preds[..., :2] = (preds[..., :2] + grid[None, ...]) * stride
        preds[..., 2:4] = torch.exp(preds[..., 2:4]) * stride  # 这一步可能由于preds[..., 2:4]值过大，进而导致exp计算后溢出得到Nan值

        for i in range(tars.size(0)):  # each image
            tar = tars[i]  # (num_bbox, 6)
            pred = preds[i]  # (num_anchors*h*w, 85)
            valid_gt_idx = tar[:, 4] >= 0  # 有效label索引（那些没有bbox的gt对应的class值为-1）
            tot_num_gt += valid_gt_idx.sum()
            if valid_gt_idx.sum() == 0:
                tar_cls_i = tar.new_zeros((0, self.num_class))  # (0, 80)
                tar_box_i = tar.new_zeros((0, 4))  # (0, 4)
                tar_cof_i = tar.new_zeros((pred.size(0), 1))  # (num_anchors*h*w, 1)
                frontground_mask = tar.new_zeros(pred.size(0)).bool()  # (num_anchors*h*w,)
                tar_box_l1_i = tar.new_zeros((0, 4))
            else:
                tar_box_i = tar[valid_gt_idx, :4]  # (valid_num_box, 4) / [x, y, w, h]
                tar_cls_i = F.one_hot(tar[valid_gt_idx, 4].long(), num_classes=self.num_class) * self.cls_smoothness # (valid_num_box, 80)
                
                # frontground_mask: (num_anchors*h*w,) 其中为True的元素个数设为Y; is_grid_in_gtbox_and_gtctr: (valid_num_box, N)
                frontground_mask, is_grid_in_gtbox_and_gtctr = self.select_grid(tar_box_i, grid, stride)
                pred_box_i = pred[frontground_mask, :4]  # (Y, 4) / [x, y, w, h] / 提取那些满足限制条件的prediction
                # 每个gt box与所有满足条件(被选为前景)的prediction计算iou
                iou = gpu_iou(xywh2xyxy(tar_box_i), xywh2xyxy(pred_box_i))  # (valid_num_box, Y)
                # (valid_num_box, Y)
                iou_loss = -torch.log(iou + 1e-8)
                pred_cof_i = pred[frontground_mask, 4]  # (Y, )
                pred_cls_i = pred[frontground_mask, 5:]  # (Y, 80)
                # (Y, 80) & (Y, 1) -> (Y, 80)
                pred_cls_i_for_loss = torch.sigmoid(pred_cls_i) * torch.sigmoid(pred_cof_i).unsqueeze(1)
                # (Y, 80) -> (valid_num_box, Y, 80)
                pred_cls_i_for_loss = torch.sqrt(pred_cls_i_for_loss.unsqueeze(0).repeat(tar_cls_i.size(0), 1, 1))
                # (valid_num_box, 80) -> (valid_num_box, Y, 80)
                tar_cls_i_for_loss = tar_cls_i.unsqueeze(1).repeat(1, pred_cls_i.size(0), 1)
                # 每个gt cls与所有满足条件的prediction计算binary cross entropy
                cls_loss = -(tar_cls_i_for_loss * torch.log(pred_cls_i_for_loss) + (1 - tar_cls_i_for_loss) * torch.log(1 - pred_cls_i_for_loss))
                # (valid_num_box, Y, 80) -> (valid_num_box, Y)
                cls_loss = cls_loss.sum(-1)
                # (valid_num_box, Y) & (valid_num_box, Y) & (valid_num_box, Y) -> (valid_num_box, Y)
                cost = cls_loss.detach() + 3 * iou_loss.detach() + 100000 * (~is_grid_in_gtbox_and_gtctr)
                # matched_iou: (M,); matched_gt_idx: (M,)
                frontground_mask, num_fg, matched_iou, matched_gt_idx = self.simple_ota(cost, iou, frontground_mask.clone())
                tot_num_fg += num_fg
                # ================================= build target =================================
                # (valid_num_box, 80) -> (M, 80) & (M, 1) -> (M, 80) / label smoothness with iou
                tar_cls_i = tar_cls_i[matched_gt_idx] * matched_iou.unsqueeze(-1)
                # (valid_num_box, 4) -> (M, 4)
                tar_box_i = tar_box_i[matched_gt_idx]
                # (h*w,) -> (h*w, 1)
                tar_cof_i = frontground_mask.unsqueeze(-1)

                if self.use_l1:
                    # (M, 4)
                    tar_box_l1_i = self.build_l1_target(grid, stride, tar_box_i, num_fg, frontground_mask)

            batch_tar_cls.append(tar_cls_i)
            batch_tar_box.append(tar_box_i)
            batch_tar_cof.append(tar_cof_i)
            fg.append(frontground_mask)
            if self.use_l1:
                batch_tar_box_l1.append(tar_box_l1_i)

        # one stage whole batch
        batch_tar_cls = torch.cat(batch_tar_cls, 0)  # (X, 80)
        batch_tar_box = torch.cat(batch_tar_box, 0)  # (X, 4)
        batch_tar_cof = torch.cat(batch_tar_cof, 0)  # (N*h*w, 1)
        fg = torch.cat(fg, 0)  # (N*h*w) / 其中为True的元素个数为X
        if self.use_l1:
            batch_tar_box_l1 = torch.cat(batch_tar_box_l1, 0)  # (X, 4)

        # =================================== compute losses ===================================
        tot_num_fg = max(tot_num_fg, 1)

        # regression loss
        reg_loss = self.iou_loss(preds[..., :4], batch_tar_box, fg, self.hyp['iou_type'])  # regression

        # cofidence loss
        cof_loss = self.bce_cof(preds[..., 4].view(-1, 1), batch_tar_cof.type(preds.type()))  # cofidence

        # classification loss
        assert preds[..., 5:].view(-1, self.num_class)[fg].size() == batch_tar_cls.size()
        if self.hyp['use_focal_loss']:
            cls_factor = self.focal_loss_factor(preds[..., 5:].view(-1, self.num_class)[fg], batch_tar_cls)
        else:
            cls_factor = torch.ones_like(batch_tar_cls)
        cls_loss = (self.bce_cls(preds[..., 5:].view(-1, self.num_class)[fg], batch_tar_cls) * cls_factor)  # classification
        
        # l1 regression loss
        if self.use_l1:
            l1_reg_loss = self.l1_loss(origin_pred_box.view(-1, 4)[fg], batch_tar_box_l1)  # l1
        else:
            l1_reg_loss = 0.0

        out_dict = {'reg_loss': reg_loss.mean(), 
                    'cls_loss': cls_loss.mean(), 
                    'l1_reg_loss': l1_reg_loss.mean(), 
                    'cof_loss': cof_loss.mean(), 
                    'num_fg': tot_num_fg, 
                    'num_gt': tot_num_gt}

        return out_dict

    def select_grid(self, tar_box, grid, stride):
        """
        根据target box选择合适的grid（选择合适的grid即是选择合适的prediction）参与loss的计算
        Args:
            tar_box: (X, 4) / [x, y, w, h] / X -> 该image包含的有效的gt box个数
            grid: (h*w, 2) / [x, y]
            stride: scalar / downsample scale
        Returns:
            is_grid_in_gtbox_or_gtctr: (h*w,) / front ground mask / 其中为True的元素个数设为Y，则Y >= N
            is_grid_in_gtbox_and_gtctr: (valid_num_box, N)
        """
        # gt_xywh: (valid_num_box, 4) / [gt_Xctr, gt_Yctr, gt_W, gt_H]
        gt_xywh = tar_box.clone().detach()
        # =========================== 某个grid的中心点坐标是否落到某个gt box的内部 ===========================
        
        # offsets: (1, 4)
        offsets = gt_xywh.new_tensor([-1, -1, 1, 1]).unsqueeze(0) * 0.5
        # (1, 4) & (valid_num_box, 4) -> (valid_num_box, 4) / [-0.5*gt_W, -0.5*gt_H, 0.5*gt_W, 0.5*gt_H]
        wh_offsets = gt_xywh[:, 2:].repeat(1, 2) * offsets
        # gt_xyxy: (valid_num_box, 4) & (valid_num_box, 4) -> (valid_num_box, 4) / [gt_Xmin, gt_Ymin, gt_Xmax, gt_Ymax]
        gt_xyxy = gt_xywh[:, :2].repeat(1, 2) + wh_offsets
        # (valid_num_box, 4) & (1, 4) -> (valid_num_box, 4) / [-gt_Xmin, -gt_Ymin, gt_Xmax, gt_Ymax]
        gt_xyxy = gt_xyxy * gt_xyxy.new_tensor([-1, -1, 1, 1]).unsqueeze(0) 
        # ctr_grid: (h*w, 2) / [grid_Xctr, grid_Yctr]
        ctr_grid = self._make_center_grid(grid.clone(), stride)
        # (h*w, 2) -> (h*w, 4); (h*w, 4) & (1, 4) -> (h*w, 4) / [grid_Xctr, grid_Yctr, -grid_Xctr, -grid_Yctr]
        ctr_grid = ctr_grid.repeat(1, 2) * ctr_grid.new_tensor([1, 1, -1, -1]).unsqueeze(0)
        # (valid_num_box, 1, 4) & (1, h*w, 4) -> (valid_num_box, h*w, 4) / []
        box_delta = gt_xyxy.unsqueeze(1) + ctr_grid.unsqueeze(0)
        # (valid_num_box, h*w, 4) -> (valid_num_box, h*w)
        is_grid_in_gtbox = box_delta.min(dim=2).values > 0.0
        # (valid_num_box, h*w) -> (h*w,) / 对该image，所有满足该条件grid的并集
        is_grid_in_gtbox_all = is_grid_in_gtbox.sum(0) > 0.0

        # 如果grid的中心点坐标均没有在任何gt box内部，则对每个gt box而言，对每个tar选取距离最近的grid作为匹配的grid
        if is_grid_in_gtbox_all.sum() == 0:
            # (valid_num_box, 1, 2) & (1, h*w, 2) -> (valid_num_box, h*w, 2) -> (valid_num_box, h*w)
            ctr_distance = torch.norm(tar_box[:, :2].unsqueeze(1) - ctr_grid[:, :2].unsqueeze(0), dim=2)
            # (valid_num_box,)
            dist_argmin = torch.argmin(ctr_distance, dim=1)
            valid_idx = torch.unique(dist_argmin)
            choose_num = int(len(valid_idx) * 0.2) if len(valid_idx) * 0.2 > 2 else 1
            random_idx = dist_argmin[torch.randperm(len(dist_argmin))]
            is_grid_in_gtbox_all[random_idx[:choose_num]] = True

        # ======== 某个gird的中心坐标是否落到以某个gt box的中心点为圆心，center_radius为半径的圆形区域内 =========
        center_radius = 2.5
        ctr_offsets = gt_xywh.new_tensor([-1, -1, 1, 1]) * center_radius
        # (valid_num_box, 2) -> (valid_num_box, 4); [gt_Xctr, gt_Yctr, gt_Xctr, gt_Yctr]
        gt_ctr = gt_xywh[:, :2].repeat(1, 2)
        # (X, 4) & (1, 4) -> (X, 4); [gt_Xctr-radius, gt_Yctr-radius, gt_Xctr+radius, gt_Yctr+radius]
        gt_ctr_offsets = gt_ctr + ctr_offsets.unsqueeze(0)
        # (X, 4) & (1, 4) -> (X, 4); [-(gt_Xctr-radius), -(gt_Yctr-radius), gt_Xctr+radius, gt_Yctr+radius]
        gt_ctr_offsets *= gt_ctr_offsets.new_tensor([-1, -1, 1, 1]).unsqueeze(0)
        # (1, h*w, 4) & (X, 1, 4) -> (X, h*w, 4); [grid_Xctr-(gt_Xctr-radius), grid_Yctr-(gt_Yctr-radius), grid_Xctr+gt_Xctr+radius, grid_Yctr+gt_Yctr+radius]
        ctr_delta = ctr_grid.unsqueeze(0) + gt_ctr_offsets.unsqueeze(1)
        # (X, h*w, 4) -> (X, h*w)
        is_grid_in_gtctr = ctr_delta.min(dim=2).values > 0.0
        # (X, h*w) -> (h*w,) / 对该image，所有满足该条件grid的并集
        is_grid_in_gtctr_all = is_grid_in_gtctr.sum(0) > 0.0

        if is_grid_in_gtctr_all.sum() == 0:
            is_grid_in_gtctr_all = is_grid_in_gtbox_all

        # ====================================== fliter by conditions ======================================
        # (h*w,) & (h*w,) -> (h*w,)
        is_grid_in_gtbox_or_gtctr = is_grid_in_gtbox_all | is_grid_in_gtctr_all
        # (X, M) & (X, M) -> (X, N)
        is_grid_in_gtbox_and_gtctr = is_grid_in_gtbox[:, is_grid_in_gtbox_or_gtctr] & is_grid_in_gtctr[:, is_grid_in_gtbox_or_gtctr]
        return is_grid_in_gtbox_or_gtctr, is_grid_in_gtbox_and_gtctr

    def simple_ota(self, cost, iou, frontground_mask):
        """
        每个gt box都可能有好几个满足条件的（位于前景）prediction，这一步需要在其中挑选出最有价值的参与loss的计算.
        通过传入的cost和iou，对每个gt box选择与之最匹配的若干个prediction，并且使得每个prediction最多只能匹配一个gt box.

        注意：
            传入的frontground_mask被inplace的修改了。
        Args:
            frontground_mask: (h*w,) / bool / 其中为True的元素个数等于Y
            cost: (valid_num_box, Y)
            iou: (valid_num_box, Y)
        Returns:
            num_fg: 选取的prediction个数，假设为M（M的取值位于[0, Y]）
            matched_gt_cls：与每个prediction最匹配的gt class id
            matched_iou：每个prediction与之最匹配的gt box之间额iou值
            matched_gt_idx: 
        """
        assert cost.size(0) == iou.size(0)

        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # 如果位于前景的prediction个数大于10，则每个gt box最多选择10个最适合的预测作为后续的loss计算
        k = min(10, iou.size(1))  
        # (valid_num_box, k)
        topk_iou, _ = torch.topk(iou, k, dim=1)
        # (valid_num_box, k) -> (valid_num_box, )
        dynamic_k = torch.clamp(topk_iou.sum(1).int(), min=1, max=cost.size(1)).tolist()
        for i in range(cost.size(0)):  # each valid gt
            # 选取最小的dynamic_k[i]个值（因为不满足条件的prediction对应的cost加上了1000000）
            _, pos_idx = torch.topk(cost[i], k=dynamic_k[i], largest=False)
            matching_matrix[i][pos_idx] = 1

        del topk_iou, dynamic_k

        # (valid_num_box, Y) -> (Y,)  / 满足条件的prediction的并集(存在某个prediction匹配到多个gt box)
        all_matching_gt = matching_matrix.sum(0) 
        # 如果存在某个prediction匹配到多个gt box的情况
        if (all_matching_gt > 1).sum() > 0:  
            _, cost_argmin = torch.min(cost[:, all_matching_gt > 1], dim=0)
            # 处理某些prediction匹配到多个gt box的情况，将这些prediction只分配到与其匹配度最高的gt box
            matching_matrix[:, all_matching_gt > 1] = 0
            matching_matrix[cost_argmin, all_matching_gt > 1] = 1

        # (valid_num_box, Y) -> (Y,) / fg_mask总共有Y个元素
        fg_mask = matching_matrix.sum(0) > 0
        # num_fg的值应该位于区间[0, Y]
        num_fg = fg_mask.sum().item()
        # update front ground mask
        assert len(frontground_mask[frontground_mask.clone()]) == len(fg_mask)
        frontground_mask[frontground_mask.clone()] = fg_mask

        # (valid_num_box, M) -> (M,) / 假设fg_mask中为True的个数为M，则Y >= M / matched_gt的取值范围为[0, valid_num_box]
        matched_gt_idx = matching_matrix[:, fg_mask].argmax(0)
        # (valid_num_box, Y) & (valid_num_box, Y) -> (valid_num_box, Y) -> (Y,) -> (M,)
        matched_iou = (matching_matrix * iou).sum(0)[fg_mask]
        return frontground_mask, num_fg, matched_iou, matched_gt_idx

    def _make_grid(self, h, w, dtype):
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w))
        # 排列成(x, y)的形式，是因为模型输出的预测结果的排列是[x, y, w, h, cof, cls1, cls2, ...]
        grid = torch.stack((xs, ys), dim=2).view(1, 1, h, w, 2).contiguous().type(dtype)
        return grid

    def _make_center_grid(self, grid, stride):
        """
        Args:
            grid: tensor / (h*w, 2) / [xmin, ymin]
            stride: downsample scale
        Return:
            ctr_grid: (h*w, 2) / [x_center, y_center]
        """
        grid_tmp = grid * stride  # 将grid还原到原始scale
        ctr_grid = grid_tmp + 0.5 * stride  # 左上角坐标 -> 中心点坐标
        return ctr_grid

    def iou_loss(self, pred_box, tar_box, fg, iou_type='iou'):
        """
        Args:
            pred_box: (N, h*w, 4) / [x, y, w, h]
            tar_box: (X, 4) / [x, y, w, h]
            fg: (N*h*w) / 其中为True的元素个数为X
        """
        assert pred_box.size(-1) == tar_box.size(-1)
        # (X, 4)
        pred = pred_box.view(-1, 4)[fg]
        assert pred.size() == tar_box.size()
        
        inter_xy_min = torch.max((pred[:, :2] - pred[:, 2:] / 2), (tar_box[:, :2] - tar_box[:, 2:] / 2))
        inter_xy_max = torch.min((pred[:, :2] + pred[:, 2:] / 2), (tar_box[:, :2] + tar_box[:, 2:] / 2))
        mask = (inter_xy_min < inter_xy_max).type(pred.type()).prod(dim=1)

        area_pred = torch.prod(pred[:, 2:], dim=1)
        area_tar = torch.prod(tar_box[:, 2:], dim=1)
        area_inter = torch.prod(inter_xy_max - inter_xy_min, dim=1) * mask
        area_union = area_pred + area_tar - area_inter
        iou = area_inter / (area_union + 1e-16)

        if iou_type == 'iou':  # 使用iou训练，使用sgd作为优化器且lr设置稍大时，训练过程中容易出现Nan
            return 1 - iou ** 2
        elif iou_type == 'giou': 
            convex_xy_min = torch.min((pred[:, :2] - pred[:, 2:] / 2), (tar_box[:, :2] - tar_box[:, 2:] / 2))
            convex_xy_max = torch.max((pred[:, :2] + pred[:, 2:] / 2), (tar_box[:, :2] + tar_box[:, 2:] / 2))
            area_convex = torch.prod(convex_xy_max - convex_xy_min, dim=1)
            giou = iou - (area_convex - area_union) / area_convex.clamp(1e-16)
            return 1 - giou.clamp(min=-1., max=1.)
        elif iou_type == 'ciou':  # 使用ciou训练较为稳定
            # convex box's diagonal length
            c_xmin = torch.min(pred[:, 0] - pred[:, 2] / 2, tar_box[:, 0] - tar_box[:, 2] / 2)  # (N,)
            c_xmax = torch.max(pred[:, 0] + pred[:, 2] / 2, tar_box[:, 0] + tar_box[:, 2] / 2)  # (N,)
            c_ymin = torch.min(pred[:, 1] - pred[:, 3] / 2, tar_box[:, 1] - tar_box[:, 3] / 2)  # (N,)
            c_ymax = torch.max(pred[:, 1] + pred[:, 3] / 2, tar_box[:, 1] + tar_box[:, 3] / 2)  # (N,)
            c_hs = c_ymax - c_ymin  # (N,)
            c_ws = c_xmax - c_xmin  # (N,)
            c_diagonal = torch.pow(c_ws, 2) + torch.pow(c_hs, 2) + 1e-16  # (N,)
            # distance of two bbox center
            ctr_ws = pred[:, 0] - tar_box[:, 0]  # (N,)
            ctr_hs = pred[:, 1] - tar_box[:, 1]  # (N,)
            ctr_distance = torch.pow(ctr_hs, 2) + torch.pow(ctr_ws, 2)  # (N,)
            h1, w1 = pred[:, [2, 3]].T
            h2, w2 = tar_box[:, [2, 3]].T
            v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)  # (N,)
            with torch.no_grad():
                alpha = v / (1 - iou + v + 1e-16)
            ciou = iou - ctr_distance / c_diagonal - v * alpha  # (N,)
            return 1 - ciou
        else:
            raise ValueError(f"Unknow iou_type '{iou_type}', must be one of ['iou', 'giou', 'ciou']")

    def build_l1_target(self, grid, stride, tar_box, num_fg, fg):
        """
        将target转换到对应stage的prediction一致的数据格式（及将(ctr_x, ctr_y)转换为相对于对应的grid左上角的偏移量，将(w, h)转换为对应尺度下的长和宽）
        Args:
            grid: (h*w, 2)
            stride: scaler
            tar_box: (M, 4) / [x, y, w, h]
            num_fg: M
            fg: (h*w,) / 其中为True的元素个数为M
        Returns:
            tar_l1: (M, 4)
        """
        assert fg.sum() == num_fg

        tar_l1 = tar_box.new_zeros((num_fg, 4))
        tar_l1[:, 0] = tar_box[:, 0] / stride - grid[fg, 0]
        tar_l1[:, 1] = tar_box[:, 1] / stride - grid[fg, 1]
        tar_l1[:, 2] = torch.log(tar_box[:, 2] / stride + 1e-16)
        tar_l1[:, 3] = torch.log(tar_box[:, 3] / stride + 1e-16)
        return tar_l1
    
    def focal_loss_factor(self, pred, target):
        """
        compute classification loss weights
        Args:
            pred: (N, 80)
            target: (N, 80)
        Return:
            focal loss factor: (N, 80)
        """
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
