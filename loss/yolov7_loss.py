import torch
import torch.nn as nn
from utils import xyxy2xywhn, xyxy2xywh, xywh2xyxy
from utils import gpu_CIoU
from utils import gpu_iou, gpu_DIoU, gpu_Giou
import torch.nn.functional as F

def smooth_bce(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps



class YOLOV7Loss:

    def __init__(self, anchors, hyp, stage_num=3):
        """
        3种loss的重要程度依次为: confidence > classification > regression 

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
        self.positive_smooth_cls, self.negative_smooth_cls = smooth_bce()
        self.use_iou_as_tar_cof = self.hyp["use_iou_as_tar_cof"]

    def __call__(self, stage_preds, targets_batch):
        """
        通过对比preds和targets, 找到与pred对应的target。
        注意: 每一个batch的targets中的bbox_num都相同(参见cocodataset.py中fixed_imgsize_collector函数)。

        :param stage_preds: (out_small, out_mid, out_large) / [(batch_size, num_anchors, H/8, W/8, 85),(batch_size, num_anchors, H/16, W/16, 85),(batch_size, num_anchors, H/32, W/32, 85)]
        :param targets_batch: 最后一个维度上的值, 表示当前batch中该target对应的img index
        :param targets_batch: tensor / (bn, bbox_num, 6) -> [xmin, ymin, xmax, ymax, cls, img_id]
        """
        assert isinstance(stage_preds, dict)
        keys = list(stage_preds.keys())  # [small, middle, large]
        assert isinstance(targets_batch, torch.Tensor), f"targets's type should be torch.Tensor but we got {type(targets_batch)}"
        assert stage_preds[keys[0]].size(0) == targets_batch.size(0), f"the length of predictions and targets should be the same, " \
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

        cls_loss, iou_loss, cof_loss = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        # balances = [4., 1., 0.4] if len(stage_preds) == 3 else [4., 1., 0.4, 0.1]
        tot_tar_num = 0
        s = 3 / len(stage_preds)
        for i in range(len(stage_preds)):  # each stage  [pred_small, pred_middle, pred_large]
            # region ============================= yolov5 matching =============================
            fm_h, fm_w = torch.tensor(stage_preds[keys[i]].shape)[[2, 3]]
            ds_scale = self.input_img_size[1] / fm_w  # downsample scale
            # anchor: (3, 2)
            anchor = self.anchors[i]
            anchor_stage = anchor / ds_scale
            anchor_num = anchor_stage.shape[0]
            # preds: (bn, 3, h, w, 85)
            preds = stage_preds[keys[i]]
            assert preds.size(-1) == self.hyp['num_class'] + 5
            # match anchor(正样本匹配) / box, cls, img_idx, anchor_idx, grid_y, grid_x
            tar_box_from_v5, tar_cls_from_v5, img_idx_from_v5, anc_idx_from_v5, gy_from_v5, gx_from_v5 = self.match(targets, anchor_stage, (fm_w, fm_h))
            # endregion ============================= yolov5 matching =============================

            # region ============================= yolox matching =============================
            tar_box_from_x, tar_cls_from_x, img_idx_from_x, anc_idx_from_x, gy_from_x, gx_from_x = self.simple_ota(batch_size, anchor_stage, ds_scale, preds, targets_batch, img_idx_from_v5, anc_idx_from_v5, gy_from_v5, gx_from_v5)
            # endregion ============================= yolox matching =============================

            cur_tar_num = tar_box_from_x.shape[0]
            tot_tar_num += cur_tar_num

            # cur_preds: (N, 85) / [pred_x, pred_y, pred_w, pred_h, confidence, c1, c2, c3, ..., c80]
            cur_preds = preds[img_idx_from_x, anc_idx_from_x, gy_from_x, gx_from_x]

            # region ====================================== compute loss ======================================
            # Classification
            # 只有正样本才参与分类损失的计算
            if self.hyp['num_class'] > 1:  # if only one class then we don't compute class loss
                # t_cls: (N, 80)
                t_cls = torch.full_like(cur_preds[:, 5:], fill_value=self.negative_smooth_cls)
                t_cls[torch.arange(tar_cls_from_x.size(0)), tar_cls_from_x.long()] = self.positive_smooth_cls

                if self.hyp['use_focal_loss']:
                    cls_factor = self.focal_loss_factor(cur_preds[:, 5:], t_cls)
                else:
                    cls_factor = torch.ones_like(t_cls)
                
                cls_loss += (self.bce_cls(cur_preds[:, 5:], t_cls) * cls_factor).mean()

            # Confidence and Regression
            # 只有正样本才参与回归损失的计算
            t_cof = torch.zeros_like(preds[..., 4])
            if cur_tar_num > 0:
                # sigmoid(-5) ≈ 0; sigmoid(0) = 0.5; sigmoid(5) ≈ 1
                # sigmoid(-5) * 2 - 0.5 = -0.5; sigmoid(0) * 2 - 0.5 = 0.5; sigmoid(5) * 2 - 0.5 = 1.5
                pred_xy = cur_preds[:, :2].sigmoid() * 2. - 0.5
                # (N, 2) & (N, 2) -> (N, 2)
                pred_wh = (cur_preds[:, 2:4].sigmoid() * 2.) ** 2 * anchor_stage[anc_idx_from_x]
                # pred_box: (N, 4)
                pred_box = torch.cat((pred_xy, pred_wh), dim=1).to(self.device)
                # because pred_box and tar_box's format is xywh, before compute iou loss we should turn it to xyxy format
                pred_box, tar_box = xywh2xyxy(pred_box), xywh2xyxy(tar_box_from_x)
                # iou: (N,)
                iou = gpu_CIoU(pred_box, tar_box)
                iou_loss += (1.0 - iou).mean()
                # t_cof: (bn, 3, h, w) / 所有grid均参与confidence loss的计算
                if self.use_iou_as_tar_cof:
                    t_cof[img_idx_from_x, anc_idx_from_x, gy_from_x, gx_from_x] = iou.detach().clamp(0).type_as(t_cof)
                else:
                    t_cof[img_idx_from_x, anc_idx_from_x, gy_from_x, gx_from_x] = 1.0

            if self.hyp['use_focal_loss']:
                cof_factor = self.focal_loss_factor(preds[..., 4], t_cof)
            else:
                cof_factor = torch.ones_like(t_cof)
            # endregion ====================================== compute loss ======================================

            # 所有样本均参与置信度损失的计算 / 在3种loss中confidence loss是最为重要的
            cof_loss_tmp = (self.bce_cof(preds[..., 4], t_cof) * cof_factor).mean()
            cof_loss_tmp *= self.balances[i]
            self.balances[i] = self.balances[i] * 0.9999 + 0.0001 / cof_loss_tmp.detach().item()
            cof_loss += cof_loss_tmp

        self.balances = [x/self.balances[1] for x in self.balances]
        iou_loss *= self.hyp['iou_loss_scale'] * s
        cof_loss *= self.hyp['cof_loss_scale'] * s * (1. if len(stage_preds) == 3 else 1.4)
        cls_loss *= self.hyp['cls_loss_scale'] * s
        tot_loss = (iou_loss + cof_loss + cls_loss) * batch_size

        loss_dict = {
            'tot_loss': tot_loss, 
            'iou_loss': iou_loss.detach().item() * batch_size, 
            'cof_loss': cof_loss.detach().item() * batch_size, 
            'cls_loss': cls_loss.detach().item() * batch_size, 
            'tar_nums': tot_tar_num,  
        }
        return loss_dict

    def match(self, targets, anchor_stage, fm_shape):
        """
        正样本分配策略(根据anchor与targets的匹配关系进行过滤)。

        并不是传入的所有target都可以参与最终loss的计算, 只有那些与anchor的width/height ratio满足一定条件的targe才有资格;
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
        # ar_mask: (3, bn, bbox_num) 为target选取符合条件的anchor
        ar_mask = torch.max(ratio, 1/ratio).max(dim=-1)[0] < self.hyp['anchor_match_thr']  # anchor ratio mask
        # targets: (3, bn, bbox_num, 7) -> (X, 7)
        t_stage = t_stage[ar_mask]
        t_stage = t_stage[t_stage[:, 4] >= 0]

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
        # grid_xys_expand: (1, X, 2) & (5, 1, 2) -> (5, X, 2) & (5, X) -> (N, 2)
        grid_xys_expand = torch.zeros_like(grid_xys)[None] + offset[:, None, :]
        grid_xys_expand = grid_xys_expand[grid_mask]
        # tar_grid_xys在对应特征图尺寸中的xy
        tar_grid_xys = t_stage[:, [0, 1]]  # (N, 2)
        # 放宽obj预测的中心坐标精度的限制, 在真实grid_center_xy范围内浮动一个单位的长度, 均认为是预测正确；tar_grid_coors表示obj所在grid的xy坐标
        tar_grid_coors = (tar_grid_xys - grid_xys_expand).long()  # (N, 2)
        # tar_grid_off:相对于所在grid的偏移量
        tar_grid_offset = tar_grid_xys - tar_grid_coors
        tar_grid_whs = t_stage[:, [2, 3]]  # (N, 2)
        # tar_box: (N, 2) & (N, 2) -> (N, 4) / (x, y, w, h)
        tar_box = torch.cat((tar_grid_offset, tar_grid_whs), dim=-1)  # (grid_off_x, grid_off_y, w_stage, h_stage)
        # tar_cls: (N, )
        tar_cls = t_stage[:, 4]
        # tar_img_idx: (N, )
        tar_img_idx = t_stage[:, 5]  # 一个batch中的img id
        # tar_anc_idx:  （N,)
        tar_anc_idx = t_stage[:, 6]  # anchor id
        # tar_grid_i: (N, ) / row index; tar_grid_j: (N, ) / cloumn index
        tar_grid_x, tar_grid_y = tar_grid_coors.T
        tar_grid_x = torch.clamp(tar_grid_x, 0, fm_shape[0]-1)
        tar_grid_y = torch.clamp(tar_grid_y, 0, fm_shape[1]-1)

        del g, offset, t_stage
        return tar_box, tar_cls.long(), tar_img_idx.long(), tar_anc_idx.long(), tar_grid_y.long(), tar_grid_x.long()

    def simple_ota(self, batch_size, anchor_stage, ds_scale, one_stage_pred, org_tar, img_idx, anc_idx, gy, gx):
        """
        正负样本匹配策略

        Args:
            batch_size:
            anchor_stage:
            one_stage_pred: (bn, anchor_num, h, w, 85)
            org_tar: (bn, bbox_num, 6); [x, y, w, h, cls, img_id]
            tar_cls: (N,)
            img_idx: (N,)
            anc_idx: (N,)
            gy: (N,)
            gx: (N,)

        Returns:
            all_matched_pred:
            all_matched_tar:


        """

       # restore xyxyn to xyxy of original image scale
        tar_box = org_tar[:, :, :4]
        tar_cls = org_tar[:, :, 4]  # (bn, bbox_num)
        # pred: (N, 85) / [pred_x, pred_y, pred_w, pred_h, confidence, c1, c2, c3, ..., c80]
        pred = one_stage_pred[img_idx, anc_idx, gy, gx]
        # sigmoid(-5) ≈ 0; sigmoid(0) = 0.5; sigmoid(5) ≈ 1
        # sigmoid(-5) * 2 - 0.5 = -0.5; sigmoid(0) * 2 - 0.5 = 0.5; sigmoid(5) * 2 - 0.5 = 1.5
        grid_xy = torch.stack((gx, gy), dim=1)  # (N, 2)
        # restore prediction to original image scale
        pred_xy = ((pred[:, :2].sigmoid() * 2. - 0.5) + grid_xy) * ds_scale
        # (N, 2) & (N, 2) -> (N, 2) / restore prediction to original image scale
        pred_wh = (pred[:, 2:4].sigmoid() * 2.) ** 2 * anchor_stage[anc_idx] * ds_scale
        # pred_box: (N, 4) / tar_box: (N, 4)
        pred_box = torch.cat((pred_xy.contiguous(), pred_wh.contiguous()), dim=1).to(self.device)
        # because pred_box and tar_box's format is xywh, before compute iou loss we should turn it to xyxy format
        pred_box = xywh2xyxy(pred_box)  # tar_box: (bn, bbox_num, 4)
        fm_h, fm_w = one_stage_pred.size(2), one_stage_pred.size(3)

        matched_tar_box, matched_tar_cls, matched_img_idx, matched_anc_idx, matched_gy, matched_gx = [], [], [], [], [], []
        for b in range(batch_size):  # one image one stage prediction
            valid_tar_idx = org_tar[b, :, 4] >= 0  # (bbox_num,)
            i = img_idx == b  # (X,)
            matched_img_idx.append(img_idx[i])
            matched_anc_idx.append(anc_idx[i])
            matched_gy.append(gy[i])
            matched_gx.append(gx[i])

            this_tar_box = tar_box[b][valid_tar_idx]  # (Xt, 4)
            this_tar_cls = tar_cls[b][valid_tar_idx]  # (Xt,)
            num_tar = len(this_tar_box)

            matched_tar_box.append(tar_box[b][valid_tar_idx])
            matched_tar_cls.append(tar_cls[b][valid_tar_idx])

            this_pred_box = pred_box[i]  # (Xp, 4)
            this_pred_cof = pred[i][:, 4]  # (Xp,)
            this_pred_cls = pred[i][:, 5:]  # (Xp, class_num)

            pairwise_iou = gpu_iou(this_tar_box, this_pred_box)  # (Xt, Xp)
            pairwise_neg_iou_loss = -torch.log(pairwise_iou + 1e-8)  # (Xt, Xp)
            # 一个pred最多可以匹配target的个数
            topk_iou_loss, _ = torch.topk(pairwise_neg_iou_loss, min(10, pairwise_neg_iou_loss.shape[1]), dim=1)
            dynamick = torch.clamp(topk_iou_loss.sum(1).int(), min=1, max=topk_iou_loss.size(1))  # (Xt,) / 为每个target分配dynamic个prediction
            if self.hyp["num_class"] > 1: 
                # (Xt,) -> (Xt, 80) -> (Xt, 1, 80) -> (Xt, Xp, 80)
                this_tar_onehot_cls = F.one_hot(this_tar_cls.long(), self.hyp['num_class']).float().unsqueeze(1).repeat(1, i.sum().int(), 1)
                # (Xp, 80) -> (1, Xp, 80) -> (Xt, Xp, 80)
                this_pred_pairwise_cls = this_pred_cls.float().unsqueeze(0).repeat(num_tar, 1, 1).sigmoid()  # (Xt, Xp, 80)
                # (Xt, Xp, 80) & (Xt, Xp, 1) -> (Xt, Xp, 80)
                this_pred_pairwise_cls *= this_pred_cof.float()[None, :, None].repeat(num_tar, 1, 1).sigmoid()  # (Xt, Xp, 80)
                # sqrt操作会放大预测的confidence值
                this_pred_pairwise_cls = this_pred_pairwise_cls.sqrt()  # (Xt, Xp, 80)
                # torch.log(this_pred_pairwise_cls / (1 - this_pred_pairwise_cls)): 将小于0.5的预测值进一步缩小，将大于0.5的预测值进一步放大
                pairwise_cls_loss = F.binary_cross_entropy_with_logits(torch.log(this_pred_pairwise_cls / (1 - this_pred_pairwise_cls)), this_tar_onehot_cls, reduction='none').sum(dim=-1)  # (X, M)
            else:
                # (Xt, Xp)
                pairwise_cls_loss = torch.zeros_like(pairwise_neg_iou_loss)
            cost = 3 * pairwise_neg_iou_loss + pairwise_cls_loss  # (Xt, Xp)

            matching_matrix = torch.zeros_like(cost)  # (Xt, Xp)
            for ti in range(num_tar):
                pos_idx = torch.topk(cost[ti], k=dynamick[ti].item(), largest=False)[1]
                # 给每个target匹配若干的prediction(至少一个)
                matching_matrix[ti][pos_idx] = 1.0

            # 当一个prediction有多个target与之匹配时，选择cost最小的那个target(一个target可以有多个prediction与之对应，但一个prediction至多只能有一个target与之对应)
            select_matching_tar = matching_matrix.sum(dim=0)
            if (select_matching_tar > 1).sum() > 0:
                cost_min_idx = torch.min(cost[:, select_matching_tar > 1], dim=0)[1]
                matching_matrix[:, select_matching_tar > 1] *= 0.0
                matching_matrix[cost_min_idx, select_matching_tar > 1] = 1.0
                fg_pred_idx = matching_matrix.sum(dim=0) > 0.0  # (Xp,) / 被选为正样本的prediction的index / 假设fg_pred_idx中为True的元素个数为M
                # (Xt, Xp) -> (Xt, Xp) -> (M,) / M <= Xp, 给每个prediction匹配一个target
                matched_tar_idx = matching_matrix[:, fg_pred_idx].argmax(dim=0)
                assert fg_pred_idx.sum() == len(matched_tar_idx)

                matched_img_idx[b] = matched_img_idx[b][fg_pred_idx]  # (M,)
                matched_anc_idx[b] = matched_anc_idx[b][fg_pred_idx]  # (M,)
                matched_gy[b] = matched_gy[b][fg_pred_idx]  # (M,)
                matched_gx[b] = matched_gx[b][fg_pred_idx]  # (M,)
                
                tmp_grid_xy = torch.stack([matched_gx[b], matched_gy[b]], dim=1)  # (M, 2)
                tmp_tbox = matched_tar_box[b][matched_tar_idx]  # (M, 4)
                tmp_tbox = xyxy2xywhn(tmp_tbox, self.input_img_size)  # (M, 4)
                tmp_tbox *= tmp_tbox.new_tensor([[fm_w, fm_h, fm_w, fm_h]])  # to stage scale
                tmp_tbox[:, :2] = tmp_tbox[:, :2] - tmp_grid_xy 
                matched_tar_box[b] = tmp_tbox  # (M, 4)
                matched_tar_cls[b] = matched_tar_cls[b][matched_tar_idx]  # (M,)
            else:
                matched_img_idx[b] = torch.tensor([], device=self.hyp['device'], dtype=torch.float)
                matched_anc_idx[b] = torch.tensor([], device=self.hyp['device'], dtype=torch.float)
                matched_tar_box[b] = torch.tensor([], device=self.hyp['device'], dtype=torch.float)
                matched_tar_cls[b] = torch.tensor([], device=self.hyp['device'], dtype=torch.float)
                matched_gy[b] = torch.tensor([], device=self.hyp['device'], dtype=torch.float)
                matched_gx[b] = torch.tensor([], device=self.hyp['device'], dtype=torch.float)
        
        matched_img_idx_out = torch.cat(matched_img_idx, dim=0)  # (Y,)
        matched_anc_idx_out = torch.cat(matched_anc_idx, dim=0)  # (Y,)
        matched_tar_box_out = torch.cat(matched_tar_box, dim=0)  # (Y, 4)
        matched_tar_cls_out = torch.cat(matched_tar_cls, dim=0)  # (Y,)
        matched_gy_out = torch.cat(matched_gy, dim=0)  # (Y,)
        matched_gx_out = torch.cat(matched_gx, dim=0)  # (Y,)

        return matched_tar_box_out, matched_tar_cls_out, matched_img_idx_out.long(), matched_anc_idx_out.long(), matched_gy_out.long(), matched_gx_out.long() 

    def focal_loss_factor(self, pred, target):
        """
        Args:
            pred: (N, 80)
            target: (N, 80)
        Return:
            focal loss factor: (N, 80)
        """
        prob = torch.sigmoid(pred)
        # target * prob: 将正样本预测正确的概率； (1.0 - target) * (1.0 - prob): 将负样本预测正确的概率
        acc_scale = target * prob + (1.0 - target) * (1.0 - prob)
        # 对那些预测错误程度越大的预测加大惩罚力度
        gamma = self.hyp.get('focal_loss_gamma', 1.5)
        gamma_factor = (1.0 - acc_scale) ** gamma
        # 当alpha值小于0.5时, 意味着更加关注将负类样本预测错误的情况
        alpha = self.hyp.get('focal_loss_alpha', 0.25)
        alpha_factor = target * alpha + (1.0 - target) * (1.0 - alpha)
        factor = gamma_factor * alpha_factor

        return factor
