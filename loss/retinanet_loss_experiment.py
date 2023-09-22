import torch
import numpy as np
from utils import xyxy2xywh, xywh2xyxy
from utils import gpu_iou, gpu_DIoU, gpu_Giou, gpu_CIoU
from utils import GPUAnchor

__all__ = ['RetinaNetLossExperiment']
class RetinaNetLossExperiment:

    def __init__(self, hyp):

        self.device = hyp['device']
        self.num_class = hyp['num_class']
        self.pos_iou_thresh = hyp["positive_iou_thr"]
        self.neg_iou_thresh = hyp["negative_iou_thr"]
        self.iou_type = hyp['iou_type']
        self.l1_loss_scale  = hyp['l1_loss_scale']
        self.cof_loss_scale = hyp['cof_loss_scale']
        self.cls_loss_scale = hyp['cls_loss_scale']
        assert self.pos_iou_thresh > self.neg_iou_thresh, f"pos_iou_thresh should greater than neg_iou_thresh"
        self.alpha = hyp["alpha"]
        self.gamma = hyp["gamma"]
        self.delta_scales = hyp["tar_box_scale_factor"]
        self.L1Loss = torch.nn.L1Loss(reduction='none')
        self.bce_cls = torch.nn.BCEWithLogitsLoss(reduction='none').to(self.device)
        self.bce_cof = torch.nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
        self.anchors = None
        if not hyp['mutil_scale_training']:
            self.anchors = GPUAnchor(hyp['input_img_size'])()

    def iou(self, gt_boxes, anchors):
        """
        :param gt_boxes: [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor / shape: (M, 4)
        :param anchors: [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor / shape: (N, 4)
        :return: shape: (M, N)
        """
        assert isinstance(gt_boxes, torch.Tensor)
        assert isinstance(anchors, torch.Tensor)
        assert (gt_boxes[:, [2, 3]] >= gt_boxes[:, [0, 1]]).bool().all()
        assert (anchors[:, [2, 3]] >= anchors[:, [0, 1]]).bool().all()
        assert gt_boxes.shape[-1] == anchors.shape[-1] == 4

        gt_area = torch.prod(gt_boxes[:, [2, 3]] - gt_boxes[:, [0, 1]], dim=-1)
        anchor_area = torch.prod(anchors[:, [2, 3]] - anchors[:, [0, 1]], dim=-1)

        intersection_ymax = torch.min(anchors[:, 3, None], gt_boxes[:, 3])
        intersection_xmax = torch.min(anchors[:, 2, None], gt_boxes[:, 2])
        intersection_ymin = torch.max(anchors[:, 1, None], gt_boxes[:, 1])
        intersection_xmin = torch.max(anchors[:, 0, None], gt_boxes[:, 0])

        intersection_w = (intersection_xmax - intersection_xmin).clamp(0)
        intersection_h = (intersection_ymax - intersection_ymin).clamp(0)
        intersection_area = intersection_w * intersection_h
        union_area = anchor_area[:, None] + gt_area - intersection_area + 1e-8
        iou = intersection_area / union_area
        del gt_area, anchor_area, union_area, intersection_area
        return iou

    # region
    def __call__(self, imgs, regression, classfication, annonations):
        """
        compute focal loss.
        :param regression: (b, (h/8xw/8+h/16xw/16+h/32xw/32+h/64xw/64+h/128xw/128)*9, 5)
        :param classfication: (b, (h/8xw/8+h/16xw/16+h/32xw/32+h/64xw/64+h/128xw/128)*9, 80)
        :param annonations: (b, M, 5) / [xmin, ymin, xmax, ymax, cls] / targets / ground truth
        :return:
        """
        batch_size, c, h, w = imgs.size()
        # anchors's format is [xmin, ymin, xmax, ymax]
        anchors = self.anchors if self.anchors is not None else GPUAnchor([h, w])()

        assert regression.size(1) == classfication.size(1), f'regression.size(1)={regression.size(1)} and classfication.size(1)={classfication.size(1)}'
        assert anchors.size(0) == regression.size(1), f'anchor.size(0)={anchors.size(0)} and regression.size(1)={regression.size(1)}'

        l1_losses, cls_losses, cof_losses = [], [], []

        for b in range(batch_size):  # each image
            gt_ann = annonations[b]
            gt_ann = gt_ann[gt_ann[:, 4] >= 0]

            if len(gt_ann) == 0:  # 如果传入的数据不包含标注的object(只计算分类损失)
                alpha_factor = 1 - torch.ones_like(classfication[b]) * self.alpha
                focal_weight = alpha_factor * torch.pow(classfication[b], self.gamma) 
                target_cls = torch.zeros_like(classfication[b])
                cls_loss = self.bce_cls(classfication[b].float(), target_cls.float())
                cls_loss *= focal_weight
                cls_losses.append(cls_loss.sum())
                l1_losses.append(torch.tensor(0., dtype=alpha_factor.dtype, device=self.device))
                cof_losses.append(torch.tensor(0., dtype=alpha_factor.dtype, device=self.device))
                continue

            # fliter predictions by iou
            # anchor_gt_iou: (X, anchor_num)
            anchor_gt_iou = self.iou(gt_ann[:, :4], anchors)
            assert anchor_gt_iou.size() == torch.Size([anchors.size(0), gt_ann.size(0)])
            # iou_max: (X,) / iou_argmax: (X,)
            iou_max, iou_argmax = torch.max(anchor_gt_iou, dim=-1)
            # positive_indices: (num_anchor,)
            positive_indices = iou_max.ge(self.pos_iou_thresh)
            negative_indices = iou_max.lt(self.neg_iou_thresh)
            num_positive_anchors = positive_indices.sum()
            target_ann = gt_ann[iou_argmax, :]

            pred_cls = torch.clamp(classfication[b], 1e-3, 1.0 - 1e-3)
            # compute classification loss
            # target_cls: (num_anchor, 80)
            target_cls = torch.ones_like(pred_cls) * -1.0 # 既不在正样本集合也不在负样本集合的预测框不计入cls loss的计算
            target_cls[negative_indices, :] = 0.0  # 负样本的cls最好都预测为0，因为负样本表示当前位置没有目标，也就不要强行给出一个类别了
            target_cls[positive_indices, :] = 0.0
            target_cls[positive_indices, target_ann[positive_indices, 4].long()] = 1.

            alpha_factor = torch.where(target_cls > 0., self.alpha, 1-self.alpha)  # 对有目标的预测框减轻惩罚，无目标的预测框加大惩罚
            focal_weight = torch.where(target_cls > 0., 1-pred_cls, pred_cls)
            focal_weight = torch.pow(focal_weight, self.gamma) * alpha_factor
            # bce: (num_anchor, 80)
            cls_loss = self.bce_cls(classfication[b].float(), target_cls.float())
            # bce = -(target_cls * torch.log(classfication[b].float()) + (1 - target_cls) * torch.log(1 - classfication[b].float()))
            assert cls_loss.size() == torch.Size([anchors.size(0), self.num_class])
            # 对那些分类错误程度越严重的给予越高的惩罚因子
            cls_loss *= focal_weight
            # 忽略那些既不属于positive也不属于negative的anchor对应的cls loss
            cls_loss = torch.where(target_cls < 0.0, cls_loss.new_zeros(1), cls_loss)
            cls_loss = torch.div(cls_loss.sum(), torch.clamp(num_positive_anchors.float(), min=1.))
            cls_losses.append(cls_loss)

            pred_cof = regression[b, :, -1]  # (num_anchor,)
            target_cof = regression[b].new_zeros(regression[b].shape[0]).float()  # (num_anchor,)

            # compute regression loss
            if num_positive_anchors > 0:
                # build targets
                keep_anchors = anchors[positive_indices, :].float()
                keep_targets = target_ann[positive_indices, :4].float()

                xywh_anchor = xyxy2xywh(keep_anchors)
                anchor_ctr_x = xywh_anchor[:, 0]
                anchor_ctr_y = xywh_anchor[:, 1]
                anchor_w = xywh_anchor[:, 2]
                anchor_h = xywh_anchor[:, 3]

                xywh_gt = xyxy2xywh(keep_targets)
                gt_ctr_x = xywh_gt[:, 0]
                gt_ctr_y = xywh_gt[:, 1]
                gt_w = xywh_gt[:, 2].clamp(min=1.)
                gt_h = xywh_gt[:, 3].clamp(min=1.)

                tars_dx = torch.div((gt_ctr_x - anchor_ctr_x), anchor_w)
                tars_dy = torch.div((gt_ctr_y - anchor_ctr_y), anchor_h)
                tars_dw = torch.log(torch.div(gt_w, anchor_w).float())
                tars_dh = torch.log(torch.div(gt_h, anchor_h).float())
                tars_box = torch.stack([tars_dx, tars_dy, tars_dw, tars_dh]).T  # (N, 4)
                # 将target_dx和target_dy放大10倍，将target_dw和target_dh放大5倍
                tars_box = torch.div(tars_box, torch.tensor(self.delta_scales, device=self.device))
                
                # regression loss
                keep_preds = regression[b][positive_indices, :4].float()
                l1_loss = self.compute_l1_loss(keep_preds, tars_box)
                l1_losses.append(l1_loss.mean())
                
                # confidence loss
                if self.cof_loss_scale > 0.0:
                    iou_loss = self.compute_iou_loss(keep_preds, tars_box, self.iou_type)
                    target_cof[positive_indices] = iou_loss
            else:
                l1_losses.append(torch.tensor(0., dtype=gt_ann.dtype, device=self.device))
            
            # compute confidence loss
            cof_losses.append(self.bce_cof(pred_cof, target_cof))

        tot_l1_loss  = torch.stack(l1_losses).mean()  * self.l1_loss_scale
        tot_cof_loss = torch.tensor(0., dtype=alpha_factor.dtype, device=self.device)
        if self.cof_loss_scale > 0.0:
            tot_cof_loss = torch.stack(cof_losses).mean() * self.cof_loss_scale
        tot_cls_loss = torch.stack(cls_losses).mean() * self.cls_loss_scale

        common_scale = 1
        tot_loss = (tot_l1_loss + tot_cof_loss + tot_cls_loss) * common_scale
        del gt_ann, anchor_gt_iou, iou_max, iou_argmax, positive_indices, negative_indices, l1_losses, cof_losses, cls_losses

        return {"l1_loss": tot_l1_loss.detach().item() * common_scale, 
                "cls_loss": tot_cls_loss.detach().item() * common_scale, 
                'cof_loss': tot_cof_loss.detach().item() * common_scale, 
                'tot_loss': tot_loss, 
                "tar_nums": num_positive_anchors}
    # endregion

    def compute_l1_loss(self, preds_box, tars_box):
        """
        Args:
            preds: (N, 4) / [x, y, w, h]
            tars: (N, 4) / [x, y, w, h]
        Returns:
            l1_reg_loss: (N, 4)
        """
        l1_loss = self.L1Loss(preds_box, tars_box)
        # 当l1_loss的值大于1/9时对应的reg_loss不小于1/18，当l1_loss的值小于1/9时对应的reg_loss不大于1/18
        l1_loss = torch.where(l1_loss <= 1/9, 0.5 * 9 * torch.pow(l1_loss, 2), l1_loss - 0.5/9)
        return l1_loss
        
    def compute_iou_loss(self, preds_box, tars_box, iou_type):
        """
        Args:
            preds: (N, 4) / [x, y, w, h]
            tars: (N, 4) / [x, y, w, h]
            iou_type: string / 'iou' or 'giou' or 'ciou'
        Returns:
            iou: (N,)
        """
        inter_xy_min = torch.max((preds_box[:, :2] - preds_box[:, 2:] / 2), (tars_box[:, :2] - tars_box[:, 2:] / 2))
        inter_xy_max = torch.min((preds_box[:, :2] + preds_box[:, 2:] / 2), (tars_box[:, :2] + tars_box[:, 2:] / 2))
        mask = (inter_xy_min < inter_xy_max).type(preds_box.type()).prod(dim=1)

        area_pred = torch.prod(preds_box[:, 2:], dim=1)
        area_tar = torch.prod(tars_box[:, 2:], dim=1)
        area_inter = torch.prod(inter_xy_max - inter_xy_min, dim=1) * mask
        area_union = area_pred + area_tar - area_inter
        iou = area_inter / (area_union + 1e-8)

        if iou_type == 'iou':  # 使用iou训练，使用sgd作为优化器且lr设置稍大时，训练过程中容易出现Nan
            return 1 - iou ** 2
        elif iou_type == 'giou': 
            convex_xy_min = torch.min((preds_box[:, :2] - preds_box[:, 2:] / 2), (tars_box[:, :2] - tars_box[:, 2:] / 2))
            convex_xy_max = torch.max((preds_box[:, :2] + preds_box[:, 2:] / 2), (tars_box[:, :2] + tars_box[:, 2:] / 2))
            area_convex = torch.prod(convex_xy_max - convex_xy_min, dim=1)
            giou = iou - (area_convex - area_union) / area_convex.clamp(1e-8)
            return 1 - giou.clamp(min=-1., max=1.)
        elif iou_type == 'ciou':  # 使用ciou训练较为稳定
            # convex box's diagonal length
            c_xmin = torch.min(preds_box[:, 0] - preds_box[:, 2] / 2, tars_box[:, 0] - tars_box[:, 2] / 2)  # (N,)
            c_xmax = torch.max(preds_box[:, 0] + preds_box[:, 2] / 2, tars_box[:, 0] + tars_box[:, 2] / 2)  # (N,)
            c_ymin = torch.min(preds_box[:, 1] - preds_box[:, 3] / 2, tars_box[:, 1] - tars_box[:, 3] / 2)  # (N,)
            c_ymax = torch.max(preds_box[:, 1] + preds_box[:, 3] / 2, tars_box[:, 1] + tars_box[:, 3] / 2)  # (N,)
            c_hs = c_ymax - c_ymin  # (N,)
            c_ws = c_xmax - c_xmin  # (N,)
            c_diagonal = torch.pow(c_ws, 2) + torch.pow(c_hs, 2) + 1e-8  # (N,)
            # distance of two bbox center
            ctr_ws = preds_box[:, 0] - tars_box[:, 0]  # (N,)
            ctr_hs = preds_box[:, 1] - tars_box[:, 1]  # (N,)
            ctr_distance = torch.pow(ctr_hs, 2) + torch.pow(ctr_ws, 2)  # (N,)
            h1, w1 = preds_box[:, [2, 3]].T
            h2, w2 = tars_box[:, [2, 3]].T
            v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)  # (N,)
            with torch.no_grad():
                alpha = v / (1 - iou + v + 1e-8)
            ciou = iou - ctr_distance / c_diagonal - v * alpha  # (N,)
            return 1 - ciou
        else:
            raise ValueError(f"Unknow iou_type '{iou_type}', must be one of ['iou', 'giou', 'ciou']")
        