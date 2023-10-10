import torch
import torch.nn as nn
from utils import xyxy2xywhn, xyxy2xywh, xywh2xyxy, reduce_mean, gather, get_local_rank
from utils import gpu_CIoU
from utils import gpu_iou, gpu_DIoU, gpu_Giou
from torch.nn import functional as F
import math

__all__ = ['FCOSLoss']

INF = 10000000.0


def smooth_bce(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class FCOSLoss:

    def __init__(self, hyp, stage_num=5):
        """

        :param loss_hyp: dict
        :param data_hyp: dict
        :param train_hyp: dict
        :param stage_num: int
        """
        self.hyp = hyp
        self.device = hyp['device']
        cls_pos_weight = torch.tensor(hyp["cls_pos_weight"], device=self.device)
        ctr_pos_weight = torch.tensor(hyp['ctr_pos_weight'], device=self.device)
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight, reduction='none').to(self.device)
        self.bce_ctr = nn.BCEWithLogitsLoss(pos_weight=ctr_pos_weight, reduction='none').to(self.device)
        self.l1_reg = nn.L1Loss()
        self.input_img_size = hyp['input_img_size']
        self.num_stage = stage_num
        self.positive_smooth_cls, self.negative_smooth_cls = smooth_bce(self.hyp['class_smooth_factor'])
        self.radius = hyp['center_sampling_radius']
        self.grids = None
        self.eps = self.hyp['eps']

    def compute_iou_loss(self, pred: torch.FloatTensor, tar: torch.FloatTensor, weight=None):
        """
        Inputs:
            pred: (m, 4) / [l, t, r, b]
            tar: (m, 4) / [l, t, r, b]
            weight: (m,)
        """
        assert pred.size(0) == tar.size(0)
        if weight is not None:
            assert pred.size(0) == len(weight)

        pred_l, pred_t, pred_r, pred_b = pred.chunk(4, -1)  # (m, 1)
        tar_l, tar_t, tar_r, tar_b = tar.chunk(4, -1)  # (m, 1)
        tar_area = (tar_l + tar_r) * (tar_t + tar_b)  # (m, 1)
        pred_area = (pred_l + pred_r) * (pred_t + pred_b)    # (m, 1)

        w_inter = torch.min(pred_l, tar_l) + torch.min(pred_r, tar_r)    # (m, 1)
        g_w_inter = torch.max(pred_l, tar_l) + torch.max(pred_r, tar_r)  # (m, 1)
        h_inter = torch.min(pred_b, tar_b) + torch.min(pred_t, tar_t)    # (m, 1)
        g_h_inter = torch.max(pred_b, tar_b) + torch.max(pred_t, tar_t)  # (m, 1)
        ac_uion = (g_w_inter * g_h_inter).clamp(self.eps)  # (m, 1)
        area_inter = w_inter.clamp_(0.0) * h_inter.clamp_(0.0)  # (m, 1)
        area_union = (tar_area + pred_area.clamp(0.0) - area_inter).clamp(self.eps)  # (m, 1)
        ious = area_inter / area_union  # (m, 1)
        gious = ious - (ac_uion - area_union) / ac_uion  # (m, 1)

        if self.hyp['iou_type'] == 'iou':
            losses = -torch.log(ious.clamp(self.eps))
        elif self.hyp['iou_type'] == 'linear_iou':
            losses = 1 - ious
        elif self.hyp['iou_type'] == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None :
            return (losses * weight).sum() / weight.sum()
        else:
            return losses.sum()

    def __call__(self, cls_fms, reg_fms, cen_fms, targets_batch):
        """
        Inputs:
            targets_batch: tensor / (bn, bbox_num, 6) -> [xmin, ymin, xmax, ymax, cls, img_id]
            cls_fms: [(b, num_class, h/8, w/8), (b, num_class, h/16, w/16), (b, num_class, h/32, w/32), (b, num_class, h/64, w/64), (b, num_class, h/128, w/128)]
            reg_fms: [(b, 4, h/8, w/8), (b, 4, h/16, w/16), (b, 4, h/32, w/32), (b, 4, h/64, w/64), (b, 4, h/128, w/128)]
            ctr_fms: [(b, 1, h/8, w/8), (b, 1, h/16, w/16), (b, 1, h/32, w/32), (b, 1, h/64, w/64), (b, 4, h/128, w/128)]
        """
        assert len(cls_fms) == len(reg_fms) and len(reg_fms) == len(cen_fms)
        assert isinstance(targets_batch, torch.Tensor), f"targets's type should be torch.Tensor but we got {type(targets_batch)}"

        batch_size = targets_batch.size(0)
        targets = targets_batch.clone().detach()

        fm_shapes  = [(f.size(2), f.size(3)) for f in cls_fms]
        self.grids = self.make_grid(fm_shapes) if self.grids is None else self.grids  # [x, y]

        # [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]
        self.object_sizes_of_interest = self.get_regression_range_of_each_level(fm_shapes)

        tot_reg_loss, tot_cls_loss, tot_ctr_loss = [], [], []
        target_num = 0
        for s in range(len(cls_fms)):  # each stage
            stage_reg_loss, stage_cls_loss, stage_ctr_loss = [], [], []
            grid = self.grids[s]  # (h, w, 2)
            for b in range(batch_size):  # each image
                pred_cls = cls_fms[s][b]  # (num_class, h, w)
                pred_reg = reg_fms[s][b]  # (4, h, w)
                pred_cen = cen_fms[s][b]  # (1, h, w)
                tar_box = targets[b, targets[b, :, 4] >= 0, :4]  # (n, 4)
                tar_cls = targets[b, targets[b, :, 4] >= 0, 4]  # (n,)
                reg_obj_size = self.object_sizes_of_interest[s]
                stride = self.input_img_size[0] / cls_fms[s].size(2)
                # pos_idx: tuple; reg_tar: (m, 4); cls_tar: (m,); cen_tar: (m,); pos_num: m
                pos_idx, reg_tar, cls_tar, ctr_tar, pos_num = self.build_targets(grid, tar_box, tar_cls, reg_obj_size, stride)
                target_num += pos_num
                
                tmp_pred_cls = pred_cls.permute(1, 2, 0).contiguous()  # (h, w, num_class)
                tmp_pred_ctr = pred_cen.permute(1, 2, 0).contiguous() # (h, w, 1)
                tmp_pred_reg = pred_reg.permute(1, 2, 0).contiguous()  # (h, w, 4)
                tmp_tars_cls = torch.ones_like(tmp_pred_cls) * self.negative_smooth_cls  # (h, w, num_class)
                tmp_tars_cen = torch.zeros_like(tmp_pred_ctr)
                if pos_num > 0:
                    # --------------------------------------------------------------------------------- centerness loss
                    tmp_tars_cen[(pos_idx[0], pos_idx[1])] = ctr_tar.unsqueeze(dim=-1).type_as(tmp_pred_ctr)
                    ctr_focal = self.focal_loss_factor(tmp_pred_ctr[(pos_idx[0], pos_idx[1])].reshape(-1, 1), tmp_tars_cen[(pos_idx[0], pos_idx[1])].reshape(-1, 1))
                    ctr_loss = self.bce_ctr(tmp_pred_ctr[(pos_idx[0], pos_idx[1])].reshape(-1, 1), tmp_tars_cen[(pos_idx[0], pos_idx[1])].reshape(-1, 1))
                    ctr_loss = (ctr_loss * ctr_focal).sum() / pos_num
                    stage_ctr_loss.append(ctr_loss)

                    # --------------------------------------------------------------------------------- regression loss
                    iou_loss = self.compute_iou_loss(tmp_pred_reg[(pos_idx[0], pos_idx[1])], reg_tar, ctr_tar) / pos_num
                    stage_reg_loss.append(iou_loss)
                    
                    # --------------------------------------------------------------------------------- classification loss
                    tmp_tars_cls[(pos_idx[0], pos_idx[1], cls_tar.long())] = self.positive_smooth_cls  # foreground class
                else:
                    # --------------------------------------------------------------------------------- classification loss
                    # stage_reg_loss.append(tmp_pred_reg.abs().sum())
                    stage_reg_loss.append(tmp_pred_reg.clamp(0.0, 0.0).sum())
                    ctr_focal = self.focal_loss_factor(tmp_pred_ctr.reshape(-1, 1), tmp_tars_cen.reshape(-1, 1))
                    stage_ctr_loss.append(self.bce_ctr(tmp_pred_ctr.reshape(-1, 1), tmp_tars_cen.reshape(-1, 1) * ctr_focal).mean())

                cls_focal = self.focal_loss_factor(tmp_pred_cls.float().reshape(-1, self.hyp['num_class']), tmp_tars_cls.float().reshape(-1, self.hyp['num_class'])) 
                cls_loss = self.bce_cls(tmp_pred_cls.float().reshape(-1, self.hyp['num_class']), tmp_tars_cls.float().reshape(-1, self.hyp['num_class']))
                cls_loss = (cls_loss * cls_focal).mean(-1).sum() / max(pos_num, 1.0)
                stage_cls_loss.append(cls_loss)

            balance_reg_loss = torch.stack(stage_reg_loss, dim=0).mean()
            balance_ctr_loss = torch.stack(stage_ctr_loss, dim=0).mean()
            balance_cls_loss = torch.stack(stage_cls_loss, dim=0).mean()
            tot_ctr_loss.append(balance_ctr_loss)
            tot_cls_loss.append(balance_cls_loss)
            tot_reg_loss.append(balance_reg_loss)

        scale = batch_size
        ctr_loss_out = torch.stack(tot_ctr_loss, dim=0).mean() * self.hyp['ctr_loss_weight']
        cls_loss_out = torch.stack(tot_cls_loss, dim=0).mean() * self.hyp['cls_loss_weight']
        reg_loss_out = torch.stack(tot_reg_loss, dim=0).mean() * self.hyp['reg_loss_weight']
        tot_loss = (ctr_loss_out + cls_loss_out + reg_loss_out) * scale
        
        return {'tot_loss': tot_loss, 
                'cen_loss': ctr_loss_out.detach().item() * scale, 
                'cls_loss': cls_loss_out.detach().item() * scale, 
                'reg_loss': reg_loss_out.detach().item() * scale, 
                'tar_nums': target_num}
    
    @torch.no_grad()
    def build_targets(self, grid, tar_box, tar_cls, reg_obj_size, stride):
        """
        Inputs:
            grid: (h, w, 2) / [x, y]
            tar_box: (n, 4) / [xmin, ymin, xmax, ymax]
            tar_cls: (n,)
            reg_obj_size: two elements list / the regression range of the level 
        Ouputs:
            match: (h, w, n)
            reg_tar: (m, 4) / [l, t, r, b]
            cls_tar: (m,)
            cen_tar: (m,)
            pos_num: m
        """
        g = (grid.repeat(1, 1, 2)[:, :, None, :]).contiguous()  # (h, w, 1, 4) / [x, y, x, y]
        g[..., 2] *= -1
        g[..., 3] *= -1  # [x, y, -x, -y]
        tar = torch.zeros_like(tar_box)
        tar[:, 0] = tar_box[:, 0] * -1
        tar[:, 1] = tar_box[:, 1] * -1  
        tar[:, 2] = tar_box[:, 2]
        tar[:, 3] = tar_box[:, 3]  # [-xmin, -ymin, xmax, ymax]

        # ------------------------------------------------------------------------------ negative & positive samples assignment
        # is location in origin target boxes
        # (h, w, 1, 4) & (1, 1, n, 4) -> (h, w, n, 4)
        reg_targets = g + tar[None, None].contiguous()  # [x-xmin, y-ymin, xmax-x, ymax-y] / [l, t, r, b]
        # (h, w, n, 4) -> (h, w, n)
        is_in_tar_boxes = (reg_targets > 0.).all(dim=-1)  # (h, w, n)

        # center sampling
        if self.hyp['do_center_sampling']:
            is_in_tar_boxes = self.center_sampling(grid, tar_box, stride)  # (h, w, n)
        
        # filter target box by the maximum corrdinate that feature level i needs to regress
        max_reg_tar = reg_targets.max(dim=-1)[0]  # (h, w, n)
        is_cared_in_the_level = (max_reg_tar >= reg_obj_size[0]) & (max_reg_tar <= reg_obj_size[1])  # (h, w, n)

        # make sure each positive location match no more than one target box
        match_matrix = self.select_unique_by_tar_box_area(tar_box, is_in_tar_boxes, is_cared_in_the_level)

        # ------------------------------------------------------------------------------ build targets
        positive_samples_num = match_matrix.sum()
        if positive_samples_num > 0: 
            reg_tars_out = reg_targets[match_matrix] / stride  # (m, 4) / [l, t, r, b]
            # positive samples的regression target都应该为正数
            assert (reg_tars_out > 0).sum() == (reg_tars_out.size(0) * reg_tars_out.size(1))

            positive_location_idx = torch.nonzero(match_matrix, as_tuple=True)  # tuple / positive grid index
            assert reg_tars_out.size(0) == len(positive_location_idx[0]), f"positive locations in regression and classification targets number should be the same, but got {reg_tars_out.size(0)} v.s {cls_tars_out.size(0)}"
            cls_tars_out = tar_cls[positive_location_idx[2]]

            # centerness targets
            ctr_tars_out = (reg_tars_out[:, [0, 2]].min(dim=-1)[0] / reg_tars_out[:, [0, 2]].max(dim=-1)[0]) * (reg_tars_out[:, [1, 3]].min(dim=-1)[0] / reg_tars_out[:, [1, 3]].max(dim=-1)[0])
            ctr_tars_out = torch.sqrt(ctr_tars_out)

            return positive_location_idx, reg_tars_out, cls_tars_out, ctr_tars_out, positive_samples_num
        
        return None, None, None, None, 0
    
    def center_sampling(self, grid, tar_box, stride):
        """
        Inputs:
            grid: (h, w, 2) / [x, y]
            tar_box: (n, 4) / [xmin, ymin, xmax, ymax]
        Ouputs:
            is_in_tar: (h, w, n) / True means grid in tar box
        """

        tar_xywh = xyxy2xywh(tar_box)  # (n, 4) / [ctr_x, ctr_y, w, h]
        tar_ctr_box = torch.zeros_like(tar_xywh)
        tar_ctr_box[:, 0] = tar_xywh[:, 0] - self.radius * stride  # xmin
        tar_ctr_box[:, 1] = tar_xywh[:, 1] - self.radius * stride  # ymin
        tar_ctr_box[:, 2] = tar_xywh[:, 0] + self.radius * stride  # xmax
        tar_ctr_box[:, 3] = tar_xywh[:, 1] + self.radius * stride  # ymax

        tar_ctr_box[:, 0] = torch.where(tar_ctr_box[:, 0] > tar_box[:, 0], tar_ctr_box[:, 0], tar_box[:, 0])
        tar_ctr_box[:, 1] = torch.where(tar_ctr_box[:, 1] > tar_box[:, 1], tar_ctr_box[:, 1], tar_box[:, 1])
        tar_ctr_box[:, 2] = torch.where(tar_ctr_box[:, 2] < tar_box[:, 2], tar_ctr_box[:, 2], tar_box[:, 2])
        tar_ctr_box[:, 3] = torch.where(tar_ctr_box[:, 3] < tar_box[:, 3], tar_ctr_box[:, 3], tar_box[:, 3])

        tar_ctr_box[:, 0] *= -1
        tar_ctr_box[:, 1] *= -1  # [-xmin, -ymin, xmax, ymax]

        g = grid.repeat(1, 1, 2)[:, :, None, :]  # (h, w, 1, 4) / [x, y, x, y]
        g[..., 2] *= -1
        g[..., 3] *= -1  # [x, y, -x, -y]

        # (h, w, 1, 4) & (1, 1, n, 4) -> (h, w, n, 4)
        # [-xmin, -ymin, xmax, ymax] & [x, y, -x, -y]
        indicator = g + tar_ctr_box[None, None, :, :]  # [x-xmin, y-ymin, xmax-x, ymax-y] / [l, t, r, b]
        is_in_tar = (indicator > 0).all(dim=-1)  # (h, w, n)
        return is_in_tar

    def select_unique_by_tar_box_area(self, tar_box, is_in_tar_boxes, is_cared_in_the_level):
        """
        if there are still more than one objects for a location, we choose the one with minimal area
        Inputs:
            tar_box: (n, 4) / [xmin, ymin, xmax, ymax]
            is_in_tar_boxes: (h, w, n) / n is the number of gt
            is_cared_in_the_level: (h, w, n)
        """

        if torch.stack((is_in_tar_boxes, is_cared_in_the_level), dim=-1).all(dim=-1).any():
        # if is_in_tar_boxes.sum() > 0 or is_cared_in_the_level.sum() > 0:
            h, w, n = is_in_tar_boxes.shape
            tar_xywh = xyxy2xywh(tar_box)
            tar_box_area = torch.prod(tar_xywh[:, [2, 3]], dim=-1)  # (n,)
            tar_box_area = tar_box_area.repeat(h, w, 1)  # (h, w, n)
            tar_box_area[~is_in_tar_boxes] = INF
            tar_box_area[~is_cared_in_the_level] = INF
            min_area_idx = torch.min(tar_box_area, dim=-1)[1]  # (h, w)
            min_area_idx = F.one_hot(min_area_idx, tar_box.size(0)).to(torch.bool)  # (h, w, n)
            min_area_idx[tar_box_area == INF] = False

            # 确保每个location都最多只有一个gt与之对应
            assert (min_area_idx.sum(dim=-1) >= 2).sum() == 0, f"each location should match no more than one target!{min_area_idx.sum(dim=-1).shape}\n{(min_area_idx.sum(dim=-1) >= 2).sum()}"
            return min_area_idx
        return torch.zeros_like(is_in_tar_boxes, dtype=torch.bool)

    def get_regression_range_of_each_level(self, fm_shapes):
        # [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]
        object_sizes_of_interest = []
        for i, (fm_h, fm_w) in enumerate(fm_shapes):
            t = math.log2(self.input_img_size[0] / fm_h) + 3
            if i == 0:
                rg = [-1, 2**t]
            elif i == len(fm_shapes)-1:
                rg = [2**(t-1), INF]
            else:
                rg = [2**(t-1), 2**t]
            object_sizes_of_interest.append(rg)
        return object_sizes_of_interest

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
        # 当alpha值小于0.5时, 意味着更加关注将负类样本预测错误的情况
        alpha = self.hyp.get('focal_loss_alpha', 0.25)
        alpha_factor = target * alpha + (1.0 - target) * (1.0 - alpha)
        factor = gamma_factor * alpha_factor
        return factor

    def make_grid(self, shape_list):
        """
        Inputs:
            shape_list: [[h/8, w/8], [h/16, w/16], [h/32, w/32], [h/64, w/64], [h/128, w/128]]
        """
        grids = []
        for h, w in shape_list:
            stride = self.input_img_size[0] / h
            shift_x = torch.arange(0, w*stride, step=stride, device=self.device, dtype=torch.float32)
            shift_y = torch.arange(0, h*stride, step=stride, device=self.device, dtype=torch.float32)
            y, x = torch.meshgrid((shift_x, shift_y), indexing='ij')
            # mesh_grid: (col_num, row_num, 2) -> (row_num, col_num, 2)
            mesh_grid = torch.stack((y, x), dim=2).flip(dims=(-1,)) + stride // 2
            # (col_num, row_num, 2)
            grids.append(mesh_grid)
        return grids
