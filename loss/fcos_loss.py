import torch
import torch.nn as nn
from utils import xyxy2xywhn, xyxy2xywh, xywh2xyxy
from utils import gpu_CIoU
from utils import gpu_iou, gpu_DIoU, gpu_Giou
from torch.nn import functional as F

__all__ = ['FCOSLoss']

class FCOSLoss:

    def __init__(self, hyp, stage_num=5):
        """
        3种loss的重要程度依次为: confidence > classification > regression 

        :param loss_hyp: dict
        :param data_hyp: dict
        :param train_hyp: dict
        :param stage_num: int
        """
        self.hyp = hyp
        self.device = hyp['device']
        cls_pos_weight = torch.tensor(hyp["cls_pos_weight"], device=self.device)
        cen_pos_weight = torch.tensor(hyp['cen_pos_weight'], device=self.device)
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight, reduction='none').to(self.device)
        self.bce_cen = nn.BCEWithLogitsLoss(pos_weight=cen_pos_weight, reduction='mean').to(self.device)
        self.input_img_size = hyp['input_img_size']
        self.balances = [4., 1., 0.4] if stage_num == 3 else [4., 1., 0.4, 0.1]
        self.radius = hyp['center_sampling_radius']

    def compute_iou_loss(self, pred, tar, weight=None):
        """
        Inputs:
            pred: (m, 4) / [l, b, r, t]
            tar: (m, 4) / [l, b, r, t]
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = tar[:, 0]
        target_top = tar[:, 1]
        target_right = tar[:, 2]
        target_bottom = tar[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area   = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.hyp['iou_type'] == 'iou':
            losses = -torch.log(ious)
        elif self.hyp['iou_type'] == 'linear_iou':
            losses = 1 - ious
        elif self.hyp['iou_type'] == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

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
        num_level = len(cls_fms)
        targets = targets_batch.clone().detach()

        fm_shapes  = [(f.size(2), f.size(3)) for f in cls_fms]
        self.grids = self.make_grid(fm_shapes)

        self.strides = [2**(3+i) for i in range(num_level)]
        # [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]
        self.object_sizes_of_interest = self.get_regression_range_of_each_level(num_level)

        tot_reg_loss, tot_cls_loss, tot_cen_loss = [], [], []
        target_num = 0
        for s in range(len(cls_fms)):  # each stage
            stage_reg_loss, stage_cls_loss, stage_cen_loss = [], [], []
            grid = self.grids[s]  # (h, w, 2)
            for b in range(batch_size):  # each image
                pred_cls = cls_fms[s, b]  # (num_class, h, w)
                pred_reg = reg_fms[s, b]  # (4, h, w)
                pred_cen = cen_fms[s, b]  # (1, h, w)
                tar_box = targets[b, targets[b, :, 4] >= 0, :4]  # (n, 4)
                tar_cls = targets[b, targets[b, :, 4] >= 0, 4]  # (n,)
                stride = self.strides[s]
                reg_obj_size = self.object_sizes_of_interest[s]
                # pos_idx: (h, w); reg_tar: (m, 4); cls_tar: (m,); cen_tar: (m,); pos_num: m
                pos_idx, reg_tar, cls_tar, cen_tar, pos_num = self.build_targets(grid, tar_box, tar_cls, stride, reg_obj_size)
                target_num += pos_num
                
                # --------------------------------------------------------------------------------- classification loss
                tmp_pred_cls = pred_cls.permute(1, 2, 0)  # (h, w, num_class)
                tmp_pred_cls[~pos_idx] = 0
                tmp_tars_cls = torch.zeros_like(tmp_pred_cls)  # (h, w, num_class)
                tmp_tars_cls[pos_idx] = F.one_hot(cls_tar.long(), self.hyp['num_class'])
                cls_loss = self.bce_cls(tmp_pred_cls.float().reshape(-1, self.hyp['num_class']), tmp_tars_cls.float().reshape(-1, self.hyp['num_class']))
                focal = self.focal_loss_factor(tmp_pred_cls.float().reshape(-1, self.hyp['num_class']), tmp_tars_cls.float().reshape(-1, self.hyp['num_class'])) 
                stage_cls_loss.append((cls_loss * focal).mean())

                if pos_num > 0:
                    # --------------------------------------------------------------------------------- centerness loss
                    tmp_pred_cen = pred_cen.permute(1, 2, 0) # (h, w, 1)
                    tmp_pred_cen[~pos_idx] = 0
                    tmp_tars_cen = torch.zeros_like(tmp_pred_cen)
                    tmp_tars_cen[pos_idx] = cen_tar
                    cen_loss = self.bce_cen(tmp_pred_cen, tmp_tars_cen)
                    stage_cen_loss.append(cen_loss)

                    # --------------------------------------------------------------------------------- regression loss
                    tmp_pred_reg = pred_reg.permute(1, 2, 0)  # (h, w, 4)
                    reg_loss = self.compute_iou_loss(tmp_pred_reg[pos_idx], reg_tar)
                    stage_reg_loss.append(reg_loss)
                else:
                    stage_cen_loss.append(pred_cen.new_tensor(0.))
                    stage_reg_loss.append(pred_reg.new_tensor(0.))

            tot_cen_loss.append(torch.stack(stage_cen_loss, dim=0).mean())
            tot_cls_loss.append(torch.stack(stage_cls_loss, dim=0).mean())
            tot_reg_loss.append(torch.stack(stage_reg_loss, dim=0).mean())


        tot_loss = torch.stack(tot_cen_loss, dim=0).mean() * self.hyp['cen_loss_weight'] + \
                   torch.stack(tot_cls_loss, dim=0).mean() * self.hyp['cls_loss_weight'] + \
                   torch.stack(tot_reg_loss, dim=0).mean() * self.hyp['reg_loss_weight']
        
        return {'tot_loss': tot_loss, 
                'cen_loss': (torch.stack(tot_cen_loss, dim=0).mean() * self.hyp['cen_loss_weight']).detach().item(), 
                'cls_loss': (torch.stack(tot_cls_loss, dim=0).mean() * self.hyp['cls_loss_weight']).detach().iten(), 
                'reg_loss': (torch.stack(tot_reg_loss, dim=0).mean() * self.hyp['reg_loss_weight']).detach().item(), 
                'tar_num': target_num}
            


    def build_targets(self, grid, tar_box, tar_cls, stride, reg_obj_size):
        """
        Inputs:
            grid: (h, w, 2)
            tar_box: (n, 4) / [xmin, ymin, xmax, ymax]
            tar_cls: (n,)
            stride: int
            reg_obj_size: two elements list / the regression range of the level 
        Ouputs:
            match: (h, w, n)
            reg_tar: (m, 4)
            cls_tar: (m,)
            cen_tar: (m,)
            pos_num: m
        """
        g = grid.repeat(1, 1, 1, 2)  # (h, w, 1, 4) / [x, y, x, y]
        g[..., 2] *= -1
        g[..., 3] *= -1  # [x, y, -x, -y]
        g *= stride
        tar = tar_box.detach().clone()
        tar[:, 0] *= -1
        tar[:, 1] *= -1  # [-xmin, -ymin, xmax, ymax]

        # ------------------------------------------------------------------------------ negative & positive samples assignment
        # is location in origin target boxes
        # (h, w, 1, 4) & (1, 1, n, 4) -> (h, w, n, 4)
        reg_targets = g + tar[None, None]  # [x-xmin, y-ymin, xmax-x, ymax-y] / [l, b, r, t]
        is_in_tar_boxes = (reg_targets > 0).all(dim=-1)  # (h, w, n)

        # center sampling
        if self.hyp['do_center_sampling']:
            is_in_tar_boxes = self.center_sampling(grid, tar_box, stride)  # (h, w, n)

        # filter target box by the maximum corrdinate that feature level i needs to regress
        max_reg_tar = reg_targets.max(dim=-1)[0]  # (h, w)
        is_cared_in_the_level = (max_reg_tar >= reg_obj_size[0]) & (max_reg_tar <= reg_obj_size[1])  # (h, w)

        # matching matrix
        match_matrix = torch.zeros_like(is_in_tar_boxes)  # (h, w, n)
        match_matrix[is_in_tar_boxes] = 1
        match_matrix[is_cared_in_the_level] = 0
        assert match_matrix.sum() >= tar_box.size(0)  # location匹配的gt个数总和一定不小于gt总数

        # make sure each positive location match no more than one target box
        match_matrix = self.select_unique_by_tar_box_area(tar_box, match_matrix)
        assert (match_matrix.sum(dim=-1) >= 2).sum() == 0, f"each location should match no more than one target!"

        # ------------------------------------------------------------------------------ build targets
        positive_samples_num = match_matrix.sum()
        if positive_samples_num > 0: 
            # reg_tars_out = torch.zeros_like(reg_targets)
            # reg_tars_out[match_matrix] = reg_targets[match_matrix]
            reg_tars_out = reg_targets[match_matrix.to(torch.bool)]  # (m, 4) / [l, b, r, t]

            positive_location_idx = match_matrix.any(dim=-1)  # (h, w)
            positive_samples = match_matrix[positive_location_idx]  # (m, n)
            assert reg_tars_out.size(0) == positive_samples.size(0), f"positive locations in regression and classification targets number should be the same, but got {reg_tars_out.size(0)} v.s {cls_tars_out.size(0)}"
            positive_samples_tar_cls_idx = positive_samples.max(dim=-1)[1]  # (m,)
            cls_tars_out = tar_cls[positive_samples_tar_cls_idx]

            # centerness targets
            cen_tars_out = (reg_tars_out[:, [0, 2]].min(dim=-1)[0] / reg_tars_out[:, [0, 2]].max(dim=-1)[0]) \
                         * (reg_tars_out[:, [1, 3]].min(dim=-1)[0] / reg_tars_out[:, [1, 3]].max(dim=-1)[0])
            cen_tars_out = torch.sqrt(cen_tars_out)

            return positive_location_idx, reg_tars_out, cls_tars_out, cen_tars_out, positive_samples_num
        
        return None, None, None, None, positive_samples_num
    
    def center_sampling(self, grid, tar_box, stride):
        """
        Inputs:
            grid: (h, w, 2)
            tar_box: (n, 4) / [xmin, ymin, xmax, ymax]
            stride: int
        Ouputs:
            is_in_tar: (h, w, n) / True means grid in tar box
        """

        tar_xywh = xyxy2xywh(tar_box)  # (n, 4) / [ctr_x, ctr_y, w, h]
        tar_ctr_box = torch.zeros_like(tar_xywh)
        tar_ctr_box[:, 0] = tar_xywh[:, 0] - self.radius * stride  # xmin
        tar_ctr_box[:, 1] = tar_xywh[:, 0] + self.radius * stride  # xmax
        tar_ctr_box[:, 2] = tar_xywh[:, 1] - self.radius * stride  # ymin
        tar_ctr_box[:, 3] = tar_xywh[:, 1] + self.radius * stride  # ymax

        g = grid.repeat(1, 1, 1, 2)  # (h, w, 1, 4) / [x, y, x, y]
        g[..., 2] *= -1
        g[..., 3] *= -1  # [x, y, -x, -y]
        g *= stride

        # (h, w, 1, 4) & (1, 1, n, 4) -> (h, w, n, 4)
        indicator = g + tar_ctr_box[None, None]  # [x-xmin, y-ymin, xmax-x, ymax-y]
        is_in_tar = (indicator > 0).all(dim=-1)  # (h, w, n)
        return is_in_tar

    def select_unique_by_tar_box_area(self, tar_box, match_matrix):
        """
        if there are still more than one objects for a location, we choose the one with minimal area
        Inputs:
            tar_box: (n, 4) / [xmin, ymin, xmax, ymax]
            match_matrix: (h, w, n) / n is the number of gt
        """
        assert torch.unique(match_matrix) == 2
        matrix = match_matrix.to(torch.bool)

        valid_locations = torch.any(matrix, dim=-1)  # (h, w)
        tar_xywh = xyxy2xywh(tar_box)
        tar_box_area = torch.prod(tar_xywh[:, [2, 3]], dim=-1)  # (n,)
        tar_box_area = tar_box_area.repeat(matrix.size(0), matrix.size(1), 1)  # (h, w, n)
        tar_box_area[~matrix] = 0.0
        min_idx = torch.min(tar_box_area, dim=-1)[1]
        location2gt_mask = torch.zeros_like(tar_box_area, dtype=torch.bool)
        location2gt_mask[min_idx] = True  # (h, w, n)
        location2gt_mask[~valid_locations] = False 

        # 确保每个location都最多只有一个gt与之对应
        assert location2gt_mask.sum() == valid_locations.sum()
        return location2gt_mask

    def get_regression_range_of_each_level(self, num_level):
        # [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]
        object_sizes_of_interest = []
        for i in range(1, num_level+1):
            if i == 0:
                rg = [-1, 2**(3 + i)]
            elif i == num_level:
                rg = [2**(3+i), float('inf')]
            else:
                rg = [2**(3+(i-1)), 2**(3+i)]
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
        for row_num, col_num in shape_list:
            y, x = torch.meshgrid([torch.arange(row_num, device=self.device), torch.arange(col_num, device=self.device)], indexing='ij')
            # mesh_grid: (col_num, row_num, 2) -> (row_num, col_num, 2)
            mesh_grid = torch.stack((x, y), dim=2).reshape(row_num, col_num, 2)
            # (col_num, row_num, 2)
            grids.append(mesh_grid)
        return grids
