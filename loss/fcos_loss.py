import torch
import torch.nn as nn
from utils import xyxy2xywhn, xyxy2xywh, xywh2xyxy, reduce_mean, gather, get_local_rank
from utils import gpu_CIoU
from utils import gpu_iou, gpu_DIoU, gpu_Giou
from torch.nn import functional as F

__all__ = ['FCOSLoss']

INF = 1000000.0


def smooth_bce(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

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
        self.bce_cen = nn.BCEWithLogitsLoss(pos_weight=cen_pos_weight, reduction='none').to(self.device)
        self.l1_reg = nn.L1Loss()
        self.input_img_size = hyp['input_img_size']
        self.cen_loss_balances = [4., 1., 0.4] if stage_num == 3 else [1., 2., 4., 0.2, 0.1]
        self.cls_loss_balances = [4., 1., 0.4] if stage_num == 3 else [1., 2., 4., 0.2, 0.1]
        self.reg_loss_balances = [4., 1., 0.4] if stage_num == 3 else [1., 2., 4., 0.2, 0.1]
        self.num_stage = stage_num
        self.positive_smooth_cls, self.negative_smooth_cls = smooth_bce(0.0)
        self.radius = hyp['center_sampling_radius']
        self.grids = None

    def compute_iou_loss(self, pred, tar, weight=None):
        """
        Inputs:
            pred: (m, 4) / [l, t, r, b]
            tar: (m, 4) / [l, t, r, b]
            weight: (m,)
        """
        assert pred.size(0) == tar.size(0)
        if weight is not None:
            assert pred.size(0) == len(weight)

        pred_left   = pred[:, 0]  # (m,)
        pred_top    = pred[:, 1]  # (m,)
        pred_right  = pred[:, 2]  # (m,)
        pred_bottom = pred[:, 3]  # (m,)

        target_left   = tar[:, 0]  # (m,)
        target_top    = tar[:, 1]  # (m,)
        target_right  = tar[:, 2]  # (m,)
        target_bottom = tar[:, 3]  # (m,)

        target_area = (target_left + target_right) * (target_top + target_bottom)  # (m,)
        pred_area   = (pred_left   + pred_right)   * (pred_top   + pred_bottom)    # (m,)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)    # (m,)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)  # (m,)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)    # (m,)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)  # (m,)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7  # (m,)
        area_intersect = w_intersect * h_intersect  # (m,)
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

        if weight is not None :
            return (losses * weight).sum()
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
        num_level = len(cls_fms)
        targets = targets_batch.clone().detach()

        self.strides = [2**(3+i) for i in range(num_level)]

        fm_shapes  = [(f.size(2), f.size(3)) for f in cls_fms]
        self.grids = self.make_grid(fm_shapes) if self.grids is None else self.grids  # [x, y]

        # [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]
        self.object_sizes_of_interest = self.get_regression_range_of_each_level(num_level)

        tot_reg_loss, tot_cls_loss, tot_cen_loss = [], [], []
        target_num = 0
        for s in range(len(cls_fms)):  # each stage
            stage_reg_loss, stage_cls_loss, stage_cen_loss = [], [], []
            grid = self.grids[s]  # (h, w, 2)
            stride = self.strides[s]
            for b in range(batch_size):  # each image
                pred_cls = cls_fms[s][b]  # (num_class, h, w)
                pred_reg = reg_fms[s][b]  # (4, h, w)
                pred_cen = cen_fms[s][b]  # (1, h, w)
                tar_box = targets[b, targets[b, :, 4] >= 0, :4]  # (n, 4)
                tar_cls = targets[b, targets[b, :, 4] >= 0, 4]  # (n,)
                reg_obj_size = self.object_sizes_of_interest[s]
                # pos_idx: tuple; reg_tar: (m, 4); cls_tar: (m,); cen_tar: (m,); pos_num: m
                pos_idx, reg_tar, cls_tar, cen_tar, pos_num = self.build_targets(grid, tar_box, tar_cls, reg_obj_size, stride)
                target_num += pos_num
                
                tmp_pred_cls = pred_cls.permute(1, 2, 0)  # (h, w, num_class)
                tmp_pred_cen = pred_cen.permute(1, 2, 0) # (h, w, 1)
                tmp_pred_reg = pred_reg.permute(1, 2, 0)  # (h, w, 4)
                tmp_tars_cls = torch.ones_like(tmp_pred_cls) * self.negative_smooth_cls  # (h, w, num_class)
                if pos_num > 0:
                    # --------------------------------------------------------------------------------- centerness loss
                    tmp_tars_cen = torch.zeros_like(tmp_pred_cen)
                    tmp_tars_cen[(pos_idx[0], pos_idx[1])] = cen_tar.unsqueeze(dim=-1).type_as(tmp_pred_cen)
                    stage_cen_loss.append(self.bce_cen(tmp_pred_cen.reshape(-1, 1), tmp_tars_cen.reshape(-1, 1)).sum() / max(pos_num, 1.0))

                    # --------------------------------------------------------------------------------- regression loss
                    ctrness_targets_sum = cen_tar.sum()
                    loss_denorm = max(ctrness_targets_sum.item(), 1e-6)
                    stage_reg_loss.append(self.compute_iou_loss(tmp_pred_reg[(pos_idx[0], pos_idx[1])], reg_tar, cen_tar) / loss_denorm)
                    
                    # --------------------------------------------------------------------------------- classification loss
                    tmp_tars_cls[(pos_idx[0], pos_idx[1], cls_tar.long())] = self.positive_smooth_cls  # foreground class
                else:
                    # --------------------------------------------------------------------------------- classification loss
                    stage_cen_loss.append(tmp_pred_cen.new_tensor(0.0))
                    stage_reg_loss.append(tmp_pred_reg.new_tensor(0.0))

                focal = self.focal_loss_factor(tmp_pred_cls.float().reshape(-1, self.hyp['num_class']), 
                                               tmp_tars_cls.float().reshape(-1, self.hyp['num_class'])) 
                cls_loss = self.bce_cls(tmp_pred_cls.float().reshape(-1, self.hyp['num_class']), 
                                        tmp_tars_cls.float().reshape(-1, self.hyp['num_class']))
                cls_loss = (cls_loss * focal).sum() / max(pos_num, 1.0)
                stage_cls_loss.append(cls_loss)

            balance_reg_loss, balance_cen_loss, balance_cls_loss = self.compute_balance_losses(s, 
                                                                                               stage_reg_loss=torch.stack(stage_reg_loss, dim=0).mean(), 
                                                                                               stage_cen_loss=torch.stack(stage_cen_loss, dim=0).mean(), 
                                                                                               stage_cls_loss=torch.stack(stage_cls_loss, dim=0).mean())
            tot_cen_loss.append(balance_cen_loss)
            tot_cls_loss.append(balance_cls_loss)
            tot_reg_loss.append(balance_reg_loss)

        self.update_balances() 

        scale = 1
        cen_loss_out = torch.stack(tot_cen_loss, dim=0).mean() * self.hyp['cen_loss_weight']
        cls_loss_out = torch.stack(tot_cls_loss, dim=0).mean() * self.hyp['cls_loss_weight']
        reg_loss_out = torch.stack(tot_reg_loss, dim=0).mean() * self.hyp['reg_loss_weight']
        tot_loss = (cen_loss_out + cls_loss_out + reg_loss_out) * scale
        
        return {'tot_loss': tot_loss, 
                'cen_loss': cen_loss_out.detach().item() * scale, 
                'cls_loss': cls_loss_out.detach().item() * scale, 
                'reg_loss': reg_loss_out.detach().item() * scale, 
                'tar_nums': target_num}
    
    def compute_balance_losses(self, stage, stage_reg_loss, stage_cen_loss, stage_cls_loss):
        if self.hyp['do_cen_loss_balance']:
            stage_cen_loss = stage_cen_loss * self.cen_loss_balances[stage]
            self.cen_loss_balances[stage] = self.cen_loss_balances[stage] * 0.9999 + 0.0001 / (stage_cen_loss.detach().item() if stage_cen_loss.detach().item() != 0 else 1.0)
            
        if self.hyp['do_cls_loss_balance']:
            stage_cls_loss = stage_cls_loss * self.cls_loss_balances[stage]
            self.cls_loss_balances[stage] = self.cls_loss_balances[stage] * 0.9999 + 0.0001 / (stage_cls_loss.detach().item() if stage_cls_loss.detach().item() != 0 else 1.0)

        if self.hyp['do_reg_loss_balance']:
            stage_reg_loss = stage_reg_loss * self.reg_loss_balances[stage]
            self.reg_loss_balances[stage] = self.reg_loss_balances[stage] * 0.9999 + 0.0001 / (stage_reg_loss.detach().item() if stage_reg_loss.detach().item() != 0 else 1.0)

        return stage_reg_loss, stage_cen_loss, stage_cls_loss
    
    def update_balances(self):
        if self.hyp['do_cen_loss_balance']:
            self.cen_loss_balances = [x/self.cen_loss_balances[self.num_stage // 2] for x in self.cen_loss_balances]

        if self.hyp['do_cls_loss_balance']:
            self.cls_loss_balances = [x/self.cls_loss_balances[self.num_stage // 2] for x in self.cls_loss_balances]
            
        if self.hyp['do_reg_loss_balance']:
            self.reg_loss_balances = [x/self.reg_loss_balances[self.num_stage // 2] for x in self.reg_loss_balances]

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
            cen_tars_out = (reg_tars_out[:, [0, 2]].min(dim=-1)[0] / reg_tars_out[:, [0, 2]].max(dim=-1)[0]) \
                         * (reg_tars_out[:, [1, 3]].min(dim=-1)[0] / reg_tars_out[:, [1, 3]].max(dim=-1)[0])
            cen_tars_out = torch.sqrt(cen_tars_out)

            return positive_location_idx, reg_tars_out, cls_tars_out, cen_tars_out, positive_samples_num
        
        return None, None, tar_cls.new_tensor([-1]), None, 0
    
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
        indicator = g + tar_ctr_box[None, None]  # [x-xmin, y-ymin, xmax-x, ymax-y] / [l, t, r, b]
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

    def get_regression_range_of_each_level(self, num_level):
        # [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]
        object_sizes_of_interest = []
        for i in range(1, num_level+1):
            if i == 1:
                rg = [-1, 2**(5 + i)]
            elif i == num_level:
                rg = [2**(5+i-1), INF]
            else:
                rg = [2**(5+(i-1)), 2**(5+i)]
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
        for s, (row_num, col_num) in enumerate(shape_list):
            stride = self.strides[s]
            shift_x = torch.arange(0, col_num*stride, step=stride, device=self.device, dtype=torch.float32)
            shift_y = torch.arange(0, row_num*stride, step=stride, device=self.device, dtype=torch.float32)
            y, x = torch.meshgrid((shift_x, shift_y), indexing='ij')
            # mesh_grid: (col_num, row_num, 2) -> (row_num, col_num, 2)
            mesh_grid = torch.stack((x, y), dim=2).reshape(row_num, col_num, 2) + stride // 2
            # (col_num, row_num, 2)
            grids.append(mesh_grid)
        return grids
