import torch
import torch.nn as nn
from utils import xyxy2xywhn, xyxy2xywh, xywh2xyxy
from utils import gpu_CIoU
from utils import gpu_iou, gpu_DIoU, gpu_Giou

__all__ = ['FCOSLoss']

class FCOSLoss:

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
        self.radius = hyp['center_sampling_radius']
        

    def __call__(self, cls_fms, reg_fms, ctr_fms, targets_batch):
        """
        Inputs:
            targets_batch: tensor / (bn, bbox_num, 6) -> [xmin, ymin, xmax, ymax, cls, img_id]
            cls_fms: [(b, num_class, h/8, w/8), (b, num_class, h/16, w/16), (b, num_class, h/32, w/32), (b, num_class, h/64, w/64), (b, num_class, h/128, w/128)]
            reg_fms: [(b, 4, h/8, w/8), (b, 4, h/16, w/16), (b, 4, h/32, w/32), (b, 4, h/64, w/64), (b, 4, h/128, w/128)]
            ctr_fms: [(b, 1, h/8, w/8), (b, 1, h/16, w/16), (b, 1, h/32, w/32), (b, 1, h/64, w/64), (b, 4, h/128, w/128)]
        """
        assert len(cls_fms) == len(reg_fms) and len(reg_fms) == len(ctr_fms)
        assert isinstance(targets_batch, torch.Tensor), f"targets's type should be torch.Tensor but we got {type(targets_batch)}"

        batch_size = targets_batch.size(0)
        num_level = len(cls_fms)
        anchor_num = self.anchors.shape[1]
        targets = targets_batch.clone().detach()

        fm_shapes  = [(f.size(2), f.size(3)) for f in cls_fms]
        self.grids = self.make_grid(fm_shapes)

        self.strides = [2**(3+i) for i in range(num_level)]
        # [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]
        self.object_sizes_of_interest = self.get_regression_range_of_each_level(num_level)
        

        for s in range(len(cls_fms)):  # each stage
            grid = self.grids[s]  # (h, w, 2)
            for b in range(batch_size):  # each image
                tar_box = targets[b, targets[b, :, 4] >= 0, :4]  # (n, 4)
                is_locations_in_tar_boxes = self.is_in_gt_bbox(grid, tar_box, self.strides[s])  # (h, w, n)
                is_locations_in_tar_boxes = self.filter_by_object_size(tar_box, is_locations_in_tar_boxes, self.object_sizes_of_interest[s])
                if self.hyp['do_center_sampling']:
                    is_locations_in_tar_boxes = self.center_sampling(grid, tar_box, self.strides[s])  # (h, w, n)
                match_matrix = self.select_unique_matching_by_tar_box_area(is_locations_in_tar_boxes)  # (h, w, n)


    def is_in_gt_bbox(self, grid, tar_box, stride):
        """
        Inputs:
            grid: (h, w, 2)
            tar_box: (n, 4) / [xmin, ymin, xmax, ymax]
            stride: int
        Ouputs:
            is_in_tar: (h, w, n) / element value is True means the location in tar box
        """
        g = grid.repeat(1, 1, 1, 2)  # (h, w, 1, 4) / [x, y, x, y]
        g[..., 2] *= -1
        g[..., 3] *= -1  # [x, y, -x, -y]
        g *= stride
        tar = tar_box.detach().clone()
        tar[:, 0] *= -1
        tar[:, 1] *= -1  # [-xmin, -ymin, xmax, ymax]

        # (h, w, 1, 4) & (1, 1, n, 4) -> (h, w, n, 4)
        indicator = g + tar[None, None]  # [x-xmin, y-ymin, xmax-x, ymax-y]
        is_in_tar = (indicator > 0).all(dim=-1)  # (h, w, n)
        return is_in_tar
        
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

    def filter_by_object_size(self, tar_box, indicator, size_range):
        """
        filter target box by the maximum corrdinate that feature level i needs to regress
        Inputs:
            grid: (h, w, 2)
            tar_box: (n, 4) / [xmin, ymin, xmax, ymax]
            indicator: (h, w, n) / bool tensor
            size_range: two elements list
        Ouputs:
            is_cared_in_the_level: (h, w, n)
        """

        locations_targets_matrix = tar_box.repeat(indicator.size(0), indicator.size(1), 1, 1)  # (h, w, n, 4)
        locations_targets_max_reg_matrix, _ = locations_targets_matrix.max(dim=-1)  # (h, w, n)
        is_cared_in_the_level = (locations_targets_max_reg_matrix >= size_range[0]) & (locations_targets_max_reg_matrix <= size_range[1])  # (h, w, n)
        assert is_cared_in_the_level.shape == indicator
        is_cared_in_the_level[~indicator] = False
        assert is_cared_in_the_level.sum() <= indicator.sum(), f"after filter by maximum coordinate of each level, the total match number should be less than before, but got {indicator.sum()} begore < {is_cared_in_the_level.sum()} after"
        return is_cared_in_the_level
    
    def select_unique_matching_by_tar_box_area(self, tar_box, indicator):
        """
        if there are still more than one objects for a location, we choose the one with minimal area
        Inputs:
            tar_box: (n, 4) / [xmin, ymin, xmax, ymax]
            indicator: (h, w, n) / n is the number of gt
        """
        valid_locations = torch.any(indicator, dim=-1)  # (h, w)
        tar_xywh = xyxy2xywh(tar_box)
        tar_box_area = torch.prod(tar_xywh[:, [2, 3]], dim=-1)  # (n,)
        tar_box_area = tar_box_area.repeat(indicator.size(0), indicator.size(1), 1)  # (h, w, n)
        tar_box_area[~indicator] = 0.0
        min_v, min_idx = torch.min(tar_box_area, dim=-1)
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



    def match(self, targets, fm_shape):
        """
        正样本分配

        """

        # select grid in gt bbox


        
        return 

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
