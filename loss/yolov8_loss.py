import torch
import torch.nn as nn
import numpy as np
from utils import tblr2xyxy, gpu_CIoU, xyxy2tblr
import torch.nn.functional as F
from torch import Tensor
import gc 

__all__ = ['YOLOV8Loss']

class YOLOV8Loss:

    def __init__(self, hyp) -> None:
        self.hyp = hyp
        self.alpha = hyp['alpha']
        self.beta = hyp['beta']
        self.topk = hyp['topk']
        self.reg = hyp['reg']
        self.grids = {}
        self.img_sz = hyp['input_img_size']
        self.num_class = hyp['num_class']
        self.iou_loss_scale = hyp.get('iou_loss_scale', 7.5)
        self.cls_loss_scale = hyp.get('cls_loss_scale', 0.5)
        self.dfl_loss_scale = hyp.get('dfl_loss_scale', 1.5)
        self.device = hyp['device']
        cls_pos_weight = torch.tensor(hyp.get("cls_pos_weight", 1.), device=self.device)
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight, reduction='sum').to(self.device)
        self.dfl_project = torch.arange(hyp['reg'], device=self.device, dtype=torch.float32)
        self.grids, self.strides = None, None

    def __call__(self, preds:Tensor, tars:Tensor):
        """
        假设输入训练的图片尺寸为640x640x3

        Args:
            preds: {'pred_xs': (b, num_class+4*reg, 160, 160), 
                    'pred_s': (b, num_class+4*reg, 80, 80), 
                    'pred_m': (b, num_class+4*reg, 40, 40), 
                    'pred_l': (b, num_class+4*reg, 20, 20)}
            tars: (b, M, 6) / [xmin, ymin, xmax, ymax, class_id, img_id]
        """

        tar_xyxy, tar_cls = tars[..., :4], tars[..., 4]  # (b, M, 4); (b, M)
        fm_shapes = [[f.size(2), f.size(3)] for f in preds.values()]
        fm_strides = [self.img_sz[0] / f.size(2) for f in preds.values()]
        if self.grids is None or self.strides is None:
            self.grids, self.strides = self.make_grid(fm_shapes, fm_strides, self.device)  # grids: (N, 2), strides: (N, 1)
        pred_reg, pred_xyxy, pred_cls = self.pred_preprocessing(preds, self.grids)  # pred_reg: (b, N, reg*4); pred_xyxy: (b, N, 4); pred_cls: (b, N, num_class)
        mask_grids_in_gt = self.select_grids_in_gt_bbox(tar_xyxy, self.grids, tar_cls >= 0, self.strides)  # (b, N, M)
        metric, metric_iou = self.compute_metric(pred_xyxy, self.strides, pred_cls, tar_xyxy, tar_cls, mask_grids_in_gt)  # metric: (b, N, M); metric_iou: (b, N, M)
        mask_topk = self.select_topk_candidates(metric, mask_grids_in_gt)  # (b, N, M)
        mask_assign = self.label_assign(mask_topk, metric_iou)  # (b, N, M)
        metric *= mask_assign.type(metric.dtype)  # (b, N, M)
        match_cls_onehot, match_box_xyxy = self.build_target(mask_assign, tar_cls, tar_xyxy)  # match_cls_onehot: (b, N, num_class); match_box_xyxy: (b, N, M, 4) / [xmin, ymin, xmax, ymax]
        # (b, N, M) & (b, N, M) -> (b, N, M) & (b, 1, M) -> (b, N, M) -> (b, N) -> (b, N, 1)
        norm_metric = ((metric * metric_iou) / (metric.amax(dim=1, keepdim=True) + 1e-9)).amax(dim=-1).unsqueeze(-1)  # (b, N, 1)
        match_cls_score = match_cls_onehot * norm_metric  # (b, N, num_class) & (b, N, 1) -> (b, N, num_class)

        cls_loss = self.bce_cls(pred_cls.reshape(-1, self.num_class).contiguous(), match_cls_score.reshape(-1, self.num_class).contiguous()) / torch.maximum(match_cls_score.sum(), torch.tensor(1.0))
        iou_loss, dfl_loss = self.dfl_box_loss(pred_reg, pred_xyxy, match_box_xyxy, match_cls_score, self.strides, mask_assign, self.grids)

        # b = tars.size(0)
        b = 1
        tot_cls_loss = cls_loss * self.cls_loss_scale * b
        tot_iou_loss = iou_loss * self.iou_loss_scale * b
        tot_dfl_loss = dfl_loss * self.dfl_loss_scale * b
        
        msg = "cls_loss: {cls_loss}; iou_loss: {iou_loss}, dfl_loss: {dfl_loss}"
        fmt = {}
        cmd = False
        if torch.isnan(tot_cls_loss):
            fmt['cls_loss'] = 'Nan'
            cmd = True
        else:
            fmt['cls_loss'] = tot_cls_loss.detach().item()

        if torch.isnan(tot_iou_loss):
            fmt['iou_loss'] = 'Nan'
            cmd = True
        else:
            fmt['iou_loss'] = tot_iou_loss.detach().item()

        if torch.isnan(tot_dfl_loss):
            fmt['dfl_loss'] = 'Nan'
            cmd = True
        else:
            fmt['dfl_loss'] = tot_dfl_loss.detach().item()

        if cmd:
            print(msg.format(**fmt))
            input()

        

        return {'tot_loss': tot_cls_loss + tot_iou_loss + tot_dfl_loss, 
                'cls_loss': tot_cls_loss.detach().item(), 
                'iou_loss': tot_iou_loss.detach().item(),
                'dfl_loss': tot_dfl_loss.detach().item(),
                'tar_nums': mask_assign.sum().detach().item()}

    def dfl_box_loss(self, pred_reg:Tensor, pred_xyxy:Tensor, match_box_xyxy:Tensor, match_cls_score:Tensor, strides:Tensor, mask_assign: Tensor, grids:Tensor):
        """
        Inputs:
            pred_reg: (b, N, 4*self.reg)
            pred_xyxy: (b, N, 4) / [xmin, ymin, xmax, ymax]
            match_box_xyxy: (b, N, M, 4) / [xmin, ymin, xmax, ymax]
            match_cls_score: (b, N, num_class)
            strides: (N, 1)
            mask_assign: (b, N, M)
            grids: (N, 2)
        Outputs:
            iou_loss: (X, )
            dfl_loss: (X, )
        """
        match_box = match_box_xyxy / strides[None, :, :, None]  # (b, N, M, 4) & (1, N, 1, 1) -> (b, N, M, 4) / [xmin, ymin, xmax, ymax]
        fg_mask = mask_assign.any(dim=-1)  # (b, N)
        pred_box = pred_xyxy[fg_mask]  # (X, 4)
        tar_box = match_box[mask_assign]  # (X, 4)
        assert pred_box.size() == tar_box.size()

        iou = gpu_CIoU(pred_box, tar_box).unsqueeze(-1)  # (X, 1)
        weight = match_cls_score[fg_mask].sum(-1).unsqueeze(-1)  # (X, 1)
        tar_score_sum = torch.maximum(match_cls_score.sum(), match_cls_score.new_tensor(1.0))
        iou_loss = ((1. - iou) * weight).sum() / tar_score_sum

        tar = xyxy2tblr(match_box, grids)[mask_assign]  # (X, 4) / [t, b, l, r]
        pos_prd = pred_reg[fg_mask].reshape(-1, self.reg).contiguous()  # (X*4, self.reg) / before softmax

        tar.clamp_(0, self.reg - 1 - 0.01)  # range: [0, 14.99]
        tar_left = tar.long()  # range: [0, 14]
        tar_right = tar.long() + 1  # range: [1, 15]
        weight_left = tar_right.float() - tar  # (X, 4)
        weight_right = tar - tar_left.float()  # (X, 4)

        dfl = (F.cross_entropy(pos_prd, tar_left.reshape(-1), reduction='none').reshape(tar.size()) * weight_left + 
               F.cross_entropy(pos_prd, tar_right.reshape(-1), reduction='none').reshape(tar.size()) * weight_right)

        dfl_loss = dfl.mean(-1, keepdim=True) * weight  # (X, 1))
        dfl_loss = dfl_loss.sum() / tar_score_sum
        return iou_loss, dfl_loss

    def build_target(self, mask_assign:Tensor, tar_cls:Tensor, tar_xyxy:Tensor):
        """
        Inputs:
            mask_assign: (b, N, M) / 一个grid只能匹配一个gt, 但一个gt可以匹配多个grid
            tar_xyxy: (b, M, 4)
            tar_cls: (b, M)
        Outputs:
            match_cls_onehot: (b, N, num_class)
            match_box_xyxy: (b, N, M, 4)
        """
        assert mask_assign.sum(-1).max() <= 1, "each grid should be assigned only one gt"
        b, N, M = mask_assign.size()
        idx = mask_assign.float().argmax(-1)  # (b, N)
        match_cls_onehot = tar_cls.new_zeros(b, N, self.num_class)  # (b, N, num_class)
        match_box_xyxy = tar_xyxy.new_zeros(b, N, M, 4)

        y, x = torch.meshgrid((torch.arange(b), torch.arange(N)), indexing='ij')
        grid = torch.stack((y, x), dim=2).contiguous()  # (b, N, 2)
        ys = grid[..., 0]  # (b, N)
        xs = grid[..., 1]  # (b, N)

        match_cls_onehot[ys.flatten(), xs.flatten(), tar_cls[ys.flatten(), idx.flatten()].long()] = 1.
        match_cls_onehot[~(mask_assign.any(dim=-1))] = 0.
        assert match_cls_onehot.sum() == mask_assign.sum()
        match_box_xyxy[ys.flatten(), xs.flatten(), idx.flatten()] = tar_xyxy[ys.flatten(), idx.flatten()]
        match_box_xyxy[~(mask_assign.any(dim=-1))] = 0.
        return match_cls_onehot, match_box_xyxy

    def label_assign(self, mask_topk: Tensor, metric_iou:Tensor):
        """
        assign one gt to every candidate grid.
        Inputs:
            mask_topk: (b, N, M)
            metric_iou: (b, N, M)
        Outputs:
            mask: (b, N, M)
        """
        b, N, M = mask_topk.size()
        if mask_topk.float().sum(-1).max() > 1:  # one grid assigned to multiple gt
            mask_assign = torch.zeros_like(metric_iou)  # (b, N, M)
            idx = (metric_iou * mask_topk.float()).argmax(-1)  # (b, N)
            ys, xs = torch.meshgrid((torch.arange(b), torch.arange(N) ), indexing='ij')
            grid = torch.stack((ys, xs), dim=2).contiguous()  # (b, N, 2)
            mask_assign[grid[..., 0].flatten(), grid[..., 1].flatten(), idx.flatten()] = 1.  # (b, N, M)
            mask_assign[~mask_topk] = 0.
            return mask_assign.type(mask_topk.dtype)
        return mask_topk

    def pred_preprocessing(self, preds, grids):
        """
        Inputs:
            preds: {'pred_xs': (b, num_class+4*reg, 160, 160), 
                    'pred_s': (b, num_class+4*reg, 80, 80), 
                    'pred_m': (b, num_class+4*reg, 40, 40), 
                    'pred_l': (b, num_class+4*reg, 20, 20)}
            grids: (160*160+80*80+40*40*20*20=N, 2) / [x, y]
        Outputs:
            pred_reg: (b, N, 4*self.reg)
            pred_xyxy: (b, N, 4) / [xmin, ymin, xmax, ymax]
            pred_cls: (b, N, num_class)
        """
        b, sf, h, w = list(preds.values())[0].size()
        all_preds = torch.cat([x.reshape(b, sf, -1) for x in preds.values()], dim=-1).permute(0, 2, 1).contiguous()  # (b, 160*160+80*80+40*40*20*20=N, num_class+4*reg)
        pred_reg, pred_cls = all_preds.split((4*self.reg, self.num_class), -1)  # (b, N, 4*reg); (b, N, num_class)
        # (b, N, 4*reg) -> (b, N, 4, reg); (b, N, 4, reg=16) & (reg=16,) -> (b, N, 4) / [t, b, l, r]
        pred_tblr = pred_reg.reshape(b, pred_reg.size(1), 4, -1).softmax(-1).matmul(self.dfl_project.type(all_preds.dtype))
        pred_xyxy = tblr2xyxy(pred_tblr, grids)  # (b, N, 4)
        
        return pred_reg, pred_xyxy, pred_cls
    
    def select_grids_in_gt_bbox(self, tar_xyxy: Tensor, grids: Tensor, gt_mask: Tensor, strides:Tensor):
        """
        Inputs:
            tar_xyxy: (b, M, 4) / [xmin, ymin, xmax, ymax]
            grids: (N, 2) / [x, y]
            gt_mask: (b, M) / bool
            strides: (N, 1)
        Outputs:
            grids_in_gt: (b, N, M) / bool
        """
        N = len(grids)
        tar_box = tar_xyxy[:, None, :, :].repeat(1, N, 1, 1)  # (b, N, M, 4)
        tar_box /= strides[None, :, :, None]
        xmin, ymin, xmax, ymax = tar_box.chunk(4, -1)  # (b, N, M, 1)
        x, y = grids.chunk(2, -1)  # (N, 1)
        l = x[None, :, :, None] - xmin  # (1, N, 1, 1) & (b, N, M, 1) -> (b, N, M, 1)
        t = y[None, :, :, None] - ymin  # (b, N, M, 1)
        r = xmax - x[None, :, :, None]  # (b, N, M, 1)
        b = ymax - y[None, :, :, None]  # (b, N, M, 1)
        delta = torch.cat((t, b, l, r), dim=-1)  # (b, N, M, 4)
        # (b, N, M, 4) -> (b, N, M)
        grids_in_gt = delta.amin(dim=-1).gt(1e-9)
        grids_in_gt = grids_in_gt.permute(0, 2, 1).contiguous()
        grids_in_gt[~gt_mask] = False 
        return grids_in_gt.permute(0, 2, 1).contiguous()
    
    def ciou(self, bbox1:Tensor, bbox2:Tensor):
        """
        Inputs:
            bbox1:(b, N, 4) / [xmin, ymin, xmax, ymax]
            bbox2:(b, M, 4) / [xmin, ymin, xmax, ymax]
        Ouputs:
            iou: (b, N, M)
        """
        bbox1_area = torch.prod(bbox1[:, :, [2, 3]] - bbox1[:, :, [0, 1]], dim=-1)  # (b, N)
        bbox2_area = torch.prod(bbox2[:, :, [2, 3]] - bbox2[:, :, [0, 1]], dim=-1)  # (b, M)

        intersection_w = torch.clamp(torch.minimum(bbox1[..., 2:3], bbox2[..., 2][:, None, :]) - torch.maximum(bbox1[..., 0:1], bbox2[..., 0][:, None, :]), min=0.)  # (b, N, M)
        intersection_h = torch.clamp(torch.minimum(bbox1[..., 3:4], bbox2[..., 3][:, None, :]) - torch.maximum(bbox1[..., 1:2], bbox2[..., 1][:, None, :]), min=0.)  # (b, N, M)
        intersection_area = intersection_w * intersection_h  # (b, N, M)

        union_area = (bbox1_area[..., None] + bbox2_area[:, None, :]) - intersection_area  # (b, N, 1) & (b, 1, M) -> (b, N, M)
        del bbox1_area, bbox2_area
        gc.collect()
        torch.cuda.empty_cache()
        iou = intersection_area / union_area.clamp(1e-6)  # (b, N, M)
        del union_area, intersection_area, intersection_h, intersection_w
        gc.collect()
        torch.cuda.empty_cache()

        c_hs = torch.maximum(bbox1[..., 3:4], bbox2[..., 3][:, None, :]) - torch.minimum(bbox1[..., 1:2], bbox2[..., 1][:, None, :])  # (b, N, M)
        c_ws = torch.maximum(bbox1[..., 2:3], bbox2[..., 2][:, None, :]) - torch.minimum(bbox1[..., 0:1], bbox2[..., 0][:, None, :])  # (b, N, M)
        c_diagonal = torch.pow(c_ws, 2) + torch.pow(c_hs, 2)  # (b, N, M)
        del c_hs, c_ws
        gc.collect()
        torch.cuda.empty_cache()

        # compute center coordinate of bboxes
        # (b, N, 1) & (b, 1, M)
        ctr_ws = ((bbox1[..., 2] + bbox1[..., 0]) / 2)[:, :, None] - ((bbox2[..., 2] + bbox2[..., 0]) / 2)[:, None, :]  # (b, N, M)
        ctr_hs = ((bbox1[..., 3] + bbox1[..., 1]) / 2)[:, :, None] - ((bbox2[..., 3] + bbox2[..., 1]) / 2)[:, None, :]  # (b, N, M)
        # ctr_distance: distance of two bbox center
        ctr_distance = torch.pow(ctr_hs, 2) + torch.pow(ctr_ws, 2)  # (b, N, M)
        del ctr_hs, ctr_ws
        # (b, N, 1) & (b, 1, M)
        v = (4 / (np.pi ** 2)) * torch.pow(torch.atan((bbox1[..., 1] - bbox1[..., 0]) / torch.clamp((bbox1[..., 3] - bbox1[..., 1]), min=1e-6))[:, :, None] - torch.atan((bbox2[..., 2] - bbox2[..., 0]) / torch.clamp((bbox2[..., 3] - bbox2[..., 1]), min=1e-6))[:, None, :], 2)  # (b, N, M)

        with torch.no_grad():
            alpha = v / torch.clamp(1 - iou + v, min=1e-6)
        c_diagonal = torch.clamp(c_diagonal, min=1e-6)
        ciou = iou - (ctr_distance / c_diagonal + v * alpha)  # (b, N, M)
        del c_diagonal, alpha, v
        gc.collect()
        torch.cuda.empty_cache()

        return ciou

    def compute_metric(self, pred_xyxy: Tensor, strides:Tensor, pred_cls: Tensor, tar_xyxy: Tensor, tar_cls: Tensor, mask_grids_in_gt: Tensor):
        """
        Inputs:
            pred_xyxy: (b, N, 4)
            strides: (N, 1)
            pred_cls: (b, N, num_class)
            tar_xyxy: (b, M, 4)
            tar_cls: (b, M)
            mask_grids_in_gt: (b, N, M)

        Outputs:
            metric: (b, N, M)
            metric_iou: (b, N, M)
        """
        b, N, M = mask_grids_in_gt.size()
        pred_box = pred_xyxy * strides[None, :, :]
        pred_box[:, :, [0, 2]] = pred_box[:, :, [0, 2]].clamp(max=self.img_sz[1]-1)
        pred_box[:, :, [1, 3]] = pred_box[:, :, [1, 3]].clamp(max=self.img_sz[0]-1)
        metric_iou = self.ciou(pred_box, tar_xyxy).clamp(min=0.)  # (b, N, M)
        metric_iou[~mask_grids_in_gt] = 0.0
        
        cls_in_grids = tar_cls[:, None, :].repeat(1, N, 1)  # (b, N, M)
        i, j, k = torch.nonzero(mask_grids_in_gt.float(), as_tuple=True)
        c = cls_in_grids[mask_grids_in_gt].long()
        assert (c < 0).sum() == 0, f"mask_grids_in_gt is error!"
        metric_score = pred_cls.new_zeros(b, N, M)
        metric_score[i, j, k] = pred_cls.sigmoid()[i, j, c].type(metric_score.dtype)

        metric = metric_iou.pow(self.beta) * metric_score.pow(self.alpha)
        return metric, metric_iou

    def select_topk_candidates(self, metric:Tensor, mask_grids_in_gt:Tensor):
        """
        select topk candidates for every gt.
        Inputs:
            metric: (b, N, M)
            mask_grids_in_gt: (b, N, M)
        Outputs:
            mask_topk: (b, N, M)
        """
        b, N, M = metric.size()
        idx = metric.topk(self.topk, dim=1)[1]  # (b, topk, M) / 每个gt可以有topk个grid作为正样本 
        zs, ys, xs = torch.meshgrid((torch.arange(b) , torch.arange(self.topk), torch.arange(M)), indexing='ij')
        grid = torch.stack((zs, ys, xs), dim=-1).contiguous()  # (b, topk, M)

        mask_topk = torch.zeros_like(metric)  # (b, N, M)
        mask_topk[grid[..., 0].flatten(), idx.flatten(), grid[..., 2].flatten()] = 1.
        assert (mask_topk.sum(dim=1) > self.topk).sum() == 0

        zero_metric_mask = metric < 1e-9
        mask_topk[~mask_grids_in_gt] = 0.
        mask_topk[zero_metric_mask] = 0.

        return mask_topk.type(torch.bool)

    def make_grid(self, shape_list, strides, device):
        """
        Inputs:
            shape_list: [[h/4, w/4], [h/8, w/8], [h/16, w/16], [h/32, w/32]]
        Outputs:
            g: (N, 2) / [x, y]
            s: (N, 1)
        """
        g = []
        s = []
        for stride, (h, w) in zip(strides, shape_list):
            shift_x = torch.arange(0, w, device=self.device) + 0.5
            shift_y = torch.arange(0, h, device=self.device) + 0.5
            y, x = torch.meshgrid((shift_x, shift_y), indexing='ij')
            # mesh_grid: (col_num, row_num, 2) -> (row_num, col_num, 2)
            mesh_grid = torch.stack((x, y), dim=2).reshape(h, w, 2)
            # (col_num, row_num, 2)
            g.append(mesh_grid.reshape(-1, 2))
            s.append(torch.ones((h*w, 1), device=device) * stride)
        return torch.cat(g, 0), torch.cat(s, 0)

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
