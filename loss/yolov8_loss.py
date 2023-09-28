import torch
import torch.nn as nn
import numpy as np
from utils import tblr2xyxy, gpu_CIoU, xyxy2tblr
import torch.nn.functional as F
from torch import Tensor, BoolTensor, FloatTensor
import gc 
from typing import Dict

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
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight, reduction='none').to(self.device)
        # self.dfl_project = torch.arange(hyp['reg'], device=self.device, dtype=torch.float32)
        self.dfl_project = torch.arange(1, hyp['reg']+1, device=self.device, dtype=torch.float32)
        self.grids, self.strides = None, None

    def __call__(self, preds:Dict, tars:Tensor):
        """
        假设输入训练的图片尺寸为640x640x3

        Args:
            preds: {'pred_xs': (b, num_class+4*reg, 160, 160), 
                    'pred_s' : (b, num_class+4*reg, 80, 80), 
                    'pred_m' : (b, num_class+4*reg, 40, 40), 
                    'pred_l' : (b, num_class+4*reg, 20, 20)}
            tars : (b, M, 6) / [xmin, ymin, xmax, ymax, class_id, img_id]
        """

        fm_shapes  = [[f.size(2), f.size(3)] for f in preds.values()]
        fm_strides = [self.img_sz[0] / f.size(2) for f in preds.values()]
        if self.grids is None or self.strides is None:
            self.grids, self.strides = self.make_grid(fm_shapes, fm_strides, self.device)  # grids: (N, 2), strides: (N, 1)

        tar_xyxy, tar_cls = tars[..., :4], tars[..., 4]  # (b, M, 4); (b, M)
        # pred_dfl: (b, N, reg*4); pred_xyxy: (b, N, 4); pred_cls: (b, N, num_class)
        pred_dfl, pred_xyxy, pred_cls = self.pred_preprocessing(preds) 
        
        mask_assign, cls_score, box_xyxy = self.build_target(pred_cls, pred_xyxy, tar_cls, tar_xyxy)
        iou_loss, dfl_loss = self.dfl_box_loss(pred_dfl, pred_xyxy, box_xyxy, cls_score, mask_assign)
        cls_loss = self.bce_cls(pred_cls.view(-1, self.num_class).contiguous(), 
                                cls_score.view(-1, self.num_class).contiguous())
        cls_loss_factor = self.focal_loss_factor(pred_cls.view(-1, self.num_class).contiguous(), 
                                                 cls_score.view(-1, self.num_class).contiguous())  # (b*N, num_class)
        cls_loss *= cls_loss_factor
        cls_loss = cls_loss.sum() / max(cls_score.sum(), 1.0)

        b = tars.size(0)
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
                'tar_nums': mask_assign.sum().long().detach().item()}
    
    def pred_preprocessing(self, 
                           preds: Dict):
        """
        Inputs:
            preds: {
                      'pred_xs': (b, num_class+4*reg, 160, 160), 
                      'pred_s' : (b, num_class+4*reg, 80, 80), 
                      'pred_m' : (b, num_class+4*reg, 40, 40), 
                      'pred_l' : (b, num_class+4*reg, 20, 20)
                    }
        Outputs:
            pred_dfl : (b, N, 4*self.reg)
            pred_xyxy: (b, N, 4) / [xmin, ymin, xmax, ymax]
            pred_cls : (b, N, num_class)
        """
        b, sf, h, w = list(preds.values())[0].size()
        # all_preds: (b, 160*160+80*80+40*40*20*20=N, num_class+4*reg)
        all_preds   = torch.cat([x.view(b, sf, -1).contiguous() for x in preds.values()], dim=-1).permute(0, 2, 1).contiguous()  
        pred_dfl, pred_cls = all_preds.split((4*self.reg, self.num_class), -1)  # (b, N, 4*reg); (b, N, num_class)
        # (b, N, 4*reg) -> (b, N, 4, reg); (b, N, 4, reg=16) & (reg=16,) -> (b, N, 4) / [t, b, l, r]
        pred_tblr = pred_dfl.view(b, pred_dfl.size(1), 4, -1).contiguous().softmax(-1).matmul(self.dfl_project.type_as(all_preds))
        pred_xyxy = tblr2xyxy(pred_tblr, self.grids)  # (b, N, 4)
        
        return pred_dfl, pred_xyxy, pred_cls
    
    def select_grids_in_gt_bbox(self, 
                                tar_xyxy: FloatTensor, 
                                gt_mask : BoolTensor):
        """
        Inputs:
            tar_xyxy: (b, M, 4) / [xmin, ymin, xmax, ymax]
            gt_mask: (b, M) / bool
        Outputs:
            grids_in_gt: (b, M, N) / bool / 这里的mask存在一个grid匹配多个gt obj和一个gt obj匹配多个grid的情况, 后续还有步骤进行近一步清理
        """
        N = len(self.grids)
        tar_box = tar_xyxy[:, :, None, :].repeat(1, 1, N, 1)  # (b, M, 4) -> (b, M, 1, 4) -> (b, M, N, 4)
        xmin, ymin, xmax, ymax = tar_box.chunk(4, -1)  # (b, M, N, 1)
        x, y = (self.grids * self.strides).chunk(2, -1)  # (N, 2) & (N, 1) -> (N, 2) -> (N, 1)
        l = x[None, None, ...] - xmin  # (1, 1, N, 1) & (b, M, N, 1) -> (b, M, N, 1)
        t = y[None, None, ...] - ymin  # (b, M, N, 1)
        r = xmax - x[None, None, ...]  # (b, M, N, 1)
        b = ymax - y[None, None, ...]  # (b, M, N, 1)
        grids_in_gt = torch.cat((t, b, l, r), dim=-1).contiguous()  # (b, M, N, 4)
        grids_in_gt = grids_in_gt.amin(-1).gt_(1e-9)  # (b, M, N, 4) -> (b, M, N) / floatTensor
        return grids_in_gt.type_as(gt_mask)
    
    def compute_metric(self, 
                       pred_xyxy: FloatTensor, 
                       pred_cls : FloatTensor, 
                       tar_xyxy : FloatTensor, 
                       tar_cls  : FloatTensor, 
                       mask_grids_in_gt: BoolTensor):
        """
        Inputs:
            pred_xyxy       : (b, N, 4)
            pred_cls        : (b, N, num_class+1)
            tar_xyxy        : (b, M, 4)
            tar_cls         : (b, M)
            mask_grids_in_gt: (b, M, N) / bool

        Outputs:
            metric: (b, M, N)
            iou   : (b, M, N)
        """
        pred_box = pred_xyxy * self.strides[None]  # (b, N, 4) & (1, N, 1) -> (b, N, 4)
        b, M, N  = mask_grids_in_gt.size()
        pred_box = pred_box.unsqueeze_(1).expand(-1, M, -1, -1)[mask_grids_in_gt]  # (b, M, N, 4) -> (X, 4)
        tar_box  = tar_xyxy.unsqueeze(2).expand(-1, -1, N, -1)[mask_grids_in_gt]   # (b, M, N, 4) -> (X, 4)
        iou = torch.zeros_like(mask_grids_in_gt).type_as(pred_box)  # (b, M, N)
        iou[mask_grids_in_gt] = self.ciou(tar_box, pred_box).clamp_(min=0.)  # (b, M, N)
        
        c = tar_cls.long()  # (b, M)
        i = torch.arange(b).view(-1, 1).expand(-1, M)  # (b, M)
        score = pred_cls.new_zeros(b, M, N)
        score[mask_grids_in_gt] = pred_cls.sigmoid()[i, :, c].type(score.dtype)[mask_grids_in_gt]  # (b, N, num_class) -> (b, M, N)

        metric = iou.pow(self.beta) * score.pow(self.alpha)  # (b, M, N)
        return metric, iou

    def select_topk_candidates(self, 
                               metric :FloatTensor, 
                               mask_gt: BoolTensor):
        """
        select topk candidates for every gt.
        Inputs:
            metric : (b, M, N)
            mask_gt: (b, M)
        Outputs:
            mask_topk: (b, M, N)
        """
        b, M, N  = metric.size()
        topk_idx = metric.topk(self.topk, dim=-1)[1]   # (b, M, N)-> (b, M, topk) / 给每个gt分配最多topk个grid
        zs, ys, xs = torch.meshgrid(torch.arange(b), torch.arange(M), torch.arange(self.topk), indexing='ij')
        grid = torch.stack((zs, ys, xs), dim=-1).contiguous()  # (b, M, topk)

        mask_topk = torch.zeros_like(metric)  # (b, M, N)
        mask_topk[grid[..., 0].flatten(), grid[..., 1].flatten(), topk_idx.flatten()] = 1.  # (b, M, N)
        mask_topk.masked_fill_(~(mask_gt)[:, :, None].expand(-1, -1, N), 0.)
        return mask_topk
    
    def select_single_candidate_by_iou(self, 
                                       mask_topk: FloatTensor, 
                                       iou      :FloatTensor):
        """
        assign one gt to every candidate grid.
        Inputs:
            mask_topk : (b, M, N)
            iou       : (b, M, N)
        Outputs:
            mask: (b, M, N)
        """
        b, M, N = mask_topk.size()
        if mask_topk.sum(1).max() > 1:  # exist the situation which one grid assigned to predict multiple gt obj
            mask_fg = mask_topk.sum(1)  # (b, M, N) -> (b, N)
            mask_one_grid_multi_gt = (mask_fg[:, None, :] > 1).expand(-1, M, -1)  # (b, N) -> (b, 1, N) -> (b, M, N)
            max_iou_idx  = iou.argmax(1)  # (b, N)
            mask_max_iou = torch.zeros_like(mask_topk)  # (b, M, N)
            mask_max_iou.scatter_(1, max_iou_idx.unsqueeze(1), 1)
            mask_assign = torch.where(mask_one_grid_multi_gt, mask_max_iou, mask_topk)  # (b, M, N)
            return mask_assign
        return mask_topk
    
    @torch.no_grad()
    def label_assign(self, pred_cls, pred_xyxy, tar_cls, tar_xyxy):
        """
        Inputs:
            pred_cls : (b, N, num_class)
            pred_xyxy: (b, N, 4)
            tar_cls  : (b, M)
            tar_xyxy : (b, M, 4)
        Outputs:
            mask_assign: (b, M, N) / 一个grid只能匹配一个gt box, 但一个gt box可以匹配多个grid
            metric     : (b, M, N)
            iou        : (b, M, N)
        """
        mask_gt          = tar_cls >= 0  # (b, M)
        mask_grids_in_gt = self.select_grids_in_gt_bbox(tar_xyxy, mask_gt)  # (b, M, N)
        metric, iou      = self.compute_metric(pred_xyxy, pred_cls, tar_xyxy, tar_cls, mask_grids_in_gt)  # metric: (b, M, N); metric_iou: (b, M, N)
        mask_topk        = self.select_topk_candidates(metric, mask_gt)  # (b, M, N)
        mask_assign      = self.select_single_candidate_by_iou(mask_topk, iou)  # (b, N, M)
        return mask_assign, metric, iou
    
    @torch.no_grad()
    def build_target(self, 
                     pred_cls :FloatTensor, 
                     pred_xyxy:FloatTensor, 
                     tar_cls  :FloatTensor, 
                     tar_xyxy :FloatTensor):
        """
        Inputs:
            pred_cls : (b, N, num_class)
            pred_xyxy: (b, N, 4)
            tar_xyxy : (b, M, 4)
            tar_cls  : (b, M)
        Outputs:
            cls_score: (b, N, num_class)
            box_xyxy : (b, N, 4)
        """
        mask_assign, metric, iou = self.label_assign(pred_cls, pred_xyxy, tar_cls, tar_xyxy)
        
        assert mask_assign.sum(1).max() <= 1, "each grid should be assigned only one gt"
        b, M, N    = mask_assign.size()
        cls_onehot = tar_cls.new_zeros(b, N, self.num_class)  # (b, N, num_class)
        box_xyxy   = tar_xyxy.new_zeros(b, N, 4)  # (b, N, 4)

        y, x = torch.meshgrid(torch.arange(b), torch.arange(N), indexing='ij')
        grid_coords = torch.stack((y, x), dim=2).contiguous()  # (b, N, 2)
        ys = grid_coords[..., 0].flatten()  # (b*N,)
        xs = grid_coords[..., 1].flatten()  # (b*N,)

        i = mask_assign.argmax(1).flatten()  # (b, M, N) -> (b, N) -> (b*N) / each grid's gt box index
        cls_onehot[ys, xs, tar_cls[ys, i].long()] = 1.  # (b, N, num_class)
        mask_fg = (mask_assign.sum(1) > 0).unsqueeze_(-1).expand(-1, -1, self.num_class)  # (b, N) -> (b, N, 1) -> (b, N, num_class)
        cls_onehot = torch.where(mask_fg, cls_onehot, 0)
        metric *= mask_assign  # (b, M, N)
        iou    *= mask_assign  # (b, M, N)
        # (b, M, N) & (b, M, 1) -> (b, M, N) & (b, M, 1) -> (b, M, N) -> (b, N) -> (b, N, 1)
        norm_metric = ((metric * iou.amax(-1, True)) / (metric.amax(-1, True) + 1e-9)).amax(dim=1).unsqueeze(-1)  # (b, N, 1)
        cls_score   = cls_onehot * norm_metric  # (b, N, num_class) & (b, N, 1) -> (b, N, num_class)

        box_xyxy[ys, xs] = tar_xyxy[ys, i]
        return mask_assign, cls_score, box_xyxy
    
    def dfl_box_loss(self, 
                     pred_reg:Tensor, 
                     pred_xyxy:Tensor, 
                     box_xyxy:Tensor, 
                     cls_score:Tensor, 
                     mask_assign: BoolTensor):
        """
        Inputs:
            pred_reg    : (b, N, 4*self.reg)
            pred_xyxy   : (b, N, 4) 
            box_xyxy    : (b, N, 4)
            cls_score   : (b, N, num_class)
            mask_assign : (b, M, N)
        Outputs:
            iou_loss: (X, )
            dfl_loss: (X, )
        """
        # iou loss
        box      = box_xyxy / self.strides[None]  # (b, N, 4) & (1, N, 1) -> (b, N, 4)
        mask_fg  = mask_assign.sum(1) > 0  # (b, N)
        pred_box = pred_xyxy[mask_fg]  # (X, 4)
        tar_xyxy = box[mask_fg]  # (X, 4)
        assert pred_box.size() == tar_xyxy.size()
        iou = gpu_CIoU(pred_box, tar_xyxy).unsqueeze(-1)  # (X, 1)
        weight = cls_score[mask_fg].sum(-1).unsqueeze(-1)  # (b, N, num_class) -> (X, num_class) -> (X,) -> (X, 1)
        tar_score_sum = max(cls_score.sum(), 1)
        iou_loss = ((1. - iou) * weight).sum() / tar_score_sum

        # distribution focal loss
        tar_tblr = xyxy2tblr(box, self.grids)[mask_fg]  # (X, 4) / [t, b, l, r]
        tar_tblr.clamp_(0, self.reg - 1 - 0.01)  # range: [0, 14.99]
        pos_pred     = pred_reg[mask_fg].view(-1, self.reg).contiguous()  # (X*4, self.reg) / before softmax
        tar_left     = tar_tblr.long()  # range: [0, 14]
        tar_right    = tar_left + 1  # range: [1, 15]
        weight_left  = tar_right.float() - tar_tblr  # (X, 4)
        weight_right = 1 - weight_left  # (X, 4)

        dfl = (F.cross_entropy(pos_pred, tar_left.view(-1).contiguous() , reduction='none').view(tar_tblr.size()) * weight_left + 
               F.cross_entropy(pos_pred, tar_right.view(-1).contiguous(), reduction='none').view(tar_tblr.size()) * weight_right)  # (X, self.reg)

        dfl_loss = dfl.mean(-1, keepdim=True) * weight  # (X, 1))
        dfl_loss = dfl_loss.sum() / tar_score_sum
        return iou_loss, dfl_loss

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
            mesh_grid = torch.stack((x, y), dim=2).view(h, w, 2)
            # (col_num, row_num, 2)
            g.append(mesh_grid.view(-1, 2))
            s.append(torch.ones((h*w, 1), device=device) * stride)
        return torch.cat(g, 0), torch.cat(s, 0)

    def focal_loss_factor(self, pred, target):
        """
        compute classification loss weights
        Args:
            pred  : (N, 80)
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

    def ciou(self, bbox1:Tensor, bbox2:Tensor):
        """
        Inputs:
            bbox1:(x, 4) / [xmin, ymin, xmax, ymax]
            bbox2:(x, 4) / [xmin, ymin, xmax, ymax]
        Ouputs:
            iou: (x,)
        """
        eps = 1e-6
        box1_xmin, box1_ymin, box1_xmax, box1_ymax = bbox1.chunk(4, -1)  # (x, 1)
        box2_xmin, box2_ymin, box2_xmax, box2_ymax = bbox2.chunk(4, -1)  # (x, 1)
        box1_w = box1_xmax - box1_xmin  # (x, 1)
        box1_h = box1_ymax - box1_ymin  # (x, 1)
        box2_w = box2_xmax - box2_xmin  # (x, 1)
        box2_h = box2_ymax - box2_ymin  # (x, 1)

        inter_area = (box1_xmax.minimum(box2_xmax) - box1_xmin.maximum(box2_xmin)).clamp_(0.0) * \
                     (box1_ymax.minimum(box2_ymax) - box1_ymin.maximum(box2_ymin)).clamp_(0.0)  # (x, 1)
        union_area = ((box1_w * box1_h).clamp_(0) + (box2_w * box2_h).clamp_(0) - inter_area).clamp_(eps)  # (x, 1)        

        iou = inter_area / union_area  # (x, 1)
        del union_area, inter_area
        gc.collect()
        torch.cuda.empty_cache()

        union_box_ws = box1_xmax.maximum(box2_xmax) - box1_xmin.minimum(box2_xmin) 
        union_box_hs = box1_ymax.maximum(box2_ymax) - box1_ymin.minimum(box2_ymin) 
        union_box_diag_square  = (union_box_hs.pow_(2) + union_box_ws.pow_(2)).clamp_(eps)
        boxes_center_dis_square = ((box1_xmax + box1_xmin - (box2_xmax + box2_xmin)) ** 2 + (box1_ymax + box1_ymin - (box2_ymin + box2_ymax)) ** 2) / 4
        del union_box_ws, union_box_hs
        gc.collect()
        torch.cuda.empty_cache()

        v = 4 / (np.pi ** 2) * (torch.atan(box1_w / box1_h) - torch.atan(box2_w / box2_h)).pow(2)
        with torch.no_grad():
            alpha = v / (1 - iou + v).clamp_(eps)  # (x,)
        ciou = iou - (boxes_center_dis_square / union_box_diag_square + v * alpha)  # (b, M, N)
        del alpha, v, iou, boxes_center_dis_square, union_box_diag_square
        gc.collect()
        torch.cuda.empty_cache()

        return ciou.squeeze_().contiguous()