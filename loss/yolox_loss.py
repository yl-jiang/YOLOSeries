import torch
import torch.nn as nn
import numpy as np
from utils import xyxy2xywhn, xyxy2xywh, xywh2xyxy
from utils import gpu_iou, gpu_DIoU, gpu_Giou, gpu_CIoU
import torch.nn.functional as F


class YOLOXLoss:

    def __init__(self, hyp) -> None:
        self.hyp = hyp
        self.num_anchors = hyp['num_anchors']
        self.num_stage = hyp.get('num_stage', 3)
        self.grids = {}
        self.img_sz = hyp['input_img_size']
        self.num_class = hyp['num_class']
        self.use_l1 = hyp.get('use_l1', True)
        self.iou_loss_scale = hyp.get('iou_loss_scale', 0.5)
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
        self.balances = [4., 1., 0.4] if self.num_stage == 3 else [4., 1., 0.4, 0.1]

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
        tot_cls_loss, tot_iou_loss, tot_cof_loss, tot_l1_reg_loss = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        for i, k in enumerate(preds.keys()):  # each stage
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
            grid = grid.repeat(1, self.num_anchors, 1, 1, 1).contiguous().reshape(1, -1, 2)
            
            stage_out_dict = self.calculate_loss_of_each_stage(tars.float(), pred.float(), grid[0].float(), stride)
            cof_loss_tmp = stage_out_dict['cof_loss'] * self.balances[i]
            self.balances[i] = self.balances[i] * 0.9999 + 0.0001 / cof_loss_tmp.detach().item()
            tot_cof_loss += cof_loss_tmp

            tot_num_fg += stage_out_dict['num_fg']
            tot_num_gt += stage_out_dict['num_gt']
            tot_cls_loss += stage_out_dict['cls_loss'] 
            tot_iou_loss += stage_out_dict['iou_loss'] 
            tot_l1_reg_loss += stage_out_dict['l1_reg_loss']
            del stage_out_dict
        
        self.balances = [x/self.balances[1] for x in self.balances]
        if self.num_class == 1:
            tot_cls_loss = tot_cof_loss.new_tensor(0.0)

        tot_iou_loss *= self.iou_loss_scale
        tot_cls_loss *= self.cls_loss_scale
        tot_cof_loss *= self.cof_loss_scale
        tot_l1_reg_loss *= self.l1_loss_scale
        tot_loss = (tot_iou_loss + tot_cls_loss + tot_cof_loss + tot_l1_reg_loss) * batch_size

        loss_dict = {'tot_loss': tot_loss, 
                    'iou_loss': tot_iou_loss, 
                    'l1_reg_loss': tot_l1_reg_loss, 
                    'cls_loss': tot_cls_loss, 
                    'cof_loss': tot_cof_loss, 
                    'fg_nums': tot_num_fg, 
                    'tar_nums': tot_num_gt}

        return loss_dict

    def calculate_loss_of_each_stage(self, tars, preds, grid, stride):
        """
        Args:
            preds: 某个stage的预测输出 (N, num_anchors*h*w, 85): (N, num_anchors*28*28, 85)或(N, num_anchors*14*14, 85)或(N, num_anchors*7*7, 85) / [x, y, w, h, cof, cls1, cls2, ...]
            tars: tensor; (N, num_bbox, 6) / [x_ctr, y_ctr, w, h, class_id, img_id]
            stride: a scalar / 下采样尺度 / 取值: 8 or 16 or 32
            grid: (num_anchors*h*w, 2)
        Return:

        """
        batch_tar_cls, batch_tar_box, batch_tar_cof, fg, batch_tar_box_l1 = [], [], [], [], []
        tot_num_gt, tot_num_fg = 0, 0
        # (N, num_anchors*h*w, 4)
        origin_pred_box = preds[..., :4].clone()
        # restore predictions to input scale / (N, num_anchors*h*w, 85)
        preds[..., :2] = (preds[..., :2] + grid[None, ...]) * stride
        preds[..., 2:4] = torch.exp(preds[..., 2:4]) * stride  # 这一步可能由于preds[..., 2:4]值过大, 进而导致exp计算后溢出得到Nan值

        for i in range(tars.size(0)):  # each image
            tar = tars[i]  # (num_bbox, 6)
            pred = preds[i]  # (num_anchors*h*w, 85)
            valid_gt_idx = tar[:, 4] >= 0  # 有效label索引(那些没有bbox的gt对应的class值为-1)
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
        iou_loss = self.iou_loss(preds[..., :4].float(), batch_tar_box.float(), fg, self.hyp['iou_type'])  # regression

        # cofidence loss
        if self.hyp['use_focal_loss']:
            cof_factor = self.focal_loss_factor(preds[..., 4].view(-1, 1), batch_tar_cof.type(preds.type()))
        else:
            cof_factor = torch.ones_like(batch_tar_cof.type(preds.type()))
        cof_loss = self.bce_cof(preds[..., 4].view(-1, 1), batch_tar_cof.type(preds.type()))  # cofidence
        cof_loss *= cof_factor

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

        out_dict = {'iou_loss': iou_loss.mean(), 
                    'l1_reg_loss': l1_reg_loss.mean(), 
                    'cls_loss': cls_loss.mean(), 
                    'cof_loss': cof_loss.mean(), 
                    'num_fg': tot_num_fg, 
                    'num_gt': tot_num_gt}

        return out_dict

    def select_grid(self, tar_box, grid, stride):
        """
        根据target box选择合适的grid(选择合适的grid即是选择合适的prediction)参与loss的计算
        Args:
            tar_box: (X, 4) / [x, y, w, h] / X -> 该image包含的有效的gt box个数
            grid: (h*w, 2) / [x, y]
            stride: scalar / downsample scale
        Returns:
            is_grid_in_gtbox_or_gtctr: (h*w,) / front ground mask / 其中为True的元素个数设为Y, 则Y >= N
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
        # (valid_num_box, h*w) -> (h*w,) / 对该image, 所有满足该条件grid的并集
        is_grid_in_gtbox_all = is_grid_in_gtbox.sum(0) > 0.0

        # 如果grid的中心点坐标均没有在任何gt box内部, 则对每个gt box而言, 对每个tar选取距离最近的grid作为匹配的grid
        if is_grid_in_gtbox_all.sum() == 0:
            # (valid_num_box, 1, 2) & (1, h*w, 2) -> (valid_num_box, h*w, 2) -> (valid_num_box, h*w)
            ctr_distance = torch.norm(tar_box[:, :2].unsqueeze(1) - ctr_grid[:, :2].unsqueeze(0), dim=2)
            # (valid_num_box,)
            dist_argmin = torch.argmin(ctr_distance, dim=1)
            valid_idx = torch.unique(dist_argmin)
            choose_num = int(len(valid_idx) * 0.2) if len(valid_idx) * 0.2 > 2 else 1
            random_idx = dist_argmin[torch.randperm(len(dist_argmin))]
            is_grid_in_gtbox_all[random_idx[:choose_num]] = True

        # ======== 某个gird的中心坐标是否落到以某个gt box的中心点为圆心, center_radius为半径的圆形区域内 =========
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
        # (X, h*w) -> (h*w,) / 对该image, 所有满足该条件grid的并集
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
        每个gt box都可能有好几个满足条件的(位于前景)prediction, 这一步需要在其中挑选出最有价值的参与loss的计算.
        通过传入的cost和iou, 对每个gt box选择与之最匹配的若干个prediction, 并且使得每个prediction最多只能匹配一个gt box.

        注意: 
            传入的frontground_mask被inplace的修改了。
        Args:
            frontground_mask: (h*w,) / bool / 其中为True的元素个数等于Y
            cost: (valid_num_box, Y)
            iou: (valid_num_box, Y)
        Returns:
            num_fg: 选取的prediction个数, 假设为M(M的取值位于[0, Y])
            matched_gt_cls: 与每个prediction最匹配的gt class id
            matched_iou: 每个prediction与之最匹配的gt box之间额iou值
            matched_gt_idx: 
        """
        assert cost.size(0) == iou.size(0)

        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # 如果位于前景的prediction个数大于10, 则每个gt box最多选择10个最适合的预测作为后续的loss计算
        k = min(10, iou.size(1))  
        # (valid_num_box, k)
        topk_iou, _ = torch.topk(iou, k, dim=1)
        # (valid_num_box, k) -> (valid_num_box, )
        dynamic_k = torch.clamp(topk_iou.sum(1).int(), min=1, max=cost.size(1)).tolist()
        for i in range(cost.size(0)):  # each valid gt
            # 选取最小的dynamic_k[i]个值(因为不满足条件的prediction对应的cost加上了1000000)
            _, pos_idx = torch.topk(cost[i], k=dynamic_k[i], largest=False)
            matching_matrix[i][pos_idx] = 1

        del topk_iou, dynamic_k

        # (valid_num_box, Y) -> (Y,)  / 满足条件的prediction的并集(存在某个prediction匹配到多个gt box)
        all_matching_gt = matching_matrix.sum(0) 
        # 如果存在某个prediction匹配到多个gt box的情况
        if (all_matching_gt > 1).sum() > 0:  
            _, cost_argmin = torch.min(cost[:, all_matching_gt > 1], dim=0)
            # 处理某些prediction匹配到多个gt box的情况, 将这些prediction只分配到与其匹配度最高的gt box
            matching_matrix[:, all_matching_gt > 1] = 0
            matching_matrix[cost_argmin, all_matching_gt > 1] = 1

        # (valid_num_box, Y) -> (Y,) / fg_mask总共有Y个元素
        fg_mask = matching_matrix.sum(0) > 0
        # num_fg的值应该位于区间[0, Y]
        num_fg = fg_mask.sum().item()
        # update front ground mask
        assert len(frontground_mask[frontground_mask.clone()]) == len(fg_mask)
        frontground_mask[frontground_mask.clone()] = fg_mask

        # (valid_num_box, M) -> (M,) / 假设fg_mask中为True的个数为M, 则Y >= M / matched_gt的取值范围为[0, valid_num_box]
        matched_gt_idx = matching_matrix[:, fg_mask].argmax(0)
        # (valid_num_box, Y) & (valid_num_box, Y) -> (valid_num_box, Y) -> (Y,) -> (M,)
        matched_iou = (matching_matrix * iou).sum(0)[fg_mask]
        return frontground_mask, num_fg, matched_iou, matched_gt_idx

    def _make_grid(self, h, w, dtype):
        ys, xs = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device))
        # 排列成(x, y)的形式, 是因为模型输出的预测结果的排列是[x, y, w, h, cof, cls1, cls2, ...]
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

        if iou_type == 'iou':  # 使用iou训练, 使用sgd作为优化器且lr设置稍大时, 训练过程中容易出现Nan
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
                alpha = v / (1 - iou + v + 1e-8)
            ciou = iou.float() - ctr_distance.float() / c_diagonal.float() - v.float() * alpha.float()  # (N,)
            return 1 - ciou
        else:
            raise ValueError(f"Unknow iou_type '{iou_type}', must be one of ['iou', 'giou', 'ciou']")

    def build_l1_target(self, grid, stride, tar_box, num_fg, fg):
        """
        将target转换到对应stage的prediction一致的数据格式(即: 将(ctr_x, ctr_y)转换为相对于对应的grid左上角的偏移量, 将(w, h)转换为对应尺度下的长和宽)
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
