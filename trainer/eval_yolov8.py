import torch
import numpy as np
import torch.nn.functional as F
from utils import gpu_nms, numba_iou, numba_nms
from utils import xywh2xyxy, tblr2xyxy
from collections import defaultdict
from utils import weighted_fusion_bbox
from torch import nn 

__all__ = ['YOLOV8Evaluator']
class YOLOV8Evaluator:

    def __init__(self, yolo, hyp, compute_metric=False):
        self.yolo = yolo
        self.hyp = hyp
        self.device = hyp['device']
        self.num_class = hyp['num_class']
        self.inp_h, self.inp_w = hyp['input_img_size']
        self.use_tta = hyp['use_tta']
        self.grids = None
        self.iou_threshold = self.hyp['compute_metric_iou_threshold'] if compute_metric else self.hyp['iou_threshold']
        self.cls_threshold = self.hyp['compute_metric_cls_threshold'] if compute_metric else self.hyp['cls_threshold']
        self.reg = hyp['reg']

    @torch.no_grad()
    def __call__(self, inputs):
        """
        Inputs:
            inputs: (b, 3, h, w) / tensor from testdataloader
        Outputs:
            (N, 6) / [xmin, ymin, xmax, ymax, score, cls_id]
        """
        torch.cuda.empty_cache()
        if self.use_tta:
            merge_preds_out, inpendent_preds_out = self.test_time_augmentation(inputs)  # (bs, N, 84)
        else:
            merge_preds_out = self.do_inference(inputs)  # (bs, N, 85)
        return [torch.from_numpy(x) if x is not None else None for x in self.numba_nms(merge_preds_out) ]

    def do_nms(self, preds_out):
        """
         Do NMS with torch
        :param preds_out: (batch_size, X, 85)
        :return: list / [(X, 6), ..., None, (Y, 6), None, ..., (Z, 6), ...]
        """
        obj_conf_mask = preds_out[:, :, 4] >= self.conf_threshold
        outputs = []
        for i in range(preds_out.size(0)):  # each image
            x = preds_out[i][obj_conf_mask[i]]
            if x.size(0) == 0:
                outputs.append(None)
                continue
            x[:, 5:] *= x[:, 4:5]  # conf = cls_conf * obj_conf
            # [centerx, centery, w, h] -> [xmin, ymin, xmax, ymax]
            box = xywh2xyxy(x[:, :4])
            if self.hyp['mutil_label']:
                row_idx, col_idx = (x[:, 5:] >= self.cls_threshold).nonzero(as_tuple=True)
                # x: [xmin, ymin, xmax, ymax, conf, cls_id]
                x = torch.cat((box[row_idx], x[row_idx, col_idx+5][:, None], col_idx[:, None].float()), dim=1)
            else:
                cls_conf, col_idx = x[:, 5:].max(dim=1, keepdim=True)
                # [xmin, ymin, xmax, ymax, conf, cls_id]
                x = torch.cat((box, cls_conf, col_idx.float()), dim=1)
                cls_conf_mask = cls_conf.view(-1).contiguous() > self.cls_threshold
                x = x[cls_conf_mask]

            bbox_num = x.size(0)
            if not bbox_num:
                outputs.append(None)
                continue

            if self.hyp['agnostic']:  # 每张image的每个class之间的bbox进行nms
                # 给每一个类别的bbox加上一个特殊的偏置，从而使得NMS时可以在每个类别间的bbox进行
                box_offset = x[:, 5] * 4096
            else:
                box_offset = x[:, 5] * 0.
            bboxes_offseted = x[:, :4] + box_offset[:, None]  # M
            scores = x[:, 4]
            keep_index = gpu_nms(bboxes_offseted, scores, self.hyp['iou_type'], self.iou_threshold)

            if len(keep_index) > self.hyp['max_predictions_per_img']:
                keep_index = keep_index[:self.hyp['max_predictions_per_img']]  # N

            # 对每个bbox进行调优(每个最终输出的bbox都由与其iou大于iou threshold的一些bbox共同merge得到的)
            if self.hyp['postprocess_bbox']:
                if 1 < bbox_num < 3000:
                    iou = self.bbox_iou(bboxes_offseted[keep_index], bboxes_offseted)  # (N, M)
                    iou_mask = iou > self.iou_threshold  # (N, M)
                    weights = iou_mask * scores[None, :]  # (N, M)
                    # (N, M) & (M, 4) & (N, 1) -> (N, 4)
                    bboxes_offseted[keep_index, :4] = torch.mm(weights, bboxes_offseted[:, :4]).float() / weights.sum(dim=1, keepdims=True)
                    # 因为如果一个区域有物体，网络应该在这一区域内给出很多不同的预测框，我们再从这些预测框中选取一个最好的作为该处obj的最终输出；
                    # 如果在某个grid处网络只给出了很少的几个预测框，则更倾向于认为这是网络预测错误所致
                    keep_index = torch.tensor(keep_index)[iou_mask.float().sum(dim=1) > 1]
            outputs.append(x[keep_index])
        return outputs

    def test_time_augmentation(self, inputs):  # just for inference not training time
        """

        :param inputs: (bn, 3, h, w)
        :return:
        """
        bs, img_h, img_w = inputs.size(0), inputs.size(2), inputs.size(3)
        scale_facotr = [1, 0.83, 0.67]
        flip_axis = [None, 2, 3]
        aug_preds = []

        for s, f in zip(scale_facotr, flip_axis):
            if f:
                img = inputs.flip(dims=(f,))
            else:
                img = inputs
            img = self.scale_img(img, s)
            # (bs, M, 85)
            ripe_preds = self.do_inference(img)
            ripe_preds[..., :4] /= s
            if f == 2:  # flip axis y
                ymin = img_h - ripe_preds[..., 3]
                ymax = img_h - ripe_preds[..., 1]
                ripe_preds[..., 1] = ymin
                ripe_preds[..., 3] = ymax
            if f == 3:  # flip axis x
                xmin = img_w - ripe_preds[..., 2]
                xmax = img_w - ripe_preds[..., 0]
                ripe_preds[..., 0] = xmin
                ripe_preds[..., 2] = xmax
            # [(bs, M, 85), (bs, N, 85), (bs, P, 85)]
            aug_preds.append(ripe_preds)
        # (bs, M+N+P, 85)
        return torch.cat(aug_preds, dim=1).contiguous(), aug_preds

    @torch.no_grad()
    def post_processing(self, pred_reg:torch.Tensor):
        b, N, c = pred_reg.size()
        conv = nn.Conv2d(self.reg, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(self.reg, dtype=torch.float)
        conv.weight.data[:] = nn.Parameter(x.view(1, self.reg, 1, 1))
        # (b, N, 4*reg) -> (b, N, 4, reg) -> (b, reg, 4, N) & convolution -> (b, 1, 4, N) -> (b, N, 4, 1) -> (b, N, 4) / [t, b, l, r]
        # pred_reg = conv(pred_reg.cpu().reshape(b, N, 4, c//4).permute(0, 3, 2, 1).contiguous().softmax(1)).permute(0, 3, 2, 1).contiguous().squeeze(-1)
        pred_reg = pred_reg.view(b, pred_reg.size(1), 4, -1).cpu().softmax(-1).matmul(x)
        return pred_reg  # (b, N, 4)

    @torch.no_grad()
    def do_inference(self, inputs):
        b, c, h, w = inputs.size()
        # preds: {'pred_xs': (b, num_class+4*reg, 160, 160), 'pred_s': (b, num_class+4*reg, 80, 80), 'pred_m': (b, num_class+4*reg, 40, 40), 'pred_l': (b, num_class+4*reg, 20, 20)}
        preds = self.yolo(inputs)
        if self.grids is None:
            fm_shapes = [[f.size(2), f.size(3)] for f in preds.values()]
            strides   = [h/f.size(2) for f in preds.values()]
            self.grids, self.strides = self.make_grid(fm_shapes, strides, self.device)  # grids: (N, 2), strides: (N, 1)
        sf = list(preds.values())[0].size(1)
        all_preds = torch.cat([x.reshape(b, sf, -1) for x in preds.values()], dim=2).permute(0, 2, 1).contiguous()  # (b, 160*160+80*80+40*40*20*20=N, num_class+4*reg)
        pred_reg, pred_cls = all_preds.split((4*self.reg, self.num_class), -1)  # (b, N, 4*reg); (b, N, num_class)
        pred_reg = self.post_processing(pred_reg)  # (b, N, 4) / [t, b, l, r]
        pred_xyxy = tblr2xyxy(pred_reg.to(self.device), self.grids) * self.strides.unsqueeze(0)  # (b, N, 4) & (1, N, 1) -> (b, N, 4)
        # (b, N, num_class+4)
        return torch.cat((pred_xyxy, pred_cls.sigmoid_()), dim=-1).contiguous()

    @staticmethod
    def scale_img(img, scale_factor):
        """

        :param img: (bn, 3, h, w)
        :param scale_factor: 输出的img shape必须能被scale_factor整除
        :return:
        """
        if scale_factor == 1.0:
            return img
        else:
            h, w = img.shape[2], img.shape[3]
            new_h, new_w = int(scale_factor * h), int(scale_factor * w)
            img = F.interpolate(img, size=(new_h, new_w), align_corners=False, mode='bilinear')
            out_h, out_w = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
            pad = [0, out_w - new_w, 0, out_h - new_h]  # [left, right, up, down]
            return F.pad(img, pad, value=0.447)

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

    @staticmethod
    def bbox_iou(bbox1, bbox2):
        """

        :param bbox1: (N, 4) / [xmin, ymin, xmax, ymax]
        :param bbox2: (M, 4) / [xmin, ymin, xmax, ymax]
        :return:
        """
        assert bbox1.ndim == 2
        assert bbox2.ndim == 2
        n, m = bbox1.size(0), bbox2.size(0)
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])  # (N,)
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])  # (M,)
        interaction_xmin = torch.maximum(bbox1[:, 0][:, None], bbox2[:, 0])  # (N, M)
        assert interaction_xmin.size(0) == n and interaction_xmin.size(1) == m
        interaction_xmax = torch.minimum(bbox1[:, 2][:, None], bbox2[:, 2])  # (N, M)
        interaction_ymin = torch.maximum(bbox1[:, 1][:, None], bbox2[:, 1])  # (N, M)
        interaction_ymax = torch.minimum(bbox1[:, 3][:, None], bbox2[:, 3])  # (N, M)
        interaction_h = interaction_ymax - interaction_ymin
        interaction_w = interaction_xmax - interaction_xmin
        interaction_area = interaction_w * interaction_h  # (N, M)
        iou = interaction_area / (area1[:, None] + area2 - interaction_area)  # (N, M)
        assert iou.size(0) == n and iou.size(1) == m
        return iou

    def numba_nms(self, preds_out:torch.Tensor):
        """
        do NMS with numba
        Inputs:
            preds_out: (b, N, 84) / [xmin, ymin, xmax, ymax, cls1, cls2, ...]
        """
        preds_out = preds_out.cpu().numpy()
        obj_conf_mask = np.amax(preds_out[:, :, 4:], axis=-1) >= self.cls_threshold  # (b, N)
        # preds_out = preds_out.float().cpu().numpy()
        outputs = []
        for i in range(preds_out.shape[0]):  # batch
            x = preds_out[i][obj_conf_mask[i]]
            if len(x) == 0:
                outputs.append(None)
                continue
            
            # [xmin, ymin, xmax, ymax]
            box = x[:, :4]
            if self.hyp['mutil_label']:
                row_idx, col_idx = (x[:, 4:] >= self.cls_threshold).nonzero()
                # x: [xmin, ymin, xmax, ymax, conf, cls_id]
                x = np.concatenate((box[row_idx], x[row_idx, col_idx+4][:, None], col_idx[:, None].astype(np.float32)), axis=1)
            else:
                cls_score = x[:, 4:].max(axis=1)[:, None]
                col_idx = x[:, 4:].argmax(axis=1)[:, None]
                # [xmin, ymin, xmax, ymax, conf, cls_id]
                x = np.concatenate((box, cls_score, col_idx.astype(np.float32)), axis=1)
                cls_score_mask = np.ascontiguousarray(cls_score.reshape(-1)) >= self.cls_threshold
                x = x[cls_score_mask]

            bbox_num = x.shape[0]
            if not bbox_num:
                outputs.append(None)
                continue

            if self.hyp['agnostic']:  # 每张image的每个class之间的bbox进行nms
                # 给每一个类别的bbox加上一个特殊的偏置，从而使得NMS时可以在每个类别间的bbox进行
                box_offset = x[:, 5] * 4096
            else:
                box_offset = x[:, 5] * 0.
            bboxes_offseted = x[:, :4] + box_offset[:, None]  # M
            scores = x[:, 4]
            keep_index = numba_nms(bboxes_offseted, scores, self.iou_threshold)

            if len(keep_index) > self.hyp['max_predictions_per_img']:
                keep_index = keep_index[:self.hyp['max_predictions_per_img']]  # N

            # 对每个bbox进行调优(每个最终输出的bbox都由与其iou大于iou threshold的一些bbox共同merge得到的)
            if self.hyp['postprocess_bbox']:
                if 1 < bbox_num < 3000:
                    iou = numba_iou(bboxes_offseted[keep_index], bboxes_offseted)  # (N, M)
                    iou_mask = iou > self.iou_threshold  # (N, M)
                    weights = iou_mask * scores[None, :]  # (N, M)
                    # (N, M) & (M, 4) & (N, 1) -> (N, 4)
                    bboxes_offseted[keep_index, :4] = np.matmul(weights, bboxes_offseted[:, :4]).astype(np.float32) / (weights.sum(axis=1, keepdims=True) + 1e-16)
                    # 因为如果一个区域有物体，网络应该在这一区域内给出很多不同的预测框，我们再从这些预测框中选取一个最好的作为该处obj的最终输出；
                    # 如果在某个grid处网络只给出了很少的几个预测框，则更倾向于认为这是网络预测错误所致
                    keep_index = np.asarray(keep_index)[iou_mask.astype(np.float32).sum(axis=1) > 1]
            outputs.append(x[keep_index])
        return outputs