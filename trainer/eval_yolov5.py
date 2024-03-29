import torch
import numpy as np
import torch.nn.functional as F
from utils import gpu_nms, numba_iou, numba_nms, numba_xywh2xyxy
from utils import xywh2xyxy
from collections import defaultdict
from utils import weighted_fusion_bbox

__all__ = ['YOLOV5Evaluator']
class YOLOV5Evaluator:

    def __init__(self, yolo, anchors, hyp, compute_metric=False):
        self.yolo = yolo
        self.hyp = hyp
        self.device = hyp['device']
        self.num_class = hyp['num_class']
        self.anchor_num = anchors.size(1)
        self.anchors = anchors
        self.num_stage = len(anchors)
        self.grid = [torch.zeros(1)] * self.num_stage
        self.ds_scales = [8, 16, 32]  # 这个下采样尺度只适用于yolov5s, yolov5m, yolov5l, yolov5x
        self.inp_h, self.inp_w = hyp['input_img_size']
        self.use_tta = hyp['use_tta']
        self.grid_coords = [self.make_grid(self.inp_h//s, self.inp_w//s).float() for s in self.ds_scales]
        self.iou_threshold = self.hyp['compute_metric_iou_threshold'] if compute_metric else self.hyp['iou_threshold']
        self.cls_threshold = self.hyp['compute_metric_cls_threshold'] if compute_metric else self.hyp['cls_threshold']
        self.conf_threshold = self.hyp['compute_metric_conf_threshold'] if compute_metric else self.hyp['conf_threshold']

    @torch.no_grad()
    def __call__(self, inputs):
        """
        :param inputs: (b, 3, h, w) / tensor from testdataloader
        :return: (N, 6) / [xmin, ymin, xmax, ymax, conf, cls_id]
        """
        torch.cuda.empty_cache()
        if self.use_tta:
            merge_preds_out, inpendent_preds_out = self.test_time_augmentation(inputs)  # (bs, N, 85)
            if self.hyp["wfb"]:
                return self.do_wfb(inpendent_preds_out)
        else:
            merge_preds_out = self.do_inference(inputs)  # (bs, N, 85)
        return [torch.from_numpy(x) if x is not None else None for x in self.numba_nms(merge_preds_out) ]

    def do_wfb(self, preds_out):
        """
        weighted fusion bbox
        :param preds_out: [(batch_size, X, 85), (batch_size, Y, 85), (batch_size, Z, 85)]
        :return:
        """
        bs = preds_out[0].size(0)
        output = []
        out_dict = defaultdict(list)

        # preprocess data
        for i, preds in enumerate(preds_out):
            weight = self.hyp.get("wfb_weights", [1. for _ in range(len(preds_out))])[i]
            obj_conf_mask = preds[:, :, 4] > self.hyp['wfb_skip_box_threshold']
            for j in range(preds.size(0)):  # do nms for each image
                x = preds[j][obj_conf_mask[j]]
                if x.size(0) == 0:
                    continue
                x[:, 5:] *= x[:, 4:5]  # conf = cls_conf * obj_conf
                # [centerx, centery, w, h] -> [xmin, ymin, xmax, ymax]
                box = xywh2xyxy(x[:, :4])
                if self.hyp['mutil_label']:
                    row_idx, col_idx = (x[:, 5:] > self.hyp['wfb_skip_box_threshold']).nonzero(as_tuple=True)
                    # x: [xmin, ymin, xmax, ymax, conf, cls_id]
                    x = torch.cat((box[row_idx], x[row_idx, col_idx+5][:, None], col_idx[:, None].float()), dim=1)
                else:
                    cls_conf, col_idx = x[:, 5:].max(dim=1, keepdim=True)
                    # [xmin, ymin, xmax, ymax, conf, cls_id]
                    x = torch.cat((box, cls_conf, col_idx.float()), dim=1)
                    cls_conf_mask = cls_conf.view(-1).contiguous() > self.hyp['wfb_skip_box_threshold']
                    x = x[cls_conf_mask]

                bbox_num = x.size(0)
                if not bbox_num:
                    continue
                weight_to_fill = torch.full(size=(bbox_num, 1), fill_value=weight)
                x = torch.cat((x, weight_to_fill), dim=-1)
                out_dict[j].append(x.cpu().numpy())

        # WFB
        for b in range(bs):
            # 如果该张图片有预测框
            if out_dict[b]:
                bbox_list = np.vstack(out_dict[b])
                _, fusion_bbox = weighted_fusion_bbox(bbox_list, self.hyp['wfb_iou_threshold'])
                output.append(fusion_bbox)
            else:
                output.append(None)
        return output

    def do_nms(self, preds_out):
        """
         Do NMS with torch
        :param preds_out: (batch_size, X, 85)
        :return: list / [(X, 6), ..., None, (Y, 6), None, ..., (Z, 6), ...]
        """
        obj_conf_mask = preds_out[:, :, 4] >= self.conf_threshold
        outputs = []
        for i in range(preds_out.size(0)):  # do nms for each image
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
                ripe_preds[..., 1] = img_h - ripe_preds[..., 1]
            if f == 3:  # flip axis x
                ripe_preds[..., 0] = img_w - ripe_preds[..., 0]
            # [(bs, M, 85), (bs, N, 85), (bs, P, 85)]
            aug_preds.append(ripe_preds)
        # (bs, M+N+P, 85)
        return torch.cat(aug_preds, dim=1).contiguous(), aug_preds

    @torch.no_grad()
    def do_inference(self, inputs):
        preds_out = []
        input_img_h, input_img_w = inputs.size(2), inputs.size(3)
        stage_preds = self.yolo(inputs)
        batch_size = inputs.size(0)
        # [(5, 255, h/8, w/8), (5, 255, h/16, w/16), (5, 255, h/32, w/32)]
        for i in range(len(stage_preds)):
            cur_preds = stage_preds[i]
            fm_h, fm_w = cur_preds.size(2), cur_preds.size(3)
            # stage_anchor: (3, 2) -> (1, 3, 1, 1, 2)
            stage_anchor = (self.anchors[i] / self.ds_scales[i])[None, :, None, None, :].contiguous().type_as(inputs)
            # cur_preds: (bn, 3, h, w, 85) / [center_x, center_y, w, h, cofidence, c1, c2, c3, ...]
            cur_preds = cur_preds.view(batch_size, self.anchor_num, -1, fm_h, fm_w).permute(0, 1, 3, 4, 2).contiguous()
            cur_preds = cur_preds.sigmoid()
            # grid_coords: (1, 1, h, w, 2) / 可以优化grid_coords的创建方式
            if input_img_h == self.inp_h and input_img_w == self.inp_w:
                grid_coords = self.grid_coords[i].type_as(inputs)
            else:
                grid_coords = self.make_grid(fm_h, fm_w).type_as(inputs)

            # (bn, 3, h, w, 2) & (1, 1, h, w, 2) -> (bn, 3, h, w, 2)
            cur_preds[..., [0, 1]] = (cur_preds[..., [0, 1]] * 2 - 0.5 + grid_coords) * self.ds_scales[i]
            # (bn, 3, h, w, 2) & (1, 3, 1, 1, 2) -> (bn, 3, h, w, 2)
            cur_preds[..., [2, 3]] = (cur_preds[..., [2, 3]] * 2) ** 2 * stage_anchor * self.ds_scales[i]
            # [(bs, 20*20*3, 85), (bs, 40*40*3, 85), (bs, 80*80*3, 85)]
            preds_out.append(cur_preds.reshape(batch_size, -1, self.num_class+5).contiguous())
        # (bs, (20*20+40*40+80*80)*3, 85)
        return torch.cat(preds_out, dim=1).contiguous()

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

    def make_grid(self, row_num, col_num):
        y, x = torch.meshgrid([torch.arange(row_num, device=self.device), torch.arange(col_num, device=self.device)], indexing='ij')
        # mesh_grid: (col_num, row_num, 2) -> (row_num, col_num, 2)
        mesh_grid = torch.stack((x, y), dim=2).reshape(row_num, col_num, 2)
        # (1, 1, col_num, row_num, 2)
        return mesh_grid[None, None, ...].contiguous()

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

    def numba_nms(self, preds_out):
        """
        do NMS with numba
        """
        preds_out = preds_out.float().cpu().numpy()
        obj_conf_mask = preds_out[:, :, 4] >= self.conf_threshold
        outputs = []
        for i in range(preds_out.shape[0]):  # do nms for each image
            x = preds_out[i][obj_conf_mask[i]]
            if len(x) == 0:
                outputs.append(None)
                continue
            x[:, 5:] *= x[:, 4:5]  # conf = cls_conf * obj_conf
            # [centerx, centery, w, h] -> [xmin, ymin, xmax, ymax]
            box = numba_xywh2xyxy(x[:, :4])
            if self.hyp['mutil_label']:
                row_idx, col_idx = (x[:, 5:] >= self.cls_threshold).nonzero()
                # x: [xmin, ymin, xmax, ymax, conf, cls_id]
                x = np.concatenate((box[row_idx], x[row_idx, col_idx+5][:, None], col_idx[:, None].astype(np.float32)), axis=1)
            else:
                cls_conf = x[:, 5:].max(axis=1)[:, None]
                col_idx = x[:, 5:].argmax(axis=1)[:, None]
                # [xmin, ymin, xmax, ymax, conf, cls_id]
                x = np.concatenate((box, cls_conf, col_idx.astype(np.float32)), axis=1)
                cls_conf_mask = np.ascontiguousarray(cls_conf.reshape(-1)) > self.cls_threshold
                x = x[cls_conf_mask]

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