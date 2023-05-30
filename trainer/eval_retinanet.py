import torch
from utils import GPUAnchor, gpu_nms
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from utils import weighted_fusion_bbox, numba_nms, numba_iou

__all__ = ['RetinaNetEvaluator']
class RetinaNetEvaluator:

    def __init__(self, model, hyp, compute_metric=False) -> None:
        self.model = model
        self.enable_tta = hyp['use_tta']
        self.device = hyp['device']
        self.num_class = hyp['num_class']
        self.iou_loss_scale = hyp["tar_box_scale_factor"]
        self.hyp = hyp
        self.anchors = None
        self.iou_threshold = self.hyp['compute_metric_iou_threshold'] if compute_metric else self.hyp['iou_threshold']
        self.cls_threshold = self.hyp['compute_metric_cls_threshold'] if compute_metric else self.hyp['cls_threshold']

    def bbox_transform(self, anchors, regressions):
        """
        Args:
            anchors: (N, 4) / [xmin, ymin, xmax, ymax]
            regressions: (b, N, 4)
        Returns:
            pred_boxes: format [xmin, ymin, xmax, ymax] / shape (b, N, 4)
        """
        assert regressions.ndim == 3
        anchor_w = anchors[:, 2] - anchors[:, 0]  # (N,)
        anchor_h = anchors[:, 3] - anchors[:, 1]  # (N,)
        anchor_ctr_x = anchors[:, 0] + anchor_w * 0.5  # (N,)
        anchor_ctr_y = anchors[:, 1] + anchor_h * 0.5  # (N,)
        # self.iou_loss_scale是一个一维数组
        delta_scales = torch.unsqueeze(torch.tensor([self.iou_loss_scale], device=self.device), dim=0)
        assert delta_scales.size() == torch.Size([1, 1, 4])
        # 因为在计算focal regression loss 时，对target_box进行了相应的放大后再计算loss,因此这里将其还原为原来的比例
        regressions *= delta_scales

        dx = regressions[:, :, 0]  # (b, N)
        dy = regressions[:, :, 1]  # (b, N)
        dw = regressions[:, :, 2]  # (b, N)
        dh = regressions[:, :, 3]  # (b, N)

        pred_ctr_x = anchor_ctr_x + dx * anchor_w  # (b, N)
        pred_ctr_y = anchor_ctr_y + dy * anchor_h  # (b, N)
        pred_w = torch.exp(dw) * anchor_w  # (b, N)
        pred_h = torch.exp(dh) * anchor_h  # (b, N)

        pred_xmin = pred_ctr_x - pred_w * 0.5  # (b, N)
        pred_ymin = pred_ctr_y - pred_h * 0.5  # (b, N)
        pred_xmax = pred_ctr_x + pred_w * 0.5  # (b, N)
        pred_ymax = pred_ctr_y + pred_h * 0.5  # (b, N)

        pred_boxes = torch.stack([pred_xmin, pred_ymin, pred_xmax, pred_ymax], dim=-1)  # (b, N, 4)
        return pred_boxes

    @torch.no_grad()
    def do_inference(self, imgs):
        """
        Args:
            imgs: (b, 3, h, w)
        Returns:
            preds_out: (b, N, num_class + 4) / [cls1, cls2, cls3, ..., xmin, ymin, xmax, ymax]
        """
        # pred_box: (b, N, 4), pred_cls: (bs, N, num_class)
        pred_box, pred_cls = self.model(imgs)
        pred_cls = torch.sigmoid(pred_cls)
        anchors = GPUAnchor([imgs.size(2), imgs.size(3)])()  # (N, 4)
        pred_box[..., :4] = self.bbox_transform(anchors, pred_box[..., :4])  # (b, N, 4)
        pred_box[..., :4] = self.bbox_clip(imgs, pred_box[..., :4])  # (b, N, 4)
        # preds_out: (b, N, 4) & (b, N, num_class) -> (b, N, num_class+4)
        preds_out = torch.cat((pred_cls, pred_box), dim=2)
        return preds_out
        
    def do_wfb(self, preds_out):
        """
        weighted fusion bbox
        Args:
            preds_out: [(batch_size, X, 84), (batch_size, Y, 84), (batch_size, Z, 84)] / [cls1, cls2, cls3, ..., xmin, ymin, xmax, ymax]
        Returns:

        """
        batchsize = preds_out[0].size(0)
        outputs = []
        out_dict = defaultdict(list)  # {img1: [preds_from_model1, preds_from_model2, ...], img2: [preds_from_model1, preds_from_model2, ...]}
        # preprocess data
        weights = self.hyp.get("wfb_weights", [1. for _ in range(len(preds_out))])
        for i, preds in enumerate(preds_out):  # each model
            weight = weights[i]
            for b in range(batchsize):  # each image
                pred_cls = preds[b, :, :self.hyp['num_class']]  # (N, num_class)
                pred_box = preds[b, :, self.hyp['num_class']:self.hyp['num_class']+4]  # (N, 4)
                if self.hyp['mutil_label']:
                    row_idx, col_idx = (pred_cls >= self.hyp['wfb_skip_box_threshold']).nonzero(as_tuple=True)
                    # x: [xmin, ymin, xmax, ymax, prob, cls_id] / (M, 6)
                    x = torch.cat((pred_box[row_idx], pred_cls[row_idx, col_idx+4][:, None], col_idx[:, None].float()), dim=1)
                else:
                    # cls_prob: (N, 1)
                    cls_prob, col_idx = pred_cls.max(dim=1, keepdim=True)
                    # [xmin, ymin, xmax, ymax, prob, cls_id] / (N, 6)
                    x = torch.cat((pred_box, cls_prob, col_idx.float()), dim=1)
                    cls_prob_mask = cls_prob.view(-1).contiguous() >= self.hyp['wfb_skip_box_threshold']  # (N, 1)
                    x = x[cls_prob_mask]  # (M, 6)

                valid_bbox_num = x.size(0)
                if not valid_bbox_num:
                    outputs.append(None)
                    continue

                weight_to_fill = torch.full(size=(valid_bbox_num, 1), fill_value=weight)
                x = torch.cat((x, weight_to_fill), dim=-1)
                out_dict[b].append(x.cpu().numpy())

        # WFB
        for b in range(batchsize):
            # 如果该张图片有预测框
            if out_dict[b]:
                bbox_list = np.vstack(out_dict[b])
                _, fusion_bbox = weighted_fusion_bbox(bbox_list, self.hyp['wfb_iou_threshold'])
                outputs.append(fusion_bbox)
            else:
                outputs.append(None)
        return outputs

    def __call__(self, imgs):
        """
        Args:
            imgs: (b, 3, h, w)
        Returns:
            
        """
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            if self.enable_tta:
                merge_preds_out, inpendent_preds_out = self.test_time_augmentation(imgs)  # (bs, N, 85)
                if self.hyp["wfb"]:
                    return self.do_wfb(inpendent_preds_out)
            else:
                merge_preds_out = self.do_inference(imgs)  # (bs, N, 85)

        # return self.do_nms(merge_preds_out) / list of tensors
        return [torch.from_numpy(x) if x is not None else None for x in self.numba_nms(merge_preds_out)]

    def test_time_augmentation(self, inputs):  # just for inference not training time
        """
        Args:
            inputs: (bn, 3, h, w)
        Returns:
            preds: (bs, M+N+P, 84) / [cls1, cls2, cls3, ..., xmin, ymin, xmax, ymax]
        """
        img_h, img_w = inputs.size(2), inputs.size(3)
        scale_facotr = [1, 0.83, 0.67]
        flip_axis = [None, 2, 3]
        tta_preds = []

        for s, f in zip(scale_facotr, flip_axis):
            if f:
                img = inputs.flip(dims=(f,))
            else:
                img = inputs
            img = self.scale_img(img, s)
            # (bs, M, 84) / [cls1, cls2, cls3, ..., xmin, ymin, xmax, ymax]
            preds = self.do_inference(img)
            preds[..., self.hyp['num_class']:self.hyp['num_class']+4] /= s
            if f == 2:  # flip axis y
                tmp_pred_ymin = preds[..., self.hyp['num_class']+1].clone()
                tmp_pred_ymax = preds[..., self.hyp['num_class']+3].clone()
                preds[..., self.hyp['num_class']+1] = img_h - tmp_pred_ymax
                preds[..., self.hyp['num_class']+3] = img_h - tmp_pred_ymin
                del tmp_pred_ymin, tmp_pred_ymax
            if f == 3:  # flip axis x
                tmp_pred_xmin = preds[..., self.hyp['num_class']+0].clone()
                tmp_pred_xmax = preds[..., self.hyp['num_class']+2].clone()
                preds[..., self.hyp['num_class']+0] = img_w - tmp_pred_xmax
                preds[..., self.hyp['num_class']+2] = img_w - tmp_pred_xmin
                del tmp_pred_xmin, tmp_pred_xmax
            # [(bs, M, 84), (bs, N, 84), (bs, P, 84)]
            tta_preds.append(preds)
        # (bs, M+N+P, 84) / [cls1, cls2, cls3, ..., xmin, ymin, xmax, ymax]
        return torch.cat(tta_preds, dim=1).contiguous(), tta_preds

    def bbox_clip(self, img, bboxes):
        """
        Args:
            img: ()
            bboxes: format [xmin, ymin, xmax, ymax] / shape (b, N, 4)
        Return:
            bboxes: shape (b, N, 4)
        """
        b, c, h, w = img.shape
        assert c == 3
        bboxes.round_()
        bboxes[:, :, 0] = torch.clamp(bboxes[..., 0], min=0, max=w)
        bboxes[:, :, 1] = torch.clamp(bboxes[..., 1], min=0, max=h)
        bboxes[:, :, 2] = torch.clamp(bboxes[..., 2], min=0, max=w)
        bboxes[:, :, 3] = torch.clamp(bboxes[..., 3], min=0, max=h)
        return bboxes

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

    @staticmethod
    def bbox_iou(bbox1, bbox2):
        """
        Args:
            bbox1: (N, 4) / [xmin, ymin, xmax, ymax]
            bbox2: (M, 4) / [xmin, ymin, xmax, ymax]
        Returns:

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

    def do_nms(self, preds_out):
        """
        Args:
            preds_out: (b, N, num_class+4) / [cls1, cls2, cls3, ..., xmin, ymin, xmax, ymax]
        Returns:

        """
        batchsize = preds_out.size(0)
        outputs = []
        for b in range(batchsize):  # each image
            pred_cls = preds_out[b, :, :self.hyp['num_class']]  # (N, num_class)
            pred_box = preds_out[b, :, self.hyp['num_class']:self.hyp['num_class']+4]  # (N, 4)
            if self.hyp['mutil_label']:
                row_idx, col_idx = (pred_cls >= self.cls_threshold).nonzero(as_tuple=True)
                # x: [xmin, ymin, xmax, ymax, prob, cls_id] / (M, 6)
                x = torch.cat((pred_box[row_idx], pred_cls[row_idx, col_idx+4][:, None], col_idx[:, None].float()), dim=1)
            else:
                # cls_prob: (N, 1)
                cls_prob, col_idx = pred_cls.max(dim=1, keepdim=True)
                # [xmin, ymin, xmax, ymax, prob, cls_id] / (N, 6)
                x = torch.cat((pred_box, cls_prob, col_idx.float()), dim=1)
                cls_prob_mask = cls_prob.view(-1).contiguous() >= self.cls_threshold  # (N, 1)
                x = x[cls_prob_mask]  # (M, 6)

                valid_bbox_num = x.size(0)
                if not valid_bbox_num:
                    outputs.append(None)
                    continue

                if self.hyp['agnostic']:  # 每张image的每个class之间的bbox进行nms
                    # 给每一个类别的bbox加上一个特殊的偏置，从而使得NMS时可以在每个类别间的bbox进行
                    box_offset = x[:, 5] * 4096
                else:
                    box_offset = x[:, 5] * 0.0

                bboxes_offseted = x[:, :4] + box_offset[:, None]  # M
                scores = x[:, 4]
                keep_index = gpu_nms(bboxes_offseted, scores, self.hyp['nms_type'], self.iou_threshold)
                keep_index = keep_index[:self.hyp['max_predictions_per_img']]  # N

                # 对每个bbox进行调优(每个最终输出的bbox都由与其iou大于iou threshold的一些bbox共同merge得到的)
                if self.hyp['postprocess_bbox']:
                    if 1 < valid_bbox_num < 3000:
                        iou = self.bbox_iou(bboxes_offseted[keep_index], bboxes_offseted)  # (N, M)
                        iou_mask = iou > self.iou_threshold  # (N, M)
                        weights = iou_mask * scores[None, :]  # (N, M)
                        # (N, M) & (M, 4) & (N, 1) -> (N, 4)
                        x[keep_index, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(dim=1, keepdims=True)
                        # 因为如果一个区域有物体，网络应该在这一区域内给出很多不同的预测框，我们再从这些预测框中选取一个最好的作为该处obj的最终输出；
                        # 如果在某个grid处网络只给出了很少的几个预测框，则更倾向于认为这是网络预测错误所致
                        keep_index = torch.tensor(keep_index)[iou_mask.float().sum(dim=1) > 1]
                outputs.append(x[keep_index])
        return outputs

    def numba_nms(self, preds_out):
        """
        do NMS with numba
        Args:
            preds_out: (bs, M+N+P, 84) / [cls1, cls2, cls3, ..., xmin, ymin, xmax, ymax]
        Returns:

        """
        batchsize = preds_out.size(0)
        preds_out = preds_out.float().cpu().numpy()
        outputs = []
        for b in range(batchsize):  # each image
            pred_cls = preds_out[b, :, :self.hyp['num_class']]  # (N, num_class)
            pred_box = preds_out[b, :, self.hyp['num_class']:self.hyp['num_class']+4]  # (N, 4)
            if self.hyp['mutil_label']:
                row_idx, col_idx = (pred_cls >= self.cls_threshold).nonzero(as_tuple=True)
                # x: [xmin, ymin, xmax, ymax, prob, cls_id] / (M, 6)
                x = np.concatenate((pred_box[row_idx], pred_cls[row_idx, col_idx+4][:, None], col_idx[:, None].astype(np.float32)), axis=1)
            else:
                # cls_prob: (N, 1)
                cls_prob = pred_cls.max(axis=1)[:, None]  # (N, 1)
                col_idx = pred_cls.argmax(axis=1)[:, None]  # (N, 1)
                # [xmin, ymin, xmax, ymax, prob, cls_id] / (N, 6)
                x = np.concatenate((pred_box, cls_prob, col_idx.astype(np.float32)), axis=1)
                cls_prob_mask = np.ascontiguousarray(cls_prob.reshape(-1)) > self.cls_threshold  # (N, 1)
                x = x[cls_prob_mask]  # (M, 6)

            valid_bbox_num = x.shape[0]
            if not valid_bbox_num:
                outputs.append(None)
                continue

            if self.hyp['agnostic']:  # 每张image的每个class之间的bbox进行nms
                # 给每一个类别的bbox加上一个特殊的偏置，从而使得NMS时可以在每个类别间的bbox进行
                box_offset = x[:, 5] * 4096
            else:
                box_offset = x[:, 5] * 0.0
            bboxes_offseted = x[:, :4] + box_offset[:, None]  # (M, 4)
            scores = x[:, 4]  # (M,)
            keep_index = numba_nms(bboxes_offseted, scores, self.iou_threshold)
            keep_index = keep_index[:self.hyp['max_predictions_per_img']]  # N

            # 对每个bbox进行调优(每个最终输出的bbox都由与其iou大于iou threshold的一些bbox共同merge得到的)
            if self.hyp['postprocess_bbox']:
                if 1 < valid_bbox_num < 3000:
                    iou = numba_iou(bboxes_offseted[keep_index], bboxes_offseted)  # (N, M)
                    iou_mask = iou > self.iou_threshold  # (N, M)
                    weights = iou_mask * scores[None, :]  # (N, M)
                    # (N, M) & (M, 4) & (N, 1) -> (N, 4)
                    x[keep_index, :4] = np.matmul(weights, x[:, :4]).astype(np.float32) / (weights.sum(axis=1, keepdims=True) + 1e-16)
                    # 因为如果一个区域有物体，网络应该在这一区域内给出很多不同的预测框，我们再从这些预测框中选取一个最好的作为该处obj的最终输出；
                    # 如果在某个grid处网络只给出了很少的几个预测框，则更倾向于认为这是网络预测错误所致
                    keep_index = np.asarray(keep_index)[iou_mask.astype(np.float32).sum(axis=1) > 1]
            outputs.append(x[keep_index])
        return outputs