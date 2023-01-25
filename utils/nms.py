import torch
import numba
import numpy as np

from .bbox_tools import gpu_CIoU, gpu_DIoU, gpu_Giou, gpu_iou

__all__ = ['gpu_nms', 'gpu_linear_soft_nms', 'gpu_exponential_soft_nms']


@numba.njit
def numba_nms(boxes, scores, iou_threshold):
    assert boxes.shape[0] == scores.shape[0]
    box_copy = boxes.copy()
    score_copy = scores.copy()
    keep_index = []
    while score_copy.sum() > 0.:
        max_score_index = np.argmax(score_copy)
        box1 = np.expand_dims(box_copy[max_score_index], 0)
        keep_index.append(max_score_index)
        score_copy[max_score_index] = 0.
        ious = numba_iou(box1, box_copy)
        ignore_index = ious >= iou_threshold
        for i, x in enumerate(ignore_index[0]):
            if x:
                score_copy[i] = 0.

    return keep_index


def gpu_nms(boxes, scores, iou_type, iou_threshold):
    """
    :param boxes: [M, 4]
    :param scores: [M, 1]
    :param iou_threshold:
    :param iou_type: str / must be one of ['iou', 'giou', 'diou', 'ciou']
    :return:
    """
    assert isinstance(boxes, torch.Tensor) and isinstance(scores, torch.Tensor)
    assert boxes.shape[0] == scores.shape[0]

    box_copy = boxes.detach().clone()
    score_copy = scores.detach().clone()
    keep_index = []
    if iou_type.lower() == 'iou':
        iou = gpu_iou
    elif iou_type.lower() == 'giou':
        iou = gpu_Giou
    elif iou_type.lower() == 'diou':
        iou = gpu_DIoU
    elif iou_type.lower() == 'ciou':
        iou = gpu_CIoU
    else:
        raise ValueError(f'Uknown paramemter: <{iou_type}>')

    while score_copy.sum() > 0.:
        # mark reserved box
        max_score_index = torch.argmax(score_copy).item()
        box1 = box_copy[[max_score_index]]
        keep_index.append(max_score_index)
        score_copy[max_score_index] = 0.
        ious = iou(box1, box_copy)
        ignore_index = ious.gt(iou_threshold)
        score_copy[ignore_index] = 0.
        # print(score_copy.sum())
    return keep_index


def gpu_linear_soft_nms(boxes, scores, iou_type, iou_threshold=0.3, thresh=0.001):
    """
    :param boxes: [M, 4]
    :param scores: [M, 1]
    :param iou_threshold:
    :param iou_type: str / must be one of ['iou', 'giou', 'diou', 'ciou']
    :return:
    """
    assert isinstance(boxes, torch.Tensor) and isinstance(scores, torch.Tensor)
    assert boxes.shape[0] == scores.shape[0]

    box_copy = boxes.detach().clone()
    score_copy = scores.detach().clone()
    processed = torch.zeros_like(score_copy)
    if iou_type == 'iou':
        iou = gpu_iou
    elif iou_type == 'giou':
        iou = gpu_Giou
    elif iou_type == 'diou':
        iou = gpu_DIoU
    elif iou_type == 'ciou':
        iou = gpu_CIoU
    else:
        raise ValueError(f'Uknown paramemter: <{iou_type}>')
    while score_copy.sum() > 0.:
        max_score_index = torch.argmax(score_copy).item()
        box1 = box_copy[[max_score_index]]
        processed[max_score_index] = score_copy[max_score_index]
        ious = iou(box1, box_copy)  # [1, M]
        sele_index = ious.gt(iou_threshold)
        # soft score
        score_copy[sele_index] *= torch.unsqueeze(1. - ious[sele_index], dim=1)

    keep_index = processed > thresh
    return keep_index.squeeze_()


def gpu_exponential_soft_nms(boxes, scores, iou_type, iou_threshold, sigmma=0.5, thresh=0.001):
    """
    :param boxes: [M, 4]
    :param scores: [M, 1]
    :param iou_threshold:
    :param iou_type: str / must be one of ['iou', 'giou', 'diou', 'ciou']
    :return:
    """
    assert isinstance(boxes, torch.Tensor) and isinstance(scores, torch.Tensor)
    assert boxes.shape[0] == scores.shape[0]

    box_copy = boxes.detach().clone()
    score_copy = scores.detach().clone()
    processed = torch.zeros_like(score_copy)
    if iou_type == 'iou':
        iou = gpu_iou
    elif iou_type == 'giou':
        iou = gpu_Giou
    elif iou_type == 'diou':
        iou = gpu_DIoU
    elif iou_type == 'ciou':
        iou = gpu_CIoU
    else:
        raise ValueError(f'Uknown paramemter: <{iou_type}>')
    while score_copy.sum() > 0.:
        # mark reserved box
        max_score_index = torch.argmax(score_copy).item()
        box1 = box_copy[[max_score_index]]
        processed[max_score_index] = score_copy[max_score_index]
        ious = iou(box1, box_copy)
        sele_index = ious.gt(iou_threshold)
        # soft score
        score_copy[sele_index] *= torch.unsqueeze(torch.exp(-(torch.pow(ious[sele_index], 2)) / sigmma), dim=1)

    keep_index = processed > thresh
    return keep_index.squeeze_()


