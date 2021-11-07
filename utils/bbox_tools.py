import torch
import numpy as np
import numba


@numba.njit
def numba_iou(bbox1, bbox2):
    """
    :param bbox1: [[xmin, ymin, xmax, ymax], ...] / (M, 4)
    :param bbox2: [[xmin, ymin, xmax, ymax], ...] / (N, 4)
    :return: iou / (M, N)
    """
    bbox1_h, bbox1_w = bbox1[:, 2]-bbox1[:, 0], bbox1[:, 3] - bbox1[:, 1]  # (M,)
    bbox2_h, bbox2_w = bbox2[:, 2]-bbox2[:, 0], bbox2[:, 3]-bbox2[:, 1]  # (N, )
    
    bbox1_area = bbox1_h * bbox1_w  # (M,)
    bbox2_area = bbox2_h * bbox2_w  # (N, )

    intersection_ymax = np.minimum(np.expand_dims(bbox1[:, 3], 1), bbox2[:, 3])
    intersection_xmax = np.minimum(np.expand_dims(bbox1[:, 2], 1), bbox2[:, 2])
    intersection_ymin = np.maximum(np.expand_dims(bbox1[:, 1], 1), bbox2[:, 1])
    intersection_xmin = np.maximum(np.expand_dims(bbox1[:, 0], 1), bbox2[:, 0])

    intersection_w = np.maximum(0., intersection_xmax - intersection_xmin)
    intersection_h = np.maximum(0., intersection_ymax - intersection_ymin)
    intersection_area = intersection_w * intersection_h
    iou_out = intersection_area / (np.expand_dims(bbox1_area, 1) + bbox2_area - intersection_area)

    return iou_out


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


def letter_resize_bbox(bboxes, letter_info):
    """
    Resize bbox corrding to letter_img_resize() do.
    :param bboxes: bbox format -> [xmin, ymin, xmax, ymax]
    :param letter_info:
    :return:
    """
    bboxes = np.asarray(bboxes) if not isinstance(bboxes, np.ndarray) else bboxes
    letter_bbox = bboxes * letter_info['scale']
    letter_bbox[:, [1, 3]] += letter_info['pad_top']
    letter_bbox[:, [0, 2]] += letter_info['pad_left']
    return letter_bbox


def minmax_bbox_resize(bboxes, scale):
    """
    Resize bbox corrding to minmax_img_resize() do.
    :param bboxes:
    :param scale:
    :return:
    """
    bboxes = np.asarray(bboxes) if not isinstance(bboxes, np.ndarray) else bboxes
    return bboxes * scale


def cpu_iou(bbox1, bbox2):
    """
    :param bbox1: [[xmin, ymin, xmax, ymax], ...]
    :param bbox2: [[xmin, ymin, xmax, ymax], ...]
    :return:
    """
    assert bbox1.shape[-1] == bbox2.shape[-1] == 4

    bbox1_area = np.prod(bbox1[:, [2, 3]] - bbox1[:, [0, 1]], axis=-1)
    bbox2_area = np.prod(bbox2[:, [2, 3]] - bbox2[:, [0, 1]], axis=-1)

    intersection_ymax = np.minimum(bbox1[:, 3], bbox2[:, 3])
    intersection_xmax = np.minimum(bbox1[:, 2], bbox2[:, 2])
    intersection_ymin = np.maximum(bbox1[:, 1], bbox2[:, 1])
    intersection_xmin = np.maximum(bbox1[:, 0], bbox2[:, 0])

    intersection_w = np.maximum(0., intersection_xmax - intersection_xmin)
    intersection_h = np.maximum(0., intersection_ymax - intersection_ymin)
    intersection_area = intersection_w * intersection_h
    iou_out = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    return iou_out


def xyxy2xywh(bboxes):
    """
    [xmin, ymin, xmax, ymax] -> [center_x, center_y, w, h]
    :param bboxes:
    :return:
    """
    new_bbox = torch.zeros_like(bboxes)
    wh = bboxes[:, [2, 3]] - bboxes[:, [0, 1]]
    # [center_x, center_y]
    xy = (bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2
    new_bbox[:, [0, 1]] = xy
    new_bbox[:, [2, 3]] = wh
    # [x, y, w, h]
    return new_bbox


def xyxy2xywhn(bboxes, img_shape):
    """

    :param bboxes: [xmin, ymin, xmax, ymax]
    :param img_shape: [w, h]
    :return:(norm_center_x, norm_center_y, norm_w, norm_h)
    """
    assert bboxes.shape[-1] == 4, f"the last dimension must equal 4"
    assert isinstance(bboxes, (np.ndarray, torch.Tensor)), f"unknown type: {type(bboxes)}"
    wh = bboxes[..., [2, 3]] - bboxes[..., [0, 1]]
    xy = (bboxes[..., [0, 1]] + bboxes[..., [2, 3]]) / 2
    bboxes_out = torch.zeros_like(bboxes)
    bboxes_out[..., 0] = xy[..., 0] / img_shape[0]  # x
    bboxes_out[..., 1] = xy[..., 1] / img_shape[1]  # y
    bboxes_out[..., 2] = wh[..., 0] / img_shape[0]  # w
    bboxes_out[..., 3] = wh[..., 1] / img_shape[1]  # h
    return bboxes_out


def xywh2xyxy(bboxes):
    """
    [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
    :param bboxes:
    """
    bbox_out = torch.zeros_like(bboxes)
    bbox_out[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
    bbox_out[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
    bbox_out[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
    bbox_out[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2
    return bbox_out


@numba.njit
def numba_xywh2xyxy(bboxes):
    """
    [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
    :param bboxes:
    """
    bbox_out = np.zeros_like(bboxes)
    bbox_out[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
    bbox_out[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
    bbox_out[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
    bbox_out[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2
    return bbox_out


def gpu_iou(bbox1, bbox2):
    """
    :param bbox1: [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor
    :param bbox2: [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor
    :return:
    """
    assert isinstance(bbox1, torch.Tensor)
    assert isinstance(bbox2, torch.Tensor)
    assert (bbox1[:, [2, 3]] >= bbox1[:, [0, 1]]).bool().all()
    assert (bbox2[:, [2, 3]] >= bbox2[:, [0, 1]]).bool().all()
    assert bbox1.shape[-1] == bbox2.shape[-1] == 4
    assert bbox1.device == bbox2.device
    device = bbox1.device

    bbox1_area = torch.prod(bbox1[:, [2, 3]] - bbox1[:, [0, 1]], dim=-1)
    bbox2_area = torch.prod(bbox2[:, [2, 3]] - bbox2[:, [0, 1]], dim=-1)

    intersection_ymax = torch.min(bbox1[:, 3], bbox2[:, 3])
    intersection_xmax = torch.min(bbox1[:, 2], bbox2[:, 2])
    intersection_ymin = torch.max(bbox1[:, 1], bbox2[:, 1])
    intersection_xmin = torch.max(bbox1[:, 0], bbox2[:, 0])

    intersection_w = torch.max(torch.tensor(0.).float().to(device), intersection_xmax - intersection_xmin)
    intersection_h = torch.max(torch.tensor(0.).float().to(device), intersection_ymax - intersection_ymin)
    intersection_area = intersection_w * intersection_h
    iou_out = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    return iou_out


def gpu_Giou(bbox1, bbox2):
    """
    :param bbox1: [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor
    :param bbox2: [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor
    :return:
    """
    assert isinstance(bbox1, torch.Tensor)
    assert isinstance(bbox2, torch.Tensor)
    assert (bbox1[:, [2, 3]] >= bbox1[:, [0, 1]]).bool().all()
    assert (bbox2[:, [2, 3]] >= bbox2[:, [0, 1]]).bool().all()
    assert bbox1.shape[-1] == bbox2.shape[-1] == 4
    assert bbox1.device == bbox2.device

    bbox1_area = torch.prod(bbox1[:, [2, 3]] - bbox1[:, [0, 1]], dim=-1)
    bbox2_area = torch.prod(bbox2[:, [2, 3]] - bbox2[:, [0, 1]], dim=-1)

    intersection_ymax = torch.min(bbox1[:, 3], bbox2[:, 3])
    intersection_xmax = torch.min(bbox1[:, 2], bbox2[:, 2])
    intersection_ymin = torch.max(bbox1[:, 1], bbox2[:, 1])
    intersection_xmin = torch.max(bbox1[:, 0], bbox2[:, 0])

    intersection_w = (intersection_xmax - intersection_xmin).clamp(0)
    intersection_h = (intersection_ymax - intersection_ymin).clamp(0)
    intersection_area = intersection_w * intersection_h

    union_area = bbox1_area + bbox2_area - intersection_area + 1e-16
    iou = intersection_area / union_area

    c_xmin = torch.min(bbox1[:, 0], bbox2[:, 0])
    c_xmax = torch.max(bbox1[:, 2], bbox2[:, 2])
    c_ymin = torch.min(bbox1[:, 1], bbox2[:, 1])
    c_ymax = torch.max(bbox1[:, 3], bbox2[:, 3])

    c_area = (c_xmax - c_xmin) * (c_ymax - c_ymin) + 1e-16
    g_iou = iou - torch.abs(c_area - union_area) / torch.abs(c_area)

    return g_iou


def gpu_DIoU(bbox1, bbox2):
    """
    :param bbox1: [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor / shape: [1, 4] or equal to bbox2's shape
    :param bbox2: [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor / shape: [M, 4]
    :return:
    """
    assert isinstance(bbox1, torch.Tensor)
    assert isinstance(bbox2, torch.Tensor)
    assert (bbox1[:, [2, 3]] >= bbox1[:, [0, 1]]).bool().all()
    assert (bbox2[:, [2, 3]] >= bbox2[:, [0, 1]]).bool().all()
    assert bbox1.shape[-1] == bbox2.shape[-1] == 4
    assert bbox1.device == bbox2.device
    device = bbox1.device

    bbox1_area = torch.prod(bbox1[:, [2, 3]] - bbox1[:, [0, 1]], dim=-1)
    bbox2_area = torch.prod(bbox2[:, [2, 3]] - bbox2[:, [0, 1]], dim=-1)

    intersection_ymax = torch.min(bbox1[:, 3], bbox2[:, 3])
    intersection_xmax = torch.min(bbox1[:, 2], bbox2[:, 2])
    intersection_ymin = torch.max(bbox1[:, 1], bbox2[:, 1])
    intersection_xmin = torch.max(bbox1[:, 0], bbox2[:, 0])

    intersection_w = torch.clamp(intersection_xmax - intersection_xmin, min=0.)
    intersection_h = torch.clamp(intersection_ymax - intersection_ymin, min=0.)
    intersection_area = intersection_w * intersection_h

    union_area = bbox1_area + bbox2_area - intersection_area + 1e-16
    iou = intersection_area / union_area

    c_xmin = torch.min(bbox1[:, 0], bbox2[:, 0])
    c_xmax = torch.max(bbox1[:, 2], bbox2[:, 2])
    c_ymin = torch.min(bbox1[:, 1], bbox2[:, 1])
    c_ymax = torch.max(bbox1[:, 3], bbox2[:, 3])
    c_hs = c_ymax - c_ymin
    c_ws = c_xmax - c_xmin
    assert torch.sum(c_hs > 0) > 0
    assert torch.sum(c_ws > 0) > 0
    c_diagonal = torch.pow(c_ws, 2) + torch.pow(c_hs, 2) + 1e-16

    # compute center coordinate of bboxes
    bbox1_ctr_x = (bbox1[:, 2] + bbox1[:, 0]) / 2
    bbox1_ctr_y = (bbox1[:, 3] + bbox1[:, 1]) / 2
    bbox2_ctr_x = (bbox2[:, 2] + bbox2[:, 0]) / 2
    bbox2_ctr_y = (bbox2[:, 3] + bbox2[:, 1]) / 2
    ctr_hs = bbox1_ctr_x - bbox2_ctr_x
    ctr_ws = bbox1_ctr_y - bbox2_ctr_y
    ctr_distance = torch.pow(ctr_hs, 2) + torch.pow(ctr_ws, 2)

    d_iou = iou - (ctr_distance / c_diagonal)
    d_iou = torch.clamp(d_iou, -1, 1)

    return d_iou


def gpu_CIoU(bbox1, bbox2):
    """

    :param bbox1:(N, 4) / [xmin, ymin, xmax, ymax]
    :param bbox2:(N, 4) / [xmin, ymin, xmax, ymax]
    """
    assert isinstance(bbox1, torch.Tensor)
    assert isinstance(bbox2, torch.Tensor)
    # assert (bbox1[:, [2, 3]] >= bbox1[:, [0, 1]]).all(), f"bbox1: {bbox1}"
    # assert (bbox2[:, [2, 3]] >= bbox2[:, [0, 1]]).all(), f"bbox2: {bbox2}"
    assert bbox1.shape[-1] == bbox2.shape[-1] == 4
    assert bbox1.device == bbox2.device

    w1, h1 = (bbox1[:, [2, 3]] - bbox1[:, [0, 1]]).T
    w2, h2 = (bbox2[:, [2, 3]] - bbox2[:, [0, 1]]).T
    bbox1_area = w1 * h1  # (N,)
    bbox2_area = w2 * h2  # (N,)

    intersection_ymax = torch.min(bbox1[:, 3], bbox2[:, 3])  # (N,)
    intersection_xmax = torch.min(bbox1[:, 2], bbox2[:, 2])  # (N,)
    intersection_ymin = torch.max(bbox1[:, 1], bbox2[:, 1])  # (N,)
    intersection_xmin = torch.max(bbox1[:, 0], bbox2[:, 0])  # (N,)

    intersection_w = torch.clamp(intersection_xmax - intersection_xmin, min=0.)  # (N,)
    intersection_h = torch.clamp(intersection_ymax - intersection_ymin, min=0.)  # (N,)
    intersection_area = intersection_w * intersection_h  # (N,)

    union_area = bbox1_area + bbox2_area - intersection_area + 1e-16  # (N,)
    iou = intersection_area / union_area  # (N,)

    c_xmin = torch.min(bbox1[:, 0], bbox2[:, 0])  # (N,)
    c_xmax = torch.max(bbox1[:, 2], bbox2[:, 2])  # (N,)
    c_ymin = torch.min(bbox1[:, 1], bbox2[:, 1])  # (N,)
    c_ymax = torch.max(bbox1[:, 3], bbox2[:, 3])  # (N,)
    c_hs = c_ymax - c_ymin  # (N,)
    c_ws = c_xmax - c_xmin  # (N,)
    c_diagonal = torch.pow(c_ws, 2) + torch.pow(c_hs, 2) + 1e-16  # (N,)

    # compute center coordinate of bboxes
    bbox1_ctr_x = (bbox1[:, 2] + bbox1[:, 0]) / 2  # (N,)
    bbox1_ctr_y = (bbox1[:, 3] + bbox1[:, 1]) / 2  # (N,)
    bbox2_ctr_x = (bbox2[:, 2] + bbox2[:, 0]) / 2  # (N,)
    bbox2_ctr_y = (bbox2[:, 3] + bbox2[:, 1]) / 2  # (N,)
    ctr_ws = bbox1_ctr_x - bbox2_ctr_x  # (N,)
    ctr_hs = bbox1_ctr_y - bbox2_ctr_y  # (N,)
    # ctr_distance: distance of two bbox center
    ctr_distance = torch.pow(ctr_hs, 2) + torch.pow(ctr_ws, 2)  # (N,)
    v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)  # (N,)

    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-16)
    ciou = iou - (ctr_distance / c_diagonal + v * alpha)  # (N,)
    return ciou


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
        iou = gpu_giou
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
        iou = gpu_giou
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
        iou = gpu_giou
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

