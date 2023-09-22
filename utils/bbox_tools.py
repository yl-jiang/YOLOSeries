import torch
import numba
import numpy as np

__all__ = ['letter_resize_bbox', 'minmax_bbox_resize', 
           'cpu_iou', 'xyxy2xywh', 'xyxy2xywhn', 'xywh2xyxy', 'numba_xywh2xyxy', 
           'gpu_iou', 'gpu_Giou', 'gpu_CIoU', 'gpu_DIoU', 'box_candidates', 
           'valid_bbox', 'numba_iou', 'numba_xyxy2xywh', 'tblr2xyxy', 'xyxy2tblr']


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
    iou_out = intersection_area / np.clip(bbox1_area + bbox2_area - intersection_area, a_min=1e-6)

    return iou_out


def xyxy2xywh(bboxes):
    """
    [xmin, ymin, xmax, ymax] -> [center_x, center_y, w, h]
    :param bboxes:
    :return:
    """
    new_bbox = torch.zeros_like(bboxes)
    wh = bboxes[..., [2, 3]] - bboxes[..., [0, 1]]
    # [center_x, center_y]
    xy = (bboxes[..., [0, 1]] + bboxes[..., [2, 3]]) / 2
    new_bbox[..., [0, 1]] = xy
    new_bbox[..., [2, 3]] = wh
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
    :param bboxes: (N, 4)
    """
    # assert bboxes.ndim == 2, f"input bboxes's shape be same like (N, 4), but got {bboxes.shape}"
    bbox_out = torch.zeros_like(bboxes)
    bbox_out[..., 0] = bboxes[..., 0] - bboxes[..., 2] / 2
    bbox_out[..., 1] = bboxes[..., 1] - bboxes[..., 3] / 2
    bbox_out[..., 2] = bboxes[..., 0] + bboxes[..., 2] / 2
    bbox_out[..., 3] = bboxes[..., 1] + bboxes[..., 3] / 2
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


@numba.njit
def numba_xyxy2xywh(bboxes):
    """
    [xmin, ymin, xmax, ymax] -> [center_x, center_y, w, h]
    :param bboxes:
    """
    bbox_out = np.zeros_like(bboxes)
    bbox_out[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
    bbox_out[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
    bbox_out[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bbox_out[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bbox_out

def gpu_iou(bbox1, bbox2):
    """
    :param bbox1: [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor / (N, 4)
    :param bbox2: [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor / (M, 4)
    :return: iou / (N, M)
    """
    bbox1_area = torch.prod(bbox1[:, [2, 3]] - bbox1[:, [0, 1]], dim=-1)  # (N,)
    bbox2_area = torch.prod(bbox2[:, [2, 3]] - bbox2[:, [0, 1]], dim=-1)  # (M,)

    # bbox1_w = bbox1[:, 2] - bbox1[:, 0]
    # bbox1_h = bbox1[:, 3] - bbox1[:, 1]
    # bbox2_w = bbox2[:, 2] - bbox2[:, 0]
    # bbox2_h = bbox2[:, 3] - bbox2[:, 1]
    # bbox1_area = bbox1_w * bbox1_h
    # bbox2_area = bbox2_w * bbox2_h

    intersection_ymax = torch.min(bbox1[:, None, 3], bbox2[:, 3])  # (N, M)
    intersection_xmax = torch.min(bbox1[:, None, 2], bbox2[:, 2])  # (N, M)
    intersection_ymin = torch.max(bbox1[:, None, 1], bbox2[:, 1])  # (N, M)
    intersection_xmin = torch.max(bbox1[:, None, 0], bbox2[:, 0])  # (N, M)

    intersection_w = torch.clamp(intersection_xmax - intersection_xmin, min=0.0)  # (N, M)
    intersection_h = torch.clamp(intersection_ymax - intersection_ymin, min=0.0)  # (N, M)
    intersection_area = intersection_w * intersection_h  # (N, M)
    iou = intersection_area / (bbox1_area[:, None] + bbox2_area - intersection_area).clamp(1e-6)  # (N, M)

    return iou  # (N, M)


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

    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area.clamp(min=1e-6)

    c_xmin = torch.min(bbox1[:, 0], bbox2[:, 0])
    c_xmax = torch.max(bbox1[:, 2], bbox2[:, 2])
    c_ymin = torch.min(bbox1[:, 1], bbox2[:, 1])
    c_ymax = torch.max(bbox1[:, 3], bbox2[:, 3])

    c_area = (c_xmax - c_xmin) * (c_ymax - c_ymin)
    g_iou = iou - torch.abs(c_area - union_area) / torch.abs(c_area.clamp(1e-6))

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

    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area.clamp(min=1e-6)

    c_xmin = torch.min(bbox1[:, 0], bbox2[:, 0])
    c_xmax = torch.max(bbox1[:, 2], bbox2[:, 2])
    c_ymin = torch.min(bbox1[:, 1], bbox2[:, 1])
    c_ymax = torch.max(bbox1[:, 3], bbox2[:, 3])
    c_hs = c_ymax - c_ymin
    c_ws = c_xmax - c_xmin
    assert torch.sum(c_hs > 0) > 0
    assert torch.sum(c_ws > 0) > 0
    c_diagonal = torch.pow(c_ws, 2) + torch.pow(c_hs, 2)

    # compute center coordinate of bboxes
    bbox1_ctr_x = (bbox1[:, 2] + bbox1[:, 0]) / 2
    bbox1_ctr_y = (bbox1[:, 3] + bbox1[:, 1]) / 2
    bbox2_ctr_x = (bbox2[:, 2] + bbox2[:, 0]) / 2
    bbox2_ctr_y = (bbox2[:, 3] + bbox2[:, 1]) / 2
    ctr_hs = bbox1_ctr_x - bbox2_ctr_x
    ctr_ws = bbox1_ctr_y - bbox2_ctr_y
    ctr_distance = torch.pow(ctr_hs, 2) + torch.pow(ctr_ws, 2)

    d_iou = iou - (ctr_distance / c_diagonal.clamp(min=1e-6))
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

    w1, h1 = (bbox1[:, [2, 3]] - bbox1[:, [0, 1]]).T  # (N, 2)
    w2, h2 = (bbox2[:, [2, 3]] - bbox2[:, [0, 1]]).T  # (N, 2)
    bbox1_area = w1 * h1  # (N,)
    bbox2_area = w2 * h2  # (N,)

    intersection_ymax = torch.min(bbox1[:, 3], bbox2[:, 3])  # (N,)
    intersection_xmax = torch.min(bbox1[:, 2], bbox2[:, 2])  # (N,)
    intersection_ymin = torch.max(bbox1[:, 1], bbox2[:, 1])  # (N,)
    intersection_xmin = torch.max(bbox1[:, 0], bbox2[:, 0])  # (N,)

    intersection_w = torch.clamp(intersection_xmax - intersection_xmin, min=0.)  # (N,)
    intersection_h = torch.clamp(intersection_ymax - intersection_ymin, min=0.)  # (N,)
    intersection_area = intersection_w * intersection_h  # (N,)

    union_area = bbox1_area + bbox2_area - intersection_area  # (N,)
    union_area = torch.clamp(union_area, min=1e-6)
    iou = intersection_area / union_area  # (N,)

    c_xmin = torch.min(bbox1[:, 0], bbox2[:, 0])  # (N,)
    c_xmax = torch.max(bbox1[:, 2], bbox2[:, 2])  # (N,)
    c_ymin = torch.min(bbox1[:, 1], bbox2[:, 1])  # (N,)
    c_ymax = torch.max(bbox1[:, 3], bbox2[:, 3])  # (N,)
    c_hs = c_ymax - c_ymin  # (N,)
    c_ws = c_xmax - c_xmin  # (N,)
    c_diagonal = torch.pow(c_ws, 2) + torch.pow(c_hs, 2)  # (N,)
    

    # compute center coordinate of bboxes
    bbox1_ctr_x = (bbox1[:, 2] + bbox1[:, 0]) / 2  # (N,)
    bbox1_ctr_y = (bbox1[:, 3] + bbox1[:, 1]) / 2  # (N,)
    bbox2_ctr_x = (bbox2[:, 2] + bbox2[:, 0]) / 2  # (N,)
    bbox2_ctr_y = (bbox2[:, 3] + bbox2[:, 1]) / 2  # (N,)
    ctr_ws = bbox1_ctr_x - bbox2_ctr_x  # (N,)
    ctr_hs = bbox1_ctr_y - bbox2_ctr_y  # (N,)
    # ctr_distance: distance of two bbox center
    ctr_distance = ctr_hs.pow_(2) + ctr_ws.pow_(2)  # (N,)
    v = (4 / (np.pi ** 2)) * (torch.atan(w2 / torch.clamp(h2, min=1e-6)) - torch.atan(w1 / torch.clamp(h1, min=1e-6))).pow_(2)  # (N,)

    with torch.no_grad():
        alpha = v / torch.clamp(1 - iou + v, min=1e-6)
    c_diagonal = torch.clamp(c_diagonal, min=1e-6)
    ciou = iou - (ctr_distance / c_diagonal + v * alpha)  # (N,)
    return ciou


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    """
    Compute candidate boxes
    :param box1: before augment
    :param box2: after augment
    :param wh_thr: wh_thr (pixels)
    :param ar_thr:
    :param area_thr:
    :return:
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def valid_bbox(bboxes, box_type='xyxy', wh_thr=2, ar_thr=10, area_thr=16):
    """
    根据bbox的width阈值,height阈值,width-height ratio阈值以及area阈值,过滤掉不满足限制条件的bbox。

    :param wh_thr:
    :param area_thr: area threshold
    :param ar_thr: aspect ratio threshold
    :param bboxes: ndarray or list; [[a, b, c, d], [e, f, g, h], ...]
    :param box_type: 'xyxy' or 'xywh'; [xmin, ymin, xmax, ymax] or [center_x, center_y, w, h]
    :return:
    """
    if box_type == 'xywh':
        labels = xywh2xyxy(np.array(bboxes))
    elif box_type:
        labels = np.array(bboxes)
    else:
        raise ValueError(f'unknow bbox format: {box_type}')

    x = labels[:, 2] > labels[:, 0]  # xmax > xmin
    y = labels[:, 3] > labels[:, 1]  # ymax > ymin
    w = labels[:, 2] - labels[:, 0]
    h = labels[:, 3] - labels[:, 1]
    w_mask = w > wh_thr
    h_mask = h > wh_thr
    area_mask = (w * h) >= area_thr
    ar_1, ar_2 = w / (h+1e-16), h / (w+1e-16)
    ar = np.where(ar_1 > ar_2, ar_1, ar_2)
    ar_mask = ar < ar_thr
    valid_idx = np.stack([x, y, ar_mask, area_mask, h_mask, w_mask], axis=1)
    valid_inx = np.all(valid_idx, axis=1)

    return valid_inx


def tblr2xyxy(tblr: torch.Tensor, grid: torch.Tensor):
    """
    Inputs:
        tblr: (b, N, 4) / [t, b, l, r]
        grid: (N, 2) / [x, y]
    Outputs:
        xyxy: (b, N, 4) / [xmin, ymin, xmax, ymax]
    """
    assert tblr.ndim == 3 and tblr.size(1) == grid.size(0)
    t, b, l, r = tblr.chunk(4, -1)  # (b, N, 1)
    xmin = grid[:, [0]][None] - l  # (1, N, 1) & (b, N, 1) -> (b, N, 1)
    ymin = grid[:, [1]][None] - t  # (1, N, 1) & (b, N, 1) -> (b, N, 1)
    xmax = grid[:, [0]][None] + r  # (1, N, 1) & (b, N, 1) -> (b, N, 1)
    ymax = grid[:, [1]][None] + b  # (1, N, 1) & (b, N, 1) -> (b, N, 1)
    
    return torch.cat((xmin, ymin, xmax, ymax), dim=-1).contiguous()  # (b, N, 4)


def xyxy2tblr(xyxy:torch.Tensor, grid:torch.Tensor):
    """
    Inputs:
        xyxy: (b, N, 4) / [xmin, ymin, xmax, ymax]
        grid: (N, 2) / [x, y]
    Outputs:
        tblr: (b, N, 4) / [t, b, l, r]
    """
    xmin, ymin, xmax, ymax = xyxy.chunk(4, -1)  # (b, N, 1)
    gx, gy = grid.chunk(2, -1)  # (N, 1)
    # (1, N, 1) & (b, N, 1)
    t = gy[None] - ymin
    b = ymax - gy[None]
    l = gx[None] - xmin
    r = xmax - gx[None]

    return torch.cat((t, b, l, r), dim=-1).contiguous()  # (b, N, 4)

