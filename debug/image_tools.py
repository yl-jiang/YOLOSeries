import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

from data_aug import CV2Transform


def letter_resize_img(img, dst_size, stride=64, fill_value=128, only_ds=False, training=True):
    """
    only scale down
    :param only_ds: only downsample
    :param fill_value:
    :param training:
    :param img:
    :param dst_size: int or [h, w]
    :param stride:
    :return:
    """
    if isinstance(dst_size, int):
        dst_size = [dst_size, dst_size]

    # 将dst_size调整到是stride的整数倍
    dst_del_h, dst_del_w = np.remainder(dst_size[0], stride), np.remainder(dst_size[1], stride)
    dst_pad_h = stride - dst_del_h if dst_del_h > 0 else 0
    dst_pad_w = stride - dst_del_w if dst_del_w > 0 else 0
    dst_size = [dst_size[0] + dst_pad_h, dst_size[1] + dst_pad_w]

    org_h, org_w = img.shape[:2]  # [height, width]
    scale = float(np.min([dst_size[0] / org_h, dst_size[1] / org_w]))
    if only_ds:
        scale = min(scale, 1.0)  # only scale down for good test performance
    if scale != 1.:
        resize_h, resize_w = int(org_h * scale), int(org_w * scale)
        img_resize = cv2.resize(img.copy(), (resize_w, resize_h), interpolation=0)
    else:
        resize_h, resize_w = img.shape[:2]
        img_resize = img.copy()

    # training时需要一个batch保持固定的尺寸，testing时尽可能少的填充像素以加速inference
    if not training:
        pad_h, pad_w = dst_size[0] - resize_h, dst_size[1] - resize_w
        pad_h, pad_w = np.remainder(pad_h, stride), np.remainder(pad_w, stride)
        top = int(round(pad_h / 2))
        left = int(round(pad_w / 2))
        bottom = pad_h - top
        right = pad_w - left
        if isinstance(fill_value, int):
            fill_value = (fill_value, fill_value, fill_value)
        img_out = cv2.copyMakeBorder(img_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_value)
    else:
        img_out = np.full(shape=dst_size+[3], fill_value=fill_value)
        pad_h, pad_w = dst_size[0] - resize_h, dst_size[1] - resize_w
        top, left = pad_h // 2, pad_w // 2
        bottom, right = pad_h - top, pad_w - left
        img_out[top:(top+resize_h), left:(left+resize_w)] = img_resize
    letter_info = {'scale': scale, 'pad_top': top, 'pad_left': left, "pad_bottom": bottom, "pad_right": right, "org_shape": (org_h, org_w)}
    return img_out.astype(np.uint8), letter_info


def minmax_img_resize(img, min_side=None, max_side=None):
    scale = compute_resize_scale(img, min_side, max_side)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img, scale


def compute_resize_scale(img, min_side, max_side):
    if min_side is None:
        min_side = 800
    if max_side is None:
        max_side = 1300
    height, width, _ = img.shape

    scale = min(min_side / height, min_side / width)
    if scale * max(height, width) > max_side:
        scale = min(max_side / height, max_side / width)
    return scale


def min_scale_resize(img, dst_size):
    """

    :param img: ndarray
    :param dst_size: [h, w]
    :returns: resized_img, img_org
    """
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    min_scale = min(np.array(dst_size) / max(img.shape[:2]))
    h, w = img.shape[:2]
    if min_scale != 1:
        img_out = cv2.resize(img, (int(w*min_scale), int(h*min_scale)), interpolation=cv2.INTER_AREA)
        return img_out, img.shape[:2]
    return img


def compute_featuremap_shape(img_shape, pyramid_level):
    """
    compute feature map's shape based on pyramid level.
    :param img_shape: 3 dimension / [h, w, c]
    :param pyramid_level: int
    :return:
    """
    img_shape = np.array(img_shape)
    fm_shape = (img_shape - 1) // (2**pyramid_level) + 1
    return fm_shape


if __name__ == '__main__':
    # test CV2Transform
    img_path = r''
    bbox_head = np.array([[96, 374, 143, 413]])
    cv_img = cv2.imread(img_path)
    s = cv_img.shape
    print(cv_img.shape)
    labels = np.array([1])
    trans = CV2Transform(cv_img, bbox_head, labels)
    # img, bbox = trans.randomFlip(cv_img, bbox_head)
    print(trans.img.shape)
    # print(trans.bboxes)
    cv2.rectangle(trans.img, (trans.bboxes[:, 1], trans.bboxes[:, 0]), (trans.bboxes[:, 3], trans.bboxes[:, 2]), (55, 255, 155), 5)

    cv2.imshow('image', trans.img)
    cv2.waitKey(50000)


