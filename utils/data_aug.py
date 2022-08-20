import random
import math

import cv2
import numpy as np
from functools import reduce

from utils import xywh2xyxy

# region
# ===================================== aug for img =====================================
class CV2Transform:
    """
    pass
    """
    def __init__(self, aug_threshold=None, strict=False, fill_value=128):

        if aug_threshold is None:
            self.aug_threshold = 0.3
        else:
            self.aug_threshold = aug_threshold

        self.strict = strict
        self.fill_value = fill_value

    def __call__(self, cv_img, bboxes=None, labels=None):
        # cv_img:bgr/(h,w,c)
        # bboxes:[[xmin,ymin,xmax,ymax],...]
        # labels:[2,4,...]
        
        self.img = cv_img
        self.bboxes = bboxes
        self.labels = labels
        self.done_yoco = False

        # inplace modify self.img and self.bboxes
        self.randomFlip()
        self.randomScale()
        self.randomBlur()
        self.RandomBrightness()
        self.randomHue()
        self.RandomSaturation()
        self.randomShift()
        self.randomCrop()
        return self.img, self.bboxes, self.labels
        

    @classmethod
    def _check_input(cls, cv_img, bboxes, labels):
        if not isinstance(cv_img, np.ndarray):
            raise ValueError("Image's type must be ndarray")
        if len(cv_img.shape) < 3:
            raise ValueError("Image must be colorful")
        if not isinstance(bboxes, np.ndarray):
            raise ValueError("bboxes's type must be ndarray")
        if not isinstance(labels, np.ndarray):
            raise ValueError("labels's type must be ndarray")
        return cls(cv_img, bboxes, labels)


    def yoco(self, img_org, img_aug, thresh):
        assert img_org.shape == img_aug.shape
        h, w, c = img_org.shape
        aug_img = img_aug
        if not self.done_yoco:
            if np.random.random() < thresh:  # 垂直切分并增强后合并
                self.done_yoco = True
                aug_img = np.concatenate((img_org[:, 0:int(w/2), :], img_aug[:, int(w/2):, :]), axis=1)
            if not self.done_yoco and np.random.random() < thresh:
                aug_img = np.concatenate((img_org[0:h//2, :, :], img_aug[h//2:, :, :]), axis=0)
        return aug_img


    def randomFlip(self):
        # 垂直翻转/y坐标不变,x坐标变化
        if random.random() < self.aug_threshold:
            img = np.fliplr(self.img).copy()
            h, w = self.img.shape[:2]
            xmax = w - self.bboxes[:, 0]
            xmin = w - self.bboxes[:, 2]
            self.bboxes[:, 0] = xmin
            self.bboxes[:, 2] = xmax
            self.img = img

    def randomScale(self):
        # 固定住高度,以0.8-1.2伸缩宽度,做图像形变
        if random.random() < self.aug_threshold:
            scale = random.uniform(0.8, 1.2)
            # cv2.resize(img, shape)/其中shape->[宽,高]
            self.img = cv2.resize(self.img, None, fx=scale, fy=1)
            self.bboxes[:, [0, 2]] *= scale

    def randomBlur(self):
        # 均值滤波平滑图像
        if random.random() < self.aug_threshold:
            img_org = self.img.copy()
            img_blur = cv2.blur(self.img, (5, 5))
            self.img = self.yoco(img_org, img_blur, self.aug_threshold)

    def randomHue(self, hgain=0.5, sgain=0.5, vgain=0.5):
        # 图片色调
        if random.random() < self.aug_threshold:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV))
            dtype = self.img.dtype  # uint8

            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
            hsv_org = cv2.merge((hue, sat, val))
            img_hsv = self.yoco(hsv_org, img_hsv, self.aug_threshold)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=self.img)  # no return needed

    def RandomSaturation(self):
        # 图片饱和度
        if random.random() < self.aug_threshold:
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            hsv_org = hsv.copy()
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s.astype(np.float32)
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            hsv = self.yoco(hsv_org, hsv, self.aug_threshold)
            cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=self.img)

    def RandomBrightness(self):
        # 图片亮度
        if random.random() < self.aug_threshold:
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            hsv_org = hsv.copy()
            # hsv分别表示：色调(H),饱和度(S),明度(V)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v.astype(np.float32)
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            hsv = self.yoco(hsv_org, hsv, self.aug_threshold)
            cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=self.img)

    def randomShift(self):
        # 随机平移
        center_y = (self.bboxes[:, 1] + self.bboxes[:, 3]) / 2
        center_x = (self.bboxes[:, 0] + self.bboxes[:, 2]) / 2
        if random.random() < self.aug_threshold:
            h, w, c = self.img.shape
            after_shfit_image = np.zeros((h, w, c), dtype=self.img.dtype)
            # after_shfit_image每行元素都设为[104,117,123]
            after_shfit_image[:, :, :] = (self.fill_value, self.fill_value, self.fill_value)  # bgr
            shift_x = int(random.uniform(-w * 0.2, w * 0.2))
            shift_y = int(random.uniform(-h * 0.2, h * 0.2))
            # 图像平移
            if shift_x >= 0 and shift_y >= 0:  # 向下向右平移
                after_shfit_image[shift_y:, shift_x:, :] = self.img[:h - shift_y, :w - shift_x, :]
                min_x, min_y, max_x, max_y = shift_x, shift_y, w, h
            elif shift_x >= 0 and shift_y < 0:  # 向上向右平移
                after_shfit_image[:h + shift_y, shift_x:, :] = self.img[-shift_y:, :w - shift_x, :]
                min_x, min_y, max_x, max_y = shift_x, 0, w, h - shift_y
            elif shift_x <= 0 and shift_y >= 0:  # 向下向左平移
                after_shfit_image[shift_y:, :w + shift_x, :] = self.img[:h - shift_y, -shift_x:, :]
                min_x, min_y, max_x, max_y = 0, shift_y, w, h
            else:  # 向上向左平移
                after_shfit_image[:h + shift_y, :w + shift_x, :] = self.img[-shift_y:, -shift_x:, :]
                min_x, min_y, max_x, max_y = 0, 0, w - shift_x, h - shift_y

            center_shift_y = center_y + shift_y
            center_shift_x = center_x + shift_x
            mask1 = (center_shift_x > 0) & (center_shift_x < w)
            mask2 = (center_shift_y > 0) & (center_shift_y < h)
            mask = np.logical_and(mask1, mask2)
            boxes_in = self.bboxes[mask]
            # 如果做完平移后bbox的中心点被移到了图像外,就撤销平移操作
            if len(boxes_in) > 0:
                # bbox平移
                boxes_in[:, [1, 3]] = np.clip(boxes_in[:, [1, 3]] + shift_y, a_min=min_y, a_max=max_y)
                boxes_in[:, [0, 2]] = np.clip(boxes_in[:, [0, 2]] + shift_x, a_min=min_x, a_max=max_x)
                labels_in = self.labels[mask]
                self.img = after_shfit_image
                self.bboxes = boxes_in
                self.labels = labels_in

    def randomCrop(self):
        # 随机裁剪
        if random.random() < self.aug_threshold:
            height, width, c = self.img.shape
            # x,y代表裁剪后的图像的中心坐标,h,w表示裁剪后的图像的高,宽
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(width / 4, 3 * width / 4)
            y = random.uniform(height / 4, 3 * height / 4)

            # 左上角
            crop_xmin = np.clip(x - (w / 2), a_min=0, a_max=width).astype(np.int32)
            crop_ymin = np.clip(y - (h / 2), a_min=0, a_max=height).astype(np.int32)
            # 右下角
            crop_xmax = np.clip(x + (w / 2), a_min=0, a_max=width).astype(np.int32)
            crop_ymax = np.clip(y + (h / 2), a_min=0, a_max=height).astype(np.int32)

            if self.strict:
                # 只留下那些bbox的中心点坐标仍在裁剪后区域内的bbox
                bbox_center_y = (self.bboxes[:, 1] + self.bboxes[:, 3]) / 2
                bbox_center_x = (self.bboxes[:, 0] + self.bboxes[:, 2]) / 2
                mask1 = (bbox_center_y < crop_ymax) & (bbox_center_x < crop_xmax)
                mask2 = (bbox_center_y > crop_ymin) & (bbox_center_x > crop_xmin)
                mask = mask1 & mask2
            else:
                # 只保留那些bbox左上角或者右上角坐标仍在裁剪区域内的bbox
                cliped_bbox = np.empty_like(self.bboxes)
                cliped_bbox[:, [0, 2]] = np.clip(self.bboxes[:, [0, 2]], crop_xmin, crop_xmax)
                cliped_bbox[:, [1, 3]] = np.clip(self.bboxes[:, [1, 3]], crop_ymin, crop_ymax)
                mask1 = (cliped_bbox[:, 2] - cliped_bbox[:, 0]) > 0
                mask2 = (cliped_bbox[:, 3] - cliped_bbox[:, 1]) > 0
                mask = mask1 & mask2

            bbox_out = self.bboxes[mask]
            labels_out = self.labels[mask]
            if len(bbox_out) > 0:
                crop_width = crop_xmax - crop_xmin
                crop_height = crop_ymax - crop_ymin
                bbox_out[:, [1, 3]] = np.clip(bbox_out[:, [1, 3]] - crop_ymin, a_min=0, a_max=crop_height)
                bbox_out[:, [0, 2]] = np.clip(bbox_out[:, [0, 2]] - crop_xmin, a_min=0, a_max=crop_width)
                new_img = self.img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
                self.img = new_img
                self.bboxes = bbox_out
                self.labels = labels_out



def RandomBlur(img, thresh):
    """

    :param img: ndarray
    :param thresh:
    :return:
    """
    assert isinstance(img, np.ndarray), f"Unkown Image Type {type(img)}"
    # 均值滤波平滑图像
    if random.random() < thresh:
        img_out = cv2.blur(img, (5, 5))
        return img_out
    return img


def RandomSaturation(img, thresh):
    # 图片饱和度
    if random.random() < thresh:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.choice([0.5, 1.5])
        s = s.astype(np.float32)
        s *= adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        img_out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img_out
    return img


def RandomBrightness(img, thresh):
    # 图片亮度
    if random.random() < thresh:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # hsv分别表示：色调(H),饱和度(S),明度(V)
        h, s, v = cv2.split(hsv)
        adjust = random.choice([0.5, 1.5])
        v = v.astype(np.float32)
        v *= adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        img_out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img_out
    return img


def RandomHSV(img, thresh, hgain=0.5, sgain=0.5, vgain=0.5):
    """
    将输入的RGB模态的image转换为HSV模态,并随机从对比度,饱和度以及亮度三个维度进行变换。

    :param img: ndarray / RGB
    :param thresh:
    :param hgain:
    :param sgain:
    :param vgain:
    :return:
    """
    assert isinstance(img, np.ndarray), f"Unkown Image Type {type(img)}"
    # 图片色调
    if random.random() < thresh:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        img_out = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)  # no return needed
        return img_out
    return img

    
def YOCO(img, aug_method):
    """
    Args:
        img: ndarray / (h, w, c)
        aug_method: augumentation functions
    Returns:
        img: image after YOCO
    """
    h, w, c = img.shape
    if np.random.random() < 0:  # 垂直切分并增强后合并
        aug_img = np.concatenate((aug_method(img[:, 0:int(w/2), :]), aug_method(img[:, int(w/2):, :])), axis=1)
    else:
        aug_img = np.concatenate((aug_method(img[0:h//2, :, :]), aug_method(img[h//2:, :, :])), axis=0)
    return aug_img

# endregion


# region
# ===================================== aug for img and bbox =====================================
def randomShift(self):
    # 随机平移
    center_y = (self.bboxes[:, 1] + self.bboxes[:, 3]) / 2
    center_x = (self.bboxes[:, 0] + self.bboxes[:, 2]) / 2
    if random.random() < self.aug_threshold:
        h, w, c = self.img.shape
        after_shfit_image = np.zeros((h, w, c), dtype=self.img.dtype)
        # after_shfit_image每行元素都设为[104,117,123]
        after_shfit_image[:, :, :] = (self.fill_value, self.fill_value, self.fill_value)  # bgr
        shift_x = int(random.uniform(-w * 0.2, w * 0.2))
        shift_y = int(random.uniform(-h * 0.2, h * 0.2))
        # 图像平移
        if shift_x >= 0 and shift_y >= 0:  # 向下向右平移
            after_shfit_image[shift_y:, shift_x:, :] = self.img[:h - shift_y, :w - shift_x, :]
            min_x, min_y, max_x, max_y = shift_x, shift_y, w, h
        elif shift_x >= 0 and shift_y < 0:  # 向上向右平移
            after_shfit_image[:h + shift_y, shift_x:, :] = self.img[-shift_y:, :w - shift_x, :]
            min_x, min_y, max_x, max_y = shift_x, 0, w, h - shift_y
        elif shift_x <= 0 and shift_y >= 0:  # 向下向左平移
            after_shfit_image[shift_y:, :w + shift_x, :] = self.img[:h - shift_y, -shift_x:, :]
            min_x, min_y, max_x, max_y = 0, shift_y, w, h
        else:  # 向上向左平移
            after_shfit_image[:h + shift_y, :w + shift_x, :] = self.img[-shift_y:, -shift_x:, :]
            min_x, min_y, max_x, max_y = 0, 0, w - shift_x, h - shift_y

        center_shift_y = center_y + shift_y
        center_shift_x = center_x + shift_x
        mask1 = (center_shift_x > 0) & (center_shift_x < w)
        mask2 = (center_shift_y > 0) & (center_shift_y < h)
        mask = np.logical_and(mask1, mask2)
        boxes_in = self.bboxes[mask]
        # 如果做完平移后bbox的中心点被移到了图像外,就撤销平移操作
        if len(boxes_in) > 0:
            # bbox平移
            boxes_in[:, [1, 3]] = np.clip(boxes_in[:, [1, 3]] + shift_y, a_min=min_y, a_max=max_y)
            boxes_in[:, [0, 2]] = np.clip(boxes_in[:, [0, 2]] + shift_x, a_min=min_x, a_max=max_x)
            labels_in = self.labels[mask]
            self.img = after_shfit_image
            self.bboxes = boxes_in
            self.labels = labels_in


def RandomScale(img, bboxes, thresh):
    """

    :param img: ndarray
    :param bboxes:
    :param thresh:
    :return:
    """
    assert isinstance(img, np.ndarray), f"Unkown Image Type {type(img)}"

    # 固定住高度,以0.8-1.2伸缩宽度,做图像形变
    if random.random() < thresh:
        scale = random.uniform(0.8, 1.2)
        # cv2.resize(img, shape)/其中shape->[宽,高]
        img_out = cv2.resize(img, None, fx=scale, fy=1)
        bboxes_out = np.zeros_like(bboxes)
        bboxes_out[:, [0, 2]] *= bboxes[:, [0, 2]] * scale
        return img_out, bboxes_out
    return img, bboxes


def randomCrop(self):
    # 随机裁剪
    if random.random() < self.aug_threshold:
        height, width, c = self.img.shape
        # x,y代表裁剪后的图像的中心坐标,h,w表示裁剪后的图像的高,宽
        h = random.uniform(0.6 * height, height)
        w = random.uniform(0.6 * width, width)
        x = random.uniform(width / 4, 3 * width / 4)
        y = random.uniform(height / 4, 3 * height / 4)

        # 左上角
        crop_xmin = np.clip(x - (w / 2), a_min=0, a_max=width).astype(np.int32)
        crop_ymin = np.clip(y - (h / 2), a_min=0, a_max=height).astype(np.int32)
        # 右下角
        crop_xmax = np.clip(x + (w / 2), a_min=0, a_max=width).astype(np.int32)
        crop_ymax = np.clip(y + (h / 2), a_min=0, a_max=height).astype(np.int32)

        if self.strict:
            # 只留下那些bbox的中心点坐标仍在裁剪后区域内的bbox
            bbox_center_y = (self.bboxes[:, 1] + self.bboxes[:, 3]) / 2
            bbox_center_x = (self.bboxes[:, 0] + self.bboxes[:, 2]) / 2
            mask1 = (bbox_center_y < crop_ymax) & (bbox_center_x < crop_xmax)
            mask2 = (bbox_center_y > crop_ymin) & (bbox_center_x > crop_xmin)
            mask = mask1 & mask2
        else:
            # 只保留那些bbox左上角或者右上角坐标仍在裁剪区域内的bbox
            cliped_bbox = np.empty_like(self.bboxes)
            cliped_bbox[:, [0, 2]] = np.clip(self.bboxes[:, [0, 2]], crop_xmin, crop_xmax)
            cliped_bbox[:, [1, 3]] = np.clip(self.bboxes[:, [1, 3]], crop_ymin, crop_ymax)
            mask1 = (cliped_bbox[:, 2] - cliped_bbox[:, 0]) > 0
            mask2 = (cliped_bbox[:, 3] - cliped_bbox[:, 1]) > 0
            mask = mask1 & mask2

        bbox_out = self.bboxes[mask]
        labels_out = self.labels[mask]
        if len(bbox_out) > 0:
            crop_width = crop_xmax - crop_xmin
            crop_height = crop_ymax - crop_ymin
            bbox_out[:, [1, 3]] = np.clip(bbox_out[:, [1, 3]] - crop_ymin, a_min=0, a_max=crop_height)
            bbox_out[:, [0, 2]] = np.clip(bbox_out[:, [0, 2]] - crop_xmin, a_min=0, a_max=crop_width)
            new_img = self.img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
            self.img = new_img
            self.bboxes = bbox_out
            self.labels = labels_out


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


def random_perspective(img, tar_bboxes, tar_labels, degrees=10, translate=.1, scale=.1, shear=10,
                       perspective=0.0, dst_size=448, fill_value=128):
    """
    random perspective one image
    :param img:
    :param tar_bboxes: [xmin, ymin, xmax, ymax] / ndarray
    :param tar_labels: [a, b, ...] / ndarray
    :param degrees:
    :param translate:
    :param scale:
    :param shear:
    :param perspective:
    :param dst_size: output image size
    :param fill_value:
    :return:
    """
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3, "only process one image once for now"
    assert isinstance(tar_labels, np.ndarray), "tar_label must be ndarray"
    assert len(tar_labels) == len(tar_bboxes), \
        f"the length of bboxes and labels should be the same, but len(bboxes)={len(tar_bboxes)} and len(labels)={len(tar_labels)}"

    if isinstance(dst_size, int):
        dst_size = [dst_size, dst_size]

    if img.shape[0] != dst_size:
        height, width = dst_size

    # Center / Translation / 平移
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale / 旋转,缩放
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear / 剪切
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation / 平移
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (dst_size[0] != img.shape[0]) or (M != np.eye(3)).any():  # image changed
        if isinstance(fill_value, (int, tuple)):
            fill_value = (fill_value, fill_value, fill_value)
        if perspective:  # 是否进行透视变换
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=fill_value)
        else:  # 只进行仿射变换
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=fill_value)

    # Transform label coordinates
    n = len(tar_bboxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        # left-top, right-bottom, left-bottom, right-top
        xy[:, :2] = tar_bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]  # x_lt, x_rb, x_lb, x_rt
        y = xy[:, [1, 3, 5, 7]]  # y_lt, y_rb, y_lb, y_rt
        # 将变换后的bbox摆正
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=tar_bboxes[:, :4].T * s, box2=xy.T)
        tar_bboxes = tar_bboxes[i]
        tar_labels = tar_labels[i]
        tar_bboxes[:, :4] = xy[i]

    return img, tar_bboxes, tar_labels


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


def mosaic(imgs, bboxes, labels, mosaic_shape=640*2, fill_value=128):
    """
    mosaic four images
    :param fill_value:
    :param labels:
    :param imgs: four image list
    :param bboxes: list of ndarray; ndarray's shape is Nx4; bbox's format is [xmin, ymin, xmax, ymax]
    :param mosaic_shape: a list: [h, w] or an integer
    """
    assert len(imgs) == 4, f"only support mosiac 4 images for now, but given {len(imgs)} images!"
    assert len(imgs) == len(bboxes), f"len(imgs) = {len(imgs)} not equal len(bboxes) = {len(bboxes)}!"
    assert len(imgs) == len(labels), f"len(imgs) = {len(imgs)} not equal len(labels) = {len(labels)}!"

    if isinstance(mosaic_shape, int):
        mosaic_shape = [mosaic_shape, mosaic_shape]

    xc, yc = [int(random.uniform(2*x/5, 4*x/5)) for x in (np.array(mosaic_shape))]
    img_out = np.full(shape=mosaic_shape + [3], dtype=np.uint8, fill_value=fill_value)
    bboxes_out = []
    labels_out = []
    for i, img in enumerate(imgs):
        h, w = img.shape[:2]
        if i == 0:
            xmin_o, ymin_o, xmax_o, ymax_o = max(xc-w, 0), max(yc-h, 0), xc, yc
        elif i == 1:
            xmin_o, ymin_o, xmax_o, ymax_o = xc, max(yc-h, 0), min(xc+w, mosaic_shape[1]), yc
        elif i == 2:
            xmin_o, ymin_o, xmax_o, ymax_o = max(xc-w, 0), yc, xc, min(yc+h, mosaic_shape[0])
        else:
            xmin_o, ymin_o, xmax_o, ymax_o = xc, yc, min(xc+w, mosaic_shape[1]), min(yc+h, mosaic_shape[0])

        # 截取各子image的中心区域贴到mosaic image上
        xc_i, yc_i, w_i, h_i = w // 2, h // 2,  xmax_o-xmin_o, ymax_o-ymin_o
        detal_w_i, detal_h_i = w_i // 2, h_i // 2
        xmin_i = xc_i - detal_w_i
        ymin_i = yc_i - detal_h_i
        xmax_i = xc_i + (w_i - detal_w_i)
        ymax_i = yc_i + (h_i - detal_h_i)
        img_out[ymin_o:ymax_o, xmin_o:xmax_o] = img[ymin_i:ymax_i, xmin_i:xmax_i]

        # 处理label
        box = bboxes[i].copy().astype(np.float32)
        # in small image
        box[:, [0, 2]] = np.clip(np.round(box[:, [0, 2]], decimals=2), xmin_i+1, xmax_i-1)
        box[:, [1, 3]] = np.clip(np.round(box[:, [1, 3]], decimals=2), ymin_i+1, ymax_i-1)
        box[:, [0, 2]] -= xmin_i
        box[:, [1, 3]] -= ymin_i

        # ===============================================================
        # 对target进行过滤,剔除掉经过裁切后与原bbox重合面积过小的bbox
        box[:, [0, 2]] += xmin_o  # x
        box[:, [1, 3]] += ymin_o  # y
        org_box_area = np.prod(bboxes[i][:, 2:4] - bboxes[i][:, 0:2], axis=1) + 1e-16
        mdy_box_area = np.prod(box[:, 2:4] - box[:, 0:2], axis=1) + 1e-16
        iou = np.round(mdy_box_area / org_box_area, decimals=3)
        assert np.sum(iou[iou > 1]) == 0
        valid_idx = iou >= 0.15
        bboxes_out.append(box[valid_idx])
        labels_out.extend(np.array(labels[i])[valid_idx])
        # ===============================================================

        # in big image(do not remove any box)
        # box[:, [0, 2]] += xmin_o  # x
        # box[:, [1, 3]] += ymin_o  # y
        # bboxes_out.append(box)
        # labels_out.extend(labels[i])

    bboxes_out = np.concatenate(bboxes_out, axis=0)
    bboxes_out = np.clip(bboxes_out, 0, mosaic_shape[0])
    labels_out = np.array(labels_out)

    return img_out, bboxes_out, labels_out


def mixup(img1, bbox1, label1, img2, bbox2, label2):
    """
    使用不同的透明度混合两张图像(保持两张图像的target box)。

    :param bbox2:
    :param bbox1:
    :param img1: ndarray
    :param label1:
    :param img2: ndarray
    :param label2:
    :return:
    """
    assert isinstance(img1, np.ndarray) and img1.dtype == 'uint8', "images's dtype must be np.ndarray with np.uint8"
    assert isinstance(img2, np.ndarray) and img2.dtype == 'uint8', "images's dtype must be np.ndarray with np.uint8"
    assert img1.shape == img2.shape, f"the shape of image1 and image2 should be the same, " \
                                     f"but len(image1)={len(img1)} and len(image2)={len(img2)}"
    assert isinstance(bbox1, np.ndarray)
    assert isinstance(bbox2, np.ndarray)
    assert len(bbox1) == len(label1)
    assert len(bbox2) == len(label2)

    mix_ratio = np.random.beta(8., 8.)
    img_out = (img1 * mix_ratio + img2 * (1 - mix_ratio)).astype(np.uint8)
    bbox_out = np.concatenate([bbox1, bbox2], axis=0)
    label_out = np.concatenate([label1, label2], axis=0)
    return img_out, bbox_out, label_out


def RandomFlipLR(img, bboxes, thresh):
    """
    随机左右翻转image。

    :param img: ndarray
    :param bboxes: [xmin, ymin, xmax, ymax]
    :param thresh:
    :return:
    """
    assert isinstance(img, np.ndarray), f"Unkown Image Type {type(img)}"

    # 水平翻转/y坐标不变,x坐标变化
    if random.random() < thresh:
        img_out = np.fliplr(img).copy()
        _, w = img.shape[:2]
        xmax = w - bboxes[:, 0]
        xmin = w - bboxes[:, 2]
        bboxes_out = bboxes.copy()
        bboxes_out[:, 0] = xmin
        bboxes_out[:, 2] = xmax
        return img_out, bboxes_out
    return img, bboxes


def RandomFlipUD(img, bboxes, thresh):
    """
    随机上下翻转image。

    :param img: ndarray
    :param bboxes: [xmin, ymin, xmax, ymax]
    :param thresh: 0 ~ 1
    """
    assert isinstance(img, np.ndarray), f"Unkown Image Type {type(img)}"

    # 竖直翻转/x坐标不变,y坐标变化
    if random.random() < thresh:
        img_out = np.flipud(img).copy()
        h, _ = img.shape[:2]
        ymax = h - bboxes[:, 1]
        ymin = h - bboxes[:, 3]
        bboxes_out = bboxes.copy()
        bboxes_out[:, 1] = ymin
        bboxes_out[:, 3] = ymax
        return img_out, bboxes_out
    return img, bboxes


def cutout(img, bbox, cls, cutout_iou_thr=0.7):
    """
    在图像中挖孔,并使用随机颜色填充。

    :param img: ndarray / (h, w, 3)
    :param bbox: ndarray / (N, 4) / [xmin, ymin, xmax, ymax]
    :param cls: ndarray / (N,) / [1, 2, 3, ...]
    :param cutout_p: 使用cutout的概率
    :param cutout_iou_thr: cutout部分图像与target的所有bbox计算iou,iou值小于等于该阈值的mask视为有效的mask(剔除与target重合过多的mask)
    """
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    h, w, _ = img.shape

    img_cut, bboxes_cut, classes_cut = img.copy(), bbox.copy(), cls.copy()
    valid_ann_idx = []

    for s in scales:
        mask_h = np.random.randint(1, int(h * s))
        mask_w = np.random.randint(1, int(w * s))

        mask_xc = np.random.randint(0, w)
        mask_yc = np.random.randint(0, h)
        mask_xmin = np.clip(mask_xc - mask_w // 2, 0, w)  # (1,)
        mask_ymin = np.clip(mask_yc - mask_h // 2, 0, h)  # (1,)
        mask_xmax = np.clip(mask_xc + mask_w // 2, 0, w)  # (1,)
        mask_ymax = np.clip(mask_yc + mask_h // 2, 0, h)  # (1,)

        mask_w = mask_xmax - mask_xmin
        mask_h = mask_ymax - mask_ymin
        mask_area = np.clip(mask_w * mask_h, 0., None)

        bbox_area = np.clip(np.prod(bbox[:, 2:4] - bbox[:, 0:2], axis=1), 0., None)

        ints_xmin = np.maximum(bbox[:, 0], mask_xmin)  # (N,)
        ints_ymin = np.maximum(bbox[:, 1], mask_ymin)  # (N,)
        ints_xmax = np.minimum(bbox[:, 2], mask_xmax)  # (N,)
        ints_ymax = np.minimum(bbox[:, 3], mask_ymax)  # (N,)

        ints_w = np.clip(ints_xmax - ints_xmin, 0., w)
        ints_h = np.clip(ints_ymax - ints_ymin, 0., h)
        ints_area = np.clip(ints_w * ints_h, 0., None)
        iou = ints_area / (mask_area + bbox_area - ints_area + 1e-16)  # (N,)
        valid_idx = iou <= cutout_iou_thr
        valid_ann_idx.append(np.nonzero(valid_idx))
        illegal_idx = iou > cutout_iou_thr

        if np.all(illegal_idx):  # 如果cutout部分图像与target的某个bbox重合面积过大,则跳过这一次的cutout操作(舍弃mask)
            continue
        else:  # 如果cutout部分图像与target中的某个bbox重合面积过大,则删除相应的target bbox(舍弃target)
            mask_color = [np.random.randint(69, 200) for _ in range(3)]
            img_cut[mask_ymin:mask_ymax, mask_xmin:mask_xmax] = mask_color

    valid_idx = reduce(np.intersect1d, valid_ann_idx)
    if 0 < len(valid_idx): 
        bboxes_cut = bboxes_cut[valid_idx]
        classes_cut = np.array(classes_cut)[valid_idx]
        return img_cut, bboxes_cut, classes_cut
    else:
        return img, bbox, cls


def scale_jitting(img, bbox, label, dst_size=None):
    """
    :param: bbox: (xmin, ymin, xmax, ymax)
    :param: dst_size: (h, w)
    将输入的image进行随机scale的缩放后,再从缩放后的image中裁剪固定尺寸的image
    """
    FLIP_LR = np.random.rand() > 0.5

    if dst_size and isinstance(dst_size, int):
        dst_size = [dst_size, dst_size]
    else:
        dst_size = img.shape[:2]

    # jit_scale = max(img.shape[0]/dst_size[0], img.shape[1]/dst_size[1], dst_size[0]/img.shape[0], dst_size[1]/img.shape[1])
    scale = min(img.shape[0]/dst_size[0], img.shape[1]/dst_size[1])
    # 确保输入图片的尺寸大于dst_size,如果不满足则放大输入的image
    if scale < 1.:
        jit_scale = max(dst_size[0]/img.shape[0], dst_size[1]/img.shape[1]) + np.random.uniform(0.5, 1.5)
    else:
        jit_scale = max(dst_size[0]/img.shape[0], dst_size[1]/img.shape[1]) + np.random.uniform(0., 0.5)

    if jit_scale != 1.:
        resized_h, resized_w = int(img.shape[0]*jit_scale), int(img.shape[1]*jit_scale)
        resized_img = cv2.resize(np.ascontiguousarray(img.copy()), (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        if FLIP_LR and img.ndim == 3:
            resized_img = resized_img[:, ::-1, :]
        if FLIP_LR and img.ndim == 2:
            resized_img = resized_img[:, ::-1]
        x_offset, y_offset = 0, 0
        if resized_h > dst_size[0]:
            y_offset = np.random.randint(0, resized_h-dst_size[0])
        if resized_w > dst_size[1]:
            x_offset = np.random.randint(0, resized_w-dst_size[1])
        img_out = resized_img[y_offset: y_offset+dst_size[0], x_offset:x_offset+dst_size[1]]

        # process bbox and label
        bbox_out = bbox.copy()
        bbox_out *= jit_scale
        if FLIP_LR:
            bbox_out[:, [0, 2]] = resized_w - bbox_out[:, [0, 2]]  # [xmin, ymin, xmax, ymax] -> [xmax, ymin, xmin, ymax]
            tmp_xmin = bbox_out[:, 2].copy()
            bbox_out[:, 2] = bbox_out[:, 0]  # [xmax, ymin, xmax, ymax]
            bbox_out[:, 0] = tmp_xmin  # [xmin, ymin, xmax, ymax]
        bbox_out[:, [0, 2]] -= x_offset
        bbox_out[:, [0, 2]] = np.clip(bbox_out[:, [0, 2]], 0, dst_size[1])
        bbox_out[:, [1, 3]] -= y_offset
        bbox_out[:, [1, 3]] = np.clip(bbox_out[:, [1, 3]], 0, dst_size[0])
        resized_hs = bbox_out[:, 2] - bbox_out[:, 0] + 1e-16
        resized_ws = bbox_out[:, 3] - bbox_out[:, 1] + 1e-16
        ar = np.maximum(resized_ws/resized_hs, resized_hs/resized_ws)
        keep = (ar < 20) & (resized_hs >= 3) & (resized_ws >= 3)
        if np.sum(keep) > 0:
            return img_out, bbox_out[keep], np.array(label)[keep]
        
    return img, bbox, label

# endregion