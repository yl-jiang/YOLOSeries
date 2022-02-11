import numpy as np
import random
import cv2
import matplotlib.pyplot as plt


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


class CV2Transform:
    """
    pass
    """
    def __init__(self, cv_img, bboxes=None, labels=None, aug_threshold=None, strict=False, fill_value=128):
        # cv_img:bgr/(h,w,c)
        # bboxes:[[xmin,ymin,xmax,ymax],...]
        # labels:[2,4,...]

        if aug_threshold is None:
            self.aug_threshold = 0.2
        else:
            self.aug_threshold = aug_threshold

        self.img = cv_img
        self.bboxes = bboxes
        self.labels = labels
        self.strict = strict
        self.fill_value = fill_value

        # inplace modify self.img and self.bboxes
        self.randomFlip()
        self.randomScale()
        self.randomBlur()
        # self.RandomBrightness()
        self.randomHue()
        # self.RandomSaturation()
        self.randomShift()
        self.randomCrop()

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

    def randomFlip(self):
        # 垂直翻转/y坐标不变，x坐标变化
        if random.random() < self.aug_threshold:
            img = np.fliplr(self.img).copy()
            h, w = self.img.shape[:2]
            xmax = w - self.bboxes[:, 0]
            xmin = w - self.bboxes[:, 2]
            self.bboxes[:, 0] = xmin
            self.bboxes[:, 2] = xmax
            self.img = img

    def randomScale(self):
        # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < self.aug_threshold:
            scale = random.uniform(0.8, 1.2)
            # cv2.resize(img, shape)/其中shape->[宽，高]
            self.img = cv2.resize(self.img, None, fx=scale, fy=1)
            self.bboxes[:, [0, 2]] *= scale

    def randomBlur(self):
        # 均值滤波平滑图像
        if random.random() < self.aug_threshold:
            cv2.blur(self.img, (5, 5), dst=self.img)

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
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=self.img)  # no return needed

    def RandomSaturation(self):
        # 图片饱和度
        if random.random() < self.aug_threshold:
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=self.img)

    def RandomBrightness(self):
        # 图片亮度
        if random.random() < self.aug_threshold:
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            # hsv分别表示：色调（H），饱和度（S），明度（V）
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
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
            # 如果做完平移后bbox的中心点被移到了图像外，就撤销平移操作
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
            # x,y代表裁剪后的图像的中心坐标，h,w表示裁剪后的图像的高，宽
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


def RandomScale(img, bboxes, thresh):
    """

    :param img: ndarray
    :param bboxes:
    :param thresh:
    :return:
    """
    assert isinstance(img, np.ndarray), f"Unkown Image Type {type(img)}"

    # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
    if random.random() < thresh:
        scale = random.uniform(0.8, 1.2)
        # cv2.resize(img, shape)/其中shape->[宽，高]
        img_out = cv2.resize(img, None, fx=scale, fy=1)
        bboxes_out = np.zeros_like(bboxes)
        bboxes_out[:, [0, 2]] *= bboxes[:, [0, 2]] * scale
        return img_out, bboxes_out
    return img, bboxes


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
        # hsv分别表示：色调（H），饱和度（S），明度（V）
        h, s, v = cv2.split(hsv)
        adjust = random.choice([0.5, 1.5])
        v *= adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        img_out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img_out
    return img


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
        # 如果做完平移后bbox的中心点被移到了图像外，就撤销平移操作
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
        # x,y代表裁剪后的图像的中心坐标，h,w表示裁剪后的图像的高，宽
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
    cv2.rectangle(trans.img, (trans.bboxes[:, 1], trans.bboxes[:, 0]), (trans.bboxes[:, 3], trans.bboxes[:, 2]),
                  (55, 255, 155), 5)

    cv2.imshow('image', trans.img)
    cv2.waitKey(50000)


