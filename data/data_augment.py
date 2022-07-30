from numpy import flipud
from sklearn.preprocessing import scale
from utils import mosaic, random_perspective, valid_bbox, mixup, cutout
from utils import RandomHSV, RandomFlipLR, RandomFlipUD, scale_jitting, YOCO
from .dataset import Dataset

import random
import numpy as np


class Transform:

    def __init__(self, hsv_p=0.1, hgain=5, sgain=30, vgain=30, cutout_p=0.1, cutout_iou_thr=0.3, fliplr_p=0.1, flipud_p=0.1, scale_jitting_p=0.1) -> None:
           self.hsv_p  = hsv_p
           self.hgain = hgain
           self.vgain = vgain
           self.sgain = sgain
           self.cutout_p = cutout_p
           self.cutout_iou_thr = cutout_iou_thr
           self.fliplr_p = fliplr_p
           self.flipud_p = flipud_p
           self.scale_jitting_p = scale_jitting_p
           
    def __call__(self, img, bboxes, labels):
        """
        Args:
            img: ndarray of shape (h, w, c)
            bboxes: ndarray like [[xmin, ymin, xmax, ymax], [...]]
            labels: list of int like [cls1, cls2, ...]
        """

        if random.random() < self.cutout_p:
            img, bboxes, labels = cutout(img, bboxes, labels, cutout_iou_thr=self.cutout_iou_thr)

        img = RandomHSV(img, self.hsv_p, self.hgain, self.sgain, self.vgain)
        img, bboxes = RandomFlipLR(img, bboxes, self.fliplr)
        img, bboxes = RandomFlipUD(img, bboxes, self.flipud)

        if random.random() < self.scale_jitting_p:
            img, bboxes, labels = scale_jitting(img, bboxes, labels)

        return np.ascontiguousarray(img), np.ascontiguousarray(bboxes), np.ascontiguousarray(labels)



class MoasicTransform(Dataset):

    def __init__(self, input_dimension, mosaic=True):
         super().__init__(input_dimension, mosaic)

    def load_mosaic(self, ix):
        """
        mosaic augumentation
        
        Args:
            ix: image index

        Returns:
            img: numpy.ndarray; (h, w, 3)
            bboxes: 
            labels: 
        """
        indices = [ix] + [random.randint(0, len(self) - 1) for _ in range(3)]
        random.shuffle(indices)
        imgs, bboxes, labels = [], [], []

        for i in indices:
            img, ann = self.dataset.load_img_and_ann(i)
            bboxes.append(ann['bboxes'])
            labels.append(ann['classes'])
            imgs.append(img)

        img, bboxes, labels = mosaic(imgs, bboxes, labels,
                                    mosaic_shape=[_*2 for _ in self.input_img_size],
                                    fill_value=self.fill_value)
        img, bboxes, labels = random_perspective(img, bboxes, labels,
                                                self.data_aug_param["degree"],
                                                self.data_aug_param['translate'],
                                                self.data_aug_param['scale'],
                                                self.data_aug_param['shear'],
                                                self.data_aug_param['presepctive'],
                                                self.input_img_size,
                                                self.fill_value)
        return img, bboxes, labels


    
    def __call__(self, img, ) -> Any:
         return super().__call__(*args, **kwds)
         if random.random() < self.data_aug_param.get('mosaic', 0.0):
                img, bboxes, labels = self.load_mosaic(ix)
                if random.random() < self.data_aug_param.get('mixup', 0.0):
                    img2, bboxes2, labels2 = self.load_mosaic(random.randint(0, len(self) - 1))
                    img, bboxes, labels = mixup(img, bboxes, labels, img2, bboxes2, labels2)