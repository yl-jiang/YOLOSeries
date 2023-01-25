from numpy import flipud
from sklearn.preprocessing import scale
from utils import mosaic, random_perspective, valid_bbox, mixup, cutout
from utils import RandomHSV, RandomFlipLR, RandomFlipUD, scale_jitting, YOCO


import random
import numpy as np

__all__ = ["Transform"]

class Transform:

    def __init__(self, 
                 hsv_p=0.1, 
                 hgain=0.015, 
                 sgain=0.5, 
                 vgain=0.5, 
                 cutout_p=0.1, 
                 cutout_iou_thr=0.7, 
                 fliplr_p=0.3, 
                 flipud_p=0.01, 
                 scale_jitting_p=0.01) -> None:
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
        Return:
            img:
            bboxes:
            labels:
        """

        if random.random() < self.cutout_p:
            img, bboxes, labels = cutout(img, bboxes, labels, cutout_iou_thr=self.cutout_iou_thr)

        img = RandomHSV(img, self.hsv_p, self.hgain, self.sgain, self.vgain)
        img, bboxes = RandomFlipLR(img, bboxes, self.fliplr_p)
        img, bboxes = RandomFlipUD(img, bboxes, self.flipud_p)

        if random.random() < self.scale_jitting_p:
            img, bboxes, labels = scale_jitting(img, bboxes, labels)

        return np.ascontiguousarray(img), np.ascontiguousarray(bboxes), np.ascontiguousarray(labels)