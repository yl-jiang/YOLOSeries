from utils import mosaic, random_perspective, valid_bbox, mixup, cutout
from .datasets_wrapper import Dataset
import random
import numpy as np
import cv2
from utils import scale_jitting

__all__ = ["MosaicTransformDataset"]


class MosaicTransformDataset(Dataset):

    def __init__(self, 
                 dataset, 
                 input_dim, 
                 do_mosaic, 
                 preproc=None,
                 mosaic_p=1.0, 
                 degree=0.2, 
                 translate=0.3, 
                 scale=0.5, 
                 shear=0.3, 
                 presepctive=False, 
                 mixup_p=0.1, 
                 mixup_scale=[0.5, 1.5],
                 fill_value=128, 
                 ):
         super().__init__(input_dim, mosaic)
         self._dataset = dataset
         self.do_mosaic = do_mosaic
         self.degree = degree
         self.translate = translate
         self.scale = scale
         self.shear = shear
         self.presepctive = presepctive
         self.mixup_p = mixup_p
         self.mixup_scale = mixup_scale
         self.fill_value = fill_value
         self.mosaic_p = mosaic_p
         self.preproc = preproc

    def __len__(self):
        return len(self._dataset)

    def load_img(self, idx):
        return self._dataset.load_img(idx)

    @property
    def num_class(self):
        return len(self._dataset.classes)

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
        indices = [ix] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]
        random.shuffle(indices)
        imgs, bboxes, labels = [], [], []

        for i in indices:
            img, ann = self._dataset.load_img_and_ann(i)
            bboxes.append(ann['bboxes'])
            labels.append(ann['classes'])
            imgs.append(img)

        img, bboxes, labels = mosaic(imgs, bboxes, labels, mosaic_shape=[_*2 for _ in self.input_dim], fill_value=self.fill_value)
        img, bboxes, labels = random_perspective(img, bboxes, labels,
                                                 self.degree,
                                                 self.translate,
                                                 self.scale,
                                                 self.shear,
                                                 self.presepctive,
                                                 self.input_dim,
                                                 self.fill_value)
        return img, bboxes, labels

    @Dataset.mosaic_getitem
    def __getitem__(self, ix):
        """
        Returns:
            img: ndarray 
            ann: dict like: {"bboxes": ndarray, "classes": list}
            ix: img id
        """
        if self.do_mosaic and random.random() < self.mosaic_p:
            img, bboxes, classes = self.load_mosaic(ix)
            if random.random() < self.mixup_p:
                img, bboxes, classes = self.powerful_mixup(img, bboxes, classes)
        else:
            img, ann = self._dataset.load_img_and_ann(ix)
            bboxes, classes = ann["bboxes"], ann["classes"]

        if self.preproc is not None:
            img, bboxes, classes = self.preproc(img, bboxes, classes)

        ann = {"bboxes": bboxes, "classes": classes}
        return img, ann, ix

    def powerful_mixup(self, org_img, org_bboxes, org_classes):
        if random.random() < 0.5:
            img2, bboxes2, classes2 = self.load_mosaic(random.randint(0, len(self._dataset) - 1))
            img, bboxes, classes = mixup(org_img, org_bboxes, org_classes, img2, bboxes2, classes2)
        else:
            jit_factor = random.uniform(*self.mixup_scale)
            FLIP = random.uniform(0, 1) > 0.5  # flip left and right
            classes2 = []
            while len(classes2) == 0:
                rnd_idx = random.randint(0, len(self._dataset) - 1)
                mixin_img, mixin_ann = self._dataset.load_img_and_ann(rnd_idx)
                mixin_bboxes = mixin_ann["bboxes"]
                classes2 = mixin_ann['classes']

            cp_img = np.ones((self.input_dim[0], self.input_dim[1], 3), dtype=np.uint8) * self.fill_value
            cp_scale_ratio = min(self.input_dim[0] / mixin_img.shape[0], self.input_dim[1] / mixin_img.shape[1])
            resized_img = cv2.resize(np.ascontiguousarray(mixin_img), 
                                     (int(mixin_img.shape[1] * cp_scale_ratio), int(mixin_img.shape[0] * cp_scale_ratio)), 
                                     interpolation=cv2.INTER_LINEAR)
            cp_img[:int(mixin_img.shape[0]*cp_scale_ratio), :int(mixin_img.shape[1]*cp_scale_ratio)] = resized_img
            cp_img = cv2.resize(np.ascontiguousarray(cp_img), (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)), interpolation=cv2.INTER_LINEAR)
            cp_scale_ratio *= jit_factor

            if FLIP:
                cp_img = cp_img[:, ::-1, :]
            
            cp_h, cp_w = cp_img.shape[:2]
            tar_h, tar_w = org_img.shape[:2]
            padded_img = np.zeros((max(cp_h, tar_h), max(cp_w, tar_w), 3), dtype=np.uint8) 
            padded_img[:cp_h, :cp_w] = cp_img
            x_offset, y_offset = 0, 0
            if padded_img.shape[0] > tar_h:
                y_offset = random.randint(0, padded_img.shape[0] - tar_h - 1)
            if padded_img.shape[1] > tar_w:
                x_offset = random.randint(0, padded_img.shape[1] - tar_w - 1)

            img2 = padded_img[y_offset:y_offset+tar_h, x_offset:x_offset+tar_w, :]
            mixin_bboxes_copy = mixin_bboxes.copy()
            mixin_bboxes[:, 0::2] = np.clip(mixin_bboxes_copy[:, 0::2] * cp_scale_ratio, 0, cp_w)
            mixin_bboxes[:, 1::2] = np.clip(mixin_bboxes_copy[:, 1::2] * cp_scale_ratio, 0, cp_h)
            if FLIP:
                mixin_bboxes[:, 0::2] = (cp_w - mixin_bboxes[:, 0::2][:, ::-1])
            bboxes2 = mixin_bboxes.copy()
            bboxes2[:, 0::2] = np.clip(mixin_bboxes[:, 0::2] - x_offset, 0, tar_w)
            bboxes2[:, 1::2] = np.clip(mixin_bboxes[:, 1::2] - y_offset, 0, tar_h)

        bboxes = np.vstack((org_bboxes, bboxes2))
        classes = np.hstack((org_classes, classes2))
        img = org_img.astype(np.float32) * 0.5 + img2.astype(np.float32) * 0.5
        return img.astype(np.uint8), bboxes, classes
        

            
            



