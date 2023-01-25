from utils import RandomHSV, RandomFlipLR, RandomFlipUD, scale_jitting, cutout, RandomPerspective, RandomBlur


__all__ = ["Transforms"]

class Transforms:

    def __init__(self, data_aug_hyp):
        self.data_aug_hyp = data_aug_hyp

    def __call__(self, img, box, lab):
        if self.data_aug_hyp is not None:
            img, box, lab = RandomPerspective(img, box, lab,
                                              self.data_aug_hyp['data_aug_prespective_p'], 
                                              self.data_aug_hyp["data_aug_degree"],
                                              self.data_aug_hyp['data_aug_translate'],
                                              self.data_aug_hyp['data_aug_scale'],
                                              self.data_aug_hyp['data_aug_shear'],
                                              self.data_aug_hyp['data_aug_prespective'],
                                              self.data_aug_hyp['input_img_size'],
                                              self.data_aug_hyp["data_aug_fill_value"])
            img, box, lab = cutout(img, box, lab, 
                                   self.data_aug_hyp['data_aug_cutout_iou_thr'], 
                                   self.data_aug_hyp['data_aug_cutout_p'])
            img = RandomHSV(img, 
                            self.data_aug_hyp['data_aug_hsv_p'], 
                            self.data_aug_hyp['data_aug_hsv_hgain'], 
                            self.data_aug_hyp['data_aug_hsv_sgain'], 
                            self.data_aug_hyp['data_aug_hsv_vgain'])
            img, box = RandomFlipLR(img, box, self.data_aug_hyp['data_aug_fliplr_p'])
            img, box = RandomFlipUD(img, box, self.data_aug_hyp['data_aug_flipud_p'])
            img, box, lab = scale_jitting(img, box, lab, self.data_aug_hyp['data_aug_scale_jitting_p'])
        return img, box, lab