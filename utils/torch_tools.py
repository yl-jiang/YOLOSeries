from torch.utils.data.sampler import Sampler
import random
import torch
import torchvision
from utils import letter_resize_bbox, letter_resize_img, maybe_mkdir, clear_dir
from tqdm import trange
import pickle
import numpy as np

def normalization(img):
    # 输入图像的格式为(h,w,3)
    assert len(img.shape) == 3 and img.shape[-1] == 3
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    return transforms(img)

class AspectRatioBatchSampler(Sampler):
    """
    按照图片的长宽比对输入网络训练图片进行从小到大的重新排列
    """

    def __init__(self, data_source, batch_size, drop_last, aspect_ratio_list=None):
        super(AspectRatioBatchSampler, self).__init__(data_source)
        assert hasattr(data_source, "aspect_ratio"), f"data_source should has method of aspect_ratio"
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.ar = aspect_ratio_list
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        """
        按照图片的长宽比对输入网络训练图片进行从小到大的重新排列
        :return:
        """
        # determine the order of the images
        order = list(range(len(self.data_source))) 
        if self.ar is None:
            order = list(range(len(self.data_source)))
            order.sort(key=lambda x: self.data_source.aspect_ratio(x))
            pickle.dump(order, open("./dataset/pkl/wheat_aspect_ratio.pkl", 'wb'))
        else:
            sort_i = np.argsort(self.ar)
            order = np.array(order)[sort_i]

        groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)]
                  for i in range(0, len(order), self.batch_size)]
        # divide into groups, one group = one batch
        return groups



def fixed_imgsize_collector(data_in, dst_size):
    """

    :param data_in: tuple, data[0] is image, data[1] is annotation, data[2] is image's id
    :param dst_size:
    :return:
    """
    # 输入图像的格式为(h,w,3)
    assert data_in[0][0].ndim == 3 and data_in[0][0].shape[-1] == 3, f"data's formate should be (h, w, 3), but got {data_in[0].shape}"
    
    batch_size = len(data_in)
    imgs = [d[0] for d in data_in]
    anns = [d[1] for d in data_in]
    img_ids = [d[2] for d in data_in]

    # batch内image的图像拥有相同的shape，batch之间image的shape不一样
    # dst_size = padding(dst_size, 32)
    imgs_out = torch.zeros(batch_size, 3, dst_size[0], dst_size[1])
    boxes_num = [len(ann['bboxes']) for ann in anns]
    # 初始化为-1是为了区分有无object的bbox，最后一个维度是为了标记一个batch中每个ann对应的img idx，for build_target()
    anns_out = torch.ones(batch_size, max(boxes_num), 6) * -1

    # resize_info在测试时恢复原始图像用
    resize_infos = []

    for b in range(batch_size):
        img = imgs[b]  # ndarray
        ann = anns[b]  # dict
        ann_bboxes = ann['bboxes']
        ann_classes = ann['classes']
        assert len(ann_bboxes) == len(ann_classes)

        img, resize_info = letter_resize_img(img, dst_size)
        imgs_out[b] = normalization(img)
        resize_infos.append(resize_info)

        # 如果img的annotations不为空
        if len(ann_classes) > 0:
            boxes = letter_resize_bbox(ann_bboxes, resize_info)
            for i in range(len(ann_classes)):
                anns_out[b, i, :4] = torch.from_numpy(boxes[i])
                anns_out[b, i, 4] = float(ann_classes[i])
                anns_out[b, i, 5] = b

    return {'img': imgs_out, 'ann': anns_out, 'resize_info': resize_infos, 'img_id': img_ids}