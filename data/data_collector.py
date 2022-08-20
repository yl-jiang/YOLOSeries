from numpy import isin
import torch
from utils import letter_resize_bbox, letter_resize_img

__all__ = ["fixed_imgsize_collector", "FixSizeCollector"]

def normal_normalization(img):
    return torch.from_numpy(img / 255.0).permute(2, 0, 1).contiguous()


class FixSizeCollector:
    def __init__(self, dst_size) -> None:
        self.dst_size = dst_size
        if isinstance(dst_size, int):
            self.dst_size = [dst_size, dst_size]

    def __call__(self, data_in):
        """
        将Dataset中__getitem__方法返回的每个值进行进一步组装。

        :param data_in: tuple, data[0] is image, data[1] is annotation, data[2] is image's id
        :param dst_size:
        :return: dictionary
        """
        # 输入图像的格式为(h,w,3)
        assert data_in[0][0].ndim == 3 and data_in[0][0].shape[-1] == 3, f"data's formate should be (h, w, 3), but got {data_in[0].shape}"
        
        batch_size = len(data_in)
        imgs = [d[0] for d in data_in]
        anns = [d[1] for d in data_in]
        img_ids = [d[2] for d in data_in]

        # batch内image的图像拥有相同的shape, batch之间image的shape不一样
        # dst_size = padding(dst_size, 32)
        imgs_out = torch.zeros(batch_size, 3, self.dst_size[0], self.dst_size[1])
        boxes_num = [len(ann['bboxes']) for ann in anns]
        # 初始化为-1是为了区分有无object的bbox, 最后一个维度是为了标记一个batch中每个ann对应的img idx, for build_target()
        anns_out = torch.ones(batch_size, max(boxes_num), 6) * -1

        # resize_info在测试时恢复原始图像用
        resize_infos = []

        for b in range(batch_size):
            img = imgs[b]  # ndarray
            ann = anns[b]  # dict
            ann_bboxes = ann['bboxes']
            ann_classes = ann['classes']
            assert len(ann_bboxes) == len(ann_classes)
            img, resize_info = letter_resize_img(img, self.dst_size)
            imgs_out[b] = normal_normalization(img)
            resize_infos.append(resize_info)

            # 如果img的annotations不为空
            if len(ann_classes) > 0:
                boxes = letter_resize_bbox(ann_bboxes, resize_info)
                for i in range(len(ann_classes)):
                    anns_out[b, i, :4] = torch.from_numpy(boxes[i])
                    anns_out[b, i, 4] = float(ann_classes[i])
                    anns_out[b, i, 5] = b

        return {'img': imgs_out, 'ann': anns_out, 'resize_info': resize_infos, 'img_id': img_ids}

def fixed_imgsize_collector(data_in, dst_size):
    """
    将Dataset中__getitem__方法返回的每个值进行进一步组装。

    :param data_in: tuple, data[0] is image, data[1] is annotation, data[2] is image's id
    :param dst_size:
    :return: dictionary
    """
    # 输入图像的格式为(h,w,3)
    assert data_in[0][0].ndim == 3 and data_in[0][0].shape[-1] == 3, f"data's formate should be (h, w, 3), but got {data_in[0].shape}"
    
    batch_size = len(data_in)
    imgs = [d[0] for d in data_in]
    anns = [d[1] for d in data_in]
    img_ids = [d[2] for d in data_in]

    # batch内image的图像拥有相同的shape, batch之间image的shape不一样
    # dst_size = padding(dst_size, 32)
    imgs_out = torch.zeros(batch_size, 3, dst_size[0], dst_size[1])
    boxes_num = [len(ann['bboxes']) for ann in anns]
    # 初始化为-1是为了区分有无object的bbox, 最后一个维度是为了标记一个batch中每个ann对应的img idx, for build_target()
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
        imgs_out[b] = normal_normalization(img)
        resize_infos.append(resize_info)

        # 如果img的annotations不为空
        if len(ann_classes) > 0:
            boxes = letter_resize_bbox(ann_bboxes, resize_info)
            for i in range(len(ann_classes)):
                anns_out[b, i, :4] = torch.from_numpy(boxes[i])
                anns_out[b, i, 4] = float(ann_classes[i])
                anns_out[b, i, 5] = b

    return {'img': imgs_out, 'ann': anns_out, 'resize_info': resize_infos, 'img_id': img_ids}