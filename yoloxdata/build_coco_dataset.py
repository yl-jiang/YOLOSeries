import sys
sys.path.insert(0, "/home/uih/JYL/Projects/Others/maketest/YOLOSeries/")

from yoloxdata.dataset import YOLODataset
from yoloxdata import Transform, InfiniteSampler, YOLOBatchSampler, YOLODataLoader, worker_init_reset_seed
from yoloxdata.dataset import MosaicTransformDataset
from yoloxdata import FixSizeCollector
from yoloxdata import InfiniteAspectRatioBatchSampler
import numpy as np
import torch
from yoloxdata import DataPrefetcher
from tqdm import trange
from pathlib import Path


def build_dataset(img_dir, lab_dir, name_path, input_size, batch_size, aspect_ratio_filepath, cache_num=0):
    dataset = YOLODataset(img_dir, lab_dir, name_path, input_size, cache_num, preproc=Transform())
    mosaic = MosaicTransformDataset(dataset, input_size, True)
    # sampler = InfiniteSampler(len(mosaic), 0)
    sampler = InfiniteAspectRatioBatchSampler(mosaic, False, aspect_ratio_filepath)
    batch_sampler = YOLOBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False, do_mosaic=True)
    dataloader_kwargs = {"batch_sampler": batch_sampler, 
                        "worker_init_fn": worker_init_reset_seed, 
                        "collate_fn": FixSizeCollector(input_size), 
                        "num_workers": 6, 
                        "pin_memory":True, 
                        }
    dataloader = YOLODataLoader(mosaic, **dataloader_kwargs)
    prefector = DataPrefetcher(dataloader)
    return mosaic, dataloader, prefector


if __name__ == "__main__":

    base_dir = Path("xxx/COCO2017/train/")
    img_dir = base_dir / "image"
    lab_dir = base_dir / "label"
    name_path = base_dir / "names.txt"
    input_size = [640, 640]
    cache_num = 0
    aspect_ratio_filepath = "./yoloxdata/aspect_ratio_pkl/coco_aspect_ratio.pkl"
    batch_size = 15
    dataset, dataloader, prefetcher = build_dataset(img_dir, lab_dir, name_path, input_size, batch_size, aspect_ratio_filepath, cache_num)
    loader = prefetcher.next()
    imgs = loader["img"].detach().cpu()
    anns = loader["ann"].detach().cpu()
    imgids = loader["img_id"]

    for img, ann, imgid in zip(imgs, anns, imgids):
        img = img.permute(1, 2, 0)
        img = np.clip(img * 255.0, 0.0, 255.0)
        img_mdy = np.ascontiguousarray(img.numpy().astype('uint8'))
        h, w, _ = img.shape
        # 该笔数据中是否有object，ann[:, 4] == -1表示没有object
        valid_index = torch.nonzero(ann[:, 4] >= 0, as_tuple=False).squeeze(dim=1)
        # 如果该笔数据有object的话，就plot出来
        if valid_index.numel() > 0:
            ann_mdy = {'bboxes': ann[valid_index][:, :4].numpy(),
                    'classes': ann[valid_index][:, 4].numpy().astype('uint8')}
        # 如果该笔数据中没有发现object，则打印出图片的路径
        else:
            ann_mdy = {'bboxes': [], 'classes': []}

        dataset.cv2_save_fig(img_mdy, ann_mdy['bboxes'], ann_mdy['classes'], f"./test_{imgid}.png")