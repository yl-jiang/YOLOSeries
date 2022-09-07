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
base_dir = Path("/Volumes/Samsung/Dataset/")
img_dir = base_dir / "COCO/train_dataset/image/"
lab_dir = base_dir / "COCO/train_dataset/label/"
name_path = base_dir / "COCO/train_dataset/names.txt"
input_size = [448, 448]
cache_num = 0
preproc = Transform(cutout_p=1)
dataset = YOLODataset(img_dir, lab_dir, name_path, input_size, cache_num, preproc)
# for i in trange(len(dataset)):
#     img, ann, filename = dataset[i]
#     bboxes, classes = ann["bboxes"], ann["classes"]
#     dataset.cv2_save_fig(img, bboxes, classes, f"/Volumes/Samsung/Dataset/Tmp/{filename}.png")

mosaic = MosaicTransformDataset(dataset, input_size, True)
# for i in trange(len(mosaic)):
#     img, ann, filename = mosaic[i]
#     bboxes, classes = ann["bboxes"], ann["classes"]
#     dataset.cv2_save_fig(img, bboxes, classes, f"{str(base_dir)}/Mosaic/{filename}.png")

sampler = InfiniteSampler(len(mosaic), 0)
sampler = InfiniteAspectRatioBatchSampler(mosaic, False, "./data/aspect_ratio_pkl/coco_aspect_ratio.pkl")
batch_sampler = YOLOBatchSampler(sampler=sampler, batch_size=10, drop_last=False, do_mosaic=True)
dataloader_kwargs = {"batch_sampler": batch_sampler, 
                     "worker_init_fn": worker_init_reset_seed, 
                     "collate_fn": FixSizeCollector(input_size), 
                     "num_workers": 0, 
                     "pin_memory":True, 
                     }
dataloader = YOLODataLoader(mosaic, **dataloader_kwargs)
prefector = DataPrefetcher(dataloader)
loader = prefector.next()
# for loader in dataloader:
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
# break