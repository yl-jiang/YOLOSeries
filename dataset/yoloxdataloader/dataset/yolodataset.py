import os
import sys
import random
import warnings
from time import time
from pathlib import Path

current_work_directionary = Path('__file__').parent.absolute()
sys.path.insert(0, str(current_work_directionary))

import cv2
import h5py
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from loguru import logger
import torch.backends.cudnn as cudnn
from multiprocessing.pool import ThreadPool

from .base_generator import Generator
from utils import maybe_mkdir, clear_dir
from utils import valid_bbox
from .datasets_wrapper import Dataset
NUM_THREADS = min(8, os.cpu_count())


def init_random_seed(seed=7):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class YOLODataset(Dataset, Generator):

    def __init__(self, img_dir, lab_dir, name_path, input_dim, cache_num=0, preproc=None) -> None:
        """
        Args:
            img_dir: 该文件夹下只存放图像文件
            lab_dir: 该文件夹下只存放label文件(.txt), 文件中的每一行存放一个bbox以及对应的class(例如: 0 134 256 448 560)
        """
        super().__init__(input_dimension=input_dim)
        
        self.img_dir = Path(img_dir)
        self.lab_dir = Path(lab_dir)
        self.preproc = preproc
        if self.lab_dir is not None:
            self._check_dataset()

        self.img_files = [_ for _ in self.img_dir.iterdir() if _.is_file()]
        self.classes, self.labels, self.cls2lab, self.lab2cls = self.parse_names(name_path)

        # 在内存中事先缓存一部分数据，方便快速读取（尽量榨干机器的全部性能），这个值根据具体的设备需要手动调整
        self.cache_num_in_ram = min(cache_num, len(self))  # 缓存到内存（RAM）中的数据量大小（正比于当前机器RAM的大小）
        self.h5_files = []
        self.cached_cls = [None] * self.cache_num_in_ram
        self.cached_box = [None] * self.cache_num_in_ram
        self.cached_img = [None] * self.cache_num_in_ram

        self.cache_dir = "./data/cache/"
        if self.cache_num_in_ram > 0:
            self.load()

    @classmethod
    def init(cls, img_dir, lab_dir):
        assert Path(img_dir).exists(), f"{img_dir} is not exists!"
        assert Path(lab_dir).exists(), f"{lab_dir} is not exists!"
        img_filenames = []
        for f in Path(img_dir).iterdir():
            if f.suffix in (".png", ".jpg", ".bmp"):
                img_filenames.append(f)
        lab_filenames = []
        for f in Path(lab_dir).iterdir():
            if f.suffix == ".txt":
                lab_filenames.append(f)
        assert len(img_filenames) == len(lab_filenames), f"found {len(img_filenames)} images but found {len(lab_filenames)} label files!"
        return cls(img_dir, lab_dir)

    def parse_names(self, name_path):
        print(f"Parser names.txt: {name_path}")
        _start = time()
        assert Path(name_path).exists(), f"{name_path} is not exists!"
        classes, labels = [], []
        with open(name_path, 'r') as f:
            for line in f.readlines():
                contents = line.strip().split()
                classes.append(int(contents[0]))
                labels.append(" ".join(contents[1:])) # 有些label的名称包含多个单词
        cls2lab = dict([(c, l) for c, l in zip(classes, labels)])
        lab2cls = dict([(l, c) for c, l in zip(classes, labels)])
        print(f"- Use time {time() - _start:.3f}s")
        return classes, labels, cls2lab, lab2cls

    def size(self):
        return len(self.img_files)

    @property
    def num_class(self):
        return len(self.classes)

    def has_label(self, label):
        return label in self.labels

    def has_class(self, c):
        return c in self.classes

    def has_name(self, name):
        return name in self.labels

    def label_to_name(self, label):
        return self.cls2lab[label]

    def name_to_label(self, name):
        return self.lab2cls[name]

    def aspect_ratio(self, idx):
        """

        :param img_id:
        """
        img_arr = self.load_img(idx)
        h, w = img_arr.shape[0], img_arr.shape[1]
        ratio = w / h
        return ratio

    def img_aspect_ratio(self, img_index):
        return self.aspect_ratio(img_index)

    def __len__(self):
        return len(self.img_files)

    def _check_dataset(self):
        print(f"Checking the consistency of dataset!")
        _start = time()
        img_filenames = set([_.stem for _ in self.img_dir.iterdir()])
        lab_filenames = set([_.stem for _ in self.lab_dir.iterdir()])
        assert len(lab_filenames) == len(img_filenames), f"there are {len(lab_filenames)} label files, but found {len(img_filenames)} image files!"

        for p in self.lab_dir.iterdir():
            if p.suffix == ".txt" and p.is_file():
                img_filenames.add(p.stem)

        assert len(img_filenames) == len(lab_filenames)
        print(f"- Use time {time() - _start:.3f}s")

    def load_img(self, img_index):
        assert 0 <= img_index < self.size(), f"img_index must be in [0, {self.size}), but got {img_index}"
        img_path = self.img_files[img_index]
        img = np.asarray(Image.open(img_path))
        if img.ndim == 2:  # gray image to rgb
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
        
    def load_annotations(self, img_index):
        """
        Args:
            img_index: image index

        Return:
            annonation array, formate is [cls, xmin, ymin, xmax, ymax] shape is (N, 5)

        formate saved in label txt is:
            class1 xmin ymin xmax ymax
            class2 xmin ymin xmax ymax
            ... ...
        """
        assert 0 <= img_index < self.size(), f"img_index must be in [0, {self.size}), but got {img_index}"
        img_path = self.img_files[img_index]
        filename = img_path.stem
        lab_path = self.lab_dir / f"{filename}.txt"
        assert lab_path.exists(), f"img_path is {img_path} but the corrding lab_path {lab_path} is not exists!"
        try:
            with warnings.catch_warnings():
                # 忽略label.txt文件存在但存储内容为空时，numpy抛出的userwarnning
                warnings.simplefilter("ignore")
                ann = np.loadtxt(str(lab_path), delimiter=' ', dtype=np.float32)
        except Exception as err:  # label.txt文件不存在或为空
            ann = np.zeros(shape=[1, 5], dtype=np.float32)
        
        if len(ann) > 0:
            if ann.ndim == 1 and len(ann) == 5:
                ann = ann[None, :]
            assert ann.ndim == 2 and ann.shape[1] == 5, f"annotation's shape must same as (N, 5) that represent 'class, xmin, ymin, xmax, ymax' for each element, but got {ann.shape}\n {ann}"
            # 过滤掉一些不合格的bbox
            whs = ann[:, [3, 4]] - ann[:, [1, 2]]
            mask = np.all(whs >= 1, axis=1)
            ann = ann[mask]
        else:
            ann = np.zeros(shape=[1, 5], dtype=np.float32)  # 不要使用np.empty()，empty()函数会生成随机数
        ann_out = {'classes': ann[:, 0], 'bboxes': ann[:, 1:]}

        return ann_out

    def save_cache(self, cache_dir):
        """
        将数据集的每张image以及对应的annotations使用h5py保存为一个.h5文件
        """
        threads = ThreadPool(NUM_THREADS).imap(lambda ix: (self.load_img(ix), self.load_annotations(ix)), range(len(self)))
        # threads = (super(COCODataset, self).load_img_and_ann(i) for i in range(len(self)))
        start = time()
        tbar = tqdm(threads, total=len(self), ncols=100, file=sys.stdout, desc="caching dataset ... ... ")
        for i, (img, ann) in enumerate(tbar):
            img_path = self.get_img_path(i)
            img_name = Path(img_path).stem
            cache_file_path = Path(cache_dir / f"{img_name}.cache.h5")
            if not cache_file_path.exists():
                self.h5_files.append(str(img_path))
                with h5py.File(str(cache_file_path), "w") as f:
                    grp = f.create_group(img_name, track_order=True)
                    grp['img'] = img
                    grp['imgsz'] = img.shape
                    grp['img_path'] = str(img_path)
                    grp['classes'] = ann['classes']
                    grp['bboxes'] = ann['bboxes']
                    if i < self.cache_num_in_ram:
                        self.cached_img[i] = img
                        self.cached_cls[i] = ann['classes']
                        self.cached_box[i] = ann['bboxes']
                    f.close()
            tbar.set_description(f"cache each image use time: {(time() - start) / (i+1):.3f}s")

    def load_cache(self, cache_filenames):
        """
        读取已经保存的h5文件名，并缓存指定数量的数据到内存中(使用多进程)
        """
        
        imgnames = []
        for i, p in enumerate(cache_filenames):
            if p.suffix == ".h5":
                img_name = p.stem.split(".")[0]
                self.h5_files.append(p)
                imgnames.append(img_name)

        if self.cache_num_in_ram > NUM_THREADS * float('inf'):  # 实测使用多进程会降低文件读取速度，故这里不使用多进程
            pool = ThreadPool(NUM_THREADS).imap(lambda x: self.read_h5(*x), zip(self.h5_files[:self.cache_num_in_ram], imgnames[:self.cache_num_in_ram]))
        else:
            pool = (self.read_h5(*x) for x in zip(self.h5_files[:self.cache_num_in_ram], imgnames[:self.cache_num_in_ram]))

        tbar = tqdm(pool, total=self.cache_num_in_ram, file=sys.stdout, desc="caching data into memory ... ... ", ncols=100)
        for i, x in enumerate(tbar):
            self.cached_img[i] = x[0]
            self.cached_cls[i] = x[1]
            self.cached_box[i] = x[2]

    @staticmethod
    def read_h5(h5_file_path, img_name):
        with h5py.File(str(h5_file_path), "r") as f:
            img = f[img_name]['img'][()]
            cls = f[img_name]['classes'][()]
            box = f[img_name]['bboxes'][()]
            return img, cls, box

    def load(self):
        cache_dir = Path(self.cache_dir) 
        if cache_dir.exists():
            tot_files = [p for p in Path(cache_dir).iterdir() if p.is_file() and p.suffix == ".h5"]
        else:
            maybe_mkdir(cache_dir)
            tot_files = []

        if len(tot_files) != len(self):
            clear_dir(cache_dir)
            self.save_cache(cache_dir)
        else:
            self.load_cache(tot_files)

    def load_img_and_ann(self, ix):
        # 如果ix没在(内存)缓存中
        if ix < self.cache_num_in_ram:
            img = self.cached_img[ix]
            cls = self.cached_cls[ix]
            box = self.cached_box[ix]
            ann = {"classes": cls, "bboxes": box}
        else:
            # ix是否在h5文件中
            if len(self.h5_files) > ix:
                with h5py.File(self.h5_files[ix], 'r') as f:
                    img_name = Path(self.h5_files[ix]).stem.split('.')[0]
                    img = f[img_name]['img'][()]
                    cls = f[img_name]['classes'][()]
                    box = f[img_name]['bboxes'][()]
                    ann = {"classes": cls, "bboxes": box}
            else:
                img = self.load_img(ix)
                ann = self.load_annotations(ix)
        
        return img, ann

    def img_path(self, img_index):
        return self.get_img_path(img_index)

    def get_img_path(self, ix):
        assert 0 <= ix < len(self), f"image index should in the range (0, {len(self)}), but got index {ix}"
        return self.img_files[ix]

    def cv2_save_fig(self, img, bboxes, classes, save_path):
        assert isinstance(img, np.ndarray)
        assert len(bboxes) == len(classes)
        names = [self.cls2lab[c] for c in classes]

        if not Path(save_path).parent.exists():
            Path(save_path).parent.mkdir(parents=True)

        if len(bboxes) > 0:
            for i, box in enumerate(bboxes):
                # pt1:左上角坐标[xmin, ymin] ; pt2:右下角坐标[xmax, ymax]
                lt = (int(round(box[0])), int(round(box[1])))
                rb = (int(round(box[2])), int(round(box[3])))
                bl = (int(round(box[0])), int(round(box[3])))
                # cv2.rectangle() parameters:
                # img: image array
                # pt1: 左上角
                # pt2: 右下角
                # color: color
                # thickness: 表示矩形边框的厚度，如果为负值，如 CV_FILLED，则表示填充整个矩形
                img = cv2.rectangle(img, pt1=lt, pt2=rb, color=[200, 0, 0], thickness=1)
                # text:显示的文本
                # org文本框左下角坐标（只接受元素为int的元组）
                # fontFace：字体类型
                # fontScale:字体大小（float）
                # thickness：int，值为-1时表示填充颜色
                font = cv2.FONT_HERSHEY_SIMPLEX
                caption = names[i]
                img = cv2.putText(img,
                                  text=caption,
                                  org=lt,
                                  fontFace=font, fontScale=0.35,
                                  color=[200, 0, 0],
                                  thickness=1, 
                                  lineType=cv2.LINE_AA)
        cv2.imwrite(str(save_path), np.ascontiguousarray(img[:, :, ::-1]))

    @logger.catch
    @Dataset.mosaic_getitem
    def __getitem__(self, ix):
        """
        Args:
            ix: index
        Returns:
            img: 
            ann: {'bboxes': [[xmin, ymin, xmax, ymax], [...]], classes: [cls1, cls2, ...]}
            img_name: string
        """

        img, ann = self.load_img_and_ann(ix)

        if len(ann['classes']) > 0:
            valid_index = valid_bbox(ann['bboxes'])
            ann['bboxes'] = ann['bboxes'][valid_index]
            ann['classes'] = ann['classes'][valid_index]

        # 如果返回没有bbox的训练数据，会造成计算loss时在匹配target和prediction时出现问题，这里采用的应对策略是再resample一个训练数据，直到满足条件为止
        while np.sum(ann['bboxes']) == 0: 
            i = random.randint(0, len(self)-1)
            img, ann = self.load_img_and_ann(i)
        
        if self.preproc is not None:
            img, bboxes, classes = self.preproc(img, ann['bboxes'], ann['classes'])
            ann = {'bboxes': bboxes, 'classes': classes}

        return img, ann, str(Path(self.get_img_path(ix)).stem)
        