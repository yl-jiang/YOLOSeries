import random
import warnings
from time import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .dataset_wrapper import *
from .base_generator import Generator
from utils import mosaic, RandomPerspective, valid_bbox, mixup
import os
from loguru import logger
import torchvision
from utils import letter_resize_img

NUM_THREADS = min(8, os.cpu_count())

__all__ = ["YOLODataset", "TestDataset"]


class YOLODataset(Dataset, Generator):

    def __init__(self, img_dir, lab_dir, name_path, input_dim, aug_hyp, cache_num=0, enable_data_aug=False, preproc=None) -> None:
        """
        Args:
            img_dir: 该文件夹下只存放图像文件
            lab_dir: 该文件夹下只存放label文件(.txt), 文件中的每一行存放一个bbox以及对应的class(例如: 0 134 256 448 560)
        """
        super().__init__(input_dimension=input_dim, enable_data_aug=enable_data_aug)
        
        self.img_dir = Path(img_dir)
        self.lab_dir = Path(lab_dir)
        print(f"Checking the consistency of dataset!")
        _start = time()
        if self.lab_dir is not None:
            self._check_dataset()

        print(f"- Use time {time() - _start:.3f}s")

        self.img_files = [_ for _ in self.img_dir.iterdir() if _.is_file()]

        print(f"Parser names!")
        _start = time()
        self.classes, self.labels, self.cls2lab, self.lab2cls = self.parse_names(name_path)
        print(f"- Use time {time() - _start:.3f}s")

        self.input_img_size = input_dim
        self.data_aug_param = aug_hyp
        self.preproc = preproc

        if isinstance(input_dim, (list, tuple)):
            self.input_img_size = np.array(input_dim)
        self.input_img_size = input_dim

        # 在内存中事先缓存一部分数据，方便快速读取（尽量榨干机器的全部性能），这个值根据具体的设备需要手动调整
        self.cache_num = min(cache_num, len(self)) if cache_num > 0 else 0  # 缓存到内存（RAM）中的数据量大小（正比于当前机器RAM的大小）
        self.imgs = None
        self._cache_image()
        

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
        assert Path(name_path).exists(), f"{name_path} is not exists!"
        classes, labels = [], []
        with open(name_path, 'r') as f:
            for line in f.readlines():
                contents = line.strip().split()
                classes.append(int(contents[0]))
                labels.append(" ".join(contents[1:])) # 有些label的名称包含多个单词
        cls2lab = dict([(c, l) for c, l in zip(classes, labels)])
        lab2cls = dict([(l, c) for c, l in zip(classes, labels)])
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

    def img_path(self, img_index):
        return self.img_files[img_index]

    def img_aspect_ratio(self, img_index):
        return self.aspect_ratio(img_index)

    def aspect_ratio(self, idx):
        """

        :param img_id:
        """
        img_arr = self.load_img(idx)
        h, w = img_arr.shape[0], img_arr.shape[1]
        ratio = w / h
        return ratio

    def __len__(self):
        return len(self.img_files)

    def _check_dataset(self):
        img_filenames = set([_.stem for _ in self.img_dir.iterdir()])
        lab_filenames = set([_.stem for _ in self.lab_dir.iterdir()])
        assert len(lab_filenames) == len(img_filenames), f"there are {len(lab_filenames)} label files, but found {len(img_filenames)} image files!"

        for p in self.lab_dir.iterdir():
            if p.suffix == ".txt" and p.is_file():
                img_filenames.add(p.stem)

        assert len(img_filenames) == len(lab_filenames)

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
                ann = np.loadtxt(str(lab_path), delimiter=' ', dtype=np.float32, skiprows=1)
        except Exception as err:  # label.txt文件不存在或为空
            ann = np.zeros(shape=[1, 5], dtype=np.float32)
        
        if len(ann) > 0:
            if ann.ndim == 1 and len(ann) == 5:
                ann = ann[None, :]
            assert ann.ndim == 2 and ann.shape[1] == 5, f"annotation's shape must same as (N, 5) that represent 'class, xmin, ymin, xmax, ymax' for each element, but got {ann.shape}\n {ann}"
            # 过滤掉一些不合格的bbox / [cls, xmin, ymin, xmax, ymax]
            whs = ann[:, [3, 4]] - ann[:, [1, 2]]
            mask = np.all(whs >= 1, axis=1)
            ann = ann[mask]
        else:
            ann = np.zeros(shape=[1, 5], dtype=np.float32)  # 不要使用np.empty()，empty()函数会生成随机数
        ann_out = {'classes': ann[:, 0], 'bboxes': ann[:, 1:]}

        return ann_out

    def mosaic(self, ix):
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
            img, ann = self.pull_item(i)
            bboxes.append(ann['bboxes'])
            labels.append(ann['classes'])
            imgs.append(img)

        img, bboxes, labels = mosaic(imgs, bboxes, labels,
                                     mosaic_shape=[_*2 for _ in self.input_img_size],
                                     fill_value=self.data_aug_param["data_aug_fill_value"], img_ids=indices)
        # img, bboxes, labels = RandomPerspective(img, bboxes, labels, self.data_aug_param["data_aug_perspective_p"],
        #                                         self.data_aug_param["data_aug_degree"],
        #                                         self.data_aug_param['data_aug_translate'],
        #                                         self.data_aug_param['data_aug_scale'],
        #                                         self.data_aug_param['data_aug_shear'],
        #                                         self.data_aug_param['data_aug_prespective'],
        #                                         self.input_img_size,
        #                                         self.data_aug_param["data_aug_fill_value"])
        return img, bboxes, labels

    def powerful_mixup(self, org_img, org_bboxes, org_classes):
        if random.random() < 0.5:
            img2, bboxes2, classes2 = self.mosaic(random.randint(0, len(self._dataset) - 1))
            img, bboxes, classes = mixup(org_img, org_bboxes, org_classes, img2, bboxes2, classes2)
        else:
            mixup_scale = [0.5, 1.5]
            jit_factor = random.uniform(*mixup_scale)
            FLIP = random.uniform(0, 1) > 0.5  # flip left and right
            classes2 = []
            while len(classes2) == 0:
                rnd_idx = random.randint(0, len(self) - 1)
                mixin_img, mixin_ann = self.pull_item(rnd_idx)
                mixin_bboxes = mixin_ann["bboxes"]
                classes2 = mixin_ann['classes']

            cp_img = np.ones((self.input_dim[0], self.input_dim[1], 3), dtype=np.uint8) * self.data_aug_param["data_aug_fill_value"]
            cp_scale_ratio = min(self.input_dim[0] / mixin_img.shape[0], self.input_dim[1] / mixin_img.shape[1])
            resized_img = cv2.resize(np.ascontiguousarray(mixin_img), 
                                     (int(mixin_img.shape[1] * cp_scale_ratio), int(mixin_img.shape[0] * cp_scale_ratio)), 
                                     interpolation=cv2.INTER_LINEAR)
            cp_img[:int(mixin_img.shape[0]*cp_scale_ratio), :int(mixin_img.shape[1]*cp_scale_ratio)] = resized_img
            cp_img = cv2.resize(np.ascontiguousarray(cp_img), 
                                (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)), 
                                interpolation=cv2.INTER_LINEAR)
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
        
    def _cache_image(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            f"{self.cache_num}({self.cache_num/len(self):3.2%}) images will be cached, there are {len(self)} images totaly.\n"
            "********************************************************************************\n"
        )
        max_h = self.input_dim[0]
        max_w = self.input_dim[1]
        
        cache_file = os.path.join(str(Path(self.img_dir).parent), f"img_{Path(self.img_dir).name}_resized_cache_h{max_h}_w{max_w}.array")
        
        if not os.path.exists(cache_file):
            logger.info(f"Caching images for the first time. This usually might take tens of minutes\ncache file: {cache_file}")
            self.imgs = np.memmap(
                cache_file,
                shape=(self.cache_num, max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(lambda i: self.load_resized_img(i), range(self.cache_num))
            pbar = tqdm(enumerate(loaded_images), total=self.cache_num)
            for k, out in pbar:
                self.imgs[k][:out.shape[0], :out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs ...")
        logger.info(f"cache_file: {cache_file}")
        self.imgs = np.memmap(
            cache_file,
            shape=(self.cache_num, max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_resized_img(self, i):
        img = self.load_img(i)
        r = min(self.input_dim[0] / img.shape[0], self.input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_resized_ann(self, i):
        assert 0 <= i < self.size(), f"img_index must be in [0, {self.size}), but got {i}"
        img_path = self.img_files[i]
        filename = img_path.stem
        lab_path = self.lab_dir / f"{filename}.txt"
        assert lab_path.exists(), f"img_path is {img_path} but the corrding lab_path {lab_path} is not exists!"

        with open(str(lab_path), 'r') as f:
            first_line = f.readlines()[0].strip()
            w, h = first_line.split(' ')

        ann = self.load_annotations(i)
        r = min(self.input_dim[0] / int(h), self.input_dim[1] / int(w))
        ann['bboxes'] = ann['bboxes'] * r
        return ann

    def pull_item(self, ix):
        # 如果ix没在(内存)缓存中
        if ix < self.cache_num and self.imgs is not None:
            img = self.imgs[ix]
            ann = self.load_resized_ann(ix)
        else:
            img = self.load_img(ix)
            ann = self.load_annotations(ix)
        
        return img, ann

    def get_img_path(self, ix):
        assert 0 <= ix < len(self), f"image index should in the range (0, {len(self)}), but got index {ix}"
        return self.img_files[ix]

    def img_path(self, img_index):
        return self.get_img_path(img_index)

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
                img = cv2.rectangle(img, pt1=lt, pt2=rb, color=[100, 149, 237], thickness=2)
                # text:显示的文本
                # org文本框左下角坐标（只接受元素为int的元组）
                # fontFace：字体类型
                # fontScale:字体大小（float）
                # thickness：int，值为-1时表示填充颜色
                font = cv2.FONT_HERSHEY_SIMPLEX
                caption = names[i]
                img = cv2.putText(img,
                                text=caption,
                                org=bl,
                                fontFace=font, fontScale=0.75,
                                color=[135, 206, 235],
                                thickness=1)
        cv2.imwrite(str(save_path), img[:, :, ::-1])

    @Dataset.aug_getitem
    def __getitem__(self, ix):
        """
        Args:
            ix: index
        Returns:
            img: 
            ann: {'bboxes': [[xmin, ymin, xmax, ymax], [...]], classes: [cls1, cls2, ...]}
            img_name: string
        """

        # traing or validation
        img, ann = self.pull_item(ix)
        bboxes, labels = ann['bboxes'], ann['classes']
        
        if self.enable_data_aug:
            if random.random() < self.data_aug_param['data_aug_mosaic_p']:
                img, bboxes, labels = self.mosaic(ix)
                if random.random() < self.data_aug_param['data_aug_mixup_p']:
                    img2, bboxes2, labels2 = self.mosaic(random.randint(0, len(self) - 1))
                    img, bboxes, labels = mixup(img, bboxes, labels, img2, bboxes2, labels2)
            
            if self.preproc is not None:
                img, bboxes, labels = self.preproc(img, bboxes, labels)

            ann.update({'classes': labels, 'bboxes': bboxes})

        if len(ann['classes']) > 0:
            valid_index = valid_bbox(ann['bboxes'])
            ann['bboxes'] = ann['bboxes'][valid_index]
            ann['classes'] = ann['classes'][valid_index]

        # 如果返回没有bbox的训练数据，会造成计算loss时在匹配target和prediction时出现问题，这里采用的应对策略是再resample一个训练数据，直到满足条件为止
        while np.sum(ann['bboxes']) == 0: 
            i = random.randint(0, len(self)-1)
            img, ann = self.pull_item(i)

        return img, ann, str(Path(self.get_img_path(ix)).stem)


# -------------------------------------------------------------------------------------------------------------------------

class TestDataset(Dataset):

    def __init__(self, img_dir, img_size):
        self.img_pathes = []
        for p in Path(img_dir).iterdir():
            if p.is_file() and p.suffix in [".png", '.jpg']:
                self.img_pathes.append(str(p))
        self.img_size = img_size
        self.num_class = 0
        self.class2label = ['lab' for _ in range(self.num_class)]

    def __len__(self):
        return len(self.img_pathes)

    def __iter__(self):
        self.count = 0
        return self

    @staticmethod
    def normalization(img):
        # 输入图像的格式为(h,w,3)
        assert len(img.shape) == 3 and img.shape[-1] == 3
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        return transforms(img)

    def __getitem__(self, item):
        img_bgr = cv2.imread(self.img_pathes[item])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized, letter_info = letter_resize_img(img_rgb, self.img_size)
        img_normed = self.normalization(img_resized)
        return img_normed, letter_info

