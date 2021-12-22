import pickle
import sys
from pathlib import Path
current_work_directionary = Path('__file__').parent.absolute()
sys.path.insert(0, str(current_work_directionary))

from dataset import Generator
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import numpy as np
import random
from utils import RandomHSV, RandomFlipLR, RandomFlipUD
from utils import maybe_mkdir, clear_dir
from utils import mosaic, random_perspective, valid_bbox, mixup, cutout
import torch
import torch.backends.cudnn as cudnn
from time import time
from tqdm import tqdm
import h5py
from multiprocessing.pool import ThreadPool
import os
from functools import partial
from utils import fixed_imgsize_collector, AspectRatioBatchSampler
import warnings
NUM_THREADS = min(8, os.cpu_count())

def init_random_seed(seed=7):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class YoloDataset(Dataset, Generator):

    def __init__(self, img_dir, lab_dir, name_path, input_img_size, aug_hyp, cache_num=0) -> None:
        """
        Args:
            img_dir: 该文件夹下只存放图像文件
            lab_dir: 该文件夹下只存放label文件（.txt），文件中的每一行存放一个bbox以及对应的class（例如：0 134 256 448 560）
        """
        super().__init__()
        
        self.img_dir = Path(img_dir)
        self.lab_dir = Path(lab_dir) if lab_dir is not None else lab_dir
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

        self.input_img_size = input_img_size
        self.is_training = aug_hyp is not None
        if self.is_training:  # training
            self.data_aug_param = {
                "scale": aug_hyp['data_aug_scale'],
                "translate": aug_hyp['data_aug_translate'],
                "degree": aug_hyp['data_aug_degree'],
                "shear": aug_hyp['data_aug_shear'],
                "presepctive": aug_hyp['data_aug_prespective'],
                "mixup": aug_hyp['data_aug_mixup_p'],
                "hsv": aug_hyp['data_aug_hsv_p'],
                "hgain": aug_hyp['data_aug_hsv_hgain'],
                "sgain": aug_hyp['data_aug_hsv_sgain'],
                "vgain": aug_hyp['data_aug_hsv_vgain'],
                "fliplr": aug_hyp['data_aug_fliplr_p'],
                "flipud": aug_hyp['data_aug_flipud_p'],
                "mosaic": aug_hyp['data_aug_mosaic_p'], 
                "cutout": aug_hyp['data_aug_cutout_p'], 
                "cutout_iou_thr": aug_hyp['data_aug_cutout_iou_thr'], 
                }
            self.fill_value = aug_hyp['data_aug_fill_value']

        if isinstance(input_img_size, (list, tuple)):
            self.input_img_size = np.array(input_img_size)
        self.input_img_size = input_img_size

        # 在内存中事先缓存一部分数据，方便快速读取（尽量榨干机器的全部性能），这个值根据具体的设备需要手动调整
        self.cache_num_in_ram = min(cache_num, len(self))  # 缓存到内存（RAM）中的数据量大小（正比于当前机器RAM的大小）
        self.h5_files = []
        self.cached_cls = [None] * self.cache_num_in_ram
        self.cached_box = [None] * self.cache_num_in_ram
        self.cached_img = [None] * self.cache_num_in_ram

        self.cache_dir = "./dataset/cache/"
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
                ann = np.loadtxt(str(lab_path), delimiter=' ', dtype=np.float32)
        except Exception as err:  # label.txt文件不存在或为空
            ann = np.zeros(shape=[1, 5], dtype=np.float32)
        
        if len(ann) > 0:
            if ann.ndim == 1 and len(ann) == 5:
                ann = ann[None, :]
            assert ann.ndim == 2 and ann.shape[1] == 5, f"annotation's shape must same as (N, 5) that represent 'class, xmin, ymin, xmax, ymax' for each element, but got {ann.shape}\n {ann}"
            # 过滤掉一些不合格的bbox
            whs = ann[:, [2, 3]] - ann[:, [0, 1]]
            mask = np.all(whs >= 1, axis=1)
            ann = ann[mask]
        else:
            ann = np.zeros(shape=[1, 5], dtype=np.float32)  # 不要使用np.empty()，empty()函数会生成随机数
        ann_out = {'classes': ann[:, 0], 'bboxes': ann[:, 1:]}

        return ann_out

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
            img, ann = self.load_img_and_ann(i)
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

    def __getitem__(self, ix):
        """

        :param ix:
        :return: [xmin, ymin, xmax, ymax]
        """
        # only detection image
        if self.lab_dir is None: 
            # 单纯的的测试模式是不需要label的，这里创建一个临时的label是为了复用fixed_imgsize_collector函数
            dummy_ann = {'bboxes': np.zeros(shape=(1,4)), "classes": np.zeros((1))} 
            return self.load_img(ix), dummy_ann, str(0)

        # traing or validation
        img, ann = self.load_img_and_ann(ix)
        bboxes, labels = ann['bboxes'], ann['classes']
        
        if self.is_training:
            if random.random() < self.data_aug_param.get('mosaic', 0.0):
                img, bboxes, labels = self.load_mosaic(ix)
            if random.random() < self.data_aug_param.get('mixup', 0.0):
                img2, bboxes2, labels2 = self.load_mosaic(random.randint(0, len(self) - 1))
                img, bboxes, labels = mixup(img, bboxes, labels, img2, bboxes2, labels2)
            if random.random() < self.data_aug_param.get('cutout', 0.0):
                img, bboxes, labels = cutout(img, bboxes, labels, cutout_iou_thr=self.data_aug_param['cutout_iou_thr'])

            img = RandomHSV(img, self.data_aug_param['hsv'], self.data_aug_param['hgain'],
                            self.data_aug_param['sgain'], self.data_aug_param['vgain'])
            img, bboxes = RandomFlipLR(img, bboxes, self.data_aug_param['fliplr'])
            img, bboxes = RandomFlipUD(img, bboxes, self.data_aug_param['flipud'])
            ann.update({'classes': labels, 'bboxes': bboxes})

        if len(ann['classes']) > 0:
            valid_index = valid_bbox(ann['bboxes'])
            ann['bboxes'] = ann['bboxes'][valid_index]
            ann['classes'] = ann['classes'][valid_index]

        # 如果返回没有bbox的训练数据，会造成计算loss时在匹配target和prediction时出现问题，这里采用的应对策略是再resample一个训练数据，直到满足条件为止
        while np.sum(ann['bboxes']) == 0: 
            i = random.randint(0, len(self)-1)
            img, ann = self.load_img_and_ann(i)

        return img, ann, str(Path(self.get_img_path(ix)).stem)
        

def YoloDataloader(hyp, is_training=True):
    """
    pytorch dataloader for cocodataset.
    :param kwargs:
    :return:
    """
    collector_fn = partial(fixed_imgsize_collector, dst_size=hyp['input_img_size'])

    if is_training:
        coco_dataset_kwargs = {
            'data_aug_scale': hyp['data_aug_scale'],
            'data_aug_shear': hyp['data_aug_shear'],
            'data_aug_translate': hyp['data_aug_translate'],
            'data_aug_degree': hyp['data_aug_degree'],
            'data_aug_prespective': hyp['data_aug_prespective'],
            'data_aug_hsv_p': hyp['data_aug_hsv_p'],
            "data_aug_hsv_hgain": hyp['data_aug_hsv_hgain'],
            "data_aug_hsv_sgain": hyp['data_aug_hsv_sgain'],
            "data_aug_hsv_vgain": hyp['data_aug_hsv_vgain'],
            'data_aug_mixup_p': hyp['data_aug_mixup_p'],
            'data_aug_fliplr_p': hyp['data_aug_fliplr_p'],
            'data_aug_flipud_p': hyp['data_aug_flipud_p'],
            'data_aug_fill_value': hyp['data_aug_fill_value'],
            "data_aug_mosaic_p": hyp['data_aug_mosaic_p'],
            "data_aug_cutout_p": hyp['data_aug_cutout_p'], 
            "data_aug_cutout_iou_thr": hyp['data_aug_cutout_iou_thr'], 
            }
        assert Path(hyp['train_img_dir']).exists() and Path(hyp['train_img_dir']).is_dir()
        assert Path(hyp['train_lab_dir']).exists() and Path(hyp['train_lab_dir']).is_dir()
        dataset = YoloDataset(hyp['train_img_dir'], hyp['train_lab_dir'], hyp['name_path'], hyp['input_img_size'], coco_dataset_kwargs, hyp['cache_num'])
        if hyp.get('aspect_ratio', None):  # 是否采用按照数据集中图片长宽比从小到大的顺序sample数据
            print(f"Build Aspect Ratio BatchSampler!")
            _start = time()
            ar = None
            if hyp.get('aspect_ratio_path', None) is not None and Path(hyp['aspect_ratio_path']).exists():
                ar = pickle.load(open(hyp['aspect_ratio_path'], 'rb'))
            sampler = AspectRatioBatchSampler(dataset, hyp['batch_size'], hyp['drop_last'], aspect_ratio_list=ar)
            print(f"- Use time {time() - _start:.3f}s")
        else:
            sampler = None

        # 使用自定义的batch_sampler时，不要给DataLoader中的batch_size,shuffle赋值，因为这些参数会在自定义的batch_sampler中已经定义了
        dataloader = DataLoader(dataset,
                                batch_sampler=sampler,
                                collate_fn=collector_fn,
                                num_workers=hyp['num_workers'],
                                pin_memory=hyp['pin_memory'], 
                                batch_size=hyp['batch_size'] if sampler is None else None)

    elif not is_training and hyp.get('val_lab_dir', None) is not None:  # validation for compute mAP
        assert Path(hyp['val_img_dir']).exists() and Path(hyp['val_img_dir']).is_dir()
        assert Path(hyp['val_lab_dir']).exists() and Path(hyp['val_lab_dir']).is_dir()
        dataset = YoloDataset(hyp['val_img_dir'], hyp['val_lab_dir'], hyp['name_path'], hyp['input_img_size'], None)
        dataloader = DataLoader(dataset, batch_size=hyp['batch_size'], 
                                shuffle=False, drop_last=False, 
                                num_workers=hyp['num_workers'], 
                                pin_memory=hyp['pin_memory'], 
                                collate_fn=collector_fn)
    else:  # just detection
        assert Path(hyp['test_img_dir']).exists() and Path(hyp['test_img_dir']).is_dir()
        dataset = YoloDataset(hyp['test_img_dir'], None, hyp['name_path'], hyp['input_img_size'], None)
        dataloader = DataLoader(dataset, batch_size=hyp['batch_size'],
                                shuffle=False, drop_last=False,
                                num_workers=hyp['num_workers'],
                                pin_memory=hyp['pin_memory'],
                                collate_fn=collector_fn)
    
    
    return dataloader, dataset


def test():
    init_random_seed()
    aug_hyp = {
              'data_aug_scale': 0.,
              'data_aug_shear': 0,
              'data_aug_translate': 0.,
              'data_aug_degree': 0,
              'data_aug_prespective': False,
              'data_aug_hsv_p': 1,
              "data_aug_hsv_hgain": 0.015,
              "data_aug_hsv_sgain": 0.7,
              "data_aug_hsv_vgain": 0.4,
              'data_aug_mixup_p': 0.0,
              'data_aug_fliplr_p': 0,
              'data_aug_flipud_p': 0,
              'data_aug_fill_value': 128,
              'data_aug_mosaic_p': 1., 
              "data_aug_cutout_p": 1.0, 
              "data_aug_cutout_iou_thr": 0.3, 
    }

    dataset = YoloDataset('/home/uih/JYL/Dataset/COCO2017/train/image/',
                          "/home/uih/JYL/Dataset/COCO2017/train/label/",
                          '/home/uih/JYL/Dataset/COCO2017/train/names.txt',
                          [448, 448], aug_hyp, 0)
    collector = partial(fixed_imgsize_collector, dst_size=[448, 448])
    batch_size = 5
    print(f"Build Aspect Ratio BatchSampler!")
    _start = time()

    ar_list = pickle.load(open("/home/uih/JYL/Programs/YOLO/dataset/pkl/coco_aspect_ratio.pkl", 'rb'))
    sampler = AspectRatioBatchSampler(dataset, batch_size, True, aspect_ratio_list=ar_list)
    print(f"- Use time {time() - _start:.3f}s")
    loader = DataLoader(dataset, collate_fn=collector, batch_sampler=sampler)
    with tqdm(total=len(loader), ncols=50) as t:
        for b, x in enumerate(loader):
            for i in range(batch_size):
                ann = x['ann'][i]
                title = x['img_id'][i]
                img = x['img'][i]
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
                save_path = current_work_directionary / "result" / "tmp" / f"org_{b*batch_size+i}.png"
                dataset.cv2_save_fig(img_mdy, ann_mdy['bboxes'], ann_mdy['classes'], str(save_path))
                # print(f"{b*batch_size+i}\t{len(ann_mdy['bboxes'])}")
            if b > 10:
                break
        t.update(batch_size)


if __name__ == '__main__':
    test()
