import numpy as np
from pycocotools.coco import COCO
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from abc import ABC
from pathlib import Path
from pycocotools.coco import COCO


class COCOGenerator(ABC):

    def __init__(self, data_dir, setname, use_crowd):
        self.setname = setname
        self.data_dir = Path(data_dir)
        self.use_crowd = use_crowd
        self.annpath = self.data_dir / 'annotations' / f'instances_{setname}.json'
        self.coco = COCO(self.annpath)
        self.img_ids = self.coco.getImgIds()
        self.classes, self.class2id, self.id2class, self.class2label, self.label2class = [], {}, {}, {}, {}
        self.get_coco_info()
        self.num_class = len(self.classes)
        super(COCOGenerator, self).__init__()

    def get_coco_info(self):
        """
        pass
        """
        category_info = self.coco.loadCats(ids=self.coco.getCatIds())
        category_info.sort(key=lambda x: int(x['id']))
        for category in category_info:
            # self.classes.append(category['id'])
            self.class2id[len(self.classes)] = category['id']
            self.id2class[category['id']] = len(self.classes)
            self.class2label[len(self.classes)] = category['name']
            self.label2class[category['name']] = len(self.classes)
            self.classes.append(category['id'])

    def __len__(self):
        return len(self.img_ids)

    def size(self):
        """
        Size of COCO dataset.
        :return:
        """
        return len(self.img_ids)

    def has_label(self, label):
        return label in self.class2label

    def has_name(self, name):
        return name in self.label2class

    def label_to_name(self, label):
        return self.class2label[label]

    def name_to_label(self, name):
        return self.label2class[name]

    def coco_id_to_class(self, coco_id):
        return self.id2class[coco_id]

    def coco_id_to_label(self, coco_id):
        return self.class2label[self.coco_id_to_class(coco_id)]

    def get_img_path(self, idx):
        return str(self.data_dir / self.setname / f'{self.img_ids[idx]:>012}.jpg')

    def load_annotations(self, idx):
        """

        :param idx:
        :return: return box formate -> [xmin, ymin, xmax, ymax]
        """
        ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[idx], iscrowd=self.use_crowd)
        img_info = self.coco.imgs[self.img_ids[idx]]
        annotations = {'classes': np.empty((0,), dtype=np.uint16), 'bboxes': np.empty((0, 4), dtype=np.float32)}
        # 有些img没有annotations
        if len(ann_ids) == 0:
            return annotations
        else:
            # anns is a list
            anns = self.coco.loadAnns(ids=ann_ids)
            for idx, ann in enumerate(anns):
                # 有些annotations中bbox的width/height值小于1，遇到这样的bbox就舍弃掉
                if (ann['bbox'][2]) < 1 or (ann['bbox'][3] < 1):
                    continue
                else:
                    annotations['classes'] = np.concatenate([annotations['classes'],
                                                             [self.coco_id_to_class(ann['category_id'])]],
                                                            axis=0)
                    annotations['bboxes'] = np.concatenate([annotations['bboxes'],
                                                            [[ann['bbox'][0],
                                                              ann['bbox'][1],
                                                              ann['bbox'][0] + ann['bbox'][2],
                                                              ann['bbox'][1] + ann['bbox'][3]]]],
                                                           axis=0)
        return annotations


if __name__ == "__main__":
    import shutil

    coco = COCOGenerator("/home/uih/JYL/Dataset/COCO2017/", "train2017", False)
    for i in tqdm(range(len(coco)), total=len(coco)):
        ann = coco.load_annotations(i)
        img_id = coco.img_ids[i]
        box = ann['bboxes']
        cls_ = ann['classes']
        img_src = coco.get_img_path(i)
        assert Path(img_src).exists()
        img_dst = Path("/home/uih/JYL/Dataset/COCO2017/image/") / f"{img_id}.jpg"
        shutil.copy(str(img_src), str(img_dst))
        with open("/home/uih/JYL/Dataset/COCO2017/label/" + f"{img_id}.txt", 'w') as f:
            for c, b in zip(cls_, box):
                f.write(f"{c} {b[0]} {b[1]} {b[2]} {b[3]}\n")

    with open("/home/uih/JYL/Dataset/COCO2017/train/names.txt", 'w') as f:
        for k, v in coco.class2label.items():
            f.write(f"{k} {v}\n")