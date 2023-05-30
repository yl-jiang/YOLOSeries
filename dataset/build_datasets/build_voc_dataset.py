import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path
import shutil


VOC_LABEL_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

def mk_name_txt(names, save_path):
    with open(save_path, 'w') as f:
        for i, name in enumerate(names):
            f.write(f"{i} {name}\n")


def read_image(path, dtype=np.float32, color=True):
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
            
    except Exception as err:
        print(err)
        
    else:
        img = np.asarray(img, dtype=dtype)

    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:  # 灰度图片
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:   # 彩色图片
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def parse_single_xml(xml_file, use_diffcult=True):
    one_file_bboxes = []
    one_file_cls = []
    bs = BeautifulSoup(open(xml_file), 'lxml')
    img_file_name = bs.find('filename').string
    for obj in bs.find_all('object'):
        try:
            diffcult = int(obj.find('difficult').string)
            if diffcult == 1 and not use_diffcult:
                continue
        except AttributeError as err:
            print(img_file_name, err)
        
        name = obj.find('name').string
        if name in VOC_LABEL_NAMES:
            cls_id = VOC_LABEL_NAMES.index(name)
            bndbox_obj = obj.find('bndbox', recursive=False)
            ymax = float(bndbox_obj.find('ymax').string)
            xmax = float(bndbox_obj.find('xmax').string)
            ymin = float(bndbox_obj.find('ymin').string)
            xmin = float(bndbox_obj.find('xmin').string)
            one_file_bboxes.append([xmin, ymin, xmax, ymax])
            one_file_cls.append(cls_id)

    return img_file_name, one_file_bboxes, one_file_cls


def parse_xmls(voc_data_dir, img_save_dir, lab_save_dir):
    img_filepathes = [p for p in (Path(voc_data_dir) / 'JPEGImages').iterdir()]
    img_filenames = {p.stem: str(p) for p in img_filepathes}
    xml_filepathes = [str(p) for p in (Path(voc_data_dir) / "Annotations").iterdir()]
    for xml_path in tqdm(xml_filepathes, total=len(xml_filepathes)):
        file_name, bboxes, cls_ids = parse_single_xml(xml_path)
        file_name = file_name.split(".")[0]
        if not img_filenames.get(file_name, None):
            continue
        if len(cls_ids) == 0:
            continue

        src = img_filenames[file_name]
        dst = Path(img_save_dir) / f"{file_name}.jpg"
        shutil.copy(src, dst)

        f = open(str(Path(lab_save_dir) / f"{file_name}.txt"), 'w')
        for c, b in zip(cls_ids, bboxes):
            f.write(f"{c} {b[0]} {b[1]} {b[2]} {b[3]}\n")
        f.close()


if __name__ == '__main__':
    # test all data parser
    parse_xmls("xxx/Dataset/VOC/VOCtest2012/VOC2012/", "xxx/Dataset/VOC/val2012/image", "xxx/Dataset/VOC/val2012/label")