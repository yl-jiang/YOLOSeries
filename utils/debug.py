import cv2
from pathlib import Path
import numpy as np
import time
from datetime import datetime

__all__ = ['cv2_save_fig', 'cv2_save_figs']

def cv2_save_fig(img, box, cla, name, save_path):
    assert isinstance(img, np.ndarray)
    assert len(box) == len(cla)
    # names = [self.cls2lab[c] for c in classes]

    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True)

    if len(box) > 0:
        for i, box in enumerate(box):
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
            img = cv2.rectangle(np.ascontiguousarray(img), pt1=lt, pt2=rb, color=[100, 149, 237], thickness=2, lineType=cv2.LINE_AA)
            # text:显示的文本
            # org文本框左下角坐标（只接受元素为int的元组）
            # fontFace：字体类型
            # fontScale:字体大小（float）
            # thickness：int，值为-1时表示填充颜色
            font = cv2.FONT_HERSHEY_SIMPLEX
            caption = name[i]
            img = cv2.putText(img,
                            text=caption,
                            org=bl,
                            fontFace=font, fontScale=0.75,
                            color=[135, 206, 235],
                            thickness=1, 
                            lineType=cv2.LINE_AA)
    cv2.imwrite(str(save_path), img[:, :, ::-1])


def cv2_save_figs(data, save_dir):
    imgs = data['img'].cpu().numpy()  # (b, 3, h, w)
    anns = data['ann'].cpu().numpy()  # (b, M, 6) / [xmin, ymin, xmax, ymax, cls, batch_id]
    classes = anns[:, :, 4]  # (b, M)
    bboxes  = anns[:, :, :4]  # (b, M, 4)

    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)

    for i, (img, box, cla) in enumerate(zip(imgs, bboxes, classes)):
        save_path = Path(save_dir) / f"{i}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}.png"
        img = np.clip(img.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        msk = np.squeeze(cla >= 0)  # (M,)
        if msk.sum() == 0:
            continue
        cla = np.squeeze(cla[msk])  # (x,)
        name = [str(int(c)) for c in cla]  # (x,)
        box = box[msk]  # (x, 4)
        cv2_save_fig(img, box, cla, name, save_path)
