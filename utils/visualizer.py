import cv2
import torch
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gc
import pickle

VOC_BBOX_LABEL_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat',
                        'chair', 'cow', 'diningtable', 'dog',
                        'horse', 'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']
COCO_BBOX_LABEL_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                         'bus', 'train', 'truck', 'boat', 'traffic light',
                         'fire hydrant', 'stop sign', 'parking meter', 'bench',
                         'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                         'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                         'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                         'sports ball', 'kite', 'baseball bat', 'baseball glove',
                         'skateboard', 'surfboard', 'tennis racket', 'bottle',
                         'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                         'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                         'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                         'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                         'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def random_colors(color_num):

    colors = [[random.randint(0, 255)/255. for _ in range(3)] for _ in range(color_num)]
    if color_num == 20:
        color_dict = dict(zip(VOC_BBOX_LABEL_NAMES, colors))
        return color_dict
    if color_num == 80:
        color_dict = dict(zip(COCO_BBOX_LABEL_NAMES, colors))
        return color_dict


colors = random_colors(80)
id2lab = {cls: lab for cls, lab in zip(range(len(COCO_BBOX_LABEL_NAMES)), COCO_BBOX_LABEL_NAMES)}


def plt_save_img(img, bboxes, labels, scores, save_path):
    """

    :param img: (h, w, 3)
    :param bboxes: format -> [xmin, ymin, xmax, ymax] / shape: (n, 4) / type: ndarray
    :param save_path:
    :param labels: type: list
    :param scores:
    :return:
    """
    # print(labels, scores)
    assert isinstance(img, np.ndarray), f"the first parameter's dtype should be np.ndarray but got {type(img)}!"
    assert img.shape[-1] == 3, f"img's shape must be (h, w, 3), but got {img.shape}"
    assert len(bboxes) == len(labels)
    assert len(bboxes) == len(scores)
    fig, ax = plt.subplots(figsize=[16, 16])
    ax.imshow(img)
    font = {'family': 'serif',
            'color': 'k',
            'weight': 'normal',
            'size': 8}

    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True)
    if len(bboxes) > 0:
        for i, box in enumerate(bboxes):
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            xy = (box[0], box[1])
            # xy = (xmin, ymin), w = box_width, h = box_height, fill = False, edgecolor = 'r', linewidth=1
            rectangle = mpatches.Rectangle(xy=xy, width=box_w, height=box_h, fill=False, edgecolor=colors[id2lab[labels[i]]], linewidth=2.5)
            ax.add_patch(rectangle)
            caption = id2lab[labels[i]] + f':{scores[i]:.3f}'
            # color: text字体颜色 / style='italic' 斜体
            ax.text(x=xy[0],
                    y=xy[1] - 3,
                    s=caption,
                    fontdict=font,
                    color='k',
                    style='italic',
                    bbox={'facecolor': colors[id2lab[labels[i]]], 'alpha': 0.5, 'pad': 3})
    ax.set_axis_off()
    plt.savefig(save_path, dpi=200)
    plt.clf()
    fig.clf()
    fig.close()
    plt.close('all')
    gc.collect()


def cv2_save_img_plot_pred_gt(img, pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, save_path):
    assert isinstance(img, np.ndarray)
    assert len(pred_bboxes) == len(pred_labels)
    assert len(gt_bboxes) == len(gt_labels)
    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True)

    img_gt, img_pred = img.copy(), img.copy()
    if len(pred_bboxes) > 0:
        for i, box in enumerate(pred_bboxes):
            # pt1:左上角坐标[xmin, ymin] ; pt2:右下角坐标[xmax, ymax]
            lt = (round(box[0]), round(box[1]))
            rb = (round(box[2]), round(box[3]))
            bl = (round(box[0]), round(box[3]))
            # cv2.rectangle() parameters:
            # img: image array
            # pt1: 左上角
            # pt2: 右下角
            # color: color
            # thickness: int / 表示矩形边框的厚度，如果为负值，如 CV_FILLED，则表示填充整个矩形
            img = cv2.rectangle(img, pt1=lt, pt2=rb, color=[0, 238, 238], thickness=1)
            # text:显示的文本
            # org文本框左下角坐标（只接受元素为int的元组）
            # fontFace：字体类型
            # fontScale:字体大小（float）
            # thickness：int，值为-1时表示填充颜色
            font = cv2.FONT_HERSHEY_SIMPLEX
            # caption = id2lab[labels[i]] + f':{scores[i]:.3f}'
            caption = f'{pred_labels[i]}:{pred_scores[i]:.1f}'
            box_h, box_w = int(box[3] - box[1]), int(box[2] - box[0])
            img = cv2.rectangle(img, pt1=lt, pt2=(round(box[0])+box_w, round(box[1])+12), color=[200, 0, 0], thickness=-1)
            h, w, c = img.shape
            img = cv2.putText(img,
                              text=caption,
                              org=(round(box[0]), round(box[1])+9),
                              fontFace=font, fontScale=0.35,
                              color=[255, 255, 255],
                              thickness=1, 
                              lineType=cv2.LINE_AA)
            img_pred = np.ascontiguousarray(img.copy())

    if len(gt_bboxes) > 0:
        for i, box in enumerate(gt_bboxes):
            lt = (round(box[0]), round(box[1]))
            rb = (round(box[2]), round(box[3]))
            bl = (round(box[0]), round(box[3]))
            img = cv2.rectangle(img, pt1=lt, pt2=rb, color=[0, 255, 255], thickness=1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            caption = f'{gt_labels[i]}'
            box_h, box_w = int(box[3] - box[1]), int(box[2] - box[0])
            img = cv2.rectangle(img, pt1=lt, pt2=(round(box[0])+box_w, round(box[1])-12), color=[0, 200, 0], thickness=-1)
            h, w, c = img.shape
            img = cv2.putText(img,
                              text=caption,
                              org=(round(box[0]), round(box[1])-9),
                              fontFace=font, fontScale=0.35,
                              color=[255, 255, 255],
                              thickness=1, 
                              lineType=cv2.LINE_AA)
            img_gt = np.ascontiguousarray(img.copy())
            
    img = np.ascontiguousarray((img_pred * 0.65 + img_gt * 0.35).astype('uint8'))
    cv2.imwrite(str(save_path), img[:, :, ::-1])


def cv2_save_img(img, bboxes, labels, scores, save_path):
    """

    :param img:
    :param bboxes: [xmin, ymin, xmax, ymax]
    :param labels:
    :param scores:
    :param save_path:
    :return:
    """
    assert isinstance(img, np.ndarray)
    assert len(bboxes) == len(labels)

    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True)

    if len(bboxes) > 0:
        for i, box in enumerate(bboxes):
            # pt1:左上角坐标[xmin, ymin] ; pt2:右下角坐标[xmax, ymax]
            lt = (round(box[0]), round(box[1]))
            rb = (round(box[2]), round(box[3]))
            bl = (round(box[0]), round(box[3]))
            # cv2.rectangle() parameters:
            # img: image array
            # pt1: 左上角
            # pt2: 右下角
            # color: color
            # thickness: 表示矩形边框的厚度，如果为负值，如 CV_FILLED，则表示填充整个矩形
            img = cv2.rectangle(img, pt1=lt, pt2=rb, color=[0, 238, 238], thickness=1)
            # text:显示的文本
            # org文本框左下角坐标（只接受元素为int的元组）
            # fontFace：字体类型
            # fontScale:字体大小（float）
            # thickness：int，值为-1时表示填充颜色
            font = cv2.FONT_HERSHEY_SIMPLEX
            # caption = id2lab[labels[i]] + f':{scores[i]:.3f}'
            caption = f'{labels[i]}:{scores[i]:.1f}'
            box_h, box_w = int(box[3] - box[1]), int(box[2] - box[0])
            img = cv2.rectangle(img, pt1=lt, pt2=(round(box[0])+box_w, round(box[1])+12), color=[200, 0, 0], thickness=-1)
            h, w, c = img.shape
            img = cv2.putText(img,
                              text=caption,
                              org=(round(box[0]), round(box[1])+9),
                              fontFace=font, fontScale=0.35,
                              color=[255, 255, 255],
                              thickness=1, 
                              lineType=cv2.LINE_AA)
    cv2.imwrite(str(save_path), img[:, :, ::-1])


def plt_plot_img(img, bboxes, labels, scores):
    """

    :param img: (h, w, 3)
    :param bboxes: format -> [xmin, ymin, xmax, ymax] / shape: (n, 4) / type: ndarray
    :param labels: type: list
    :param scores:
    :return:
    """
    assert isinstance(img, np.ndarray)
    assert img.shape[-1] == 3
    assert len(bboxes) == len(labels)
    assert len(bboxes) == len(scores)
    fig, ax = plt.subplots(figsize=[16, 10])
    ax.imshow(img)
    font = {'family': 'serif',
            'color': 'k',
            'weight': 'normal',
            'size': 10}

    if len(bboxes) > 0:
        for i, box in enumerate(bboxes):
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            xy = (box[0], box[1])
            # xy = (xmin, ymin), w = box_width, h = box_height, fill = False, edgecolor = 'r', linewidth=1
            rectangle = mpatches.Rectangle(xy=xy, width=box_w, height=box_h, fill=False, edgecolor='g', linewidth=1.5)
            ax.add_patch(rectangle)
            caption = labels[i] + f':{scores[i]:.3f}'
            # color: text字体颜色 / style='italic' 斜体
            ax.text(x=xy[0],
                    y=xy[1] - 8,
                    s=caption,
                    fontdict=font,
                    color='k',
                    style='italic',
                    bbox={'facecolor': 'y', 'alpha': 0.8, 'pad': 3})
    ax.set_axis_off()
    plt.show()
    plt.clf()
    plt.close('all')


def plt_plot_all(img, pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels):
    """

    :param img: (h, w, 3)
    :param bboxes: format -> [xmin, ymin, xmax, ymax] / shape: (n, 4) / type: ndarray
    :param labels: type: list
    :param scores:
    :return:
    """
    assert isinstance(img, np.ndarray)
    assert img.shape[-1] == 3
    assert len(pred_bboxes) == len(pred_labels)
    assert len(pred_bboxes) == len(pred_scores)
    fig, ax = plt.subplots(figsize=[16, 10])
    ax.imshow(img)
    font = {'family': 'serif',
            'color': 'k',
            'weight': 'normal',
            'size': 8}

    if len(pred_bboxes) > 0:
        for i, box in enumerate(pred_bboxes):
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            xy = (box[0], box[1])
            # xy = (xmin, ymin), w = box_width, h = box_height, fill = False, edgecolor = 'r', linewidth=1
            rectangle = mpatches.Rectangle(xy=xy, width=box_w, height=box_h, fill=False, edgecolor='g', linewidth=1.5)
            ax.add_patch(rectangle)
            caption = pred_labels[i] + f':{pred_scores[i]:.3f}'
            # color: text字体颜色 / style='italic' 斜体
            ax.text(x=xy[0],
                    y=xy[1] - 3,
                    s=caption,
                    fontdict=font,
                    color='k',
                    style='italic',
                    bbox={'facecolor': 'y', 'alpha': 0.5, 'pad': 3})

    if len(gt_bboxes) > 0:
        for i, box in enumerate(gt_bboxes):
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            xy = (box[0], box[1])
            # xy = (xmin, ymin), w = box_width, h = box_height, fill = False, edgecolor = 'r', linewidth=1
            rectangle = mpatches.Rectangle(xy=xy, width=box_w, height=box_h, fill=False, edgecolor='r', linewidth=1.5)
            ax.add_patch(rectangle)
            caption = gt_labels[i]
            # color: text字体颜色 / style='italic' 斜体
            ax.text(x=xy[0],
                    y=xy[1] - 3,
                    s=caption,
                    fontdict=font,
                    color='k',
                    style='italic',
                    bbox={'facecolor': 'r', 'alpha': 0.5, 'pad': 3})

    ax.set_axis_off()
    plt.show()
    plt.clf()
    plt.close('all')