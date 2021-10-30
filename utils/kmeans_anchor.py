#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/22 15:30
# @Author  : jyl
# @File    : kmeans_anchor.py
import numpy as np

def alias_setup(probs):
    """
    probs： 某个概率分布
    返回: Alias数组与Prob数组
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    smaller = []
    larger = []
    for i, prob in enumerate(probs):
        q[i] = K * prob  # 概率
        if q[i] < 1.0:
            smaller.append(i)
        else:
            larger.append(i)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    输入: Prob数组和Alias数组
    输出: 一次采样结果
    '''
    K = len(J)
    k = int(np.floor(np.random.rand() * K))  # 随机取一列
    if np.random.rand() < q[k]:
        return k
    else:
        return J[k]

def alias_sample(probs, samples):
    assert isinstance(samples, int), 'Samples must be a integer.'
    sample_result = []
    J, p = alias_setup(probs)
    for i in range(samples):
        sample_result.append(alias_draw(J, p))
    return sample_result

cluster_x = []
cluster_y = []

for img in data_list:
    img_width = img['width']
    img_height = img['height']
    # box: [ymax, xmax, ymin, xmin]
    for box in img['obj']['bbox']:
        box_width = box[1] - box[-1]
        box_height = box[0]  - box[2]
        cluster_x.append(box_width / img_width)
        cluster_y.append(box_height / img_height)



def iou(center_box, other_boxes):
    intersection_box = np.where(center_box < other_boxes, center_box, other_boxes)
    intersection_area = np.prod(intersection_box, axis=1)
    center_box_area = np.prod(center_box)
    otherbox_areas = np.prod(other_boxes, axis=1)
    ious = intersection_area / (center_box_area + otherbox_areas - intersection_area)
    return ious


def classification(k, bboxes, use_alias):
    length = len(bboxes)
    center_index = get_centers(k, bboxes, use_alias)
    center_coord = bboxes[center_index]
    center_tmp = np.zeros_like(center_coord)
    ori_dis = np.full(shape=length, fill_value=np.inf)
    class_list = np.zeros(shape=length) - 1

    times = 1
    while np.sum(np.square(center_coord - center_tmp)) > 1e-7:
        times += 1
        center_tmp = center_coord.copy()
        for i in range(k):
            new_dis = 1 - iou(center_coord[i], bboxes)
            class_list = np.where(ori_dis < new_dis, class_list, i)
            ori_dis = np.where(ori_dis < new_dis, ori_dis, new_dis)
        # update center
        for i in range(k):
            center_coord[i] = np.mean(bboxes[class_list == i], axis=0)

    return class_list, center_coord


def show_result(class_list, raw_data, center_coordinate):
    print('Showing... ...')
    colors = [
        '#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#228B22',
        '#0000FF', '#FF1493', '#EE82EE', '#000000', '#FFA500',
        '#00FF00', '#006400', '#00FFFF', '#0000FF', '#FFFACD',
    ]

    use_color = []
    for node in class_list:
        use_color.append(colors[int(node)])

    plt.figure(num=1, figsize=(16, 9))
    plt.scatter(x=raw_data[:, 0], y=raw_data[:, 1], c=use_color, s=50, marker='o', alpha=0.3)
    plt.scatter(x=center_coordinate[:, 0], y=center_coordinate[:, 1], c='b', s=200, marker='+', alpha=0.8)
    plt.show()


def get_centers(k, bboxes, use_alias):
    if use_alias:
        centers = [random.randint(a=0, b=len(bboxes))]
        tmp_dis = np.full(shape=len(bboxes), fill_value=np.inf)
        while len(centers) < k:
            for i, center in enumerate(centers):
                dis = 1 - iou(center, bboxes)
                dis = np.where(dis < tmp_dis, dis, tmp_dis)
            probs = dis / np.sum(dis)
            #             centers.append(np.random.choice(a=len(bboxes), size=1, p=probs)[0])
            centers.append(alias_sample(probs, 1)[0])
        return centers
    else:
        return np.random.choice(a=np.arange(len(bboxes)), size=k)


def kmeans(raw_data, k, use_alias, show):
    class_list, center_coordinate = classification(k, raw_data, use_alias)
    if show:
        show_result(class_list, raw_data, center_coordinate)
    return class_list, center_coordinate