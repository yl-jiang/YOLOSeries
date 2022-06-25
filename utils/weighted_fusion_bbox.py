import numpy as np

from utils import cpu_iou

bboxes = [[
    [0.00, 0.51, 0.81, 0.91, 0.9, 0],
    [0.10, 0.31, 0.71, 0.61, 0.8, 1],
    [0.01, 0.32, 0.83, 0.93, 0.2, 0],
    [0.02, 0.53, 0.11, 0.94, 0.4, 1],
    [0.03, 0.24, 0.12, 0.35, 0.7, 1],
],[
    [0.04, 0.56, 0.84, 0.92, 0.5, 1],
    [0.12, 0.33, 0.72, 0.64, 0.8, 1],
    [0.38, 0.66, 0.79, 0.95, 0.7, 1],
    [0.08, 0.49, 0.21, 0.89, 0.3, 0],
]]

weights = [2, 1]
# iou_thr = 0.5
skip_box_thr = 0.0001
sigma = 0.1


def preprocess(bboxes, weights):
    """

    :param bboxes: [xmin, ymin, xmax, ymax, score, class]
    :param weights: list
    :return: [xmin, ymin, xmax, ymax, score, class, weight]
    """
    model_num = len(weights)
    bbox_list = []
    for i in range(model_num):
        weight = np.full(shape=(len(bboxes[i]), 1), fill_value=weights[i])
        bbox = np.concatenate((np.array(bboxes[i]), weight), axis=-1)
        bbox_list.append(bbox)
        bbox_list = np.concatenate(bbox_list, axis=0)
    return bbox_list


def update_fusion_bbox(cluster_bbox):
    """
    :param cluster_bbox: [xmin, ymin, xmax, ymax, score, class, weight]
    """
    fusion_bbox = []
    for i in range(len(cluster_bbox)):
        bbox = np.array(cluster_bbox[i])[:, :4]  # (N, 4)
        score = np.array(cluster_bbox[i])[:, 4]  # (N,)
        lab = np.array(cluster_bbox[i])[:, 5]
        # 确保每个cluster中所有bbox的label都相同
        assert np.power(lab - lab[0], 2).sum() == 0
        w = np.array(cluster_bbox[i])[:, 6]  # (N,)
        weighted_bbox = bbox * score.reshape(-1, 1)
        weighted_bbox /= np.sum(score)
        weighted_bbox = np.mean(weighted_bbox, axis=0)
        weighted_score = score * w
        weighted_score = np.sum(weighted_score) / np.sum(w)
        # fusion: [xmin, ymin, xmax, ymax, score, class]
        fusion = np.append(weighted_bbox, [weighted_score, lab[0]])
        fusion_bbox.append(fusion)
    return fusion_bbox


def weighted_fusion_bbox(bbox_list, iou_thr=0.5):
    """

    :param bbox_list: shape: (N, 7) / [xmin, ymin, xmax, ymax, score, class, weight]
    :param iou_thr: float
    :return:
    """
    Cluster, Fusion = [], []
    unique_lab = np.unique(bbox_list[:, 5])
    for lab in unique_lab:
        lab_mask = bbox_list[:, 5] == lab
        bbox = bbox_list[lab_mask]
        sort_index = np.argsort(bbox[:, 4])[::-1]
        fusion_bbox = [bbox[sort_index[0]][:7].tolist()]  # 将score值最大的bbox放入fusiin，作为fusion的初始化值
        cluster_bbox = [[]]
        for i in sort_index:
            assert len(cluster_bbox) == len(fusion_bbox)
            cur_bbox = bbox[i]
            ious = cpu_iou(np.array(cur_bbox)[:4][None,:], np.array(fusion_bbox)[:, :4])
            iou_mask = np.greater_equal(ious, iou_thr)
            # 如果当前的bbox与fusion中的所有bbox的iou值都不超过阈值，则将其作为一个新元素分别添加到custer和fusion中
            if len(iou_mask.nonzero()[0]) == 0:
                fusion_bbox.append(cur_bbox.tolist())
                cluster_bbox.append([cur_bbox.tolist()])
            # 否则将当前bbox添加到相应的cluster中，最为构成最终对应fusion box的一员
            else:
                for j in iou_mask.nonzero()[0].astype(np.int32):
                    cluster_bbox[j].append(cur_bbox.tolist())
            fusion_bbox = update_fusion_bbox(cluster_bbox)
        Cluster.append(cluster_bbox)
        Fusion.append(fusion_bbox)

    return Cluster, Fusion
