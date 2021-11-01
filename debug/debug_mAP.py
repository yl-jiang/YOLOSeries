import numpy as np
from collections import defaultdict
from numpy.lib.arraysetops import unique
import matplotlib.pyplot as plt
from pathlib import Path
from utils import maybe_mkdir
import torch



class mAP_v2:

    def __init__(self, ground_truth, predict, plot_save_dir):
        """
        :param predict: [batch_size, ]
            目标检测算法的输出(已经经过NMS等一系列处理)，对一张图片而言，算法可能会输出M个预测框
            every element in predict has shape [M, 6], here number 5 represent [xim, ymin, xmax, ymax, conf, cls]
        :param ground_truth: [batch_size, ]
            与predict一一对应的每张图片的ground truth bbox，GT_bbox的数目很可能与算法预测的不一致
            every element in ground_truth has shape [N, 5], here number 4 represent [xmin, ymin, xmax, ymax, cls]
        :param iou_threshold: scalar
            对于elevenInterpolation，iou_threshold一般取0.5
            对于everyInterpolation，iou_threshold可以取任意[0, 1]之间的数
        """
        assert len(predict) == len(ground_truth)
        self.pred, self.gt = [], []

        # 剔除ground_truth为空的数据
        for i in range(len(ground_truth)):
            if len(ground_truth[i]) > 0:
                self.gt.append(ground_truth[i])
                self.pred.append(predict[i])

        self.iou_thr = np.linspace(0.5, 0.95, 10)
        self.save_dir = Path(plot_save_dir)
        maybe_mkdir(self.save_dir)
        del predict, ground_truth


    def compute_tp(self, gt, pred):
        """
        compute tp for single image.
        Arguments:
            gt: length N; element formate is (xmin, ymin, xmax, ymax, cls)
            pred: length M; element formate is (xmin, ymin, xmax, ymax, conf, cls)
        """
        # (M, 1 or 10)
        tp = np.zeros(shape=(pred.shape[0], len(self.iou_thr)), dtype=np.bool)
        # (N, 4) & (M, 4) -> (N, M)
        ious = self.iou(gt[:, :4], pred[:, :4]) 
        iou_mask = ious >= self.iou_thr[0]  # (N, M)
        lab_mask = gt[:, [4]] == pred[:, 5]  # (N, M)
        # iou符合条件而且label匹配
        mask = iou_mask & lab_mask
        if mask.sum() > 0:
            gt_i, pred_i = np.nonzero(mask)
            match_iou = ious[mask]
            match = np.concatenate((np.stack((gt_i, pred_i), axis=1), match_iou[:, None]), axis=1)  # (X, 3) / [gt, pred, iou]
            if mask.sum() > 1:  # 只有一个元素的情况下不需要排序和筛选
                match = match[match[:, 2].argsort()[::-1]]
                # 一个预测框只负责一个gt
                match = match[np.unique(match[:, 1], return_index=True)[1]]  # (Y, 3)
                # 一个gt只能匹配一个预测框
                match = match[np.unique(match[:, 0], return_index=True)[1]]  # (Z, 3)
            tp[match[:, 1].astype(np.int32)] = match[:, [2]] >= self.iou_thr

        return tp
    

    @staticmethod
    def iou(box1, box2):
        """
        :param box1: [M, 4]
        :param box2: [N, 4]
        :return: [M, N]
        """
        # expand dim for broadcast computing
        # shape: [M, 1, 4]
        box1 = np.expand_dims(box1, axis=1)
        # shape: [M, 1]
        box1_area = np.prod(box1[..., [2, 3]] - box1[..., [0, 1]], axis=-1)
        # shape: [N,]
        box2_area = np.prod(box2[:, [2, 3]] - box2[:, [0, 1]], axis=-1)
        # [M, 1] & [N,] -> [M, N]
        intersection_xmin = np.maximum(box1[..., 0],box2[:, 0])
        intersection_ymin = np.maximum(box1[..., 1],box2[:, 1])
        intersection_xmax = np.minimum(box1[..., 2],box2[:, 2])
        intersection_ymax = np.minimum(box1[..., 3],box2[:, 3])
        # [M, N] & [M, N] -> [M, N]
        intersection_w = np.maximum(0., intersection_xmax - intersection_xmin)
        intersection_h = np.maximum(0., intersection_ymax - intersection_ymin)
        intersection_area = intersection_w * intersection_h + 1e-16
        # [M, N] & [M, 1] & [N,] & [M, N] -> [M, N]
        ious = intersection_area / (box1_area + box2_area - intersection_area + 1e-16)
        return ious


    def compute_ap_per_class(self):
        """
        compute ap for all images.
        """
        tps = []
        for gt, pred in zip(self.gt, self.pred):
            tps.append(self.compute_tp(gt, pred))

        tps = np.concatenate(tps, axis=0)  # (M, X)
        pred_all = np.concatenate(self.pred, axis=0)
        gt_all = np.concatenate(self.gt, axis=0)
        assert len(tps) == len(pred_all)
        
        pred_confs = pred_all[:, 4]  # (M,)
        pred_cls = pred_all[:, 5]  # (M,)
        tar_cls = gt_all[:, 4]  # (N,)

        # 按照从大到小的顺序进行排列
        sort_i = np.argsort(pred_confs)[::-1]
        tps = tps[sort_i]
        pred_confs = pred_confs[sort_i]
        pred_cls = pred_cls[sort_i]

        unique_tar_cls = np.unique(tar_cls)
        # ap for each iou threshold
        ap = np.zeros((len(unique_tar_cls), tps.shape[1]))
        # precision for each class
        precision = np.zeros(shape=[len(unique_tar_cls), 1000])
        # recall for each class
        recall = np.zeros(shape=[len(unique_tar_cls), 1000])
        # precision-recall for mAP@0.5
        xs, pr = np.linspace(0, 1, 1000), []

        for i, c in enumerate(unique_tar_cls):  # each class
            match_i = pred_cls == c  # current class mask
            num_tar = (tar_cls == c).sum()  # TP of current class
            if match_i.sum() > 0 and num_tar > 0:
                cumsum_fp = (~tps[match_i]).cumsum(0)  # (M, X)
                cumsum_tp = tps[match_i].cumsum(0)  # (M, X)
                # 随着测试的case增多，recall越来越大， precision越来越小（因为同一class的num_tar是固定的）
                cumsum_recall = cumsum_tp / (num_tar + 1e-16) 
                cumsum_precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-16)
                # 将np.interp的前两个参数取负号，主要是为了照顾使用np.interp方法插值时x超出xp两端时默认值设置的问题
                recall[i] = np.interp(-xs, -pred_confs[match_i], cumsum_recall[:, 0], left = 0)  # recall of curremt class
                precision[i] = np.interp(-xs, -pred_confs[match_i], cumsum_precision[:, 0], left = 1)  # precision of current class
                for j in range(tps.shape[1]):  # each iou threshold
                    ap[i, j], rec, pre = self.compute_ap(cumsum_recall[:, j], cumsum_precision[:, j])  # average precision with each iou threshold of current class
                    if j == 0:  # only mAP@0.5
                        pr.append(np.interp(xs, rec, pre))  # precision-recall curve of 0.5 iou of current class  
            else:
                continue
        f1_score = 2 * precision * recall / (precision + recall + 1e-16)

        self.plot_curve(xs, precision, self.save_dir/"Precision.png", "Precision", "Precision-Confidence")
        self.plot_curve(xs, recall, self.save_dir/"Recall.png", "Recall", 'Recall-Conficence')
        self.plot_curve(xs, f1_score, self.save_dir/"F1Score.png", "F1", 'F1 Score-Confidence')
        self.plot_pr_curve(xs, pr, ap, self.save_dir/"PRCurve.png")

        best_i = f1_score.mean(axis=0).argmax(axis=0)
        metrics = {"precision": precision[:, best_i], 
                   "recall": recall[:, best_i], 
                   "ap": ap, 
                   "f1": f1_score[:, best_i], 
                   "unique_cls": unique_tar_cls}
        return metrics


    def compute_ap(self, recall, precision):
        rec = np.concatenate(([0.], recall, [1.]))
        pre = np.concatenate(([1.], precision, [0.]))

        pre = np.flip(np.maximum.accumulate(np.flip(pre)))

        method = 'coco'
        if method == 'coco':
            xs = np.linspace(0, 1, 101)
            # 计算每个recall和precision刻度围成的梯形面积之和
            ap = np.trapz(y=np.interp(xs, rec, pre), x=xs)
        else:  # continous
            i = np.nonzero(rec[1:], rec[:-1])
            ap = np.sum(rec[i+1] - rec[i]) * pre[i+1]
        return ap, rec, pre
    
    @staticmethod
    def plot_curve(xs, ys, save_path, ylabel,  title):
        """
        plot Precision-Confidence, Recall-Confidence, F1-Confidence curve.
        """
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        ax.plot(xs, ys.T, linewidth=1, color='grey')
        my = ys.mean(0)
        ax.plot(xs, my, linewidth=2, color='red', label='all classes')
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        plt.legend(loc=0)
        fig.savefig(str(save_path), dpi=300)
        plt.close()

    @staticmethod
    def plot_pr_curve(xs, ys, ap, save_path):
        """
        plot Precision-Recall curve.
        Arguments:
            ys: precision-recall(mAP@0.5) curve list of each class
        """
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        ys = np.stack(ys, axis=1)
        ax.plot(xs, ys, linewidth=1, color='grey')
        # 曲线上的点代表所有class在同一recall（x轴）下的precision值（y轴）
        ax.plot(xs, ys.mean(axis=1), linewidth=2, color='red', label=f'all classes {ap[:, 0].mean():.2f} mAP@0.5')
        ax.set_ylabel('Precision')
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Precision-Recall")
        plt.legend(loc=0)
        fig.savefig(str(save_path), dpi=300)
        plt.close()


    def get_mean_metrics(self):
        metrics = self.compute_ap_per_class()
        ap = metrics['ap']
        ap50 = ap[:, 0]
        apm = ap.mean(axis=1)
        map50 = ap50.mean()
        map = apm.mean()
        mp = metrics['precision'].mean()
        mr = metrics['recall'].mean()
        return map, map50, mp, mr



def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)



if __name__ == "__main__":
    import pickle
    gts = pickle.load(open("/pkl/gt_bbox.pkl", 'rb'))
    preds = pickle.load(open("/pkl/pred_coco_bbox_640_xlarge.pkl", 'rb'))
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    
    tps = []
    confs = []
    pred_cls = []
    gt_cls = []
    for i in range(len(gts)):
        gtc = gts[i][:, 4]
        x = np.zeros_like(gts[i])
        x[:, 0] = gts[i][:, 4]
        x[:, 1:] = gts[i][:, :4]
        tps.append(process_batch(torch.from_numpy(preds[i]), torch.from_numpy(x), iouv).cpu().numpy())
        confs.append(preds[i][:, 4])
        pred_cls.append(preds[i][:, 5])
        gt_cls.append(gts[i][:, 4])
    tps = np.concatenate(tps, axis=0)
    confs = np.concatenate(confs, 0)
    pred_cls = np.concatenate(pred_cls, 0)
    gt_cls = np.concatenate(gt_cls, 0)
    p, r, ap, f1, ap_class = ap_per_class(tps, confs, pred_cls, gt_cls, True)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    a = 1