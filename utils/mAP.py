import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gc
import seaborn as sn
current_work_directionary = Path('__file__').parent.absolute()
sys.path.insert(0, str(current_work_directionary))

from .common import maybe_mkdir
def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

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
    intersection_area = intersection_w * intersection_h
    # [M, N] & [M, 1] & [N,] & [M, N] -> [M, N]
    ious = intersection_area / np.clip(box1_area + box2_area - intersection_area, a_min=1e-6, a_max= 10000000)
    return ious

class mAP_v2:

    def __init__(self, ground_truth, predict, plot_save_dir, type='coco'):
        """
        Inputs:
            predict: [batch_size, ]
                目标检测算法的输出(已经经过NMS等一系列处理), 对一张图片而言, 算法可能会输出M个预测框
                every element in predict has shape [M, 6], the last dimension represent [xim, ymin, xmax, ymax, conf, cls]
            ground_truth: [batch_size, ]
                与predict一一对应的每张图片的ground truth bbox, GT_bbox的数目很可能与算法预测的不一致
                every element in ground_truth has shape [N, 5], the last dimension represent [xmin, ymin, xmax, ymax, cls]
        """
        assert len(predict) == len(ground_truth)
        self.pred, self.gt = [], []

        # 剔除ground_truth为空的数据
        for i in range(len(ground_truth)):
            if len(ground_truth[i]) > 0 and len(predict[i]) > 0:
                self.gt.append(ground_truth[i])
                self.pred.append(predict[i])

        self.iou_thr = np.linspace(0.5, 0.95, 10)
        self.save_dir = Path(plot_save_dir)
        self.type = type
        maybe_mkdir(self.save_dir)

    def compute_tp(self, gt, pred):
        """
        compute tp for single image.
        Inputs:
            gt: length N; element formate is (xmin, ymin, xmax, ymax, cls)
            pred: length M; element formate is (xmin, ymin, xmax, ymax, score, cls)
        Outputs:
            tp: (M, 10) / bool
        """
        # (M, 10)
        tp = np.zeros(shape=(pred.shape[0], len(self.iou_thr)), dtype=bool)
        # (N, 4) & (M, 4) -> (N, M)
        ious = iou(gt[:, :4], pred[:, :4]) 
        iou_mask = ious >= self.iou_thr[0]  # (N, M)
        lab_mask = gt[:, [4]] == pred[:, 5]  # (N, M)
        # iou符合条件而且label匹配
        mask = iou_mask & lab_mask
        if mask.sum() > 0:
            gt_i, pred_i = np.nonzero(mask)
            match_iou = ious[mask]
            match = np.concatenate((np.stack((gt_i, pred_i), axis=1), match_iou[:, None]), axis=1)  # (X, 3) / [gt, pred, iou]
            if mask.sum() > 1:  # 只有一个元素的情况下不需要排序和筛选
                # 根据iou从大到小排序
                match = match[match[:, 2].argsort()[::-1]]
                # 一个预测框只负责一个gt
                match = match[np.unique(match[:, 1], return_index=True)[1]]  # (Y, 3)
                # 一个gt只能匹配一个预测框
                match = match[np.unique(match[:, 0], return_index=True)[1]]  # (Z, 3)
            tp[match[:, 1].astype(np.int32)] = match[:, [2]] >= self.iou_thr

        return tp
    
    def compute_ap_per_class(self):
        """
        compute ap for all images.
        """
        tps = []
        for gt, pred in zip(self.gt, self.pred):
            tps.append(self.compute_tp(gt, pred))  # [(m1, 10), (m2, 10), ...]

        tps = np.concatenate(tps, axis=0)  # (M, 10)
        pred_all = np.concatenate(self.pred, axis=0)  # (M, 6)
        assert len(tps) == len(pred_all)

        gt_all = np.concatenate(self.gt, axis=0)  # (N, 5)
        
        pred_confs = pred_all[:, 4]  # (M,)
        pred_cls = pred_all[:, 5]  # (M,)
        tar_cls = gt_all[:, 4]  # (N,)

        # 按照从大到小的顺序进行排列
        sort_i = np.argsort(pred_confs)[::-1]
        sorted_tps = tps[sort_i]  # (M, 10)
        sorted_cof = pred_confs[sort_i]  # (M,)
        sorted_cls = pred_cls[sort_i]  # (M,)

        tot_cls = np.unique(tar_cls)  # (num_class,)
        # ap for each iou threshold
        ap = np.zeros((len(tot_cls), sorted_tps.shape[1]))  # (num_class, 10)
        # precision for each class
        precision = np.zeros(shape=[len(tot_cls), 1000])
        # recall for each class
        recall = np.zeros(shape=[len(tot_cls), 1000])
        # precision-recall for mAP@0.5
        xs, pr = np.linspace(0, 1, 1000), []

        for i, c in enumerate(tot_cls):  # each class
            match_i = sorted_cls == c  # current class mask
            num_tar = (tar_cls == c).sum()  # TP of current class
            if match_i.sum() > 0 and num_tar > 0:
                cumsum_fp = (~sorted_tps[match_i]).cumsum(0)  # (X, 10)
                cumsum_tp = sorted_tps[match_i].cumsum(0)  # (X, 10)
                # 随着测试的case增多, recall越来越大,  precision越来越小（因为同一class的num_tar是固定的）
                cumsum_recall = cumsum_tp / (num_tar + 1e-16)   # (X, 10)
                cumsum_precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-16)  # (X, 10)
                # 将np.interp的前两个参数取负号, 主要是为了照顾使用np.interp方法插值时x超出xp两端时默认值设置问题
                recall[i] = np.interp(-xs, -sorted_cof[match_i], cumsum_recall[:, 0], left = 0)  # recall of curremt class
                precision[i] = np.interp(-xs, -sorted_cof[match_i], cumsum_precision[:, 0], left = 1)  # precision of current class
                for j in range(sorted_tps.shape[1]):  # each iou threshold
                    ap[i, j], rec, pre = self.compute_ap(cumsum_recall[:, j], cumsum_precision[:, j], self.type)  # average precision with each iou threshold of current class
                    if j == 0:  # only mAP@0.5
                        pr.append(np.interp(xs, rec, pre))  # precision-recall curve of 0.5 iou of current class  
            else:
                continue
        f1_score = 2 * precision * recall / (precision + recall + 1e-16)
        best_i = smooth(f1_score.mean(0), 0.1).argmax()
        try:
            self.plot_curve(xs, precision, self.save_dir/"Precision.png", "Precision", "Precision-Confidence")
            self.plot_curve(xs, recall, self.save_dir/"Recall.png", "Recall", 'Recall-Conficence')
            self.plot_curve(xs, f1_score, self.save_dir/"F1Score.png", "F1", 'F1 Score-Confidence')
            self.plot_pr_curve(xs, pr, ap, self.save_dir/"PRCurve.png")
        except Exception as err:
            print(err)

        metrics = {"precision": precision[:, best_i],   # (num_class,)
                   "recall": recall[:, best_i],   # (num_class,)
                   "ap": ap,   # (num_class, 10)
                   "f1": f1_score[:, best_i],   # (num_class,)
                   "unique_cls": tot_cls}
        return metrics

    def compute_ap(self, recall, precision, type):
        """
        Inputs:
            recall: (X,)
            precision: (X,)
        """
        rec = np.concatenate(([0.], recall, [1.]))
        pre = np.concatenate(([1.], precision, [0.]))

        pre = np.flip(np.maximum.accumulate(np.flip(pre)))

        if type == 'coco':
            xs = np.linspace(0, 1, 101)
            # 计算每个recall和precision刻度围成的梯形面积之和
            ap = np.trapz(y=np.interp(xs, rec, pre), x=xs)
        else:  # continous
            i = np.where(rec[1:] != rec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((rec[i+1] - rec[i]) * pre[i+1])
        return ap, rec, pre
    
    @staticmethod
    def plot_curve(xs, ys, save_path, ylabel,  title):
        """
        plot Precision-Confidence, Recall-Confidence, F1-Confidence curve.
        """
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        ax.plot(xs, ys.T, linewidth=1, color='gray')
        my = ys.mean(0)
        ax.plot(xs, my, linewidth=2, color='red', label='all classes')
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        plt.legend(loc=0)
        fig.savefig(str(save_path), dpi=300)
        plt.close('all')
        fig.clear()
        del ax, fig, my
        gc.collect()

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

    def plot_ap_per_class(self, apm, cls2lab=None):
        """
        Inputs:
            ap: (num_class,)
            cls2id: dict like {0: 'person', 1: 'bus', ...}

        """
        clsid = np.arange(len(apm))
        sorted_i = np.argsort(apm)
        fig = plt.figure(figsize=[10, 10])
        category_colors = plt.colormaps['RdYlGn'](np.linspace(0.15, 0.85, len(apm)))
        if cls2lab is None:
            plt.barh([str(i) for i in clsid[sorted_i]], apm[sorted_i], height=0.8, align='center', color=category_colors)
        else:
            labs = [cls2lab[i] for i in sorted_i]
            plt.barh(labs, apm[sorted_i], height=0.8, align='center', color=category_colors)

        for i, score in enumerate(apm[sorted_i]):
            plt.text(score, i, f"{score:.3f}", fontweight='bold')

        plt.xlabel('mAP', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        plt.ylabel('Category', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        plt.title(f"mAP {apm.mean():.3f}", fontdict={'fontsize': 14, 'fontweight': 'bold'})
        plt.tight_layout()
        plt.savefig(str(self.save_dir / 'AP_Per_Class.png'), dpi=300)
        fig.clear()
        plt.close('all')
        del fig
        gc.collect()

    def get_mean_metrics(self):
        try:
            metrics = self.compute_ap_per_class()
            ap = metrics['ap']  # (num_class, 10)
            apm = ap.mean(axis=1)  # (num_class,)
            map50 = ap[:, 0].mean()
            map = apm.mean()
            mp = metrics['precision'].mean()
            mr = metrics['recall'].mean()
            self.plot_ap_per_class(apm)
        except Exception as err:
            map, map50, mp, mr = 0., 0., 0., 0.
            print(err)
        return map, map50, mp, mr


class ConfusionMatrix:

    def __init__(self, predict, ground_truth, conf_thres, num_class, iou_thres) -> None:
        """
        :param predict: [batch_size, ]
            目标检测算法的输出(已经经过NMS等一系列处理), 对一张图片而言, 算法可能会输出M个预测框
            every element in predict has shape [M, 6], here number 5 represent [xim, ymin, xmax, ymax, conf, cls]
        :param ground_truth: [batch_size, ]
            与predict一一对应的每张图片的ground truth bbox, GT_bbox的数目很可能与算法预测的不一致
            every element in ground_truth has shape [N, 5], here number 4 represent [xmin, ymin, xmax, ymax, cls]
        :param iou_threshold: scalar
            对于elevenInterpolation, iou_threshold一般取0.5
            对于everyInterpolation, iou_threshold可以取任意[0, 1]之间的数
        """
        assert len(predict) == len(ground_truth)
        self.preds, self.gts = [], []

        # 剔除ground_truth为空的数据
        for i in range(len(ground_truth)):
            if len(ground_truth[i]) > 0:
                self.gts.append(ground_truth[i])
                self.preds.append(predict[i])
        del predict, ground_truth


        # self.preds = np.concatenate(self.preds, axis=0)
        # self.gts = np.concatenate(self.gts, 0)
        self.iou_thr = iou_thres
        self.conf_thr = conf_thres
        # 增加的一行一列表示将某个gt cls预测为背景（也就是没有预测到该gt cls）
        # row: prediction; column: label
        self.matrix = np.zeros(shape=(num_class+1, num_class+1)) 
        self.num_class = num_class

    def process_batch(self, preds, gts):
        """
        Arguments:
            preds: (M, 6) / [xmin, ymin, xmax, ymax, conf, cls]
            gts: (N, 5) / [xmin, ymin, xmax, ymax, cls]
        """
        matrix = np.zeros(shape=(self.num_class+1, self.num_class+1))
        preds = preds[preds[:, 4] > self.conf_thr]
        gt_cls = gts[:, 4].astype(np.int16)
        pred_cls = preds[:, 5].astype(np.int16)
        ious = iou(gts[:, :4], preds[:, :4])  # (N, M)

        gt_i, pred_i = np.nonzero(ious > self.iou_thr)
        if len(gt_i) > 0:
            match_i = np.concatenate((np.stack((gt_i, pred_i), axis=1), ious[gt_i, pred_i][:, None]), axis=1)  # (gt, pred, iou)
            if len(gt_i) > 1:
                # 按照iou值进行排序
                match_i = match_i[match_i[:, 2].argsort()[::-1]]
                # 每个预测框只能负责一个gt
                match_i = match_i[np.unique(match_i[:, 1], return_index=True)[1]]
                match_i = match_i[match_i[:, 2].argsort()[::-1]]
                # 一个gt只能被匹配一次
                match_i = match_i[np.unique(match_i[:, 0], return_index=True)[1]]
            else:
                match_i = np.zeros((0, 3))
            
            gt_mi, pred_mi, ious = match_i.astype(np.int16).transpose()
            for i, c in enumerate(gt_cls):
                match = gt_mi == i
                # sum(match) == 1是因为：如果一个gt有与之匹配的预测框, 那么该gt能且只能被匹配一次
                if len(match_i) > 0 and sum(match) == 1:  
                    matrix[pred_cls[pred_mi[match]], c] += 1
                else:
                    matrix[-1, c] += 1  # FP(在某一张image的预测中, 模型输出了gt cls中没有的类别)

            if len(match_i) == 0:  # 所有的gt都没有与之匹配的预测框, 意味着如果模型有输出, 那么其输出的pred都是错的
                for i, c in enumerate(pred_cls):
                    if not any(pred_mi == i):
                        matrix[c, -1] += 1  # FN
        return matrix

    def compute_matrix(self):
        for i in range(len(self.preds)):
            self.matrix += self.process_batch(self.preds[i], self.gts[i])
            

    def plot(self, names, save_path):
        matrix = self.matrix / (self.matrix.sum(0).reshape(1, -1)) + 1e-6
        matrix[matrix < 0.005] = np.nan
        fig = plt.figure(figsize=[12, 10], tight_layout=True)
        sn.set(font_scale=0.8)
        if names:
            xlabels = names + ['background FN']
            ylabels = names + ['background FP']
        else:
            xlabels = 'auto'
            ylabels = 'auto'

        sn.heatmap(matrix, annot=self.num_class < 30, annot_kws={'size': 8, 'weight':'bold'}, cmap='Reds', fmt='.2f',
                   square=True, xticklabels=xlabels, yticklabels=ylabels).set_facecolor((1, 1, 1))
       
        fig.axes[0].set_xlabel('Label')
        fig.axes[0].set_ylabel('Predict')
        fig.savefig(str(save_path), dpi=300)
        plt.close()



if __name__ == "__main__":
    # all_gts = [np.array(
    #      [[2  , 10 , 173, 238, 0],
    #       [439, 157, 556, 241, 1],
    #       [437, 246, 518, 351, 1],
    #       [272, 190, 316, 259, 2]])]
    
    # all_preds = [np.array(
    #         [[0  , 13 , 174, 244, 0.471781, 0],
    #          [274, 226, 301, 265, 0.414941, 3],
    #          [429, 219, 528, 247, 0.460851, 1],
    #          [0  , 199, 88 , 436, 0.292345, 4],
    #          [433, 260, 506, 336, 0.269833, 1]])]

    import pickle

    all_gts = pickle.load(open('/E/JYL/Git/mAP/input/gt.pkl', 'rb'))
    all_preds = pickle.load(open('/E/JYL/Git/mAP/input/pr.pkl', 'rb'))
    
    mapv2 = mAP_v2(all_gts, all_preds, "/home/uih/JYL/GitHub/YOLOSeries/result/curve", 'coco')
    map, map50, mp, mr = mapv2.get_mean_metrics()
    print(f'mAP = {map:.3f}')
    print(f'mAP@0.5 = {map50:.3f}')
    print(f'mp = {mp:.3f}')
    print(f'mr = {mr:.3f}')

    # cm = ConfusionMatrix(all_preds, all_gts, 0.25, len(names), 0.45)
    # cm.compute_matrix()
    # cm.plot(names, "/utils/confusion-matrix.png")