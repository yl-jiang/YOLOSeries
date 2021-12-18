import numpy as np
from collections import defaultdict
from numpy.lib.arraysetops import unique
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from numpy.matrixlib import matrix
current_work_directionary = Path('__file__').parent.absolute()
sys.path.insert(0, str(current_work_directionary))
from utils import maybe_mkdir
import seaborn as sn


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

class mAP:

    def __init__(self, predict, ground_truth, iou_threshold=0.5):
        """
        :param predict: [batch_size, ]
            目标检测算法的输出(已经经过NMS等一系列处理)，对一张图片而言，算法可能会输出M个预测框
            every element in predict has shape [M, 5], here number 5 represent [xim, ymin, xmax, ymax, conf]
        :param ground_truth: [batch_size, ]
            与predict一一对应的每张图片的ground truth bbox，GT_bbox的数目很可能与算法预测的不一致
            every element in ground_truth has shape [N, 4], here number 4 represent [xmin, ymin, xmax, ymax]
        :param iou_threshold: scalar
            对于elevenInterpolation，iou_threshold一般取0.5
            对于everyInterpolation，iou_threshold可以取任意[0, 1]之间的数
        """
        assert len(predict) == len(ground_truth)
        self.pred, self.gt_box = [], []

        # 剔除ground_truth为空的数据
        for i in range(len(ground_truth)):
            if len(ground_truth[i]) > 0:
                self.gt_box.append(ground_truth[i])
                self.pred.append(predict[i])
                
        del predict, ground_truth

        self.iou_threshold = iou_threshold
        self.ap_dict = self.make_ap_dict()
        self.precision, self.recall = self.compute_pr(self.ap_dict)
        self.elevenPointAP = self.elevenPointInterpolation()
        self.everyPointAP = self.everyPointInterpolation()

    def make_ap_dict(self):
        ap_dict = defaultdict(list)
        for pred, gt_box in zip(self.pred, self.gt_box):
            if len(pred) > 0:  # 如果预测框不为空（gt box不肯能为空，因为在__init__中已经将可能为空的gt box过滤掉了）
                pred, gt_box = np.asarray(pred), np.asarray(gt_box)
                tpfp, conf, gt_num = self.get_tpfp(pred[:, 4], pred[:, 0:4], gt_box)
                ap_dict['tpfp'].extend(tpfp)
                ap_dict['conf'].extend(conf)
                ap_dict['gt_num'].append(gt_num)
            else:
                ap_dict['tpfp'].append(0)
                ap_dict['conf'].append(0)
                ap_dict['gt_num'].append(gt_box.shape[0])
        return ap_dict

    def get_tpfp(self, pred_conf, pred_box, gt_box):
        """
        每次调用只处理一张图片的预测结果，主要功能是判断该张图片中每个预测框为TP还是FP
        :param pred_conf: [M, 1]
        :param pred_box: [M, 4]
        :param gt_box: [N, 4]
        :return:
        """
        assert pred_conf.shape[0] == pred_box.shape[0]
        gt_num = gt_box.shape[0]
        # [M, 4] & [N, 4] -> [M, N]
        ious = iou(pred_box, gt_box)
        tp_num, descend_index = self.make_pr_mask(pred_conf, ious)
        conf = pred_conf[descend_index]
        return tp_num, conf, gt_num

    def make_pr_mask(self, pred_conf, ious):
        """
        每次调用只处理一张图片的预测结果，主要功能是确保每个预测框最多只负责一个gt_box的预测
        :param pred_conf:
        :param pred2gt_mask:
        :return:
        """
        tpfp_list, gt_list = [], []
        descend_index = np.argsort(pred_conf)[::-1]
        for i in descend_index:
            if np.max(ious[i]) >= self.iou_threshold:  # 确保每个预测框最多只能匹配一个gt box
                gt_index = np.argmax(ious[i])
                if gt_index not in gt_list:  #  一个gt box最多只能被一个预测框预测到
                    tpfp_list.append(1)
                    gt_list.append(gt_index)
                else:
                    tpfp_list.append(0)
            else:
                tpfp_list.append(0)
        
        return tpfp_list, descend_index

    @staticmethod
    def compute_pr(ap_dict):
        """
        对得到的tpfp_list按照pred_conf降序排序后，分别计算每个位置的precision和recall
        :param ap_dict:
        :return:
        """
        sorted_order = np.argsort(ap_dict['conf'])[::-1]
        all_gt_num = np.sum(ap_dict['gt_num'])
        ordered_tpfp = np.array(ap_dict['tpfp'])[sorted_order]
        recall = np.cumsum(ordered_tpfp) / all_gt_num
        ones = np.ones_like(recall)
        precision = np.cumsum(ordered_tpfp) / np.cumsum(ones)
        return precision, recall

    def elevenPointInterpolation(self):
        precision_list = []
        recall_thres = np.arange(0, 1.1, 0.1)
        for thres in recall_thres:
            index = np.greater(self.recall, thres)
            if index.sum() > 0:
                precision_list.append(np.max(self.precision[self.recall >= thres]))
            else:
                precision_list.append(0.)
        return np.mean(precision_list)

    def everyPointInterpolation(self):
        last_recall = 0.
        auc = 0.
        for recall in self.recall:
            precision = np.max(self.precision[self.recall >= recall])
            auc += (recall - last_recall) * precision
            last_recall = recall
        return auc


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

        try:
            self.plot_curve(xs, precision, self.save_dir/"Precision.png", "Precision", "Precision-Confidence")
            self.plot_curve(xs, recall, self.save_dir/"Recall.png", "Recall", 'Recall-Conficence')
            self.plot_curve(xs, f1_score, self.save_dir/"F1Score.png", "F1", 'F1 Score-Confidence')
            self.plot_pr_curve(xs, pr, ap, self.save_dir/"PRCurve.png")
        except Exception as err:
            print(err)

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


class ConfusionMatrix:

    def __init__(self, predict, ground_truth, conf_thres, num_class, iou_thres) -> None:
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
                # sum(match) == 1是因为：如果一个gt有与之匹配的预测框，那么该gt能且只能被匹配一次
                if len(match_i) > 0 and sum(match) == 1:  
                    matrix[pred_cls[pred_mi[match]], c] += 1
                else:
                    matrix[-1, c] += 1  # FP(在某一张image的预测中，模型输出了gt cls中没有的类别)

            if len(match_i) == 0:  # 所有的gt都没有与之匹配的预测框，意味着如果模型有输出，那么其输出的pred都是错的
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
    import pickle
    all_gts = pickle.load(open("/pkl/gt_bbox.pkl", 'rb'))
    all_preds = pickle.load(open("/pkl/pred_coco_bbox_640_xlarge.pkl", 'rb'))
    names = pickle.load(open("/pkl/label_names.pkl", 'rb'))['coco']
    names = list(names.values())

    
    mapv2 = mAP_v2(all_gts, all_preds, "/result/curve")
    mapv2.compute_ap_per_class()

    map, map50, mp, mr = mapv2.get_mean_metrics()
    print(f'mAP = {map:.3f}')
    print(f'mAP@0.5 = {map50 * 100:.1f}')
    print(f'mp = {mp * 100:.1f}')
    print(f'mr = {mr * 100:.1f}')

    cm = ConfusionMatrix(all_preds, all_gts, 0.25, len(names), 0.45)
    cm.compute_matrix()
    cm.plot(names, "/utils/confusion-matrix.png")