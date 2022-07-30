import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

current_work_directionary = Path('__file__').parent.absolute()
sys.path.insert(0, str(current_work_directionary))

import cv2
import emoji
import pickle
import torch.cuda
import numpy as np
from tqdm import tqdm
from loguru import logger

from config import Config
from trainer import Evaluate
from utils import maybe_mkdir
from utils import cv2_save_img
from utils import time_synchronize
from data import YoloDataloader
from trainer import ExponentialMovingAverageModel
from utils import cv2_save_img_plot_pred_gt, ConvBnAct, fuse_conv_bn, summary_model, mAP_v2
from models import Yolov5Small, Yolov5SmallWithPlainBscp, Yolov5Large, Yolov5Middle, Yolov5XLarge
from models import Yolov5SmallDW, Yolov5MiddleDW, Yolov5LargeDW, Yolov5XLargeDW

class Validation:

    def __init__(self, anchors, hyp):
        self.hyp = hyp
        # parameters
        self.select_device()
        self.use_cuda = self.hyp['device'] == 'cuda'
        self.anchors = anchors

        if isinstance(anchors, (list, tuple)):
            self.anchors = torch.tensor(anchors)  # (3, 3, 2)
        self.anchors = self.anchors.to(self.hyp['device'])
        anchor_num_per_stage = self.anchors.size(0)  # 3

        # 确保输入图片的shape必须能够被32整除（对yolov5s而言），如果不满足条件则对设置的输入shape进行调整
        self.hyp['input_img_size'] = self.padding(self.hyp['input_img_size'], 32)
    
        self.testdataset, self.testdataloader = self.load_dataset(False)
        if self.hyp['current_work_dir'] is None:
            self.cwd = Path('./').absolute()
        else:
            self.cwd = Path(self.hyp['current_work_dir'])

        if self.testdataset.num_class == 0:
            num_class = int(input("Please input class num of this dataset: "))
            self.testdataset.num_class = num_class
            self.testdataset.cls2lab = ['lab' for _ in range(num_class)]

        self.hyp['num_class'] = self.testdataset.num_class
        # model, optimizer, loss, lr_scheduler, ema
        self.model = self.select_model(anchor_num_per_stage, self.testdataset.num_class).to(self.hyp['device'])
        self.ema_model = ExponentialMovingAverageModel(self.model)

        self.load_model('cpu')
        if self.hyp['ema_model']:
            del self.model
            self.fuse_conv_bn(self.ema_model.ema)
            if self.hyp['half'] and self.hyp['device'] == 'cuda':
                self.ema_model.ema = self.ema_model.ema.half()    
            self.validate = Evaluate(self.ema_model.ema,  self.anchors, self.hyp)
        else:
            del self.ema_model
            self.fuse_conv_bn(self.model)
            if self.hyp['half'] and self.hyp['device'] == 'cuda':
                self.model = self.model.half()
            self.validate = Evaluate(self.model,  self.anchors, self.hyp)
        
    def load_dataset(self, is_training):
        dataloader, dataset = YoloDataloader(self.hyp, is_training=is_training)
        return dataset, dataloader

    @property
    def select_model(self):
        if self.hyp['model_type'].lower() == "plainsmall":
            return Yolov5SmallWithPlainBscp
        elif self.hyp['model_type'].lower() == "middle":
            return Yolov5Middle
        elif self.hyp['model_type'].lower() == "large":
            return Yolov5Large
        elif self.hyp['model_type'].lower() == "xlarge":
            return Yolov5XLarge
        elif self.hyp['model_type'].lower() == "smalldw":
            return Yolov5SmallDW
        elif self.hyp['model_type'].lower() == "middledw":
            return Yolov5MiddleDW
        elif self.hyp['model_type'].lower() == "largedw":
            return Yolov5LargeDW
        elif self.hyp['model_type'].lower() == "xlargedw":
            return Yolov5XLargeDW
        elif self.hyp['model_type'].lower() == "small":
            return Yolov5Small
        else:
            raise ValueError(f'unknown model type "{self.hyp["model_type"]}"')

    def fuse_conv_bn(self, model):
        summary_model(model, verbose=True, prefix="Before Fuse Conv and Bn\t")
        for m in model.modules():
            if isinstance(m, ConvBnAct) and hasattr(m, 'bn'):
                m.conv = fuse_conv_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        summary_model(model, verbose=True, prefix="After Fuse Conv and Bn\t")
        
    @staticmethod
    def padding(hw, factor=32):
        h, w = hw
        h_mod = h % factor
        w_mod = w % factor
        if h_mod > 0:
            h = (h // factor + 1) * factor
        if w_mod > 0:
            w = (w // factor + 1) * factor
        return h, w

    def preds_postprocess(self, inp, outputs, info):
        """

        :param inp: normalization image
        :param outputs:
        :param info:
        :return:
        """
        processed_preds = []
        processed_inp = []
        for i in range(len(outputs)):
            scale, pad_top, pad_left = info[i]['scale'], info[i]['pad_top'], info[i]['pad_left']
            pad_bot, pad_right = info[i]['pad_bottom'], info[i]['pad_right']
            pred = outputs[i]
            org_h, org_w = info[i]['org_shape']
            cur_h, cur_w = inp[i].size(1), inp[i].size(2)

            img = inp[i].permute(1, 2, 0)
            img *= 255.0
            img = img.numpy().astype(np.uint8)
            img = img[pad_top:(cur_h - pad_bot), pad_left:(cur_w - pad_right), :]
            img = cv2.resize(img, (org_w, org_h), interpolation=0)

            if pred is not None and pred.size(0) > 0:
                pred[:, [0, 2]] -= pad_left
                pred[:, [1, 3]] -= pad_top
                pred[:, [0, 1, 2, 3]] /= scale
                pred[:, [0, 2]] = pred[:, [0, 2]].clamp(1, org_w - 1)
                pred[:, [1, 3]] = pred[:, [1, 3]].clamp(1, org_h - 1)
                if self.hyp['use_auxiliary_classifier']:
                    # 将每个预测框中的物体抠出来，放到一个额外的分类器再进行预测一次是否存在对象
                    pass
                processed_preds.append(pred.cpu().numpy())
            else:
                processed_preds.append(np.ones((1, 6)) * -1.)
            processed_inp.append(img)
        return processed_inp, processed_preds

    def select_device(self):
        if self.hyp['device'].lower() != 'cpu':
            if torch.cuda.is_available():
                self.hyp['device'] = 'cuda'
            else:
                self.hyp['device'] = 'cpu'

    def load_model(self, map_location):
        """
        尝试load pretrained模型，如果失败则退出程序
        """
        if self.hyp["pretrained_model_path"] is not None:
            model_path = Path(self.hyp["pretrained_model_path"]).resolve()
            if Path(model_path).exists():
                try:
                    state_dict = torch.load(model_path, map_location=map_location)
                    if "model_state_dict" not in state_dict:
                        raise ValueError("not found model's state_dict in this file, load model failed!")
                    else:  # load training model
                        self.model.load_state_dict(state_dict["model_state_dict"])
                        print(f"successful load pretrained model {model_path}")

                    if "ema" in state_dict:  # load EMA model
                        self.ema_model.ema.load_state_dict(state_dict['ema'])
                        print(f"successful load pretrained EMA model {model_path}")
                    else:
                        self.hyp['ema_model'] = False
                        print(f"load EMA model falied, use plain model instead!")
                    del state_dict
                except Exception as err:
                    print(f"Error\t➡️ {err}")
    
    def gt_bbox_postprocess(self, anns, infoes):
        """
        testdataloader出来的gt bboxes经过了letter resize，这里将其还原到原始的bboxes

        :param: anns: dict
        """
        ppb = []  # post processed bboxes
        ppc = []  # post processed classes
        for i in range(anns.shape[0]):
            scale, pad_top, pad_left = infoes[i]['scale'], infoes[i]['pad_top'], infoes[i]['pad_left']
            valid_idx = anns[i][:, 4] >= 0
            ann_valid = anns[i][valid_idx]
            ann_valid[:, [0, 2]] -= pad_left
            ann_valid[:, [1, 3]] -= pad_top
            ann_valid[:, :4] /= scale
            ppb.append(ann_valid[:, :4].cpu().numpy())
            ppc.append(ann_valid[:, 4].cpu().numpy().astype('uint16'))
        return ppb, ppc

    @logger.catch
    def predtict(self):
        """
        测试testdataloader中的所有图片并将结果保存到磁盘
        """
        for i, x in enumerate(self.testdataloader):
            imgs = x['imgs']  # (bn, 3, h, w)
            infoes = x['resize_infoes']
            try:
                gt_bbox, gt_cls = self.gt_bbox_postprocess(x['anns'], infoes)
            except Exception as err:
                gt_bbox, gt_cls = None, None
            imgs = imgs.to(self.hyp['device'])
            if self.hyp['half'] and self.hyp['device'] == 'cuda':
                imgs = imgs.half()
            outputs = self.validate(imgs)
            imgs, preds = self.preds_postprocess(imgs.cpu(), outputs, infoes)
            pred_cls = [preds[j][:, 5] for j in range(len(imgs))]

            if self.hyp['save_img']:
                for k in range(len(imgs)):
                    save_path = str(self.cwd / 'result' / 'tmp' / f"{i * self.hyp['batch_size'] + k} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png")
                    maybe_mkdir(Path(save_path).parent)
                    pred_lab = [self.testdataset.cls2lab[int(c)] for c in pred_cls[k]]
                    if gt_cls is not None:
                        gt_lab = [self.testdataset.cls2lab[int(c)] for c in gt_cls[k]]
                    if self.hyp['show_gt_bbox'] and gt_bbox is not None:
                        cv2_save_img_plot_pred_gt(imgs[k], preds[k][:, :4], pred_lab, preds[k][:, 4], gt_bbox[k], gt_lab, save_path)
                    else:
                        cv2_save_img(imgs[k], preds[k][:, :4], pred_lab, preds[k][:, 4], save_path)
            del imgs, preds

    def count_object(self, pred_lab):
        """
        按照object的个数降序输出

        :param pred_lab: [(X, ), (Y, ), (Z, ), ...]
        """
        msg = []
        for lab in pred_lab:
            counter = Counter(lab)
            names, numbers = [], []
            for nam, num in counter.items():
                names.append(nam)
                numbers.append(str(num))
            sort_index = np.argsort([int(i) for i in numbers])[::-1]
            ascending_numbers = [numbers[i] for i in sort_index]
            ascending_names = [names[i] for i in sort_index]
            if len(numbers) > 0:
                if (self.cwd / "result" / 'pkl' / "emoji_names.pkl").exists():
                    coco_emoji = pickle.load(open(str(self.cwd / "result" / 'pkl' / "emoji_names.pkl"), 'rb'))
                    msg_ls = [" ".join([number, coco_emoji[name]]) for name, number in zip(ascending_names, ascending_numbers)]
                else:
                    msg_ls = [" ".join([number, name]) for name, number in zip(ascending_names, ascending_numbers)]
            else:
                msg_ls = ["No object has been found!"]
            msg.append(emoji.emojize("; ".join(msg_ls)))
        return msg

    def calculate_mAP(self):
        """
        计算testdataloader中所有数据的map
        """
        start_t = time_synchronize()
        pred_bboxes, pred_classes, pred_confidences, pred_labels, gt_bboxes, gt_classes = [], [], [], [], [], []
        for i, x in enumerate(self.testdataloader):
            imgs = x['img']  # (bn, 3, h, w)
            infoes = x['resize_info']

            # gt_bbox: [(M, 4), (N, 4), (P, 4), ...]; gt_cls: [(M,), (N, ), (P, ), ...]
            # coco val2017 dataset中存在有些图片没有对应的gt bboxes的情况
            gt_bbox, gt_cls = self.gt_bbox_postprocess(x['ann'], infoes)
            gt_bboxes.extend(gt_bbox)
            gt_classes.extend(gt_cls)

            # 统计预测一个batch需要花费的时间
            t1 = time_synchronize()
            imgs = imgs.to(self.hyp['device'])
            if self.hyp['half'] and self.hyp['device'] == 'cuda':
                imgs = imgs.half()
            outputs = self.validate(imgs)
            # preds: [(X, 6), (Y, 6), (Z, 6), ...]
            imgs, preds = self.preds_postprocess(imgs.cpu(), outputs, infoes)
            t = time_synchronize() - t1

            batch_pred_box, batch_pred_cof, batch_pred_cls, batch_pred_lab = [], [], [], []
            for j in range(len(imgs)):
                valid_idx = preds[j][:, 5] >= 0
                if valid_idx.sum() == 0:
                    pred_box, pred_cls, pred_cof, pred_lab = [], [], [], []
                else:
                    pred_box = preds[j][valid_idx, :4]
                    pred_cof = preds[j][valid_idx, 4]
                    pred_cls = preds[j][valid_idx, 5]
                    pred_lab = [self.testdataset.cls2lab[int(c)] for c in pred_cls]

                batch_pred_box.append(pred_box)
                batch_pred_cls.append(pred_cls)
                batch_pred_cof.append(pred_cof)
                batch_pred_lab.append(pred_lab)

            pred_bboxes.extend(batch_pred_box)
            pred_classes.extend(batch_pred_cls)
            pred_confidences.extend(batch_pred_cof)
            pred_labels.extend(batch_pred_lab)
            
            obj_msg = self.count_object(batch_pred_lab)
            
            for k in range(len(imgs)):
                count = i * len(imgs) + k + 1
                print(f"[{count:>05}/{len(self.testdataset)}] ➡️ " + obj_msg[k] + f" ({(t/len(imgs)):.2f}s)")
                if self.hyp['save_img']:
                    save_path = str(self.cwd / 'result' / 'tmp' / f"{i * self.hyp['batch_size'] + k} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png")
                    if self.hyp['show_gt_bbox']:
                        gt_lab = [self.testdataset.cls2lab[int(c)] for c in gt_cls[k]]
                        cv2_save_img_plot_pred_gt(imgs[k], batch_pred_box[k], batch_pred_lab[k], batch_pred_cof[k], gt_bbox[k], gt_lab, save_path)
                    else:
                        cv2_save_img(imgs[k], batch_pred_box[k], batch_pred_lab[k], batch_pred_cof[k], save_path)
            del imgs, preds

        total_use_time = time_synchronize() - start_t

        all_preds = []
        for pred_box, pred_cof, pred_cls in zip(pred_bboxes, pred_confidences, pred_classes):
            if len(pred_box) == 0:
                all_preds.append(np.zeros((0, 6)))
            else:
                all_preds.append(np.concatenate((pred_box, pred_cof[:, None], pred_cls[:, None]), axis=1))
        
        all_gts = []
        for gt_box, gt_cls in zip(gt_bboxes, gt_classes):
            all_gts.append(np.concatenate((gt_box, gt_cls[:, None]), axis=1))

        # 如果测试的数据较多，计算一次mAP需花费较多时间，这里将结果保存以便后续统计
        if self.hyp['save_pred_bbox']:
            save_path = self.cwd / "result" / "pkl" / f"pred_bbox_{self.hyp['input_img_size'][0]}_{self.hyp['model_type']}.pkl"
            pickle.dump(all_preds, open(str(save_path), 'wb'))
            pickle.dump(all_gts, open(self.cwd / "result" / "pkl" / "gt_bbox.pkl", "wb"))

        mapv2 = mAP_v2(all_gts, all_preds, self.cwd / "result" / "curve")
        map, map50, mp, mr = mapv2.get_mean_metrics()
        print(f"use time: {total_use_time:.2f}s")
        print(f'mAP = {map * 100:.3f}')
        print(f'mAP@0.5 = {map50 * 100:.1f}')
        print(f'mp = {mp * 100:.1f}')
        print(f'mr = {mr * 100:.1f}')


if __name__ == '__main__':
    config_ = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, dest='cfg', help='path to config file')
    parser.add_argument('--val_img_dir', required=True, dest='val_img_dir', type=str)
    parser.add_argument('--val_lab_dir', required=True, dest='val_lab_dir', type=str)
    parser.add_argument('--pretrained_model_path', required=True, dest='pretrained_model_path', type=str)
    parser.add_argument('--model_type', required=True, dest='model_type', type=str)
    parser.add_argument('--name_path', required=True, dest='name_path', type=str)
    parser.add_argument('--batch_size', default=8, dest='name_path', type=str)
    args = parser.parse_args()
    
    # # ======================================================================
    # class Args:
    #     cfg = "/home/uih/JYL/Programs/YOLO/config/validation.yaml"
    #     val_lab_dir = '/home/uih/JYL/Dataset/VOC/val2012/label'
    #     val_img_dir = '/home/uih/JYL/Dataset/VOC/val2012/image/'
    #     name_path = '/home/uih/JYL/Dataset/VOC/val2012/names.txt'
    #     pretrained_model_path = "/home/uih/JYL/Programs/YOLO_ckpts/yolov5_small_for_voc.pth"
    # args = Args()
    # # ======================================================================

    hyp = config_.get_config(args.cfg, args)
    anchors = torch.tensor([[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]])
    val = Validation(anchors, hyp)
    val.calculate_mAP()
