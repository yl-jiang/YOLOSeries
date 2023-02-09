import sys
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

current_work_directionary = Path('__file__').parent.absolute()
sys.path.insert(0, str(current_work_directionary))

import cv2
import emoji
import torch.cuda
import numpy as np
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

from config import Config
from trainer import YOLOV5Evaluator as Evaluate
from trainer import ExponentialMovingAverageModel
from utils import cv2_save_img, cv2_save_img_plot_pred_gt
from utils import clear_dir
from utils import time_synchronize
from dataset import build_val_dataloader
from utils import mAP_v2
from models import *

from utils import (configure_nccl, configure_omp, get_local_rank,
                   get_rank, get_world_size, occupy_mem, padding, 
                   is_parallel, adjust_status, synchronize, 
                   configure_module, launch)
import torch.distributed as dist
import gc


class Training:

    def __init__(self, anchors, hyp):
        configure_omp()
        configure_nccl()

        # parameters
        self.anchors = anchors
        self.hyp = hyp
        self.select_device()

        # rank, device
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.hyp['device'] = self.device
        self.rank = get_rank()
        self.use_cuda = True if torch.cuda.is_available() else False
        self.is_distributed = get_world_size() > 1

        # current work directory
        self.cwd = Path('./').absolute()
        self.hyp['current_work_dir'] = str(self.cwd)

        self.before_validation()

    def load_dataset(self):
        dataset, dataloader, prefetcher = build_val_dataloader(img_dir=self.hyp['val_img_dir'], 
                                                            lab_dir=self.hyp['val_lab_dir'], 
                                                            name_path=self.hyp['name_path'], 
                                                            input_dim=self.hyp['input_img_size'], 
                                                            aug_hyp=None, 
                                                            cache_num=self.hyp['cache_num'], 
                                                            enable_data_aug=False, 
                                                            seed=self.hyp['random_seed'], 
                                                            batch_size=self.hyp['batch_size'], 
                                                            num_workers=self.hyp['num_workers'], 
                                                            pin_memory=self.hyp['pin_memory'],
                                                            shuffle=False, 
                                                            drop_last=False)

        return dataset, dataloader, prefetcher

    @property
    def select_model(self):
        if self.hyp['model_type'].lower() == "plainsmall":
            return YOLOV5SmallWithPlainBscp
        elif self.hyp['model_type'].lower() == "middle":
            return YOLOV5Middle
        elif self.hyp['model_type'].lower() == "large":
            return YOLOV5Large
        elif self.hyp['model_type'].lower() == "xlarge":
            return YOLOV5XLarge
        elif self.hyp['model_type'].lower() == "smalldw":
            return YOLOV5SmallDW
        elif self.hyp['model_type'].lower() == "middledw":
            return YOLOV5MiddleDW
        elif self.hyp['model_type'].lower() == "largedw":
            return YOLOV5LargeDW
        elif self.hyp['model_type'].lower() == "xlargedw":
            return YOLOV5XLargeDW
        else:
            return YOLOV5Small

    def before_validation(self):
        occupy_mem(self.local_rank)

        # input_dim
        self.hyp['input_img_size'] = padding(self.hyp['input_img_size'], 32)

        # batch_size
        if dist.is_available() and dist.is_initialized():
            self.hyp['batch_size'] = self.hyp['batch_size'] // dist.get_world_size()

        # dataset
        self.val_dataset, self.val_dataloader, self.val_prefetcher = self.load_dataset()

        # update hyper parameters
        self.hyp['num_class'] = self.val_dataset.num_class

        # anchor
        if isinstance(self.anchors, (list, tuple)):
            self.anchors = torch.tensor(self.anchors)  # (3, 3, 2)
        self.anchors = self.anchors.to(self.device)
        anchor_num_per_stage = self.anchors.size(0)  # 3

        # model
        torch.cuda.set_device(self.local_rank)
        model = self.select_model(anchor_num_per_stage, self.val_dataset.num_class)
        model = model.to(self.device)

        # EMA
        if self.hyp['do_ema']:
            self.ema_model = ExponentialMovingAverageModel(model)
        else:
            self.ema_model = None

        # ddp
        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)
        self.model = model

        # load pretrained model
        self.load_model()

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
            org_h, org_w = info[i]['org_shape']
            cur_h, cur_w = inp[i].size(1), inp[i].size(2)

            img = inp[i].permute(1, 2, 0)
            img *= 255.0
            img = np.clip(img, 0, 255.0)
            img = img.numpy().astype(np.uint8)
            img = img[pad_top:(cur_h - pad_bot), pad_left:(cur_w - pad_right), :]
            img = cv2.resize(img, (org_w, org_h), interpolation=0)
            processed_inp.append(img)

            if outputs[i] is None:
                processed_preds.append(None)
                continue
            else:
                pred = outputs[i]
                pred[:, [0, 2]] -= pad_left
                pred[:, [1, 3]] -= pad_top
                pred[:, [0, 1, 2, 3]] /= scale
                pred[:, [0, 2]] = pred[:, [0, 2]].clamp(1, org_w - 1)
                pred[:, [1, 3]] = pred[:, [1, 3]].clamp(1, org_h - 1)
                if self.hyp['use_auxiliary_classifier']:
                    # 将每个预测框中的物体抠出来, 放到一个额外的分类器再进行预测一次是否存在对象
                    pass
                processed_preds.append(pred.cpu().numpy())
            
        return processed_inp, processed_preds

    def select_device(self):
        if self.hyp['device'].lower() != 'cpu':
            if torch.cuda.is_available():
                self.hyp['device'] = 'cuda'
                # region (GPU Tags)
                # 获取当前使用的GPU的属性并打印出来
                gpu_num = torch.cuda.device_count()
                cur_gpu_id = torch.cuda.current_device()
                cur_gpu_name = torch.cuda.get_device_name()
                cur_gpu_properties = torch.cuda.get_device_properties(cur_gpu_id)
                gpu_total_memory = cur_gpu_properties.total_memory
                gpu_major = cur_gpu_properties.major
                gpu_minor = cur_gpu_properties.minor
                gpu_multi_processor_count = cur_gpu_properties.multi_processor_count
                # endregion
                msg = f"Use Nvidia GPU {cur_gpu_name}, find {gpu_num} GPU devices, current device id: {cur_gpu_id}, "
                msg += f"total memory={gpu_total_memory/(2**20):.1f}MB, major={gpu_major}, minor={gpu_minor}, multi_processor_count={gpu_multi_processor_count}"
                print(msg)
            else:
                self.hyp['device'] = 'cpu'

    def load_model(self, map_location='cpu'):
        """
        load pretrained model, EMA model, optimizer(注意: __init_weights()方法并不适用于所有数据集)
        """
        # self._init_bias()
        if self.hyp.get("pretrained_model_path", None):
            model_path = self.hyp["pretrained_model_path"]
            if Path(model_path).exists():
                try:
                    state_dict = torch.load(model_path, map_location=map_location)
                    
                except Exception as err:
                    print(err)
                
                else:
                    if "model_state_dict" not in state_dict:
                        print(f"can't load pretrained model from {model_path}")
    
                    else:  # load pretrained model
                        self.model.load_state_dict(state_dict["model_state_dict"])
                        print(f"use pretrained model {model_path}")

                    if self.ema_model is not None and "ema" in state_dict:  # load EMA model
                        self.ema_model.ema.load_state_dict(state_dict['ema'])
                        print(f"use pretrained EMA model from {model_path}")
                    else:
                        print(f"can't load EMA model from {model_path}")

                    if self.ema_model is not None and 'ema_update_num' in state_dict:
                        self.ema_model.update_num = state_dict['ema_update_num']

                    del state_dict
        else:
            print('training from stratch!')
        
    def gt_bbox_postprocess(self, anns, infoes):
        """
        valdataloader出来的gt bboxes经过了letter resize, 这里将其还原为原始的bboxes
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
                if (self.cwd / "result" / 'pkl' / "voc_emoji_names.pkl").exists():
                    emoji_names = pickle.load(open(str(self.cwd / "result" / 'pkl' / "voc_emoji_names.pkl"), 'rb'))
                    msg_ls = [" ".join([number, emoji_names[name]]) for name, number in zip(ascending_names, ascending_numbers)]
                else:
                    msg_ls = [" ".join([number, name]) for name, number in zip(ascending_names, ascending_numbers)]
            else:
                msg_ls = ["No object has been found!"]
            msg.append(emoji.emojize("; ".join(msg_ls)))
        return msg

    def step(self):
        """
        计算dataloader中所有数据的map
        """
        torch.cuda.empty_cache()
        gc.collect()

        start_t = time_synchronize()
        pred_bboxes, pred_classes, pred_confidences, pred_labels, gt_bboxes, gt_classes = [], [], [], [], [], []
        iters_num = len(self.val_dataloader)

        if self.hyp['do_ema']:
            eval_model = self.ema_model.ema
        else:
            eval_model = self.model
            if is_parallel(eval_model):
                eval_model = eval_model.module

        with adjust_status(eval_model, training=False) as m:
            # validater
            validater = Evaluate(m, self.anchors, self.hyp, compute_metric=True)

            for i in range(iters_num):
                t1 = time_synchronize()
                if self.use_cuda:
                    x = self.val_prefetcher.next()
                else:
                    x = next(self.val_dataloader)

                imgs = x['img']  # (bn, 3, h, w)
                infoes = x['resize_info']

                # gt_bbox: [(M, 4), (N, 4), (P, 4), ...]; gt_cls: [(M,), (N, ), (P, ), ...]
                # coco val2017 dataset中存在有些图片没有对应的gt bboxes的情况
                gt_bbox, gt_cls = self.gt_bbox_postprocess(x['ann'], infoes)
                gt_bboxes.extend(gt_bbox)
                gt_classes.extend(gt_cls)

                imgs = imgs.to(self.hyp['device'])
                if self.hyp['half'] and 'cuda' in self.hyp['device']:
                    imgs = imgs.half()
                
                outputs = validater(imgs)
                # preds: [(X, 6), (Y, 6), (Z, 6), ...]
                imgs, preds = self.preds_postprocess(imgs.cpu(), outputs, infoes)
                t = time_synchronize() - t1

                batch_pred_box, batch_pred_cof, batch_pred_cls, batch_pred_lab = [], [], [], []
                for j in range(len(imgs)):
                    pred_box, pred_cls, pred_cof, pred_lab = [], [], [], []
                    if preds[j] is not None:
                        valid_idx = preds[j][:, 5] >= 0
                        if valid_idx.sum() > 0:
                            pred_box = preds[j][valid_idx, :4]
                            pred_cof = preds[j][valid_idx,  4]
                            pred_cls = preds[j][valid_idx,  5]
                            pred_lab = [self.val_dataset.cls2lab[int(c)] for c in pred_cls]

                    batch_pred_box.append(pred_box)
                    batch_pred_cls.append(pred_cls)
                    batch_pred_cof.append(pred_cof)
                    batch_pred_lab.append(pred_lab)

                obj_msg = self.count_object(batch_pred_lab)
                for k in range(len(imgs)):
                    count = i * self.hyp['batch_size'] + k + 1
                    print(f"[{count:>05} / {len(self.val_dataset)}] ➡️  " + obj_msg[k] + f" ({(t/len(imgs)):.2f}s)")
                    if self.hyp['save_img']:
                        save_path = str(self.cwd / 'result' / 'tmp' / f"{count} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png")
                        if self.hyp['show_gt_bbox']:
                            gt_lab = [self.val_dataset.cls2lab[int(c)] for c in gt_cls[k]]
                            cv2_save_img_plot_pred_gt(imgs[k], batch_pred_box[k], batch_pred_lab[k], batch_pred_cof[k], gt_bbox[k], gt_lab, save_path)
                        else:
                            cv2_save_img(imgs[k], batch_pred_box[k], batch_pred_lab[k], batch_pred_cof[k], save_path)

                del x, imgs, preds, outputs, infoes
                pred_bboxes.extend(batch_pred_box)
                pred_classes.extend(batch_pred_cls)
                pred_confidences.extend(batch_pred_cof)
                pred_labels.extend(batch_pred_lab)

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

            # 如果测试的数据较多, 计算一次mAP需花费较多时间, 这里将结果保存以便后续统计
            if self.hyp['save_pred_bbox']:
                save_path = self.cwd / "result" / "pkl" / f"pred_bbox_{self.hyp['input_img_size'][0]}_{self.hyp['model_type']}.pkl"
                pickle.dump(all_preds, open(str(save_path), 'wb'))
            if self.hyp['save_gt_bbox']:
                pickle.dump(all_gts, open(self.cwd / "result" / "pkl" / "gt_bbox.pkl", "wb"))

            mapv2 = mAP_v2(all_gts, all_preds, self.cwd / "result" / "curve")
            map, map50, mp, mr = mapv2.get_mean_metrics()
            print(f"map={map}, map50={map50}, mp={mp}, mr={mr}")

        del validater, all_preds, all_gts, pred_bboxes, pred_classes, pred_confidences, pred_labels
        
            
@logger.catch
def main(x):
    configure_module()
    
    config_ = Config()
    class Args:
        def __init__(self) -> None:
            self.cfg = "./config/train_yolov5.yaml"
    args = Args()

    anchors = torch.tensor([[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]])
    hyp = config_.get_config(args.cfg, args)
    train = Training(anchors, hyp)
    train.step()


if __name__ == '__main__':
    import os
    
    config_ = Config()

    from utils import launch, get_num_devices
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    num_gpu = get_num_devices()
    clear_dir(str(current_work_directionary / 'result' / 'tmp'))
    launch(
        main, 
        num_gpus_per_machine= num_gpu, 
        num_machines= 1, 
        machine_rank= 0, 
        backend= "nccl", 
        dist_url= "auto", 
        args=(None,),
    )

    
