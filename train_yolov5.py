from pathlib import Path
import sys

current_work_directionary = Path('__file__').parent.absolute()
sys.path.insert(0, str(current_work_directionary))

import cv2
import torch.cuda
from torch.utils.tensorboard import SummaryWriter
from models import Yolov5Small, Yolov5Middle, Yolov5Large, Yolov5SmallWithPlainBscp, Yolov5XLarge
from models import Yolov5SmallDW, Yolov5MiddleDW, Yolov5LargeDW, Yolov5XLargeDW
from tqdm import tqdm
from config import Config
import logging
import numpy as np
from utils import cv2_save_img
from utils import maybe_mkdir, clear_dir
from utils import time_synchronize, summary_model
from datetime import datetime
import random
import torch.nn.functional as F
import math
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp
from loss import YOLOV5Loss
from trainer import ExponentialMovingAverageModel
import torch.optim as optim
from torchnet.meter import AverageValueMeter
from trainer import Evaluate
from torch import nn
from dataset import testdataloader, YoloDataloader
import argparse
import pickle
from utils import mAP_v2, cv2_save_img_plot_pred_gt
from collections import Counter
import emoji
from loguru import logger


class Training:

    def __init__(self, anchors, hyp):
        # parameters
        self.hyp = hyp
        self.select_device()
        self.use_cuda = self.hyp['device'] == 'cuda'
        self.anchors = anchors

        if isinstance(anchors, (list, tuple)):
            self.anchors = torch.tensor(anchors)  # (3, 3, 2)
        self.anchors = self.anchors.to(self.hyp['device'])
        anchor_num_per_stage = self.anchors.size(0)  # 3

        # 确保输入图片的shape必须能够被32整除（对yolov5s而言），如果不满足条件则对设置的输入shape进行调整
        self.hyp['input_img_size'] = self.padding(self.hyp['input_img_size'], 32)
       
        # dataset, scaler, loss_meter
        self.traindataloader, self.traindataset = self.load_dataset(is_training=True)
        self.valdataloader, self.valdataset = self.load_dataset(is_training=False)
        self.testdataloader, self.testdataset = testdataloader(self.hyp['test_data_dir'], self.hyp['input_img_size'])
        self.hyp['num_class'] = self.traindataset.num_class
        self.scaler = amp.GradScaler(enabled=self.use_cuda)  # mix precision training
        self.tot_loss_meter = AverageValueMeter()
        self.box_loss_meter = AverageValueMeter()
        self.cof_loss_meter = AverageValueMeter()
        self.cls_loss_meter = AverageValueMeter()

        # current work path
        if self.hyp['current_work_path'] is None:
            self.cwd = Path('./').absolute()
        else:
            self.cwd = Path(self.hyp['current_work_path'])
        
        self.writer = SummaryWriter(log_dir=str(self.cwd / 'log'))
        self.init_lr = self.hyp['init_lr']
        
        # model, optimizer, loss, lr_scheduler, ema
        self.model = self.select_model(anchor_num_per_stage, self.hyp['num_class']).to(hyp['device'])
        self.optimizer = self._init_optimizer()
        self.optim_scheduler = lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=self._lr_lambda)
        self.loss_fcn = YOLOV5Loss(self.anchors, self.hyp)
        self.ema_model = ExponentialMovingAverageModel(self.model)
        self.validate = Evaluate(self.ema_model.ema, self.anchors, self.hyp)

        # load pretrained model or initialize model's parameters
        self.load_model(False, 'cpu')

        # logger
        self.logger = self._config_logger()
        tbar_tags = ("epoch", "tot", "box", "cof", "cls", "tnum", "imgsz", "lr", 'AP@.5', 'mAP', "time(s)")
        msg = "%10s" * len(tbar_tags)
        print(msg % tbar_tags)

        # cudnn settings
        if not self.hyp['mutil_scale_training'] and self.hyp['device'] == 'cuda':
            # 对于输入数据的维度恒定的网络，使用如下配置可加速训练
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # config warmup step
        if self.hyp['do_warmup']:
            self.hyp['warmup_steps'] = max(self.hyp.get('warmup_epoch', 3) * len(self.traindataloader), 1000)
        self.accumulate = self.hyp['accumulate_loss_step'] / self.hyp['batch_size']

    def load_dataset(self, is_training):
        dataloader, dataset = YoloDataloader(self.hyp, is_training)
        return dataloader, dataset

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
        else:
            return Yolov5Small

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

    def _config_logger(self):
        clear_dir(str(self.cwd / 'log'))  # 再写入log文件前先清空log文件夹
        model_summary = summary_model(self.model, self.hyp['input_img_size'], verbose=False)
        logger = logging.getLogger("SimpleYolov5")
        logger.setLevel(logging.INFO)
        if self.hyp['save_log_txt']:
            if self.hyp.get('log_save_path', None) and Path(self.hyp['log_save_path']).exists():
                txt_log_path = self.hyp['log_save_path']
            else:
                txt_log_path = str(self.cwd / 'log' / 'log.txt')
                maybe_mkdir(Path(txt_log_path).parent)
        else:
            return None
        handler = logging.FileHandler(txt_log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        msg = f"\n{'=' * 70} Model Summary {'=' * 70}\n"
        msg += f"Model Summary:\tlayers {model_summary['number_layers']}; parameters {model_summary['number_params']}; gradients {model_summary['number_gradients']}; flops {model_summary['flops']}GFLOPs"
        msg += f"\n{'=' * 70} Training {'=' * 70}\n"
        logger.info(msg)
        tags = ("all_mem(G)", "cac_mem(G)", "epoch", "step", "batchsz", "img_shape", "tot_loss", "reg_loss",
                "cof_loss", "cls_loss", "use_time(s)", "tar_num", "model_saved")
        logger.info("{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}".format(*tags))
        return logger

    def _lr_lambda(self, epoch, scheduler_type='linear'):
        lr_bias = self.hyp['lr_scheculer_bias']  # lr_bias越大lr的下降速度越慢，整个epoch跑完最后的lr值也越大
        if scheduler_type == 'linear':
            return (1 - epoch / (self.hyp['total_epoch'] - 1)) * (1. - lr_bias) + lr_bias
        elif scheduler_type == 'cosine':
            return ((1 + math.cos(epoch * math.pi / self.hyp['total_epoch'])) / 2) * (1. - lr_bias) + lr_bias  # cosine
        else:
            return math.pow(1 - epoch / self.hyp['total_epoch'], 0.9)

    def _init_optimizer(self):
        param_group_weight, param_group_bias, param_group_other = [], [], []
        for m in self.model.modules():
            if hasattr(m, "bias") and isinstance(m.bias, nn.Parameter):
                param_group_bias.append(m.bias)
            
            if isinstance(m, nn.BatchNorm2d):
                param_group_other.append(m.weight)
            elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
                param_group_weight.append(m.weight)

        if self.hyp['optimizer'].lower() == "sgd":
            optimizer = optim.SGD(params=param_group_other, lr=self.hyp['init_lr'], nesterov=True, momentum=self.hyp['momentum'])
        elif self.hyp['optimizer'].lower() == "adam":
            optimizer = optim.Adam(params=param_group_other, lr=self.hyp['init_lr'], betas=(self.hyp['momentum'], 0.999))
        else:
            RuntimeError(f"Unkown optim_type {self.hyp['optimizer']}")

        optimizer.add_param_group({"params": param_group_weight, "weight_decay": self.hyp['weight_decay']})
        optimizer.add_param_group({"params": param_group_bias})

        del param_group_weight, param_group_bias, param_group_other
        return optimizer

    def _init_bias(self):
        """
        初始化模型参数，主要是对detection layers的bias参数进行特殊初始化，参考RetinaNet那篇论文，这种初始化方法可让网络较容易度过前期训练困难阶段
        （使用该初始化方法可能针对coco数据集有效，在对global wheat数据集的测试中，该方法根本train不起来）
        """
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, (nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True

        detect_layer = [self.model.detect.detect_small,
                        self.model.detect.detect_mid,
                        self.model.detect.detect_large]

        input_img_shape = 128
        stage_outputs = self.model(torch.zeros(1, 3, input_img_shape, input_img_shape).float().to(self.hyp['device']))
        strides = torch.tensor([input_img_shape / x.size(2) for x in stage_outputs])
        class_frequency = None  # 各类别占整个数据集的比例
        for m, stride in zip(detect_layer, strides):
            bias = m.bias.view(self.model.num_anchor, -1)  # (255,) -> (3, 85)
            with torch.no_grad():
                bias[:, 4] += math.log(8 / (self.hyp['input_img_size'][0] / stride) ** 2)
                if class_frequency is None:
                    bias[:, 5:] += math.log(0.6 / (self.model.num_class - 0.99))  # cls
                else:
                    # 类别较多的那一类给予较大的对应bias值，类别较少的那些类给予较小的bias。
                    # 这样做的目的是为了解决类别不平衡问题，因为这种初始化方式只在分类层进行，会使得网络在进行分类预测时，预测到类别较少的那一类case较为容易（因为对应的bias较小，容易在预测这些类别时给出较大的预测值）
                    # 使用这种初始化方式的好处主要是为了解决数据类别不平衡问题造成的早期训练不稳定情况。
                    # 注：这种初始化方法只针对二分类（因为多分类不能针对各个class给予不同的bias）
                    assert isinstance(class_frequency, torch.Tensor), f"class_frequency should be a tensor but we got {type(class_frequency)}"
                    bias[:, 5:] += torch.log(class_frequency / class_frequency.sum())
                m.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)
        del stage_outputs

    @logger.catch
    def step(self):
        self.optimizer.zero_grad()
        tot_loss_before = float('inf')
        map50, map, = 0, 0
        epoch_period = 0
        for epoch in range(self.hyp['total_epoch']):
            self.model.train()
            epoch_start = time_synchronize()
            with tqdm(total=len(self.traindataloader), ncols=160, file=sys.stdout) as tbar:
                for i, x in enumerate(self.traindataloader):
                    start_t = time_synchronize()
                    cur_steps = len(self.traindataloader) * epoch + i + 1
                    ann = x['ann'].to(self.hyp['device'])  # (bn, bbox_num, 6)
                    img = x['img'].to(self.hyp['device'])  # (bn, 3, h, w)
                    img, ann = self.mutil_scale_training(img, ann)
                    batchsz, inp_c, inp_h, inp_w = img.shape

                    # warmup
                    self.warmup(epoch, cur_steps)

                    # forward
                    with amp.autocast(enabled=self.use_cuda):
                        stage_preds = self.model(img)
                        tot_loss, reg_loss, cof_loss, cls_loss, targets_num = self.loss_fcn(stage_preds, ann)

                    # backward
                    self.scaler.scale(tot_loss).backward()

                    # optimize
                    if cur_steps % self.accumulate == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        # maintain a model and update it every time, but it only using for inference
                        if self.hyp['do_ema']:
                            self.ema_model.update(self.model)

                    # tensorboard
                    tot_loss, reg_loss, cof_loss, cls_loss = self.update_loss_meter(tot_loss.item(), reg_loss, cof_loss, cls_loss)
                    is_best = tot_loss < tot_loss_before
                    self.summarywriter(cur_steps, tot_loss, reg_loss, cof_loss, cls_loss, map)
                    tot_loss_before = tot_loss

                    # testing
                    if cur_steps % int(self.hyp.get('validation_every', 0.5)*len(self.traindataloader))== 0:
                        for j, y in enumerate(self.testdataloader):
                            inp = y['img'].to(self.hyp['device'])  # (1, 3, h, w)
                            info = y['resize_info']
                            outputs = self.validate(inp)
                            if outputs:
                                imgs, preds = self.preds_postprocess(inp.cpu(), outputs, info)
                                for k in range(len(imgs)):
                                    save_path = str(self.cwd / 'result' / 'predictions' / f"{j + k} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png")
                                    maybe_mkdir(Path(save_path).parent)
                                    cv2_save_img(imgs[k], preds[k][:, :4], preds[k][:, 5].astype(np.uint8), preds[k][:, 4], save_path)
                                del imgs, preds, outputs
                    
                    # mAP / validation
                    if self.hyp['calculate_map_every'] is not None and cur_steps % int(self.hyp.get('calculate_map_every', 0.5) * len(self.traindataloader))== 0:
                        map, map50, mp, mr = self.calculate_mAP()

                    # verbose
                    if cur_steps % int(self.hyp.get('show_tbar_every', 5)) == 0:
                        self.show_tbar(tbar, epoch+1, i, batchsz, start_t, is_best, tot_loss, reg_loss, cof_loss, cls_loss, targets_num, inp_h, map50, map, epoch_period)

                    # save model
                    if cur_steps % int(self.hyp['save_ckpt_every'] * len(self.traindataloader)) == 0:
                        self.save(tot_loss, epoch+1, cur_steps, True)

                    del img, ann, tot_loss, reg_loss, cof_loss, cls_loss
                    tbar.update()
                tbar.close()

            epoch_period = time_synchronize() - epoch_start
            # update lr
            self.optim_scheduler.step()

        # save the lastest model
        if self.hyp.get('model_save_dir', None) and Path(self.hyp['model_save_dir']).exists():
            save_path = self.hyp['model_save_dir']
        else:
            save_path = str(self.cwd / 'checkpoints' / f'final.pth')
        self.save(tot_loss, 'finally', 'finally', True, save_path)

    def show_tbar(self, tbar, epoch, step, batchsz, start_t, is_best, tot_loss, box_loss, cof_loss, cls_loss, 
                  targets_num, img_shape, elevenPointAP, everyPointAP, epoch_period):
        # tbar
        lrs = [x['lr'] for x in self.optimizer.param_groups]
        if epoch_period == 0.0:  # 不显示第一个epoch的用时
            epoch_period = ""
            tbar_msg = "#  {:^10d}{:^10.3f}{:^10.3f}{:^10.3f}{:^10.3f}{:^10d}{:^10d}{:^10.6f}{:^10.1f}{:^10.1f}{:^10s}"
        else:
            tbar_msg = "#  {:^10d}{:^10.3f}{:^10.3f}{:^10.3f}{:^10.3f}{:^10d}{:^10d}{:^10.6f}{:^10.1f}{:^10.1f}{:^10.1f}"
        values = (epoch, tot_loss, box_loss, cof_loss, cls_loss, targets_num, img_shape, lrs[0], elevenPointAP*100, everyPointAP*100, epoch_period)
        tbar.set_description_str(tbar_msg.format(*values))

        # maybe save info to log.txt
        if self.hyp['device'].lower() == "cuda" and torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated() / 2 ** 30
            cached_memory = torch.cuda.memory_reserved() / 2 ** 30
        else:
            allocated_memory = 0.
            cached_memory = 0.
        if self.logger is not None:
            log_msg = f"{allocated_memory:^15.2f}{cached_memory:^15.2f}{(epoch+1):^15d}{step:^15d}{batchsz:^15d}{str(img_shape):^15s}"
            log_msg += f"{tot_loss:^15.5f}{box_loss:^15.5f}{cof_loss:^15.5f}{cls_loss:^15.5f}"
            period_t = time_synchronize() - start_t
            self.logger.info(log_msg + f"{period_t:^15.1e}" + f"{targets_num:^15d}" + f"{'yes' if is_best else 'no':^15s}")

    def warmup(self, epoch, cur_steps):
        if self.hyp['do_warmup'] and cur_steps < self.hyp["warmup_steps"]:
            self.accumulate = max(1, np.interp(cur_steps,
                                               [0., self.hyp['warmup_steps']],
                                               [1, self.hyp['accumulate_loss_step'] / self.hyp['batch_size']]).round())
            # optimizer有3各param_group，分别是parm_other, param_weight, param_bias
            for j, para_g in enumerate(self.optimizer.param_groups):
                if j != 2:  # param_other and param_weight(该部分参数的learning rate逐渐增大)
                    para_g['lr'] = np.interp(cur_steps,
                                             [0., self.hyp['warmup_steps']],
                                             [0., para_g['initial_lr'] * self._lr_lambda(epoch)])
                else:  # param_bias(该部分参数的learning rate逐渐减小，因为warmup_bias_lr大于initial_lr)
                    para_g['lr'] = np.interp(cur_steps,
                                             [0., self.hyp['warmup_steps']],
                                             [self.hyp['warmup_bias_lr'], para_g['initial_lr'] * self._lr_lambda(epoch)])
                if "momentum" in para_g:  # momentum(momentum在warmup阶段逐渐增大)
                    para_g['momentum'] = np.interp(cur_steps,
                                                   [0., self.hyp['warmup_steps']],
                                                   [self.hyp['warmup_momentum'], self.hyp['momentum']])

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
            img = np.clip(img, 0, 255.0)
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
                processed_preds.append(np.zeros((1, 6)))
            processed_inp.append(img)
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

    def summarywriter(self, steps, tot_loss, reg_loss, cof_loss, cls_loss, l1_reg_loss, map):
        lrs = [x['lr'] for x in self.optimizer.param_groups]
        self.writer.add_scalar(tag='train/tot_loss', scalar_value=tot_loss, global_step=steps)
        self.writer.add_scalar('train/reg_loss', reg_loss, steps)
        self.writer.add_scalar('train/cof_loss', cof_loss, steps)
        self.writer.add_scalar('train/cls_loss', cls_loss, steps)
        self.writer.add_scalar('train/l1_reg_loss', l1_reg_loss, steps)
        self.writer.add_scalar('train/map', map, steps//int(self.hyp['calculate_map_every'] * self.traindataloader))
        self.writer.add_scalar(f'train/{self.hyp["optimizer"]}_lr', lrs[0], steps)

    def mutil_scale_training(self, imgs, targets):
        """
        对传入的imgs和相应的targets进行缩放，从而达到输入的每个batch中image shape不同的目的；
        :param imgs: input image tensor from dataloader / tensor / (bn, 3, h, w)
        :param targets: targets of corrding images / tensor / (bn, bbox_num, 6)
        :return:

        todo：
            随着训练的进行，image size逐渐增大。
        """
        if self.hyp['mutil_scale_training']:
            input_img_size = max(self.hyp['input_img_size'])
            random_shape = random.randrange(input_img_size * 0.5, input_img_size * 1. + 32) // 32 * 32
            scale = random_shape / max(imgs.shape[2:])
            if scale != 1.:
                new_shape = [math.ceil(x * scale / 32) * 32 for x in imgs.shape[2:]]
                imgs = F.interpolate(imgs, size=new_shape, mode='bilinear', align_corners=False)
                targets[:, :, :4] *= scale
        return imgs, targets

    def load_model(self, load_optimizer, map_location):
        """
        load pretrained model, EMA model, optimizer(注意：__init_weights()方法并不适用于所有数据集)
        """
        self._init_bias()
        if self.hyp.get("pretrained_model_path", None):
            model_path = self.hyp["pretrained_model_path"]
            if Path(model_path).exists():
                try:
                    state_dict = torch.load(model_path, map_location=map_location)
                    if "model_state_dict" not in state_dict:
                        print(f"can't load pretrained model from {model_path}")
    
                    else:  # load pretrained model
                        self.model.load_state_dict(state_dict["model_state_dict"])
                        print(f"use pretrained model {model_path}")

                    if load_optimizer and "optim_state_dict" in state_dict and state_dict.get("optim_type", None) == self.hyp['optimizer']:  # load optimizer
                        self.optimizer.load_state_dict(state_dict['optim_state_dict'])
                        print(f"use pretrained optimizer {model_path}")

                    if "ema" in state_dict:  # load EMA model
                        self.ema_model.ema.load_state_dict(state_dict['ema'])
                        print(f"use pretrained EMA model from {model_path}")
                    else:
                        print(f"can't load EMA model from {model_path}")
                    if 'ema_update_num' in state_dict:
                        self.ema_model.update_num = state_dict['ema_update_num']

                    del state_dict

                except Exception as err:
                    print(err)
            else:
                print('training from stratch!')
        else:
            print('training from stratch!')

    def save(self, loss, epoch, step, save_optimizer, save_path=None):
        if self.hyp.get('model_save_dir', None) and Path(self.hyp['model_save_dir']).exists():
            save_path = self.hyp['model_save_dir']
        else:
            save_path = str(self.cwd / 'checkpoints' / f'every_{self.hyp["model_type"]}.pth')            

        if not Path(save_path).exists():
            maybe_mkdir(Path(save_path).parent)

        hyp = {"hyp": self.hyp}

        optim_state = self.optimizer.state_dict() if save_optimizer else None
        state_dict = {
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": optim_state,
            "optim_type": self.hyp['optimizer'], 
            "loss": loss,
            "epoch": epoch,
            "step": step, 
            "ema": self.ema_model.ema.state_dict(), 
            "ema_update_num": self.ema_model.update_num, 
            "hyp": hyp, 
        }
        torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)
        del state_dict, optim_state, hyp

    def update_loss_meter(self, tot_loss, box_loss, cof_loss, cls_loss):
        if not math.isnan(tot_loss):
            self.tot_loss_meter.add(tot_loss)
        if not math.isnan(box_loss):
            self.box_loss_meter.add(box_loss)
        if not math.isnan(cof_loss):
            self.cof_loss_meter.add(cof_loss)
        if not math.isnan(cls_loss):
            self.cls_loss_meter.add(cls_loss)
        return self.tot_loss_meter.value()[0], self.box_loss_meter.value()[0], self.cof_loss_meter.value()[0], self.cls_loss_meter.value()[0]
    
    def gt_bbox_postprocess(self, anns, infoes):
        """
        valdataloader出来的gt bboxes经过了letter resize，这里将其还原为原始的bboxes
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

    def count_and_sort_object(self, pred_lab):
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

    def calculate_mAP(self):
        """
        计算dataloader中所有数据的map
        """
        start_t = time_synchronize()
        pred_bboxes, pred_classes, pred_confidences, pred_labels, gt_bboxes, gt_classes = [], [], [], [], [], []
        for x in self.valdataloader:
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
                    pred_lab = [self.valdataset.cls2lab[int(c)] for c in pred_cls]

                batch_pred_box.append(pred_box)
                batch_pred_cls.append(pred_cls)
                batch_pred_cof.append(pred_cof)
                batch_pred_lab.append(pred_lab)
            del imgs, preds
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

        # 如果测试的数据较多，计算一次mAP需花费较多时间，这里将结果保存以便后续统计
        if self.hyp['save_pred_bbox']:
            save_path = self.cwd / "result" / "pkl" / f"pred_bbox_{self.hyp['input_img_size'][0]}_{self.hyp['model_type']}.pkl"
            pickle.dump(all_preds, open(str(save_path), 'wb'))
            pickle.dump(all_gts, open(self.cwd / "result" / "pkl" / "gt_bbox.pkl", "wb"))

        mapv2 = mAP_v2(all_gts, all_preds, self.cwd / "result" / "curve")
        map, map50, mp, mr = mapv2.get_mean_metrics()
        del all_preds, all_gts, pred_bboxes, pred_classes, pred_confidences, pred_labels
        return map, map50, mp, mr


if __name__ == '__main__':
    config_ = Config()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, required=True,dest='cfg', help='path to config file')
    # parser.add_argument('--batch_size', type=int, default=2, dest='batch_size')
    # parser.add_argument("--input_img_size", default=640, type=int, dest='input_img_size')
    # parser.add_argument('--img_dir', required=True, dest='img_dir', type=str)
    # parser.add_argument('--lab_dir', required=True, dest='lab_dir', type=str)
    # parser.add_argument('--model_save_dir', default="", type=str, dest='model_save_dir')
    # parser.add_argument('--log_save_path', default="", type=str, dest="log_save_path")
    # parser.add_argument('--name_path', required=True, dest='name_path', type=str)
    # parser.add_argument('--aspect_ratio_path', default=None, dest='aspect_ratio_path', type=str, help="aspect ratio list for dataloader sampler, only support serialization object by pickle")
    # parser.add_argument('--cache_num', default=0, dest='cache_num', type=int)
    # parser.add_argument('--total_epoch', default=300, dest='total_epoch', type=int)
    # parser.add_argument('--do_warmup', default=True, type=bool, dest='do_warmup')
    # parser.add_argument('--use_tta', default=True, type=bool, dest='use_tta')
    # parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'], dest='optimizer')
    # parser.add_argument('--iou_threshold', default=0.2, type=float, dest='iou_threshold')
    # parser.add_argument('--conf_threshold', default=0.3, type=float, dest='conf_threshold')
    # parser.add_argument('--cls_threshold', default=0.3, type=float, dest='cls_threshold')
    # parser.add_argument('--agnostic', default=True, type=bool, dest='agnostic', help='whether do NMS among the same class predictions.') 
    # parser.add_argument('--init_lr', default=0.01, type=float, dest='init_lr', help='initialization learning rate')
    # parser.add_argument('--pretrained_model_path',default="", dest='pretrained_model_path') 
    # args = parser.parse_args()

    class Args:
        cfg = "/home/uih/JYL/Programs/YOLO/config/train_yolov5.yaml"
        pretrained_model_path = "/home/uih/JYL/Programs/YOLO_ckpts/yolov5_small_for_voc.pth"
        # lab_dir = '/home/uih/JYL/Dataset/COCO2017/train/label'
        # img_dir = '/home/uih/JYL/Dataset/COCO2017/train/image/'
        # name_path = '/home/uih/JYL/Dataset/COCO2017/train/names.txt'
        train_lab_dir = '/home/uih/JYL/Dataset/VOC/train2012/label'
        train_img_dir = '/home/uih/JYL/Dataset/VOC/train2012/image'
        name_path = '/home/uih/JYL/Dataset/VOC/train2012/names.txt'
        val_img_dir = "/home/uih/JYL/Dataset/VOC/val2012/image"
        val_lab_dir = "/home/uih/JYL/Dataset/VOC/val2012/label"
    args = Args()

    hyp = config_.get_config(args.cfg, args)
    anchors = torch.tensor([[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]])
    train = Training(anchors, hyp)
    train.step()
