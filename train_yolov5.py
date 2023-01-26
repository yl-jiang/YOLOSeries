import sys
import math
import pickle
import random
import logging
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
from torch import nn
from tqdm import tqdm
from loguru import logger
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary as ModelSummary
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

from config import Config
from loss import YOLOV5Loss
from trainer import Evaluate
from utils import cv2_save_img
from utils import maybe_mkdir, clear_dir
from trainer import ExponentialMovingAverageModel
from utils import time_synchronize, summary_model
from dataset import build_dataloader, build_test_dataloader
from utils import mAP_v2, catch_warnnings
from models import Yolov5Small, Yolov5Middle, Yolov5Large, Yolov5SmallWithPlainBscp, Yolov5XLarge
from models import Yolov5SmallDW, Yolov5MiddleDW, Yolov5LargeDW, Yolov5XLargeDW

from utils import (configure_nccl, configure_omp, get_local_rank, print_config, 
                   get_rank, get_world_size, occupy_mem, padding, MeterBuffer, 
                   all_reduce_norm, is_parallel, adjust_status, synchronize, 
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

        self.before_train()

    def load_dataset(self, data_type='train'):
        if data_type == 'train':  # training set
            dataset, dataloader, prefetcher = build_dataloader(img_dir=self.hyp['train_img_dir'], 
                                                               lab_dir=self.hyp['train_lab_dir'], 
                                                               name_path=self.hyp['name_path'], 
                                                               input_dim=self.hyp['input_img_size'], 
                                                               aug_hyp=self.hyp, 
                                                               cache_num=self.hyp['cache_num'], 
                                                               enable_data_aug=self.hyp['enable_data_aug'], 
                                                               seed=self.hyp['random_seed'], 
                                                               batch_size=self.hyp['batch_size'], 
                                                               num_workers=self.hyp['num_workers'], 
                                                               pin_memory=self.hyp['pin_memory'],
                                                               shuffle=self.hyp['shuffle'], 
                                                               drop_last=self.hyp['drop_last'])
        elif data_type == 'val':  # validation set
            dataset, dataloader, prefetcher = build_dataloader(img_dir=self.hyp['val_img_dir'], 
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
        elif data_type == 'test':
            dataset, dataloader, prefetcher = build_test_dataloader(img_dir=self.hyp['test_img_dir'], 
                                                                    input_dim=self.hyp['input_img_size'], 
                                                                    batch_size=1)
        else:
            raise ValueError(f"unknow data_type '{data_type}'")
        return dataset, dataloader, prefetcher

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

    def _init_logger(self):
        # clear_dir(str(self.cwd / 'log'))  # 再写入log文件前先清空log文件夹
        model_summary = summary_model(self.model, self.hyp['input_img_size'], verbose=True)
        logger = logging.getLogger(f"Yolov5_rank_{self.rank}")
        formated_config = print_config(self.hyp)  # record training parameters in log.txt
        logger.setLevel(logging.INFO)
        txt_log_path = str(self.cwd / 'log' / f'log_rank_{self.rank}' / f'log_{self.model.__class__.__name__}_{datetime.now().strftime("%Y%m%d-%H:%M:%S")}_{self.hyp["log_identifier"]}.txt')
        maybe_mkdir(Path(txt_log_path).parent)
        handler = logging.FileHandler(txt_log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info("\n" + formated_config)
        msg = f"\n{'=' * 70} Model Summary {'=' * 70}"
        msg += f"\nModel Summary:\tlayers {model_summary['number_layers']}; parameters {model_summary['number_params']}; gradients {model_summary['number_gradients']}; flops {model_summary['flops']}GFLOPs"
        msg += f"\n{'=' * 70} Training {'=' * 70}\n"
        logger.info(msg)
        return logger

    def _init_scheduler(self):
        if self.hyp['scheduler_type'].lower() == "onecycle":   # onecycle lr scheduler
            scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01, epochs=self.hyp['total_epoch'], steps_per_epoch=len(self.train_dataloader), three_phase=True)
        elif self.hyp['scheduler_type'].lower() == 'linear':  # linear lr scheduler
            lr_max_ds_scale = self.hyp['lr_max_ds_scale']
            linear_lr = lambda epoch: (1 - epoch / (self.hyp['total_epoch'] - 1)) * (1. - lr_max_ds_scale) + lr_max_ds_scale  # lr_bias越大lr的下降速度越慢,整个epoch跑完最后的lr值也越大
            scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_lr)
        else:  # consin lr scheduler
            lr_max_ds_scale = self.hyp['lr_max_ds_scale']  # 整个训练过程中lr的最小值等于: max_ds_rate * init_lr
            cosin_lr = lambda epoch: ((1 + math.cos(epoch * math.pi / self.hyp['total_epoch'])) / 2) * (1. - lr_max_ds_scale) + lr_max_ds_scale  # cosine
            scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=cosin_lr)
        return scheduler

    def before_train(self):
        occupy_mem(self.local_rank)

        # input_dim
        self.hyp['input_img_size'] = padding(self.hyp['input_img_size'], 32)

        # cudnn
        if not self.hyp['mutil_scale_training']:
            # 对于输入数据的维度恒定的网络, 使用如下配置可加速训练
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # batch_size
        if dist.is_available() and dist.is_initialized():
            self.hyp['batch_size'] = self.hyp['batch_size'] // dist.get_world_size()

        # init lr
        self.hyp['lr'] = self.hyp['basic_lr_per_img'] * self.hyp['batch_size'] 

        # dataset
        self.train_dataset, self.train_dataloader, self.train_prefetcher = self.load_dataset('train')
        self.val_dataset  , self.val_dataloader  , self.val_prefetcher   = self.load_dataset('val')
        self.test_dataset , self.test_dataloader , self.test_prefetcher  = self.load_dataset('test')

        # update hyper parameters
        self.hyp['num_class'] = self.train_dataset.num_class

        # mix precision training
        self.scaler = amp.GradScaler(enabled=self.use_cuda)  

        # anchor
        if isinstance(self.anchors, (list, tuple)):
            self.anchors = torch.tensor(self.anchors)  # (3, 3, 2)
        self.anchors = self.anchors.to(self.device)
        anchor_num_per_stage = self.anchors.size(0)  # 3

        # model
        torch.cuda.set_device(self.local_rank)
        model = self.select_model(anchor_num_per_stage, self.train_dataset.num_class)
        ModelSummary(model, input_size=(1, 3, self.hyp['input_img_size'][0], self.hyp['input_img_size'][1]), device=next(model.parameters()).device)

        # summarywriter
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=str(self.cwd / 'log' / f'log_rank_{self.rank}'))

        model = model.to(self.device)

        # maybe load pretrained model
        self.load_model(False, 'cpu')

        # ddp
        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        # loss
        self.loss_fcn = YOLOV5Loss(self.anchors, self.hyp)

        # EMA
        if self.hyp['do_ema']:
            self.ema_model = ExponentialMovingAverageModel(model)

        self.model = model

        # optimizer
        self.optimizer = self._init_optimizer()

        # lr_scheduler
        # self.lr_scheduler = lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=self._lr_lambda)
        self.lr_scheduler = self._init_scheduler()

        # logger
        self.logger = self._init_logger()

        # meters
        self.meter = MeterBuffer()

        # warmup step
        if self.hyp['do_warmup']:
            self.hyp['warmup_steps'] = max(self.hyp.get('warmup_epoch', 3) * len(self.train_dataloader), 1000)

        # accumulate step
        self.accumulate = self.hyp['accumulate_loss_step'] / self.hyp['batch_size']
        

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
            optimizer = optim.SGD(params=param_group_other, lr=self.hyp['lr'], nesterov=True, momentum=self.hyp['momentum'])
        elif self.hyp['optimizer'].lower() == "adam":
            optimizer = optim.Adam(params=param_group_other, lr=self.hyp['lr'], betas=(self.hyp['momentum'], 0.999))
        else:
            RuntimeError(f"Unkown optim_type {self.hyp['optimizer']}")

        optimizer.add_param_group({"params": param_group_weight, "weight_decay": self.hyp['weight_decay']})
        optimizer.add_param_group({"params": param_group_bias})

        del param_group_weight, param_group_bias, param_group_other
        return optimizer

    def _init_bias(self):
        """
        初始化模型参数, 主要是对detection layers的bias参数进行特殊初始化, 参考RetinaNet那篇论文, 这种初始化方法可让网络较容易度过前期训练困难阶段
        (使用该初始化方法可能针对coco数据集有效, 在对global wheat数据集的测试中, 该方法根本train不起来)
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
        stage_outputs = self.model(torch.zeros(1, 3, input_img_shape, input_img_shape, device=self.hyp['device']).float())
        strides = torch.tensor([input_img_shape / x.size(2) for x in stage_outputs])
        class_frequency = None  # 各类别占整个数据集的比例
        for m, stride in zip(detect_layer, strides):
            bias = m.bias.view(self.model.num_anchor, -1)  # (255,) -> (3, 85)
            with torch.no_grad():
                bias[:, 4] += math.log(8 / (self.hyp['input_img_size'][0] / stride) ** 2)
                if class_frequency is None:
                    bias[:, 5:] += math.log(0.6 / (self.model.num_class - 0.99))  # cls
                else:
                    # 类别较多的那一类给予较大的对应bias值, 类别较少的那些类给予较小的bias。
                    # 这样做的目的是为了解决类别不平衡问题, 因为这种初始化方式只在分类层进行, 会使得网络在进行分类预测时, 预测到类别较少的那一类case较为容易（因为对应的bias较小, 容易在预测这些类别时给出较大的预测值）
                    # 使用这种初始化方式的好处主要是为了解决数据类别不平衡问题造成的早期训练不稳定情况。
                    # 注：这种初始化方法只针对二分类（因为多分类不能针对各个class给予不同的bias）
                    assert isinstance(class_frequency, torch.Tensor), f"class_frequency should be a tensor but we got {type(class_frequency)}"
                    bias[:, 5:] += torch.log(class_frequency / class_frequency.sum())
                m.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)
        del stage_outputs

    @logger.catch
    @catch_warnnings
    def step(self):
        self.model.zero_grad()
        tot_loss_before = float('inf')
        one_epoch_iters = len(self.train_dataloader)
        for cur_epoch in range(self.hyp['total_epoch']):
            torch.cuda.empty_cache()
            self.model.train()
            start_epoch_t = time_synchronize()
            for i in range(one_epoch_iters):
                start_iter = time_synchronize()
                if self.use_cuda:
                    x = self.train_prefetcher.next()
                else:
                    x = next(self.train_dataloader)
                end_data_t = time_synchronize()
                step_in_epoch = i + 1
                step_in_total = one_epoch_iters * cur_epoch + i + 1
                ann = x['ann'].to(self.hyp['device'])  # (bn, bbox_num, 6)
                img = x['img'].to(self.hyp['device'])  # (bn, 3, h, w)
                img, ann = self.mutil_scale_training(img, ann)

                # warmup
                self.warmup(step_in_total)

                # forward
                my_context = self.model.no_sync if self.is_distributed and step_in_epoch % self.accumulate != 0 else nullcontext
                with my_context():
                    with amp.autocast(enabled=self.use_cuda):
                        stage_preds = self.model(img)
                        loss_dict = self.loss_fcn(stage_preds, ann)
                        loss_dict['tot_loss'] *= get_world_size()

                tot_loss = loss_dict['tot_loss']

                # backward
                self.scaler.scale(tot_loss).backward()
                end_iter_t = time_synchronize()
                loss_dict.update({'tot_loss': tot_loss.detach().item()})

                # optimize
                if step_in_epoch % self.accumulate == 0:
                    # self.scaler.unscale_(self.optimizer)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    # maintain a model and update it every time, but it only using for inference
                    if self.hyp['do_ema']:
                        self.ema_model.update(self.model)

                data_time = end_data_t - start_iter
                iter_time = end_iter_t - start_iter
                is_best = tot_loss < tot_loss_before
                tot_loss_before = tot_loss.item()

                self.update_meter(cur_epoch, step_in_epoch, step_in_total, img.size(2), img.size(0), iter_time, data_time, loss_dict, is_best)
                self.update_summarywriter()
                self.update_logger(step_in_total)
                self.save_model(cur_epoch, step_in_epoch, loss_dict, True)
                self.test(step_in_total)
                self.calculate_metric(step_in_total)

                # onecycle scheduler需要在iter尺度update
                if self.hyp['scheduler_type'].lower() == "onecycle":
                    self.lr_scheduler.step()  # 因为self.accumulate会从1开始增长, 因此第一次执行训练时self.optimizer.step()一定会在self.lr_scheduler.step()之前被执行
                
                del x, img, ann, tot_loss, stage_preds, loss_dict
            # 其余scheduler在epoch尺度update
            if self.hyp['scheduler_type'].lower() != "onecycle":
                self.lr_scheduler.step()
            gc.collect()
            epoch_time = time_synchronize() - start_epoch_t
            self.logger.info(f'\n\n{"-" * 600}\n')


    def update_logger(self, step_in_total):
        tags = ('percentage', "tot_loss", "iou_loss", 'cof_loss'  , 'cls_loss'  , "accumulate", "iter_time", 'data_time', "lr"  , "cur_epoch", "step_in_epoch", "batch_size", "input_dim", "allo_mem", "cach_mem")
        fmts = ('3.2%'      , '5.3f'    , '5.3f'    , '>5.3f'     , '>5.3f'     , '>02d'      , '5.3f'     , '5.3f'     , '5.3e', '>04d'     , '>05d'    , '>02d'      , '>03d'     , '5.3f'    ,  '5.3f')
        if step_in_total % self.hyp['save_log_every'] == 0:
            show_dict = {}
            for tag in tags:
                show_dict[tag] = self.meter.get_filtered_meter(tag)[tag].latest
            if not math.isnan(show_dict['tot_loss']):
                log_msg = ''
                for tag, fmt in zip(tags, fmts):
                    log_msg += tag + '=' + '{' + tag +  ':' + fmt + '}' + ", "
                self.logger.info(log_msg.format(**show_dict))

    def update_meter(self, cur_epoch, step_in_epoch, step_in_total, input_dim, batch_size, iter_time, data_time, loss_dict, is_best):
        self.meter.update(iter_time  = float(iter_time), 
                          data_time  = float(data_time), 
                          input_dim  = int(input_dim),
                          percentage = float(step_in_epoch / len(self.train_dataloader)),
                          step_in_epoch   = int(step_in_epoch), 
                          step_in_total   = int(step_in_total), 
                          cur_epoch  = int(cur_epoch), 
                          batch_size = int(batch_size),
                          is_best    = is_best, 
                          accumulate = int(self.accumulate),   
                          allo_mem   = torch.cuda.memory_allocated() / 2 ** 30 if self.use_cuda else 0.0,
                          cach_mem   = torch.cuda.memory_reserved() / 2 ** 30  if self.use_cuda else 0.0,
                          lr         = float([x['lr'] for x in self.optimizer.param_groups][0]), 
                          mp     = self.meter.get_filtered_meter('mp')['mp'].global_avg if len(self.meter.get_filtered_meter('mp')) > 0 else 0.0, 
                          map50  = self.meter.get_filtered_meter('map50')['map50'].global_avg if len(self.meter.get_filtered_meter('map50')) > 0 else 0.0, 
                          **loss_dict)

    def warmup(self, step_in_total):
        if self.hyp['do_warmup'] and step_in_total < self.hyp["warmup_steps"]:
            self.accumulate = max(1, np.interp(step_in_total,
                                               [0., self.hyp['warmup_steps']],
                                               [1, self.hyp['accumulate_loss_step'] / self.hyp['batch_size'] / get_world_size()]).round())

            # optimizer有3各param_group, 分别是parm_other, param_weight, param_bias
            for j, para_g in enumerate(self.optimizer.param_groups):
                if j != 2:  # param_other and param_weight(该部分参数的learning rate逐渐增大)
                    para_g['lr'] = np.interp(step_in_total,
                                             [0., self.hyp['warmup_steps']],
                                             [0., para_g['initial_lr']])
                else:  # param_bias(该部分参数的learning rate逐渐减小, 因为warmup_bias_lr大于initial_lr)
                    para_g['lr'] = np.interp(step_in_total,
                                             [0., self.hyp['warmup_steps']],
                                             [self.hyp['warmup_bias_max_lr'], para_g['initial_lr']])
                if "momentum" in para_g:  # momentum(momentum在warmup阶段逐渐增大)
                    para_g['momentum'] = np.interp(step_in_total,
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

    def update_summarywriter(self):
        if self.rank == 0 and self.hyp['enable_tensorboard']:
            step_in_total = self.meter.get_filtered_meter('step_in_total')['step_in_total'].latest
            for k, v in self.meter.items():
                self.writer.add_scalar(tag=f'train/{k}', scalar_value=v.latest, global_step=step_in_total)

    def mutil_scale_training(self, imgs, targets):
        """
        对传入的imgs和相应的targets进行缩放, 从而达到输入的每个batch中image shape不同的目的;
        :param imgs: input image tensor from dataloader / tensor / (bn, 3, h, w)
        :param targets: targets of corrding images / tensor / (bn, bbox_num, 6)
        :return:

        todo: 
            随着训练的进行, image size逐渐增大。
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
        load pretrained model, EMA model, optimizer(注意: __init_weights()方法并不适用于所有数据集)
        """
        # self._init_bias()
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

    def save_model(self, epoch, step, loss_dict, save_optimizer):
        if self.rank == 0 and step % int(self.hyp['save_ckpt_every'] * len(self.train_dataloader)) == 0:
            save_path = str(self.cwd / 'checkpoints' / f'every_{self.hyp["model_type"]}.pth')            
            if not Path(save_path).exists():
                maybe_mkdir(Path(save_path).parent)

            state_dict = {
                "model_state_dict": self.model.state_dict(),
                "optim_state_dict": self.optimizer.state_dict() if save_optimizer else None,
                "optim_type": self.hyp['optimizer'], 
                "scaler_state_dict": self.scaler.state_dict(),
                'lr_scheduler_type': self.hyp['scheduler_type'], 
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "loss": loss_dict,
                "epoch": epoch,
                "step": step, 
                "ema": self.ema_model.ema.state_dict() if self.hyp['do_ema'] else None, 
                "ema_update_num": self.ema_model.update_num  if self.hyp['do_ema'] else 0, 
                "hyp": self.hyp, 
            }
            torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)
            del state_dict
    
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

    def calculate_metric(self, step_in_total):
        """
        计算dataloader中所有数据的map
        """
        if self.hyp['calculate_map_every'] is not None and step_in_total % int(self.hyp.get('calculate_map_every', 0.5) * len(self.train_dataloader))== 0:
            torch.cuda.empty_cache()

            start_t = time_synchronize()
            pred_bboxes, pred_classes, pred_confidences, pred_labels, gt_bboxes, gt_classes = [], [], [], [], [], []
            iters_num = len(self.val_dataloader)

            all_reduce_norm(self.model)  # 该函数只对batchnorm和instancenorm有效
            if self.hyp['do_ema']:
                eval_model = self.ema_model.ema
            else:
                eval_model = self.model
                if is_parallel(eval_model):
                    eval_model = eval_model.module

            with adjust_status(eval_model, training=False) as m:
                # validater
                validater = Evaluate(m, self.anchors, self.hyp)

                for _ in range(iters_num):
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
                    if self.hyp['half'] and self.hyp['device'] == 'cuda':
                        imgs = imgs.half()
                    
                    outputs = validater(imgs)
                    # preds: [(X, 6), (Y, 6), (Z, 6), ...]
                    imgs, preds = self.preds_postprocess(imgs.cpu(), outputs, infoes)

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

            del validater, all_preds, all_gts, pred_bboxes, pred_classes, pred_confidences, pred_labels
            self.meter.update(mp=mp, map50=map50) 
            gc.collect()
            
    def test(self, step_in_epoch):
        # testing
        if self.rank == 0 and step_in_epoch % int(self.hyp.get('validation_every', 0.5) * len(self.train_dataloader)) == 0:
            torch.cuda.empty_cache()
            all_reduce_norm(self.model)  # 该函数只对batchnorm和instancenorm有效
            if self.hyp['do_ema']:
                eval_model = self.ema_model.ema
            else:
                eval_model = self.model
                if is_parallel(eval_model):
                    eval_model = eval_model.module
            with adjust_status(eval_model, training=False) as m:
                # validater
                validater = Evaluate(m, self.anchors, self.hyp)
                iters_num = len(self.test_dataloader)
                for i in range(iters_num):
                    if self.use_cuda:
                        y = self.test_prefetcher.next()
                    else:
                        y = next(self.test_dataloader)
                    inp = y['img'].to(self.hyp['device'])  # (bs, 3, h, w)
                    info = y['resize_info']
                    output = validater(inp)  # list of ndarray
                    imgs, preds = self.preds_postprocess(inp.cpu(), output, info)
                    for k in range(inp.size(0)):
                        save_path = str(self.cwd / 'result' / 'predictions' / f"{i + k} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png")
                        maybe_mkdir(Path(save_path).parent)
                        if preds[k] is None:
                            cv2_save_img(imgs[k], [], [], [], save_path)
                        else:
                            preds_lab = [self.train_dataset.cls2lab[c] for c in preds[k][:, 5].astype(np.uint8)]
                            cv2_save_img(imgs[k], preds[k][:, :4], preds_lab, preds[k][:, 4], save_path)
                    del y, inp, info, imgs, preds, output
                del validater
                gc.collect()
                synchronize()



def main(x):
    configure_module()
    
    config_ = Config()
    class Args:
        def __init__(self) -> None:
            self.cfg = "./config/train_yolov5.yaml"
    args = Args()

    anchors = torch.tensor([[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]])
    hyp = config_.get_config(args.cfg, args)
    formated_config = print_config(hyp)
    print(formated_config)
    train = Training(anchors, hyp)
    train.step()


if __name__ == '__main__':
    import os
    
    config_ = Config()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, required=True, dest='cfg', help='path to config file')
    # parser.add_argument('--pretrained_model_path',default="", dest='pretrained_model_path') 
    # parser.add_argument('--batch_size', type=int, default=2, dest='batch_size')
    # parser.add_argument("--input_img_size", default=640, type=int, dest='input_img_size')
    # parser.add_argument('--train_img_dir', required=True, dest='train_img_dir', type=str)
    # parser.add_argument('--train_lab_dir', required=True, dest='train_lab_dir', type=str)
    # parser.add_argument('--val_img_dir', required=True, dest='val_img_dir', type=str)
    # parser.add_argument('--val_lab_dir', required=True, dest='val_lab_dir', type=str)
    # parser.add_argument('--test_img_dir', required=True, dest='test_img_dir', type=str)
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
    # args = parser.parse_args()

    from utils import launch, get_num_devices
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    num_gpu = get_num_devices()
    clear_dir(str(current_work_directionary / 'log'))
    launch(
        main, 
        num_gpus_per_machine= num_gpu, 
        num_machines= 1, 
        machine_rank= 0, 
        backend= "nccl", 
        dist_url= "auto", 
        args=(None,),
    )

    
