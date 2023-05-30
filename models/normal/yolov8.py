import math
import torch
from torch import nn
from utils import ConvBnAct, Upsample, Concat, FastSPP, C2f, DistributionFocalLoss

__all__ = ['YOLOV8']


class Detect(nn.Module):
    def __init__(self, num_class, in_channels, strides):
        """
        Inputs:
            in_channels: [c_xsmall, c_small, c_mid, c_large]
        """

        super().__init__()
        self.in_channels = in_channels
        self.num_class = num_class
        self.strides = strides

        self.detect_xsmall_bbox = nn.Sequential(ConvBnAct(in_channels[0]   , in_channels[0]//4, 3, 1, 1), 
                                                ConvBnAct(in_channels[0]//4, in_channels[0]//4, 3, 1, 1), 
                                                nn.Conv2d(in_channels[0]//4, 64               , 1, 1, 0))
        
        self.detect_xsmall_cls = nn.Sequential(ConvBnAct(in_channels[0], 128      , 3, 1, 1), 
                                               ConvBnAct(128           , 128      , 3, 1, 1),
                                               nn.Conv2d(128           , num_class, 1, 1, 0))
        

        self.detect_small_bbox = nn.Sequential(ConvBnAct(in_channels[1]   , in_channels[1]//4, 3, 1, 1), 
                                               ConvBnAct(in_channels[1]//4, in_channels[1]//4, 3, 1, 1), 
                                               nn.Conv2d(in_channels[1]//4, 64               , 1, 1, 0))
        
        self.detect_small_cls = nn.Sequential(ConvBnAct(in_channels[1], 128      , 3, 1, 1), 
                                              ConvBnAct(128           , 128      , 3, 1, 1),
                                              nn.Conv2d(128           , num_class, 1, 1, 0))
        
        
        self.detect_mid_bbox = nn.Sequential(ConvBnAct(in_channels[2]   , in_channels[2]//4, 3, 1, 1), 
                                             ConvBnAct(in_channels[2]//4, in_channels[2]//4, 3, 1, 1), 
                                             nn.Conv2d(in_channels[2]//4, 64               , 1, 1, 0))
        
        self.detect_mid_cls = nn.Sequential(ConvBnAct(in_channels[2], 128      , 3, 1, 1), 
                                            ConvBnAct(128           , 128      , 3, 1, 1),
                                            nn.Conv2d(128           , num_class, 1, 1, 0))
        
        
        self.detect_large_bbox = nn.Sequential(ConvBnAct(in_channels[3]   , in_channels[3]//4, 3, 1, 1), 
                                               ConvBnAct(in_channels[3]//4, in_channels[3]//4, 3, 1, 1), 
                                               nn.Conv2d(in_channels[3]//4, 64               , 1, 1, 0))
        
        self.detect_large_cls = nn.Sequential(ConvBnAct(in_channels[3], 128      , 3, 1, 1), 
                                              ConvBnAct(128           , 128      , 3, 1, 1),
                                              nn.Conv2d(128           , num_class, 1, 1, 0))
        
        self.bias_init()
        

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        bbox_layers = [self.detect_xsmall_bbox, self.detect_small_bbox, self.detect_mid_bbox, self.detect_large_bbox]
        cls_layers = [self.detect_xsmall_cls, self.detect_small_cls, self.detect_mid_cls, self.detect_large_cls]
        
        for a, b, s in zip(bbox_layers, cls_layers, self.strides):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:self.num_class] = math.log(5 / self.num_class / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def forward(self, xs):
        """
        Inputs:
            xs: [(b, c_xsmall, h/4, w/4), (b, c_small, h/8, w/8), (b, c_mid, h/16, w/16), (b, c_large, h/32, w/32)]
        
        """
        ys = {}
        ys['pred_xs'] = torch.cat((self.detect_xsmall_bbox(xs[0]), self.detect_xsmall_cls(xs[0])), 1)
        ys['pred_x'] = torch.cat((self.detect_small_bbox(xs[1]) , self.detect_small_cls(xs[1])) , 1)
        ys['pred_m'] = torch.cat((self.detect_mid_bbox(xs[2])   , self.detect_mid_cls(xs[2]))   , 1)
        ys['pred_l'] = torch.cat((self.detect_large_bbox(xs[3]) , self.detect_large_cls(xs[3])) , 1)
        return ys


class YOLOV8(nn.Module):

    def __init__(self, in_channel=3, num_class=80, scale=1.0):
        super().__init__()
        self.num_class = num_class
        # common layers
        self.head_upsample = Upsample()
        self.head_concat = Concat()

        # ============================== backbone ==============================
        self.backbone_stem1 = ConvBnAct(in_channel, 64, 3, 2, 1)  # /2
        self.backbone_stem2 = ConvBnAct(64, 128, 3, 2, 1)  # /4

        self.backbone_stage1_c2f = C2f(128, 128, shortcut=True, num_block=int(3*scale))
        self.backbone_stage1_conv = ConvBnAct(128, 256, 3, 2, 1)  # /8

        self.backbone_stage2_c2f = C2f(256, 256, shortcut=True, num_block=int(6*scale))
        self.backbone_stage2_conv = ConvBnAct(256, 512, 3, 2, 1)  # /16

        self.backbone_stage3_c2f = C2f(512, 512, shortcut=True, num_block=int(6*scale))
        self.backbone_stage3_conv = ConvBnAct(512, 1024, 3, 2, 1)  # /32

        self.backbone_stage4_c2f = C2f(1024, 1024, shortcut=True, num_block=int(3*scale))
        self.backbone_stage4_spp = FastSPP(1024, 1024, kernel=5)

        # ============================== head ==============================
        self.head_stage1_c2f1 = C2f(1024+512, 512, shortcut=False, num_block=int(3*scale))
        self.head_stage2_c2f1 = C2f(512+256, 256, shortcut=False, num_block=int(3*scale))
        self.head_stage3_c2f1 = C2f(256+128, 128, shortcut=False, num_block=int(3*scale))

        self.head_stage3_conv = ConvBnAct(128, 128, 3, 2, 1)
        self.head_stage3_c2f2 = C2f(128+256, 256, shortcut=False, num_block=int(3*scale))

        self.head_stage2_conv = ConvBnAct(256, 256, 3, 2, 1)
        self.head_stage2_c2f2 = C2f(256+512, 512, shortcut=False, num_block=int(3*scale))

        self.head_stage1_conv = ConvBnAct(512, 512, 3, 2, 1)
        self.head_stage1_c2f2 = C2f(512+1024, 1024, shortcut=False, num_block=int(3*scale))

        # detect layers
        self.detect = Detect(num_class=num_class, in_channels=[128, 256, 512, 1024], strides=[4, 8, 16, 32])


    def forward(self, x):
        """

        :param x: tensor / (bn, 3, 640, 640)
        :return:
        """
        # ---------------------------------------------------------------------- Backbone
        x = self.backbone_stem2(self.backbone_stem1(x))

        x_2 = self.backbone_stage1_c2f(x)
        x = self.backbone_stage1_conv(x_2)

        x_4 = self.backbone_stage2_c2f(x)
        x = self.backbone_stage2_conv(x_4)

        x_6 = self.backbone_stage3_c2f(x)
        x = self.backbone_stage3_conv(x_6)

        x_8 = self.backbone_stage4_c2f(x)
        x_9 = self.backbone_stage4_spp(x_8)

        # ---------------------------------------------------------------------- Head
        x = self.head_upsample(x_9)
        x = self.head_concat((x, x_6))
        x_12 = self.head_stage1_c2f1(x)

        x = self.head_upsample(x_12)
        x = self.head_concat((x, x_4))
        x_15 = self.head_stage2_c2f1(x)

        x = self.head_upsample(x_15)
        x = self.head_concat((x, x_2))
        x_18 = self.head_stage3_c2f1(x)

        x = self.head_stage3_conv(x_18)
        x = self.head_concat((x, x_15))
        x_21 = self.head_stage3_c2f2(x)

        x = self.head_stage2_conv(x_21)
        x = self.head_concat((x, x_12))
        x_24 = self.head_stage2_c2f2(x)

        x = self.head_stage1_conv(x_24)
        x = self.head_concat((x, x_9))
        x_27 = self.head_stage1_c2f2(x)

        feats = (x_18, x_21, x_24, x_27)
        # [(5, 255, 80, 80), (5, 255, 40, 40), (5, 255, 20, 20)] / 返回的prediction中关于坐标的预测为[x, y, w, h, cofidence, c1, c2, ...]
        return self.detect(feats)

if __name__ == '__main__':
    yolo = YOLOV8(3, 80)
    dummy_img = torch.rand(5, 3, 640, 640)
    out = yolo(dummy_img)
    print(out[0].shape, out[1].shape, out[2].shape)

