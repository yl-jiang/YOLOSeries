import sys
from pathlib import Path
current_work_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(current_work_dir))

import torch
from torch import nn
from utils import Focus, BottleneckCSP, ConvBnAct, SPP, Upsample, Concat, Detect, SEBottleneckCSP, C3BottleneckCSP, FastSPP


class MiddleYOLOXBackboneAndNeck(nn.Module):

    def __init__(self, in_channel=3):
        super().__init__()
        # ============================== backbone ==============================
        # focus layer
        # self.focus = Focus(in_channel, 48, 3, 1, 1)
        self.focus = ConvBnAct(3, 48, 6, 2, 2)

        self.backbone_stage1_conv = ConvBnAct(48, 96, 3, 2, 1)  # /2
        self.backbone_stage1_bscp = C3BottleneckCSP(96, 96, shortcut=True, num_block=2)
        self.backbone_stage2_conv = ConvBnAct(96, 192, 3, 2, 1)  # /2
        self.backbone_stage2_bscp = C3BottleneckCSP(192, 192, shortcut=True, num_block=4)
        self.backbone_stage3_conv = ConvBnAct(192, 384, 3, 2, 1)  # /2
        self.backbone_stage3_bscp = C3BottleneckCSP(384, 384, shortcut=True, num_block=6)
        self.backbone_stage4_conv = ConvBnAct(384, 768, 3, 2, 1)  # /2
        self.backbone_stage4_bscp = C3BottleneckCSP(768, 768, shortcut=True, num_block=2)
        self.backbone_stage4_spp = FastSPP(768, 768, kernel=5)
        # ============================== head ==============================

        # common layers
        self.head_upsample = Upsample()
        self.head_concat = Concat()

        self.head_stage1_conv = ConvBnAct(768, 384, 1, 1, 0)
        self.head_stage1_bscp = C3BottleneckCSP(768, 384, shortcut=False, num_block=2)
        self.head_stage2_conv = ConvBnAct(384, 192, 1, 1, 0)
        self.head_stage2_bscp = C3BottleneckCSP(384, 192, shortcut=False, num_block=2)
        self.head_stage3_conv = ConvBnAct(192, 192, 3, 2, 1)
        self.head_stage3_bscp = C3BottleneckCSP(384, 384, shortcut=False, num_block=2)
        self.head_stage4_conv = ConvBnAct(384, 384, 3, 2, 1)
        self.head_stage4_bscp = C3BottleneckCSP(768, 768, shortcut=False, num_block=2)


    def forward(self, x):
        """

        :param x: tensor / (bn, 3, 640, 640)
        :return:
        """
        x = self.focus(x)  # (bn, 32, 320, 320)
        x = self.backbone_stage1_conv(x)  # (bn, 64, 160, 160)
        x = self.backbone_stage1_bscp(x)  # (bn, 64, 160, 160)
        stage1_x = self.backbone_stage2_bscp(self.backbone_stage2_conv(x))  # (bn, 128, 80, 80)
        stage2_x = self.backbone_stage3_bscp(self.backbone_stage3_conv(stage1_x))  # (bn, 256, 40, 40)
        x = self.backbone_stage4_conv(stage2_x)  # (bn, 512, 20, 20)
        x = self.backbone_stage4_bscp(x)  # (bn, 512, 20, 20)
        x = self.backbone_stage4_spp(x)  # (bn, 512, 20, 20)
        head1_x = self.head_stage1_conv(x)  # (bn, 256, 20, 20)
        x = self.head_upsample(head1_x)  # (bn, 256, 40, 40)
        x = self.head_concat([x, stage2_x])  # (bn, 512, 40, 40)
        x = self.head_stage1_bscp(x)  # (bn, 256, 40, 40)
        head2_x = self.head_stage2_conv(x)  # (bn, 128, 40, 40)
        x = self.head_upsample(head2_x)  # (bn, 128, 80, 80)
        x = self.head_concat([x, stage1_x])  # (bn, 256, 80, 80)
        small_x = self.head_stage2_bscp(x)  # (bn, 128, 80, 80)
        x = self.head_stage3_conv(small_x)  # (bn, 128, 40, 40)
        x = self.head_concat([x, head2_x])  # (bn, 256, 40, 40)
        mid_x = self.head_stage3_bscp(x)  # (bn, 256, 40, 40)
        x = self.head_stage4_conv(mid_x)  # (bn, 256, 20, 20)
        x = self.head_concat([x, head1_x])  # (bn, 512, 20, 20)
        large_x = self.head_stage4_bscp(x)  # (bn, 512, 20, 20)

        out = {}
        out['fea_s'] = small_x
        out['fea_m'] = mid_x
        out['fea_l'] = large_x
        return out


class Detect(nn.Module):

    def __init__(self, num_anchors=1, in_channels=[256, 512, 1024], mid_channel=256, wid_mul=1.0, num_classes=80):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.pred_small = self._make_layers(int(in_channels[0] * wid_mul), int(mid_channel * wid_mul))
        self.pred_middle = self._make_layers(int(in_channels[1] * wid_mul), int(mid_channel * wid_mul))
        self.pred_large = self._make_layers(int(in_channels[2] * wid_mul), int(mid_channel * wid_mul))


    def _make_layers(self, in_c, mid_c):
        stem = ConvBnAct(in_c, mid_c, 1, 1, act=True)
        cls = nn.Sequential(
                ConvBnAct(mid_c, mid_c, 3, 1, 1, act=True), 
                ConvBnAct(mid_c, mid_c, 3, 1, 1, act=True), 
                nn.Conv2d(mid_c, int(self.num_anchors * self.num_classes), 1, 1)
        )

        conv = nn.Sequential(
                ConvBnAct(mid_c, mid_c, 3, 1, 1, act=True), 
                ConvBnAct(mid_c, mid_c, 3, 1, 1, act=True)
        )

        reg = nn.Conv2d(mid_c, self.num_anchors * 4, 1, 1)
        cof = nn.Conv2d(mid_c, int(self.num_anchors * 1), 1, 1)
        return nn.ModuleDict({'stem': stem, 'conv': conv, 'cls': cls, 'reg': reg, 'cof': cof})

    def forward_each(self, layers, x):
        x = layers['stem'](x)  # stem
        cls_pred = layers['cls'](x)  # classification
        feat = layers['conv'](x)  # extract features for regression and confidence
        reg_pred = layers['reg'](feat)  # regression
        cof_pred = layers['cof'](feat)  # confidence
        # (batch_size, 4+1+80, h, w)
        output = torch.cat((reg_pred, cof_pred, cls_pred), dim=1).contiguous()
        return output

    def forward(self, x):
        x_s, x_m, x_l = x['x_s'], x['x_m'], x['x_l']
        batch_size = x_s.size(0)

        pred_s = self.forward_each(self.pred_small, x_s)
        _, _, h_s, w_s = pred_s.size() 
        pred_s = pred_s.reshape(batch_size, self.num_anchors, -1, h_s, w_s)

        pred_m = self.forward_each(self.pred_middle, x_m)
        _, _, h_m, w_m = pred_m.size() 
        pred_m = pred_m.reshape(batch_size, self.num_anchors, -1, h_m, w_m)

        pred_l = self.forward_each(self.pred_large, x_l)
        _, _, h_l, w_l = pred_l.size() 
        pred_l = pred_l.reshape(batch_size, self.num_anchors, -1, h_l, w_l)

        # pred_l: (batch_size, num_anchors, 85, H/32, W/32)
        # pred_m: (batch_size, num_anchors, 85, H/16, W/16)
        # pred_s: (batch_size, num_anchors, 85, H/8, W/8)
        return {'pred_s': pred_s, 'pred_m': pred_m, 'pred_l': pred_l}


class YoloXMiddle(nn.Module):

    def __init__(self, num_anchors=1, in_channel=3, num_classes=80):
        super().__init__()
        self.neck = MiddleYOLOXBackboneAndNeck(in_channel)
        self.detect = Detect(num_anchors=num_anchors, in_channels=[192, 384, 768], mid_channel=192, wid_mul=1.0, num_classes=num_classes)

    def forward(self, x):
        # backbone & neck
        neck = self.neck(x)
        x_s = neck['fea_s']
        x_m = neck['fea_m']
        x_l = neck['fea_l']

        neck = {'x_s': x_s, 'x_m': x_m, 'x_l': x_l}

        # head
        preds = self.detect(neck)

        return preds



if __name__ == "__main__":

    dummy = torch.rand(1, 3, 224, 224)
    yolox = YoloXMiddle(2)
    out = yolox(dummy)
    
    for k, v in out.items():
        print(f"{k}\t{v.shape}")
    

