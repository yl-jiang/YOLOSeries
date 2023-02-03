import sys
import torch
from torch import nn
from utils import ConvBnAct, Upsample, Concat, Detect, C3BottleneckCSP, FastSPP

__all__ = ['YOLOV5Large']

class YOLOV5Large(nn.Module):

    def __init__(self, anchor_num, num_class, in_channel=3):
        super(YOLOV5Large, self).__init__()
        self.num_class = num_class

        # ============================== backbone ==============================
        # focus layer
        # self.focus = Focus(3, 64, 3, 1, 1)
        self.focus = ConvBnAct(in_channel, 64, 6, 2, 2)

        self.backbone_stage1_conv = ConvBnAct(64, 128, 3, 2, 1)  # /2
        self.backbone_stage1_bscp = C3BottleneckCSP(128, 128, shortcut=True, num_block=3)
        self.backbone_stage2_conv = ConvBnAct(128, 256, 3, 2, 1)  # /2
        self.backbone_stage2_bscp = C3BottleneckCSP(256, 256, shortcut=True, num_block=6)
        self.backbone_stage3_conv = ConvBnAct(256, 512, 3, 2, 1)  # /2
        self.backbone_stage3_bscp = C3BottleneckCSP(512, 512, shortcut=True, num_block=9)
        self.backbone_stage4_conv = ConvBnAct(512, 1024, 3, 2, 1)  # /2
        self.backbone_stage4_bscp = C3BottleneckCSP(1024, 1024, shortcut=True, num_block=3)
        self.backbone_stage4_spp = FastSPP(1024, 1024, kernel=5)
        # ============================== head ==============================

        # common layers
        self.head_upsample = Upsample()
        self.head_concat = Concat()

        self.head_stage1_conv = ConvBnAct(1024, 512, 1, 1, 0)
        self.head_stage1_bscp = C3BottleneckCSP(1024, 512, shortcut=False, num_block=3)
        self.head_stage2_conv = ConvBnAct(512, 256, 1, 1, 0)
        self.head_stage2_bscp = C3BottleneckCSP(512, 256, shortcut=False, num_block=3)
        self.head_stage3_conv = ConvBnAct(256, 256, 3, 2, 1)
        self.head_stage3_bscp = C3BottleneckCSP(512, 512, shortcut=False, num_block=3)
        self.head_stage4_conv = ConvBnAct(512, 512, 3, 2, 1)
        self.head_stage4_bscp = C3BottleneckCSP(1024, 1024, shortcut=False, num_block=3)

        # detect layers
        self.num_anchor = anchor_num  # number anchor for each stage
        self.detect = Detect(in_channels=[256, 512, 1024], out_channel=self.num_anchor*(num_class+5))

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
        # [(5, 255, 80, 80), (5, 255, 40, 40), (5, 255, 20, 20)] / 返回的prediction中关于坐标的预测为[x, y, w, h, cofidence, c1, c2, ...]
        return self.detect([small_x, mid_x, large_x])

if __name__ == '__main__':
    anchors = [[10, 13, 16, 30, 33, 23],
               [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    yolo = YOLOV5Large(3, 80)

    dummy_img = torch.rand(5, 3, 640, 640)
    out = yolo(dummy_img)
    print(out[0].shape, out[1].shape, out[2].shape)


