import torch
from torch import nn
from utils import Focus, DepthWiseConvBnAct, SPP, Upsample, Concat, Detect, DepthWiseC3BottleneckCSP

__all__ = ['YOLOV5XLargeDW']

class YOLOV5XLargeDW(nn.Module):

    def __init__(self, anchor_num, num_class):
        super(YOLOV5XLargeDW, self).__init__()
        self.num_class = num_class

        # ============================== backbone ==============================

        # focus layer
        self.focus = Focus(3, 80, 3, 1, 1)

        self.backbone_stage1_conv = DepthWiseConvBnAct(80, 160, 3, 2, 1)  # /2
        self.backbone_stage1_bscp = DepthWiseC3BottleneckCSP(160, 160, shortcut=True, num_block=4)
        self.backbone_stage2_conv = DepthWiseConvBnAct(160, 320, 3, 2, 1)  # /2
        self.backbone_stage2_bscp = DepthWiseC3BottleneckCSP(320, 320, shortcut=True, num_block=12)
        self.backbone_stage3_conv = DepthWiseConvBnAct(320, 640, 3, 2, 1)  # /2
        self.backbone_stage3_bscp = DepthWiseC3BottleneckCSP(640, 640, shortcut=True, num_block=12)
        self.backbone_stage4_conv = DepthWiseConvBnAct(640, 1280, 3, 2, 1)  # /2
        self.backbone_stage4_spp = SPP(1280, 1280, kernels=[5, 9, 13])
        self.backbone_stage4_bscp = DepthWiseC3BottleneckCSP(1280, 1280, shortcut=False, num_block=4)
        # ============================== head ==============================

        # common layers
        self.head_upsample = Upsample()
        self.head_concat = Concat()

        self.head_stage1_conv = DepthWiseConvBnAct(1280, 640, 1, 1, 0)
        self.head_stage1_bscp = DepthWiseC3BottleneckCSP(1280, 640, shortcut=False, num_block=4)
        self.head_stage2_conv = DepthWiseConvBnAct(640, 320, 1, 1, 0)
        self.head_stage2_bscp = DepthWiseC3BottleneckCSP(640, 320, shortcut=False, num_block=4)
        self.head_stage3_conv = DepthWiseConvBnAct(320, 320, 3, 2, 1)
        self.head_stage3_bscp = DepthWiseC3BottleneckCSP(640, 640, shortcut=False, num_block=4)
        self.head_stage4_conv = DepthWiseConvBnAct(640, 640, 3, 2, 1)
        self.head_stage4_bscp = DepthWiseC3BottleneckCSP(1280, 1280, shortcut=False, num_block=4)

        # detect layers
        self.num_anchor = anchor_num  # number anchor for each stage
        self.detect = Detect(in_channels=[320, 640, 1280], out_channel=self.num_anchor*(num_class+5))

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
        x = self.backbone_stage4_spp(x)  # (bn, 512, 20, 20)
        x = self.backbone_stage4_bscp(x)  # (bn, 512, 20, 20)
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
    import sys
    from pathlib import Path
    FILE = Path("__file__").resolve().parent
    sys.path.insert(0, str(FILE))
    
    anchors = [[10, 13, 16, 30, 33, 23],
               [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    yolo = YOLOV5XLargeDW(3, 80)

    dummy_img = torch.rand(5, 3, 640, 640)
    out = yolo(dummy_img)
    print(out[0].shape, out[1].shape, out[2].shape)


