import torch
import math
from torch import nn
from utils import Focus, BottleneckCSP, ConvBnAct, SPP, Upsample, Concat, Detect

__all__ = ['YOLOV5SmallWithPlainBscp']
class YOLOV5SmallWithPlainBscp(nn.Module):

    def __init__(self, anchor_num, num_class, in_channel=3):
        super(YOLOV5SmallWithPlainBscp, self).__init__()
        self.num_class = num_class

        # focus
        self.focus = Focus(in_channel, 32, 3, 1, 1)

        # backbone
        self.backbone_stage1_conv = ConvBnAct(32, 64, 3, 2, 1)  # /2
        self.backbone_stage1_bscp = BottleneckCSP(64, 64)
        self.backbone_stage2_conv = ConvBnAct(64, 128, 3, 2, 1)  # /2
        self.backbone_stage2_bscp = BottleneckCSP(128, 128, num_block=3)
        self.backbone_stage3_conv = ConvBnAct(128, 256, 3, 2, 1)  # /2
        self.backbone_stage3_bscp = BottleneckCSP(256, 256, num_block=3)
        self.backbone_stage4_conv = ConvBnAct(256, 512, 3, 2, 1)  # /2
        self.backbone_stage4_spp = SPP(512, 512, kernels=[5, 9, 13])
        self.backbone_stage4_bscp = BottleneckCSP(512, 512, shortcut=False)

        # head
        self.head_upsample = Upsample()
        self.head_concat = Concat()
        self.head_stage1_conv = ConvBnAct(512, 256, 1, 1, 0)
        self.head_stage1_bscp = BottleneckCSP(512, 256, shortcut=False)
        self.head_stage2_conv = ConvBnAct(256, 128, 1, 1, 0)
        self.head_stage2_bscp = BottleneckCSP(256, 128, shortcut=False)
        self.head_stage3_conv = ConvBnAct(128, 128, 3, 2, 1)
        self.head_stage3_bscp = BottleneckCSP(256, 256, shortcut=False)
        self.head_stage4_conv = ConvBnAct(256, 256, 3, 2, 1)
        self.head_stage4_bscp = BottleneckCSP(512, 512, shortcut=False)

        # detect
        self.num_anchor = anchor_num  # number anchor for each stage
        self.detect = Detect(in_channels=[128, 256, 512], out_channel=self.num_anchor*(num_class+5))

        self._init_bias()

    def _init_bias(self):
        """
        初始化模型参数, 主要是对detection layers的bias参数进行特殊初始化, 参考RetinaNet那篇论文, 这种初始化方法可让网络较容易度过前期训练困难阶段
        (使用该初始化方法可能针对coco数据集有效, 在对global wheat数据集的测试中, 该方法根本train不起来)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, (nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True

        detect_layer = [self.detect.detect_small,
                        self.detect.detect_mid,
                        self.detect.detect_large]

        input_img_shape = 128
        stage_outputs = self(torch.zeros(1, 3, input_img_shape, input_img_shape).float())
        strides = torch.tensor([input_img_shape / x.size(2) for x in stage_outputs])
        class_frequency = None  # 各类别占整个数据集的比例
        for m, stride in zip(detect_layer, strides):
            bias = m.bias.view(self.num_anchor, -1)  # (255,) -> (3, 85)
            with torch.no_grad():
                bias[:, 4] += math.log(8 / (512 / stride) ** 2)
                if class_frequency is None:
                    bias[:, 5:] += math.log(0.6 / (self.num_class - 0.99))  # cls
                else:
                    # 类别较多的那一类给予较大的对应bias值, 类别较少的那些类给予较小的bias。
                    # 这样做的目的是为了解决类别不平衡问题, 因为这种初始化方式只在分类层进行, 会使得网络在进行分类预测时, 预测到类别较少的那一类case较为容易（因为对应的bias较小, 容易在预测这些类别时给出较大的预测值）
                    # 使用这种初始化方式的好处主要是为了解决数据类别不平衡问题造成的早期训练不稳定情况。
                    # 注：这种初始化方法只针对二分类（因为多分类不能针对各个class给予不同的bias）
                    assert isinstance(class_frequency, torch.Tensor), f"class_frequency should be a tensor but we got {type(class_frequency)}"
                    bias[:, 5:] += torch.log(class_frequency / class_frequency.sum())
                m.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)
        del stage_outputs

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
    anchors = [[10, 13, 16, 30, 33, 23],
               [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    yolo = YOLOV5SmallWithPlainBscp(3, 80)

    dummy_img = torch.rand(5, 3, 640, 640)
    out = yolo(dummy_img)
    print(out[0].shape, out[1].shape, out[2].shape)


