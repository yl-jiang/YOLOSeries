import sys    
from pathlib import Path
cwd = Path("./").parent.resolve()
sys.path.insert(0, str(cwd))
import torch
from torch import nn
from utils import ConvBnAct, CSPCSPP, Upsample, Concat, ImplicitMul, ImplicitAdd, RepConv, fuse_conv_bn
from collections import OrderedDict
import pickle
import math
import numpy as np


class BaselineBackbone(nn.Module):

    def __init__(self, in_channel=3) -> None:
        super(BaselineBackbone, self).__init__()

        # ============================== backbone ==============================
        # focus layer
        # self.focus = Focus(3, 32, 3, 1, 1)
        self.stem = ConvBnAct(in_channel, 32, 3, 1, 1, inplace=False)

        self.backbone_stage1_conv1 = ConvBnAct(32, 64, 3, 2, 1, inplace=False)  # /2
        self.backbone_stage1_conv2 = ConvBnAct(64, 64, 3, 1, 1, inplace=False)

        self.backbone_stage2_conv1 = ConvBnAct(64, 128, 3, 2, 1, inplace=False)  # /2
        self.backbone_stage2_conv2 = ConvBnAct(128, 64, 1, 1, 0, inplace=False)  
        self.backbone_stage2_conv3 = ConvBnAct(128, 64, 1, 1, 0, inplace=False)  
        self.backbone_stage2_conv4 = ConvBnAct(64, 64, 3, 1, 1, inplace=False)  
        self.backbone_stage2_conv5 = ConvBnAct(64, 64, 3, 1, 1, inplace=False)  
        self.backbone_stage2_conv6 = ConvBnAct(64, 64, 3, 1, 1, inplace=False)  
        self.backbone_stage2_conv7 = ConvBnAct(64, 64, 3, 1, 1, inplace=False)  
        self.backbone_stage2_concat = Concat(1)
        self.backbone_stage2_conv8 = ConvBnAct(64*4, 256, 1, 1, 0, inplace=False)  

        self.backbone_stage3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.backbone_stage3_conv1 = ConvBnAct(256, 128, 1, 1, 0, inplace=False)  
        self.backbone_stage3_conv2 = ConvBnAct(256, 128, 1, 1, 0, inplace=False)  
        self.backbone_stage3_conv3 = ConvBnAct(128, 128, 3, 2, 1, inplace=False)  # /2 
        self.backbone_stage3_concat1 = Concat(1)
        self.backbone_stage3_conv4 = ConvBnAct(128*2, 128, 1, 1, 0, inplace=False)  
        self.backbone_stage3_conv5 = ConvBnAct(128*2, 128, 1, 1, 0, inplace=False)  
        self.backbone_stage3_conv6 = ConvBnAct(128, 128, 3, 1, 1, inplace=False)  
        self.backbone_stage3_conv7 = ConvBnAct(128, 128, 3, 1, 1, inplace=False)  
        self.backbone_stage3_conv8 = ConvBnAct(128, 128, 3, 1, 1, inplace=False)  
        self.backbone_stage3_conv9 = ConvBnAct(128, 128, 3, 1, 1, inplace=False)  
        self.backbone_stage3_concat2 = Concat(1)
        self.backbone_stage3_conv10 = ConvBnAct(128*4, 512, 1, 1, 0, inplace=False)  

        self.backbone_stage4_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.backbone_stage4_conv1 = ConvBnAct(512, 256, 1, 1, 0, inplace=False)  
        self.backbone_stage4_conv2 = ConvBnAct(512, 256, 1, 1, 0, inplace=False)  
        self.backbone_stage4_conv3 = ConvBnAct(256, 256, 3, 2, 1, inplace=False)  # /2 
        self.backbone_stage4_concat1 = Concat(1)
        self.backbone_stage4_conv4 = ConvBnAct(256*2, 256, 1, 1, 0, inplace=False)  
        self.backbone_stage4_conv5 = ConvBnAct(256*2, 256, 1, 1, 0, inplace=False)  
        self.backbone_stage4_conv6 = ConvBnAct(256, 256, 3, 1, 1, inplace=False)  
        self.backbone_stage4_conv7 = ConvBnAct(256, 256, 3, 1, 1, inplace=False)  
        self.backbone_stage4_conv8 = ConvBnAct(256, 256, 3, 1, 1, inplace=False)  
        self.backbone_stage4_conv9 = ConvBnAct(256, 256, 3, 1, 1, inplace=False)  
        self.backbone_stage4_concat2 = Concat(1)
        self.backbone_stage4_conv10 = ConvBnAct(256*4, 1024, 1, 1, 0, inplace=False)  

        self.backbone_stage5_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.backbone_stage5_conv1 = ConvBnAct(1024, 512, 1, 1, 0, inplace=False)  
        self.backbone_stage5_conv2 = ConvBnAct(1024, 512, 1, 1, 0, inplace=False)  
        self.backbone_stage5_conv3 = ConvBnAct(512, 512, 3, 2, 1, inplace=False)  # /2 
        self.backbone_stage5_concat1 = Concat(1)
        self.backbone_stage5_conv4 = ConvBnAct(512*2, 256, 1, 1, 0, inplace=False)  
        self.backbone_stage5_conv5 = ConvBnAct(512*2, 256, 1, 1, 0, inplace=False)  
        self.backbone_stage5_conv6 = ConvBnAct(256, 256, 3, 1, 1, inplace=False)  
        self.backbone_stage5_conv7 = ConvBnAct(256, 256, 3, 1, 1, inplace=False)  
        self.backbone_stage5_conv8 = ConvBnAct(256, 256, 3, 1, 1, inplace=False)  
        self.backbone_stage5_conv9 = ConvBnAct(256, 256, 3, 1, 1, inplace=False)  
        self.backbone_stage5_concat2 = Concat(1)
        self.backbone_stage5_conv10 = ConvBnAct(256*4, 1024, 1, 1, 0, inplace=False)  

    def fuseforward(self):
        for m in self.modules():
            if isinstance(m, ConvBnAct):
                m.conv = fuse_conv_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse

    def forward(self, x):
        x = self.backbone_stage1_conv2(self.backbone_stage1_conv1(self.stem(x)))

        stage2_feats = []
        feat = self.backbone_stage2_conv1(x)
        x = self.backbone_stage2_conv2(feat)
        stage2_feats.append(x)
        x = self.backbone_stage2_conv3(feat)
        stage2_feats.append(x)
        x = self.backbone_stage2_conv5(self.backbone_stage2_conv4(x))
        stage2_feats.append(x)
        x = self.backbone_stage2_conv7(self.backbone_stage2_conv6(x))
        stage2_feats.append(x)
        feat = self.backbone_stage2_conv8(self.backbone_stage2_concat(stage2_feats[::-1]))  # 11

        stage3_feats = []
        x = self.backbone_stage3_conv1(self.backbone_stage3_maxpool(feat))
        stage3_feats.append(x)
        x = self.backbone_stage3_conv3(self.backbone_stage3_conv2(feat))
        stage3_feats.append(x)
        feat = self.backbone_stage3_concat1(stage3_feats[::-1])  # 16
        stage3_feats.clear()
        x = self.backbone_stage3_conv4(feat)
        stage3_feats.append(x)
        x = self.backbone_stage3_conv5(feat)
        stage3_feats.append(x)
        x = self.backbone_stage3_conv7(self.backbone_stage3_conv6(x))
        stage3_feats.append(x)
        x = self.backbone_stage3_conv9(self.backbone_stage3_conv8(x))
        stage3_feats.append(x)
        feat = self.backbone_stage3_conv10(self.backbone_stage3_concat2(stage3_feats[::-1]))  # 24
        stage3_route_feat = feat

        stage4_feats = []
        x = self.backbone_stage4_conv1(self.backbone_stage4_maxpool(feat))
        stage4_feats.append(x)
        x = self.backbone_stage4_conv3(self.backbone_stage4_conv2(feat))  # 28
        stage4_feats.append(x)
        feat = self.backbone_stage4_concat2(stage4_feats[::-1])  # 29
        stage4_feats.clear()
        x = self.backbone_stage4_conv4(feat)
        stage4_feats.append(x)
        x = self.backbone_stage4_conv5(feat)
        stage4_feats.append(x)
        x = self.backbone_stage4_conv7(self.backbone_stage4_conv6(x))
        stage4_feats.append(x)
        x = self.backbone_stage4_conv9(self.backbone_stage4_conv8(x))
        stage4_feats.append(x)
        x = self.backbone_stage4_concat2(stage4_feats[::-1])
        feat = self.backbone_stage4_conv10(x)  # 37
        stage4_route_feat = feat

        stage5_feats = []
        x = self.backbone_stage5_conv1(self.backbone_stage5_maxpool(feat))
        stage5_feats.append(x)
        x = self.backbone_stage5_conv3(self.backbone_stage5_conv2(feat))
        stage5_feats.append(x)
        feat = self.backbone_stage5_concat1(stage5_feats[::-1])
        stage5_feats.clear()
        x = self.backbone_stage5_conv4(feat)
        stage5_feats.append(x)
        x = self.backbone_stage5_conv5(feat)
        stage5_feats.append(x)
        x = self.backbone_stage5_conv7(self.backbone_stage5_conv6(x))
        stage5_feats.append(x)
        x = self.backbone_stage5_conv9(self.backbone_stage5_conv8(x))
        stage5_feats.append(x)
        x = self.backbone_stage5_conv10(self.backbone_stage5_concat2(stage5_feats[::-1]))  # 50

        return OrderedDict({'stage3_route_feat': stage3_route_feat, "stage4_route_feat": stage4_route_feat, "stage5_route_feat": x})


class BaselineHead(nn.Module):

    def __init__(self) -> None:
        super(BaselineHead, self).__init__()

        # ================= Head =================
        self.head_spp = CSPCSPP(1024, 512, inplace=False)
        
        self.head_eelan1_conv1 = ConvBnAct(512, 256, 1, 1, 0, inplace=False)
        self.head_eelan1_upsample = Upsample()
        self.head_eelan1_conv2 = ConvBnAct(1024, 256, 1, 1, 0, inplace=False)
        self.head_eelan1_concat1 = Concat()
        self.head_eelan1_conv3 = ConvBnAct(256*2, 256, 1, 1, 0, inplace=False)
        self.head_eelan1_conv4 = ConvBnAct(256*2, 256, 1, 1, 0, inplace=False)
        self.head_eelan1_conv5 = ConvBnAct(256, 128, 3, 1, 1, inplace=False)
        self.head_eelan1_conv6 = ConvBnAct(128, 128, 3, 1, 1, inplace=False)
        self.head_eelan1_conv7 = ConvBnAct(128, 128, 3, 1, 1, inplace=False)
        self.head_eelan1_conv8 = ConvBnAct(128, 128, 3, 1, 1, inplace=False)
        self.head_eelan1_concat2 = Concat()
        self.head_eelan1_conv9 = ConvBnAct(256*2+128*4, 256, 1, 1, 0, inplace=False)
        
        self.head_eelan2_conv1 = ConvBnAct(256, 128, 1, 1, 0, inplace=False)
        self.head_eelan2_upsample = Upsample()
        self.head_eelan2_conv2 = ConvBnAct(512, 128, 1, 1, 0, inplace=False)
        self.head_eelan2_concat1 = Concat()
        self.head_eelan2_conv3 = ConvBnAct(128*2, 128, 1, 1, 0, inplace=False)
        self.head_eelan2_conv4 = ConvBnAct(128*2, 128, 1, 1, 0, inplace=False)
        self.head_eelan2_conv5 = ConvBnAct(128, 64, 3, 1, 1, inplace=False)
        self.head_eelan2_conv6 = ConvBnAct(64, 64, 3, 1, 1, inplace=False)
        self.head_eelan2_conv7 = ConvBnAct(64, 64, 3, 1, 1, inplace=False)
        self.head_eelan2_conv8 = ConvBnAct(64, 64, 3, 1, 1, inplace=False)
        self.head_eelan2_concat2 = Concat()
        self.head_eelan2_conv9 = ConvBnAct(128*2+64*4, 128, 1, 1, 0, inplace=False)

        self.head_eelan3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.head_eelan3_conv1 = ConvBnAct(128, 128, 1, 1, 0, inplace=False)
        self.head_eelan3_conv2 = ConvBnAct(128, 128, 1, 1, 0, inplace=False)
        self.head_eelan3_conv3 = ConvBnAct(128, 128, 3, 2, 1, inplace=False)
        self.head_eelan3_concat1 = Concat()
        self.head_eelan3_conv4 = ConvBnAct(128*2+256, 256, 1, 1, 0, inplace=False)
        self.head_eelan3_conv5 = ConvBnAct(128*2+256, 256, 1, 1, 0, inplace=False)
        self.head_eelan3_conv6 = ConvBnAct(256, 128, 3, 1, 1, inplace=False)
        self.head_eelan3_conv7 = ConvBnAct(128, 128, 3, 1, 1, inplace=False)
        self.head_eelan3_conv8 = ConvBnAct(128, 128, 3, 1, 1, inplace=False)
        self.head_eelan3_conv9 = ConvBnAct(128, 128, 3, 1, 1, inplace=False)
        self.head_eelan3_concat2 = Concat()
        self.head_eelan3_conv10 = ConvBnAct(128*4+256*2, 256, 1, 1, 0, inplace=False)
        
        self.head_eelan4_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.head_eelan4_conv1 = ConvBnAct(256, 256, 1, 1, 0, inplace=False)
        self.head_eelan4_conv2 = ConvBnAct(256, 256, 1, 1, 0, inplace=False)
        self.head_eelan4_conv3 = ConvBnAct(256, 256, 3, 2, 1, inplace=False)
        self.head_eelan4_concat1 = Concat()
        self.head_eelan4_conv4 = ConvBnAct(256*2+512, 512, 1, 1, 0, inplace=False)
        self.head_eelan4_conv5 = ConvBnAct(256*2+512, 512, 1, 1, 0, inplace=False)
        self.head_eelan4_conv6 = ConvBnAct(512, 256, 3, 1, 1, inplace=False)
        self.head_eelan4_conv7 = ConvBnAct(256, 256, 3, 1, 1, inplace=False)
        self.head_eelan4_conv8 = ConvBnAct(256, 256, 3, 1, 1, inplace=False)
        self.head_eelan4_conv9 = ConvBnAct(256, 256, 3, 1, 1, inplace=False)
        self.head_eelan4_concat2 = Concat()
        self.head_eelan4_conv10 = ConvBnAct(256*4+512*2, 512, 1, 1, 0, inplace=False)

        self.head_output_repconv1 = RepConv(128, 256, 3, 1, 1)
        self.head_output_repconv2 = RepConv(256, 512, 3, 1, 1)
        self.head_output_repconv3 = RepConv(512, 1024, 3, 1, 1)

    def fuseforward(self):
        for m in self.modules():
            if isinstance(m, ConvBnAct):
                m.conv = fuse_conv_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
            if isinstance(m, RepConv):
                m.switch_to_deploy()
        

    def forward(self, inputs):
        """
        Args:
            inputs: [feat_backbone_p4, feat_backbone_p3, x]
        """
        x = self.head_spp(inputs['stage5_route_feat'])  # 51
        spp_route_feat = x

        eelan1_feats = []
        x = self.head_eelan1_upsample(self.head_eelan1_conv1(x))
        eelan1_feats.append(x)
        x = self.head_eelan1_conv2(inputs['stage4_route_feat'])
        eelan1_feats.append(x)
        feat = self.head_eelan1_concat1(eelan1_feats[::-1])
        eelan1_feats.clear()
        x = self.head_eelan1_conv3(feat)
        eelan1_feats.append(x)
        x = self.head_eelan1_conv4(feat)
        eelan1_feats.append(x)
        x = self.head_eelan1_conv5(x)
        eelan1_feats.append(x)
        x = self.head_eelan1_conv6(x)
        eelan1_feats.append(x)
        x = self.head_eelan1_conv7(x)
        eelan1_feats.append(x)
        x = self.head_eelan1_conv8(x)
        eelan1_feats.append(x)
        x = self.head_eelan1_concat2(eelan1_feats[::-1])
        feat = self.head_eelan1_conv9(x)  # 63
        eelan1_route_feat = feat

        eelan2_feats = []
        x = self.head_eelan2_upsample(self.head_eelan2_conv1(feat))
        eelan2_feats.append(x)
        x = self.head_eelan2_conv2(inputs['stage3_route_feat'])
        eelan2_feats.append(x)
        feat = self.head_eelan2_concat2(eelan2_feats[::-1])
        eelan2_feats.clear()
        x = self.head_eelan2_conv3(feat)
        eelan2_feats.append(x)
        x = self.head_eelan2_conv4(feat)
        eelan2_feats.append(x)
        x = self.head_eelan2_conv5(x)
        eelan2_feats.append(x)
        x = self.head_eelan2_conv6(x)
        eelan2_feats.append(x)
        x = self.head_eelan2_conv7(x)
        eelan2_feats.append(x)
        x = self.head_eelan2_conv8(x)
        eelan2_feats.append(x)
        x = self.head_eelan2_concat2(eelan2_feats[::-1])
        feat = self.head_eelan2_conv9(x)  # 75
        eelan2_route_feat = feat

        eelan3_feats = []
        x = self.head_eelan3_conv1(self.head_eelan3_maxpool(feat))  # 77
        eelan3_feats.append(eelan1_route_feat)
        eelan3_feats.append(x)
        x = self.head_eelan3_conv3(self.head_eelan3_conv2(feat))  # 79
        eelan3_feats.append(x)
        feat = self.head_eelan3_concat1(eelan3_feats[::-1])  # 80
        eelan3_feats.clear()
        x = self.head_eelan3_conv4(feat)
        eelan3_feats.append(x)
        x = self.head_eelan3_conv5(feat)
        eelan3_feats.append(x)
        x = self.head_eelan3_conv6(x)
        eelan3_feats.append(x)
        x = self.head_eelan3_conv7(x)
        eelan3_feats.append(x)
        x = self.head_eelan3_conv8(x)
        eelan3_feats.append(x)
        x = self.head_eelan3_conv9(x)
        eelan3_feats.append(x)
        x = self.head_eelan3_concat2(eelan3_feats[::-1])
        feat = self.head_eelan3_conv10(x)  # 88
        eelan3_route_feat = feat

        eelan4_feats = []
        x = self.head_eelan4_conv1(self.head_eelan4_maxpool(feat))
        eelan4_feats.append(spp_route_feat)
        eelan4_feats.append(x)
        x = self.head_eelan4_conv3(self.head_eelan4_conv2(feat))
        eelan4_feats.append(x)
        feat = self.head_eelan3_concat1(eelan4_feats[::-1])  # 93
        eelan4_feats.clear()
        x = self.head_eelan4_conv4(feat)
        eelan4_feats.append(x)
        x = self.head_eelan4_conv5(feat)
        eelan4_feats.append(x)
        x = self.head_eelan4_conv6(x)
        eelan4_feats.append(x)
        x = self.head_eelan4_conv7(x)
        eelan4_feats.append(x)
        x = self.head_eelan4_conv8(x)
        eelan4_feats.append(x)
        x = self.head_eelan4_conv9(x)
        eelan4_feats.append(x)
        x = self.head_eelan4_concat2(eelan4_feats[::-1])
        feat = self.head_eelan4_conv10(x)  # 101
        eelan4_route_feat = feat

        eelan2_route_feat = self.head_output_repconv1(eelan2_route_feat)  # 102
        eelan3_route_feat = self.head_output_repconv2(eelan3_route_feat)  # 103
        eelan4_route_feat = self.head_output_repconv3(eelan4_route_feat)  # 104
        return OrderedDict({"eelan2_route_feat": eelan2_route_feat, "eelan3_route_feat": eelan3_route_feat, "eelan4_route_feat": eelan4_route_feat})


class BaselineDetect(nn.Module):

    def __init__(self, num_anchors=3, in_channels=[256, 512, 1024], num_classes=80, class_frequency=None):
        super(BaselineDetect, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.class_frequency = torch.from_numpy(np.asarray(class_frequency)) if class_frequency is not None else class_frequency

        self.detect_s = nn.Conv2d(in_channels=in_channels[0], out_channels=(num_classes + 5) * num_anchors, kernel_size=1, stride=1, padding=0) 
        self.detect_m = nn.Conv2d(in_channels=in_channels[1], out_channels=(num_classes + 5) * num_anchors, kernel_size=1, stride=1, padding=0)
        self.detect_l = nn.Conv2d(in_channels=in_channels[2], out_channels=(num_classes + 5) * num_anchors, kernel_size=1, stride=1, padding=0)

        self.detect_s.dss = 8  # downsample scale
        self.detect_m.dss = 16
        self.detect_l.dss = 32

        self.implicitadd_s = ImplicitAdd(in_channels[0])
        self.implicitadd_m = ImplicitAdd(in_channels[1])
        self.implicitadd_l = ImplicitAdd(in_channels[2])

        self.implicitmul_s = ImplicitMul((num_classes + 5) * num_anchors)
        self.implicitmul_m = ImplicitMul((num_classes + 5) * num_anchors)
        self.implicitmul_l = ImplicitMul((num_classes + 5) * num_anchors)

        # self.apply(self._initialize_biases)  

    def _initialize_biases(self, m):
        """
        initialize detection convolution layers's bias as RetinaNet do.
        """
        if isinstance(m, nn.Conv2d):
            b = m.bias.view(self.num_anchors, -1)
            b.data[:, 4] += math.log(8/(640/m.dss)**2)  # confidence bias
            if self.class_frequency is not None:  # classfication bias
                b.data[:, 5:] += torch.log(self.class_frequency / self.class_frequency.sum())
            else:
                b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))
            m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def forward(self, inputs):
        """
        Args:
            inputs: dict/ {"eelan2_route_feat": eelan2_route_feat, "eelan3_route_feat": eelan3_route_feat, "eelan4_route_feat": eelan4_route_feat}
        """
        assert len(inputs) == self.num_anchors
        
        feat_small = self.implicitmul_s(self.detect_s(self.implicitadd_s(inputs['eelan2_route_feat'])))
        b, c, h, w = feat_small.shape
        feat_small = feat_small.view(b, self.num_anchors, self.num_classes+5, h, w).permute(0, 1, 3, 4, 2).contiguous()

        feat_middle = self.implicitmul_m(self.detect_m(self.implicitadd_m(inputs['eelan3_route_feat'])))
        b, c, h, w = feat_middle.shape
        feat_middle = feat_middle.view(b, self.num_anchors, self.num_classes+5, h, w).permute(0, 1, 3, 4, 2).contiguous()

        feat_large = self.implicitmul_l(self.detect_l(self.implicitadd_l(inputs['eelan4_route_feat'])))
        b, c, h, w = feat_large.shape
        feat_large = feat_large.view(b, self.num_anchors, self.num_classes+5, h, w).permute(0, 1, 3, 4, 2).contiguous()

        # pred_l: (batch_size, num_anchors, H/32, W/32, 85)
        # pred_m: (batch_size, num_anchors, H/16, W/16, 85)
        # pred_s: (batch_size, num_anchors, H/8, W/8, 85)
        pred_out = OrderedDict()
        pred_out['pred_s'] = feat_small
        pred_out['pred_m'] = feat_middle
        pred_out['pred_l'] = feat_large
        return pred_out



class YOLOv7Baseline(nn.Module):

    def __init__(self, in_channel=3, num_anchor=3, num_classes=80) -> None:
        super(YOLOv7Baseline, self).__init__()

        self.backbone = BaselineBackbone(in_channel=in_channel)
        self.head = BaselineHead()
        self.detect = BaselineDetect(num_anchors=num_anchor, in_channels=[256, 512, 1024], num_classes=num_classes)

    def fuseforward(self):
        self.backbone.fuseforward()
        self.head.fuseforward()
        return self
        
    def forward(self, x):
        backbone_feat = self.backbone(x)
        head_feat = self.head(backbone_feat)
        detect_feat = self.detect(head_feat)

        return detect_feat


if __name__ == "__main__":
    dummy = torch.rand(8, 3, 448, 448).float().contiguous()
    net = YOLOv7Baseline()
    net = net.fuseforward()
    out = net(dummy)
    for k, v in out.items():
        print(k, v.shape)


        
