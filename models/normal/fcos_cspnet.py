import torch
import torch.nn as nn
from utils import Scale
import math
import torch.nn.functional as F
from functools import partial
from utils import ConvBnAct, Upsample, Concat, C3BottleneckCSP, FastSPP

__all__ = ['FCOSCSPNet']


class FCOSCSPNet(nn.Module):

    def __init__(self, num_class, in_channel=3, head_norm_layer_type='group_norm', enable_head_scale=False, freeze_bn=False):
        super(FCOSCSPNet, self).__init__()
        self.num_class = num_class

        # ============================== backbone ==============================
        self.focus = ConvBnAct(in_channel, 32, 6, 2, 2)

        self.backbone_stage1_conv = ConvBnAct(32, 64, 3, 2, 1)  # /2
        self.backbone_stage1_bscp = C3BottleneckCSP(64, 64, shortcut=True, num_block=1)
        self.backbone_stage2_conv = ConvBnAct(64, 128, 3, 2, 1)  # /2
        self.backbone_stage2_bscp = C3BottleneckCSP(128, 128, shortcut=True, num_block=2)
        self.backbone_stage3_conv = ConvBnAct(128, 256, 3, 2, 1)  # /2
        self.backbone_stage3_bscp = C3BottleneckCSP(256, 256, shortcut=True, num_block=3)
        self.backbone_stage4_conv = ConvBnAct(256, 512, 3, 2, 1)  # /2
        self.backbone_stage4_bscp = C3BottleneckCSP(512, 512, shortcut=True, num_block=1)
        self.backbone_stage4_spp = FastSPP(512, 512, kernel=5)
    
        # ============================== head ==============================
        # common layers
        self.head_upsample = Upsample()
        self.head_concat = Concat()

        self.head_stage1_conv = ConvBnAct(512, 256, 1, 1, 0)
        self.head_stage1_bscp = C3BottleneckCSP(512, 256, shortcut=False, num_block=1)
        self.head_stage2_conv = ConvBnAct(256, 128, 1, 1, 0)
        self.head_stage2_bscp = C3BottleneckCSP(256, 128, shortcut=False, num_block=1)
        self.head_stage3_conv = ConvBnAct(128, 128, 3, 2, 1)
        self.head_stage3_bscp = C3BottleneckCSP(256, 256, shortcut=False, num_block=1)
        self.head_stage4_conv = ConvBnAct(256, 256, 3, 2, 1)
        self.head_stage4_bscp = C3BottleneckCSP(512, 512, shortcut=False, num_block=1)

        # detect layers
        self.detect = FCOSHead(num_stage=3, in_channels=[128, 256, 512], num_class=num_class, head_norm_layer_type=head_norm_layer_type, enable_head_scale=enable_head_scale)
        
        if freeze_bn:  # only do this for training
            self._freeze_bn()

    def _freeze_bn(self):
        """
        https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/12
        """
        print("Freeze parameters of BatchNorm layers.")
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m, 'weight'):
                    m.weight.requires_grad_(False)
                if hasattr(m, 'bias'):
                    m.bias.requires_grad_(False)
                m.eval()

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
        
        return self.detect([small_x, mid_x, large_x])


class FCOSHead(nn.Module):

    def __init__(self, num_stage, in_channels, num_class, head_norm_layer_type='group_norm', enable_head_scale=False):
        super(FCOSHead, self).__init__()

        if head_norm_layer_type.lower() == "batch_norm":
            NormLayer = nn.BatchNorm2d
        elif head_norm_layer_type.lower() == 'group_norm':
            NormLayer = partial(nn.GroupNorm, num_groups=32)
        else:
            raise RuntimeError(f'unknow head_norm_layer_type {head_norm_layer_type}, must be "batch_norm" or "group_norm".')
        cls_layers, reg_layers = [], []
        self.head_stem = nn.ModuleList()
        
        feat_plane = 256
        for i in range(num_stage):
            self.head_stem.append(nn.Sequential(nn.Conv2d(in_channels[i], feat_plane, 1, 1, 0, bias=False), 
                                                NormLayer(num_channels=feat_plane) if head_norm_layer_type == 'group_norm' else NormLayer(num_features=feat_plane), 
                                                nn.ReLU(inplace=True)))
            
        for _ in range(4):
            cls_layers.append(nn.Sequential(nn.Conv2d(feat_plane, feat_plane, 3, 1, 1, bias=False), 
                                            NormLayer(num_channels=feat_plane) if head_norm_layer_type == 'group_norm' else NormLayer(num_features=feat_plane), 
                                            nn.ReLU(inplace=True)))
            
            reg_layers.append(nn.Sequential(nn.Conv2d(feat_plane, feat_plane, 3, 1, 1, bias=False), 
                                            NormLayer(num_channels=feat_plane) if head_norm_layer_type == 'group_norm' else NormLayer(num_features=feat_plane), 
                                            nn.ReLU(inplace=True)))
            
        self.cls_layers = nn.Sequential(*cls_layers)
        self.reg_layers = nn.Sequential(*reg_layers)
        self.cls_out_layer = nn.Conv2d(feat_plane, num_class, 3, 1, 1)
        self.reg_out_layer = nn.Conv2d(feat_plane, 4, 3, 1, 1)
        self.ctr_out_layer = nn.Conv2d(feat_plane, 1, 3, 1, 1)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_stage)]) if enable_head_scale else None
        self.enable_head_scale = enable_head_scale
        self._init_weights()

    def _init_weights(self):
        for l in self.modules():
            if isinstance(l, nn.Conv2d):
                torch.nn.init.normal_(l.weight, std=0.01)
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_out_layer.bias, bias_value)

    def forward(self, x):
        """
        Inputs:
            x: list of backbone features:
                # fpn_features_1: (b, 128, h/8, w/8);
                # fpn_features_2: (b, 256, h/16, w/16);
                # fpn_features_3: (b, 512, h/32, w/32);
        Outputs:
            cls_fms: list of feature maps
            reg_fms: list of feature maps
            ctr_fms: list of feature maps
        """
        
        cls_fms, reg_fms, ctr_fms = [], [], []

        for l, fpn in enumerate(x):
            f = self.head_stem[l](fpn)
            cls_f = self.cls_layers(f)
            reg_f = self.reg_layers(f)

            cls_fms.append(self.cls_out_layer(cls_f))
            ctr_fms.append(self.ctr_out_layer(reg_f))

            # https://github.com/aim-uofa/AdelaiDet/blob/4a3a1f7372c35b48ebf5f6adc59f135a0fa28d60/adet/modeling/fcos/fcos.py#L231
            reg = self.reg_out_layer(reg_f)
            if self.enable_head_scale:
                reg = self.scales[l](reg)
            reg = F.relu(reg)
            reg_fms.append(reg)

        # cls_fms: [(b, num_class+1, h/8, w/8), (b, num_class+1, h/16, w/16), (b, num_class+1, h/32, w/32)] / [l, t, r, b]
        # reg_fms: [(b, 4, h/8, w/8), (b, 4, h/16, w/16), (b, 4, h/32, w/32)]
        # cen_fms: [(b, 1, h/8, w/8), (b, 1, h/16, w/16), (b, 1, h/32, w/32)]
        return cls_fms, reg_fms, ctr_fms



