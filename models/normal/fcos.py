import torch
import torch.nn as nn
from utils import resnet50, Scale
import math
import torch.nn.functional as F
from functools import partial

__all__ = ['FCOSBaseline']


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, out_channels)
        self.downsample = downsample

    def forward(self, x):
        identify = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            out = self.downsample(out)
        out += identify
        return self.relu(out)


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn3 = nn.GroupNorm(32, out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identify = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identify = self.downsample(x)
        out += identify
        return self.relu(out)

class GroupNormResNet(nn.Module):

    def __init__(self, inplane, layers, block):
        super(GroupNormResNet, self).__init__()
        assert isinstance(layers, list) and len(layers) == 4
        self.inplane_upd = inplane
        self.layers = layers

        # head layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=inplane, kernel_size=(7, 7), stride=(2, ), padding=3, bias=False)
        self.bn1 = nn.GroupNorm(32, inplane, 1e-5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # residual blocks
        self.layer1 = self._make_layer(block, inplane*1, self.layers[0], 1)
        self.layer2 = self._make_layer(block, inplane*2, self.layers[1], 2)
        self.layer3 = self._make_layer(block, inplane*4, self.layers[2], 2)
        self.layer4 = self._make_layer(block, inplane*8, self.layers[3], 2)

        # initialization
        self._initialize(self)
        self._initialize_last_bn(self)

    def _initialize(self, modules):
        """
        ordinary model initialization.
        :param modules:
        :return:
        """
        for m in modules.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.bias, 0.)

    def _initialize_last_bn(self, modules):
        """
        Zero-initialize the last BN in each residual branch,
        so that the residual branch starts with zeros, and each residual block behaves like an identity.
        This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        :param modules:
        :return:
        """
        for m in modules.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0.)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0.)

    def _make_layer(self, block, planes, block_num, stride):
        # stride = 1的Bottleneck会扩充channel，stride = 2的Bottleneck会downsample image且会扩充channel
        if stride != 1 or self.inplane_upd != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplane_upd, out_channels=planes*block.expansion, kernel_size=(1, 1), stride=stride, padding=0, bias=False),
                nn.GroupNorm(32, planes*block.expansion))
        else:
            downsample = None

        layers = [block(self.inplane_upd, planes, stride, downsample)]
        self.inplane_upd = planes * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.inplane_upd, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x2, x3, x4


def groupnorm_resnet(inplane=64, layers=None, block=None):
    if layers is None:
        layers = [3, 4, 6, 3]
    if block is None:
        block = Bottleneck
    model = GroupNormResNet(inplane, layers, block)
    return model


class FCOSFPN(nn.Module):

    def __init__(self, c3_size, c4_size, c5_size, feature_size):
        super(FCOSFPN, self).__init__()

        self.p5_1 = nn.Conv2d(in_channels=c5_size, out_channels=feature_size, kernel_size=1, stride=1, padding=0)
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_2 = nn.Conv2d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, stride=1, padding=1)

        self.p4_1 = nn.Conv2d(c4_size, feature_size, 1, 1, 0)
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_2 = nn.Conv2d(feature_size, feature_size, 3, 1, 1)

        self.p3_1 = nn.Conv2d(c3_size, feature_size, 1, 1, 0)
        self.p3_2 = nn.Conv2d(feature_size, feature_size, 3, 1, 1)

        self.p6 = nn.Conv2d(feature_size, feature_size, 3, 2, 1)
        self.p7 = nn.Conv2d(feature_size, feature_size, 3, 2, 1)

    def forward(self, x):
        assert len(x) == 3

        c3, c4, c5 = x
        p5 = self.p5_2(self.p5_1(c5))

        p4 = self.p4_1(c4)
        p5_upsample = self.p5_upsample(p5)
        p4 += p5_upsample
        p4 = self.p4_2(p4)

        p3 = self.p3_1(c3)
        p4_upsample = self.p4_upsample(p4)
        p3 += p4_upsample
        p3 = self.p3_2(p3)

        p6 = self.p6(p5)
        p7 = self.p7(p6)

        return p3, p4, p5, p6, p7

class FCOSHead(nn.Module):

    def __init__(self, in_channels, num_class, norm_layer_type='group_norm', enable_head_scale=False):
        super(FCOSHead, self).__init__()

        cls_layers, reg_layers = [], []
        if norm_layer_type.lower() == "batch_norm":
            NormLayer = nn.BatchNorm2d
        elif norm_layer_type.lower() == 'group_norm':
            NormLayer = partial(nn.GroupNorm, num_groups=32)
        else:
            raise RuntimeError(f'unknow head_norm_layer_type {norm_layer_type}, must be "batch_norm" or "group_norm".')
        
        for _ in range(4):
            cls_layers.append(nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False), 
                                            NormLayer(num_channels=in_channels) if norm_layer_type == 'group_norm' else NormLayer(num_features=in_channels), 
                                            nn.ReLU(inplace=True)))
            reg_layers.append(nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False), 
                                            NormLayer(num_channels=in_channels) if norm_layer_type == 'group_norm' else NormLayer(num_features=in_channels), 
                                            nn.ReLU(inplace=True)))

        self.cls_layers = nn.Sequential(*cls_layers)
        self.reg_layers = nn.Sequential(*reg_layers)
        self.cls_out_layer = nn.Conv2d(in_channels, num_class, 3, 1, 1)
        self.reg_out_layer = nn.Conv2d(in_channels, 4, 3, 1, 1)
        self.ctr_out_layer = nn.Conv2d(in_channels, 1, 3, 1, 1)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)]) if enable_head_scale else None
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
                # fpn_features_1: (b, 256, h/8, w/8);
                # fpn_features_2: (b, 256, h/16, w/16);
                # fpn_features_3: (b, 256, h/32, w/32);
                # fpn_features_4: (b, 256, h/64, w/64);
                # fpn_features_5: (b, 256, h/128, w/128);
        Outputs:
            cls_fms: list of feature maps
            reg_fms: list of feature maps
            ctr_fms: list of feature maps
        """
        cls_fms, reg_fms, ctr_fms = [], [], []

        for l, fpn in enumerate(x):
            cls_f = self.cls_layers(fpn)
            reg_f = self.reg_layers(fpn)

            cls_fms.append(self.cls_out_layer(cls_f))
            ctr_fms.append(self.ctr_out_layer(reg_f))

            # https://github.com/aim-uofa/AdelaiDet/blob/4a3a1f7372c35b48ebf5f6adc59f135a0fa28d60/adet/modeling/fcos/fcos.py#L231
            reg = self.reg_out_layer(reg_f)
            if self.enable_head_scale:
                reg = self.scales[l](reg)
            reg = F.relu(reg)
            reg_fms.append(reg)

        return cls_fms, reg_fms, ctr_fms


class FCOSBaseline(nn.Module):

    def __init__(self, num_class, resnet_layers=[3, 4, 6, 3], freeze_bn=False, norm_layer_type='group_norm', enable_head_scale=False):
        super(FCOSBaseline, self).__init__()

        if resnet_layers is None:
            resnet_layers = [3, 4, 6, 3]
        if norm_layer_type == "batch_norm":
            self.backbone = resnet50(inplane=64, layers=resnet_layers)
        elif norm_layer_type == 'group_norm':
            self.backbone = groupnorm_resnet(inplane=64, layers=resnet_layers)

        self.use_pretrained_resnet = False

        fpn_size = [self.backbone.layer2[resnet_layers[1]-1].conv3.out_channels,
                    self.backbone.layer3[resnet_layers[2]-1].conv3.out_channels,
                    self.backbone.layer4[resnet_layers[3]-1].conv3.out_channels]

        self.fpn = FCOSFPN(c3_size=fpn_size[0], c4_size=fpn_size[1], c5_size=fpn_size[2], feature_size=256)
        # 增加一个背景类
        self.head = FCOSHead(in_channels=256, num_class=num_class, norm_layer_type=norm_layer_type, enable_head_scale=enable_head_scale)

        self._init_weights()

        if freeze_bn:  # only do this for training
            self._freeze_bn()

    def _init_weights(self):
        # initialization
        for modules in [self.head.cls_layers, self.head.reg_layers,
                        self.head.ctr_out_layer, self.head.reg_out_layer,
                        self.head.cls_out_layer]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)

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
        # c3:(b,512,h/8,w/8); c4:(b,1024,h/16,w/16); c5:(b,2048,h/32,w/32)
        c3, c4, c5 = self.backbone(x)

        # fpn_features_1: (b, 256, h/8, w/8);
        # fpn_features_2: (b, 256, h/16, w/16);
        # fpn_features_3: (b, 256, h/32, w/32);
        # fpn_features_4: (b, 256, h/64, w/64);
        # fpn_features_5: (b, 256, h/128, w/128);
        fpn_features = self.fpn((c3, c4, c5))

        # cls_fms: [(b, num_class, h/8, w/8), (b, num_class, h/16, w/16), (b, num_class, h/32, w/32), (b, num_class, h/64, w/64), (b, num_class, h/128, w/128)] / [l, t, r, b]
        # reg_fms: [(b, 4, h/8, w/8), (b, 4, h/16, w/16), (b, 4, h/32, w/32), (b, 4, h/64, w/64), (b, 4, h/128, w/128)]
        # cen_fms: [(b, 1, h/8, w/8), (b, 1, h/16, w/16), (b, 1, h/32, w/32), (b, 1, h/64, w/64), (b, 4, h/128, w/128)]
        cls_fms, reg_fms, cen_fms = self.head(fpn_features)

        return cls_fms, reg_fms, cen_fms


if __name__ == '__main__':
    params = {'num_class': 80, 'resnet_layers': None}
    model = FCOSBaseline(**params)
    cls_fms, reg_fms, ctr_fms = model(torch.rand(5, 3, 448, 448))

    for c, r, p in zip(cls_fms, reg_fms, ctr_fms):
        print(f"{c.shape}; {r.shape}; {p.shape}")

