import torch
import torch.nn as nn
from utils import resnet50, Scale, RetinaNetPyramidFeatures
import math
import torch.nn.functional as F
from functools import partial

__all__ = ['FCOSBaseline']

class FCOSHead(nn.Module):

    def __init__(self, in_channels, num_class, head_norm_layer_type='group_norm', enable_head_scale=False):
        super(FCOSHead, self).__init__()

        cls_layers, reg_layers = [], []
        if head_norm_layer_type.lower() == "barch_norm":
            NormLayer = nn.BatchNorm2d
        elif head_norm_layer_type.lower() == 'group_norm':
            NormLayer = partial(nn.GroupNorm, num_groups=32)
        else:
            raise RuntimeError(f'unknow head_norm_layer_type {head_norm_layer_type}, must be "batch_norm" or "group_norm".')
        
        for _ in range(4):
            cls_layers.append(nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False), 
                                            NormLayer(num_channels=in_channels) if head_norm_layer_type == 'group_norm' else NormLayer(num_features=in_channels), 
                                            nn.ReLU(inplace=True)))
            reg_layers.append(nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False), 
                                            NormLayer(num_channels=in_channels) if head_norm_layer_type == 'group_norm' else NormLayer(num_features=in_channels), 
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

    def __init__(self, num_class, resnet_layers, freeze_bn=False, head_norm_layer_type='group_norm', enable_head_scale=False):
        super(FCOSBaseline, self).__init__()

        if resnet_layers is None:
            resnet_layers = [3, 4, 6, 3]

        self.backbone = resnet50(inplane=64, layers=resnet_layers)
        self.use_pretrained_resnet = False

        fpn_size = [self.backbone.layer2[resnet_layers[1]-1].conv3.out_channels,
                    self.backbone.layer3[resnet_layers[2]-1].conv3.out_channels,
                    self.backbone.layer4[resnet_layers[3]-1].conv3.out_channels]

        self.fpn = RetinaNetPyramidFeatures(c3_size=fpn_size[0], c4_size=fpn_size[1], c5_size=fpn_size[2], feature_size=256)
        self.head = FCOSHead(in_channels=256, num_class=num_class, head_norm_layer_type=head_norm_layer_type, enable_head_scale=enable_head_scale)

        if freeze_bn:  # only do this for training
            self._freeze_bn()

    def _init_weights(self):
        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
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

