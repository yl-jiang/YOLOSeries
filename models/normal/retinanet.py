import torch
import torch.nn as nn
from utils import resnet50, RetinaNetClassification, RetinaNetRegression, RetinaNetPyramidFeatures


class RetinaNet(nn.Module):

    def __init__(self, num_anchor, num_class, resnet_layers, freeze_bn=False):
        super(RetinaNet, self).__init__()

        if resnet_layers is None:
            resnet_layers = [3, 4, 6, 3]

        self.backbone = resnet50(inplane=64, layers=resnet_layers)
        self.use_pretrained_resnet = False

        fpn_size = [self.backbone.layer2[resnet_layers[1]-1].conv3.out_channels,
                    self.backbone.layer3[resnet_layers[2]-1].conv3.out_channels,
                    self.backbone.layer4[resnet_layers[3]-1].conv3.out_channels]

        self.fpn = RetinaNetPyramidFeatures(c3_size=fpn_size[0], c4_size=fpn_size[1], c5_size=fpn_size[2], feature_size=256)
        self.classification = RetinaNetClassification(in_channels=256, inner_channels=256, num_anchor=num_anchor, num_class=num_class)
        self.regression = RetinaNetRegression(in_channels=256, inner_channels=256, num_anchor=num_anchor)

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
        # c3:(b,512,h/8,w/8); c4:(b,1024,h/16,w/16); c5:(b,2048,h/32,w/32)
        c3, c4, c5 = self.backbone(x)

        # fpn_features_1: (b, h/8, w/8, 256);
        # fpn_features_2: (b, h/16, w/16, 256);
        # fpn_features_3: (b, h/32, w/32, 256);
        # fpn_features_4: (b, h/64, w/64, 256);
        # fpn_features_5: (b, h/128, w/128, 256);
        fpn_features = self.fpn((c3, c4, c5))

        # reg_1: (b, h/8, w/8, 9*4) -> (b, h/8 * w/8 * 9, 4);
        # reg_2: (b, h/16, w/16, 9*4) -> (b, h/16 * w/16 * 9, 4);
        # reg_3: (b, h/32, w/32, 9*4) -> (b, h/32 * w/32 * 9, 4);
        # reg_4: (b, h/64, w/64, 9*4) -> (b, h/64 * w/64 * 9, 4);
        # reg_5: (b, h/16, w/128, 9*4) -> (b, h/16 * w/128 * 9, 4);
        regression = torch.cat([self.regression(feature) for feature in fpn_features], dim=1)

        # cls_1: (b, h/8, w/8, 80*9) -> (b, h/8 * w/8 * 9, 80);
        # cls_2: (b, h/16, w/16, 80*9) -> (b, h/16 * w/16 * 9, 80);
        # cls_3: (b, h/32, w/32, 80*9) -> (b, h/32 * w/32 * 9, 80);
        # cls_4: (b, h/64, w/64, 80*9) -> (b, h/64 * w/64 * 9, 80);
        # cls_5: (b, h/16, w/128, 80*9) -> (b, h/16 * w/128 * 9, 80);
        classification = torch.cat([self.classification(feature) for feature in fpn_features], dim=1)

        return regression, classification


if __name__ == '__main__':
    params = {'mode': "train", 'num_anchor': 9, 'num_class': 80, 'resnet_layers': None}
    model = RetinaNet(**params)
    reg, cls = model(torch.rand(5, 3, 448, 448))
    print(reg.shape)
    print(cls.shape)


