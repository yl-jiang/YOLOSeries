import torch
import torch.nn as nn
from utils import resnet50, RetinaNetPyramidFeatures

__all__ = ['RetinaNetExperiment']

class RetinaNetRegression(nn.Module):

    def __init__(self, in_channels, inner_channels, num_anchor):
        super(RetinaNetRegression, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, inner_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(inner_channels, inner_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(inner_channels, inner_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(inner_channels, inner_channels, 3, 1, 1)
        self.output = nn.Conv2d(inner_channels, num_anchor * 5, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.output(out)
        # print(f'reg: {out.shape}')
        # (b, 4x9, h, w)
        b, c, h, w = out.size()
        out = out.permute(0, 2, 3, 1)
        # (b, hxwx9, 5) / [, conf]
        out = torch.reshape(out, shape=(b, -1, 5))
        return out


class RetinaNetClassification(nn.Module):

    def __init__(self, in_channels, inner_channels, num_anchor, num_class):
        super(RetinaNetClassification, self).__init__()

        self.num_anchor = num_anchor
        self.num_class = num_class

        self.conv1 = nn.Conv2d(in_channels, inner_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(inner_channels, inner_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(inner_channels, inner_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(inner_channels, inner_channels, 3, 1, 1)
        self.output = nn.Conv2d(inner_channels, num_anchor * num_class, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.output(out)
        # out = self.sigmoid(self.output(out))
        # print(f'cls: {out.shape}')
        # (b, 80x9, h, w)
        b, c, h, w = out.size()
        # (b, h, w, 720)
        out = out.permute(0, 2, 3, 1)
        # (b, hxwx9, 80)
        out = torch.reshape(out, shape=(b, -1, self.num_class))
        return out

class RetinaNetExperiment(nn.Module):

    def __init__(self, num_anchor, num_class, resnet_layers, freeze_bn=False):
        super(RetinaNetExperiment, self).__init__()

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
        
        self._init_weights()

    def _init_weights(self):
        """
        初始化模型参数, 主要是对detection layers的bias参数进行特殊初始化, 参考RetinaNet那篇论文, 这种初始化方法可让网络较容易度过前期训练困难阶段
        (使用该初始化方法可能针对coco数据集有效, 在对global wheat数据集的测试中, 该方法根本train不起来)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, (nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True

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

        # reg_1: (b, h/8 , w/8  , 9*5) -> (b, h/8 * w/8 * 9   , 5);
        # reg_2: (b, h/16, w/16 , 9*5) -> (b, h/16 * w/16 * 9 , 5);
        # reg_3: (b, h/32, w/32 , 9*5) -> (b, h/32 * w/32 * 9 , 5);
        # reg_4: (b, h/64, w/64 , 9*5) -> (b, h/64 * w/64 * 9 , 5);
        # reg_5: (b, h/16, w/128, 9*5) -> (b, h/16 * w/128 * 9, 5);
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
    model = RetinaNetExperiment(**params)
    reg, cls = model(torch.rand(5, 3, 448, 448))
    print(reg.shape)
    print(cls.shape)


