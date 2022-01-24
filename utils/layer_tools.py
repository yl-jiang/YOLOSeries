import torch
import torch.nn as nn
import math


def freeze_bn(m):
    """
    https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/12
    """
    if isinstance(m, nn.BatchNorm2d):
        if hasattr(m, 'weight'):
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias'):
            m.bias.requires_grad_(False)
        m.eval()  # for freeze bn layer's parameters 'running_mean' and 'running_var


class Concat(nn.Module):

    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.dim = dimension

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        return torch.cat(x, dim=self.dim)


def autopad(kernel, padding):
    if padding is None:
        return kernel // 2 if isinstance(kernel, int) else [p // 2 for p in kernel]
    else:
        return padding


class ConvBnAct(nn.Module):

    def __init__(self, in_channel, out_channel, kernel, stride, padding=0, groups=1, bias=False, act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=autopad(kernel, padding), groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class BasicBottleneck(nn.Module):

    def __init__(self, in_channel, out_channel, shorcut, groups=1, expand_ratio=0.5):
        super(BasicBottleneck, self).__init__()
        mid_channel = int(in_channel * expand_ratio)
        self.conv_bn_act_1 = ConvBnAct(in_channel, mid_channel, 1, 1)  # 不改变输入tensor的h,w
        self.conv_bn_act_2 = ConvBnAct(mid_channel, out_channel, 3, 1, 1, groups=groups)  # padding = 1, 不改变输入tensor的h,w
        self.residual = shorcut and (in_channel == out_channel)

    def forward(self, x):
        if self.residual:
            x_resiual = x.clone()
            x = self.conv_bn_act_1(x)
            x = self.conv_bn_act_2(x)
            x += x_resiual
            return x
        else:
            return self.conv_bn_act_2(self.conv_bn_act_1(x))


class BottleneckCSP(nn.Module):

    def __init__(self, in_channel, out_channel, shortcut=True, num_block=1, groups=1, bias=False):
        super(BottleneckCSP, self).__init__()
        mid_channel = int(out_channel / 2)
        self.conv_bn_act_1 = ConvBnAct(in_channel, mid_channel, 1, 1, 0)  # 不改变输入tensor的h,w
        # module2
        self.conv_2 = nn.Conv2d(in_channel, mid_channel, 1, 1, bias=bias)
        # module1
        self.conv_1 = nn.Conv2d(mid_channel, mid_channel, 1, 1, bias=bias)
        # module3
        self.concat = Concat()
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv_bn_act_2 = ConvBnAct(mid_channel * 2, out_channel, 1, 1)  # 不改变输入tensor的h,w
        self.bn = nn.BatchNorm2d(mid_channel * 2)
        self.blocks = nn.Sequential(
            *[BasicBottleneck(mid_channel, mid_channel, shortcut, groups, expand_ratio=1) for _ in range(num_block)])

    def forward(self, x):
        """

        :param x: (bn, c, h, w)
        :return:
        """
        x_resiual = x.clone()
        x_resiual = self.conv_2(x_resiual)  # (bn, c/2, h, w)
        x = self.conv_bn_act_1(x)  # (bn, c/2, h, w)
        x = self.blocks(x)  # (bn, c/2, h, w)
        x = self.conv_1(x)  # (bn, c/2, h, w)
        x = self.concat([x, x_resiual])  # (bn, c, h, w)
        x = self.act(self.bn(x))  # (bn, c, h, w)
        x = self.conv_bn_act_2(x)
        return x


class C3BottleneckCSP(nn.Module):
    """
    3 convolution layers with BottleneckCSP, a convolution layer is reduced compare to plain BottleneckCSP layer
    """
    def __init__(self, in_channel, out_channel, shortcut=True, num_block=1, groups=1, bias=False):
        super(C3BottleneckCSP, self).__init__()
        mid_channel = out_channel // 2
        self.cba1 = ConvBnAct(in_channel, mid_channel, 1, 1, groups=groups, bias=bias)
        self.cba2 = ConvBnAct(in_channel, mid_channel, 1, 1, groups=groups, bias=bias)
        self.cba3 = ConvBnAct(mid_channel*2, out_channel, 1, 1, groups=groups, bias=bias)
        self.blocks = nn.Sequential(*[BasicBottleneck(mid_channel, mid_channel, shortcut, expand_ratio=1.0) for _ in range(num_block)])
        self.concat = Concat()

    def forward(self, x):
        y1 = self.blocks(self.cba1(x))
        y2 = self.cba2(x)
        y = self.cba3(self.concat((y1, y2)))
        return y


class SEBottleneckCSP(nn.Module):
    """
    Fusion last CSP Bolck
    """

    def __init__(self, in_channel, out_channel, shortcut=True, num_block=1, groups=1, bias=False):
        super(SEBottleneckCSP, self).__init__()
        mid_channel = int(out_channel / 2)
        self.conv_bn_act_1 = ConvBnAct(in_channel, mid_channel, 1, 1, 0)  # 不改变输入tensor的h,w
        # module2
        self.conv_2 = nn.Conv2d(in_channel, mid_channel, 1, 1, bias=bias)
        # module1
        self.conv_1 = nn.Conv2d(mid_channel, mid_channel, 1, 1, bias=bias)
        # module3
        self.concat = Concat()
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv_bn_act_2 = ConvBnAct(mid_channel * 2, out_channel, 1, 1)  # 不改变输入tensor的h,w
        self.bn = nn.BatchNorm2d(mid_channel * 2)
        self.blocks = nn.Sequential(
            *[BasicBottleneck(mid_channel, mid_channel, shortcut, groups, expand_ratio=1) for _ in range(num_block)])

        self.se = SqueezeExacitationBlock(out_channel)

    def forward(self, x):
        """

        :param x: (bn, c, h, w)
        :return:
        """
        x_resiual = x.clone()
        x_resiual = self.conv_2(x_resiual)  # (bn, c/2, h, w)
        x = self.conv_bn_act_1(x)  # (bn, c/2, h, w)
        x = self.blocks(x)  # (bn, c/2, h, w)
        x = self.conv_1(x)  # (bn, c/2, h, w)
        x = self.concat([x, x_resiual])  # (bn, c, h, w)
        # add se block
        x = self.se(x)
        x = self.act(self.bn(x))  # (bn, c, h, w)
        x = self.conv_bn_act_2(x)
        return x

class SqueezeExacitationBlock(nn.Module):

    def __init__(self, in_channels):
        super(SqueezeExacitationBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        :param x: (bn, c, h, w)
        """
        features = x
        x = self.avg_pool(x)  # (bn, c, 1, 1)
        x = self.fc(x)  # (bn, c, 1, 1)
        x = self.sigmoid(x)  # (bn, c, 1, 1)
        return features * x  # (bn, c, h, w)

class Focus(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=1, stride=1, padding=0, groups=1, bias=False, act=True):
        super(Focus, self).__init__()
        self.concat = Concat()
        self.conv_bn_act = ConvBnAct(in_channel * 4, out_channel, kernel=kernel, stride=stride, padding=padding, groups=groups, bias=bias, act=act)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.concat([x[:, :, ::2, ::2], x[:, :, 1::2, ::2], x[:, :, ::2, 1::2], x[:, :, 1::2, 1::2]])  # (bn, c*4, h/2,w/2)
        x = self.conv_bn_act(x)
        return x

class SPP(nn.Module):

    def __init__(self, in_channel, out_channel, kernels=(5, 9, 13)):
        super(SPP, self).__init__()
        mid_channel = in_channel // 2
        self.conv_bn_act_1 = ConvBnAct(in_channel, mid_channel, 1, 1, 0)
        self.conv_bn_act_2 = ConvBnAct(mid_channel * (len(kernels) + 1), out_channel, 1, 1)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(k, 1, k//2) for k in kernels])
        self.concat = Concat()

    def forward(self, x):
        """

        :param x: (bn, c, h, w)
        :return:
        """
        x = self.conv_bn_act_1(x)  # (bn, c/2, h, w)
        x = [x] + [m(x) for m in self.maxpools]  # [(bn, c/2, h, w), (bn, c/2, h, w), (bn, c/2, h, w), (bn, c/2, h, w)]
        x = self.concat(x)  # (bn, 2c, h, w)
        x = self.conv_bn_act_2(x)  # (bn, out_c, h, w)
        return x

class FastSPP(nn.Module):
    """
    FastSPP与SPP的区别是，SPP是直接对输入x进行kernel分别为5，9，13的maxpooling，然后再将不同感受野的特征进行整合。
    FastSPP是仿照卷积通过层级关系的堆叠从而间接达到提取不同感受野特征的目的。
    """
    def __init__(self, in_channel, out_channel, kernel=5):
        super().__init__()
        mid_channel = in_channel // 2
        self.cba1 = ConvBnAct(in_channel, mid_channel, 1, 1, 0)
        self.cba2 = ConvBnAct(mid_channel * 4, out_channel, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel//2)

    def forward(self, x):
        x = self.cba1(x)
        x2 = self.maxpool(x)
        x3 = self.maxpool(x2)
        x4 = self.maxpool(x3)
        x5 = self.cba2(torch.cat((x, x2, x3, x4), dim=1))
        return x5


class Upsample(nn.Module):

    def __init__(self, size=None, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(size, scale_factor, mode)

    def forward(self, x):
        return self.upsample(x)


class Detect(nn.Module):

    def __init__(self, in_channels=None, out_channel=3 * 85):
        super(Detect, self).__init__()
        if in_channels is None:
            in_channels = [256, 512, 1024]
        self.detect_small = nn.Conv2d(in_channels[0], out_channel, (1,1), (1, 1), (0,0))
        self.detect_mid = nn.Conv2d(in_channels[1], out_channel, (1,1), (1, 1), (0,0))
        self.detect_large = nn.Conv2d(in_channels[2], out_channel, (1,1), (1, 1), (0,0))

    def forward(self, x):
        assert isinstance(x, list)
        assert len(x) == 3
        out_small = self.detect_small(x[0])  # n, 255, hsmall, wsmall
        out_mid = self.detect_mid(x[1])  # n, 255, hmid, wmid
        out_large = self.detect_large(x[2])  # n, 255, hlarge, wlarge
        return out_small, out_mid, out_large


def fuse_conv_bn(conv_layer, bn_layer):
    """
    BN层的参数有：alpha(BN层的weight), beta(BN层的bias), running_mean, running_var

    将batch norm层看作卷积核为1x1的卷积层:
        1. 卷积核的参数为(alpha / (sqrt(running_var) + eps));
        2. 卷积层的bias为(-(alpha * running_mean) / (sqrt(running_var) + eps) + beta)
    """
    fuseconv = nn.Conv2d(in_channels=conv_layer.in_channels, 
                            out_channels=conv_layer.out_channels, 
                            kernel_size=conv_layer.kernel_size, 
                            stride=conv_layer.stride, 
                            padding=conv_layer.padding, 
                            groups=conv_layer.groups, 
                            bias=True).requires_grad_(False).to(conv_layer.weight.device)

    conv_w = conv_layer.weight.clone().view(conv_layer.out_channels, -1)
    bn_w = torch.diag(bn_layer.weight.div(torch.sqrt(bn_layer.running_var + bn_layer.eps)))  # 为BN层创建卷积核
    fuseconv.weight.copy_(torch.mm(bn_w, conv_w).view(fuseconv.weight.shape))

    if conv_layer.bias is None:
        conv_b = torch.zeros(conv_layer.weight.size(0), device=conv_layer.weight.device)
    else:
        conv_b = conv_layer.bias
    bn_b = bn_layer.bias - (bn_layer.weight.mul(bn_layer.running_mean).div(torch.sqrt(bn_layer.eps + bn_layer.running_var)))
    fuseconv.bias.copy_(torch.mm(bn_w, conv_b.reshape(-1, 1)).reshape(-1) + bn_b)

    return fuseconv


# ======================================= DepthWise Convolution Layers ===============================================

class DepthWiseConvBnAct(nn.Module):

    def __init__(self, in_channel, out_channel, kernel, stride, padding=0, bias=False, act=True):
        super(DepthWiseConvBnAct, self).__init__()
        groups = math.gcd(in_channel, out_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=autopad(kernel, padding), groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DepthWiseBasicBottleneck(nn.Module):

    def __init__(self, in_channel, out_channel, shorcut, expand_ratio=0.5):
        super(DepthWiseBasicBottleneck, self).__init__()
        mid_channel = int(in_channel * expand_ratio)
        self.conv_bn_act_1 = DepthWiseConvBnAct(in_channel, mid_channel, 1, 1)  # 不改变输入tensor的h,w
        self.conv_bn_act_2 = DepthWiseConvBnAct(mid_channel, out_channel, 3, 1, 1)  # padding = 1, 不改变输入tensor的h,w
        self.residual = shorcut and (in_channel == out_channel)

    def forward(self, x):
        if self.residual:
            x_resiual = x.clone()
            x = self.conv_bn_act_1(x)
            x = self.conv_bn_act_2(x)
            x += x_resiual
            return x
        else:
            return self.conv_bn_act_2(self.conv_bn_act_1(x))


class DepthWiseBottleneckCSP(nn.Module):

    def __init__(self, in_channel, out_channel, shortcut=True, num_block=1, bias=False):
        super(DepthWiseBottleneckCSP, self).__init__()
        mid_channel = int(out_channel / 2)
        self.conv_bn_act_1 = DepthWiseConvBnAct(in_channel, mid_channel, 1, 1, 0)  # 不改变输入tensor的h,w
        # module2
        self.conv_2 = nn.Conv2d(in_channel, mid_channel, 1, 1, bias=bias)
        # module1
        self.conv_1 = nn.Conv2d(mid_channel, mid_channel, 1, 1, bias=bias)
        # module3
        self.concat = Concat()
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv_bn_act_2 = DepthWiseConvBnAct(mid_channel * 2, out_channel, 1, 1)  # 不改变输入tensor的h,w
        self.bn = nn.BatchNorm2d(mid_channel * 2)
        self.blocks = nn.Sequential(
            *[DepthWiseBasicBottleneck(mid_channel, mid_channel, shortcut, expand_ratio=1) for _ in range(num_block)])


class DepthWiseC3BottleneckCSP(nn.Module):
    """
    3 convolution layers with BottleneckCSP, a convolution layer is reduced compare to plain BottleneckCSP layer
    """
    def __init__(self, in_channel, out_channel, shortcut=True, num_block=1, bias=False):
        super(DepthWiseC3BottleneckCSP, self).__init__()
        mid_channel = out_channel // 2
        self.cba1 = DepthWiseConvBnAct(in_channel, mid_channel, 1, 1)
        self.cba2 = DepthWiseConvBnAct(in_channel, mid_channel, 1, 1)
        self.cba3 = DepthWiseConvBnAct(mid_channel*2, out_channel, 1, 1)
        self.blocks = nn.Sequential(*[BasicBottleneck(mid_channel, mid_channel, shortcut, expand_ratio=1.0) for _ in range(num_block)])
        self.concat = Concat()

    def forward(self, x):
        y1 = self.blocks(self.cba1(x))
        y2 = self.cba2(x)
        y = self.cba3(self.concat((y1, y2)))
        return y


# ======================================== layers for RetinaNet ==========================================

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
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
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels*self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, inplane, layers, block, num_class):
        super(ResNet, self).__init__()
        assert isinstance(layers, list) and len(layers) == 4
        self.inplane_upd = inplane
        self.layers = layers
        self.num_class = num_class

        # head layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=inplane, kernel_size=(7, 7), stride=(2, ), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=inplane)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # residual blocks
        self.layer1 = self._make_layer(block, inplane*1, self.layers[0], 1)
        self.layer2 = self._make_layer(block, inplane*2, self.layers[1], 2)
        self.layer3 = self._make_layer(block, inplane*4, self.layers[2], 2)
        self.layer4 = self._make_layer(block, inplane*8, self.layers[3], 2)

        # classifier
        if num_class is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(in_features=self.inplane_upd, out_features=num_class)

        # initialization
        # self._initialize(self)
        # self._initialize_last_bn(self)

    def _initialize(self, modules):
        """
        ordinary model initialization.
        :param modules:
        :return:
        """
        for m in modules.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
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
                nn.BatchNorm2d(num_features=planes*block.expansion))
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

        if self.num_class is not None:
            x = self.avgpool(x4)
            x = torch.flatten(x)
            return self.fc(x)
        else:
            return x2, x3, x4


def resnet50(inplane=64, layers=None, block=None, num_class=None):
    if layers is None:
        layers = [3, 4, 6, 3]
    if block is None:
        block = Bottleneck
    model = ResNet(inplane, layers, block, num_class)
    return model


class RetinaNetRegression(nn.Module):

    def __init__(self, in_channels, inner_channels, num_anchor):
        super(RetinaNetRegression, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, inner_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(inner_channels, inner_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(inner_channels, inner_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(inner_channels, inner_channels, 3, 1, 1)
        self.output = nn.Conv2d(inner_channels, num_anchor * 4, 3, 1, 1)
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
        # (b, hxwx9, 4)
        out = torch.reshape(out, shape=(b, -1, 4))
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


class RetinaNetPyramidFeatures(nn.Module):

    def __init__(self, c3_size, c4_size, c5_size, feature_size):
        super(RetinaNetPyramidFeatures, self).__init__()

        self.p5_1 = nn.Conv2d(in_channels=c5_size, out_channels=feature_size, kernel_size=1, stride=1, padding=0)
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_2 = nn.Conv2d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, stride=1, padding=1)

        self.p4_1 = nn.Conv2d(c4_size, feature_size, 1, 1, 0)
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_2 = nn.Conv2d(feature_size, feature_size, 3, 1, 1)

        self.p3_1 = nn.Conv2d(c3_size, feature_size, 1, 1, 0)
        self.p3_2 = nn.Conv2d(feature_size, feature_size, 3, 1, 1)

        self.p6 = nn.Conv2d(c5_size, feature_size, 3, 2, 1)

        self.relu = nn.ReLU(inplace=False)
        self.p7 = nn.Conv2d(feature_size, feature_size, 3, 2, 1)

    def forward(self, x):
        assert len(x) == 3

        c3, c4, c5 = x
        # print(f'c3 shape: {c3.shape} c4 shape: {c4.shape} c5 shape: {c5.shape}')
        p5 = self.p5_1(c5)
        # print(f' p5 shape: {p5.shape}')
        p5_upsample = self.p5_upsample(p5)
        # print(f' p5_upsample shape: {p5_upsample.shape}')
        p5 = self.p5_2(p5)

        p4 = self.p4_1(c4)
        # print(f'P4 shape: {p4.shape} ')
        p4 += p5_upsample
        p4_upsample = self.p4_upsample(p4)
        # print(f' p4_upsample shape: {p4_upsample.shape}')
        p4 = self.p4_2(p4)

        p3 = self.p3_1(c3)
        # print(f'P3 shape: {p3.shape} p4_upsample shape: {p4_upsample.shape}')
        p3 += p4_upsample
        p3 = self.p3_2(p3)

        p6 = self.p6(c5)

        p7 = self.relu(p6)
        p7 = self.p7(p7)

        return p3, p4, p5, p6, p7
