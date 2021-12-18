import torch
import torch.nn as nn
import math


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
        self.cba1 = ConvBnAct(in_channel, mid_channel, 1, 1)
        self.cba2 = ConvBnAct(in_channel, mid_channel, 1, 1)
        self.cba3 = ConvBnAct(mid_channel*2, out_channel, 1, 1)
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
