import sys
from pathlib import Path
current_work_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(current_work_dir))

from torch import nn
from utils import ConvBnAct, BasicBottleneck, Upsample, Concat, maybe_mkdir, SPP
from pathlib import Path
import torch


class DarkNet21(nn.Module):

    def __init__(self, in_channel=3):
        super().__init__()
        num_blocks = [1, 2, 2, 1]
        self.conv_1 = ConvBnAct(in_channel, 32, 3, 1, 1, act=True)
        self.stage_1 = self._make_layers(num_block=1, in_channel=32*1, stride=2)
        self.stage_2 = self._make_layers(num_block=num_blocks[0], in_channel=32*2, stride=2)
        self.stage_3 = self._make_layers(num_block=num_blocks[1], in_channel=32*4, stride=2)
        self.stage_4 = self._make_layers(num_block=num_blocks[2], in_channel=32*8, stride=2)
        self.stage_5 = self._make_layers(num_block=num_blocks[3], in_channel=32*16, stride=2)
        self.conv_2 = ConvBnAct(32*32, 32*16, 1, 1)
        self.conv_3 = ConvBnAct(32*16, 32*32, 3, 1, 1)
        self.spp = SPP(32*32, 32*16)
        self.conv_4 = ConvBnAct(32*16, 32*32, 3, 1, 1, act=True)
        self.conv_5 = ConvBnAct(32*32, 32*16, 1, 1, act=True)
        self.output_features = ['stage_3', 'stage_4', 'stage_5']


    def _make_layers(self, num_block, in_channel, stride):
        cba = ConvBnAct(in_channel, in_channel*2, 3, stride, 1, act=True)
        res = [(BasicBottleneck(in_channel*2, in_channel*2, True)) for _ in range(num_block)]
        return nn.Sequential(*[cba, *res])


    def forward(self, x):
        out = {}
        x = self.stage_1(self.conv_1(x))
        out['stage_1'] = x
        x = self.stage_2(x)
        out['stage_2'] = x
        x = self.stage_3(x)
        out['stage_3'] = x
        x = self.stage_4(x)
        out['stage_4'] = x
        x = self.conv_5(self.conv_4(self.spp(self.conv_3(self.conv_2(self.stage_5(x))))))
        out['stage_5'] = x
        return {k: v for k, v in out.items() if k in self.output_features}


class DarkNet53(nn.Module):

    def __init__(self, in_channel=3):
        super().__init__()
        num_blocks = [2, 8, 8, 4]
        self.conv_1 = ConvBnAct(in_channel, 32, 3, 1, 1, act=True)
        self.stage_1 = self._make_layers(num_block=1, in_channel=32*1, stride=2)
        self.stage_2 = self._make_layers(num_block=num_blocks[0], in_channel=32*2, stride=2)
        self.stage_3 = self._make_layers(num_block=num_blocks[1], in_channel=32*4, stride=2)
        self.stage_4 = self._make_layers(num_block=num_blocks[2], in_channel=32*8, stride=2)
        self.stage_5 = self._make_layers(num_block=num_blocks[3], in_channel=32*16, stride=2)
        self.conv_2 = ConvBnAct(32*32, 32*16, 1, 1)
        self.conv_3 = ConvBnAct(32*16, 32*32, 3, 1, 1)
        self.spp = SPP(32*32, 32*16)
        self.conv_4 = ConvBnAct(32*16, 32*32, 3, 1, 1, act=True)
        self.conv_5 = ConvBnAct(32*32, 32*16, 1, 1, act=True)
        self.output_features = ['stage_3', 'stage_4', 'stage_5']


    def _make_layers(self, num_block, in_channel, stride):
        cba = ConvBnAct(in_channel, in_channel*2, 3, stride, 1, act=True)
        res = [(BasicBottleneck(in_channel*2, in_channel*2, True)) for _ in range(num_block)]
        return nn.Sequential(*[cba, *res])


    def forward(self, x):
        out = {}
        x = self.stage_1(self.conv_1(x))
        out['stage_1'] = x
        x = self.stage_2(x)
        out['stage_2'] = x
        x = self.stage_3(x)
        out['stage_3'] = x
        x = self.stage_4(x)
        out['stage_4'] = x
        x = self.conv_5(self.conv_4(self.spp(self.conv_3(self.conv_2(self.stage_5(x))))))
        out['stage_5'] = x
        return {k: v for k, v in out.items() if k in self.output_features}


class Detect(nn.Module):

    def __init__(self, in_channels=[256, 512, 1024], mid_channel=256, wid_mul=1.0, num_classes=80):
        super().__init__()
        self.num_anchors = 1
        self.num_classes = num_classes

        self.pred_small = self._make_layers(int(in_channels[0] * wid_mul), int(mid_channel * wid_mul))
        self.pred_middle = self._make_layers(int(in_channels[1] * wid_mul), int(mid_channel * wid_mul))
        self.pred_large = self._make_layers(int(in_channels[2] * wid_mul), int(mid_channel * wid_mul))


    def _make_layers(self, in_c, mid_c):
        stem = ConvBnAct(in_c, mid_c, 1, 1, act=True)
        cls = nn.Sequential(
                ConvBnAct(mid_c, mid_c, 3, 1, 1, act=True), 
                ConvBnAct(mid_c, mid_c, 3, 1, 1, act=True), 
                nn.Conv2d(mid_c, int(self.num_anchors * self.num_classes), 1, 1)
        )

        conv = nn.Sequential(
                ConvBnAct(mid_c, mid_c, 3, 1, 1, act=True), 
                ConvBnAct(mid_c, mid_c, 3, 1, 1, act=True))

        reg = nn.Conv2d(mid_c, 4, 1, 1)
        cof = nn.Conv2d(mid_c, int(self.num_anchors * 1), 1, 1)
        return nn.ModuleDict({'stem': stem, 'conv': conv, 'cls': cls, 'reg': reg, 'cof': cof})

    def forward_each(self, layers, x):
        x = layers['stem'](x)  # stem
        cls_pred = layers['cls'](x)  # classification
        feat = layers['conv'](x)  # extract features for regression and confidence
        reg_pred = layers['reg'](feat)  # regression
        cof_pred = layers['cof'](feat)  # confidence
        return {'cls_pred': cls_pred, 'ref_pred': reg_pred, 'cof_pred': cof_pred}

    def forward(self, x):
        x_s, x_m, x_l = x['x_s'], x['x_m'], x['x_l']
        pred_s = self.forward_each(self.pred_small, x_s)
        pred_m = self.forward_each(self.pred_middle, x_m)
        pred_l = self.forward_each(self.pred_large, x_l)
        return {'pred_s': pred_s, 'pred_m': pred_m, 'pred_l': pred_l}


class YoloXSmall(nn.Module):

    def __init__(self, in_channel=3, num_classes=80):
        super().__init__()
        self.backbone = DarkNet53(in_channel)
        self.upsample = Upsample()
        self.cat = Concat()
        self.cba_middle = ConvBnAct(512, 256, 1, 1, act=True)
        self.cba_small = ConvBnAct(256, 128, 1, 1, act=True)
        self.branch_middle = self._make_layers(512+256, 512, 256)
        self.branch_small = self._make_layers(256+128, 256, 128)
        self.head = Detect(in_channels=[128, 256, 512], mid_channel=128, wid_mul=1.0, num_classes=num_classes)

    def _make_layers(self, in_c, mid_c, out_c):  # in_c: 512; out_c 256
        layers = [
            ConvBnAct(in_c, out_c, 1, 1, act=True),   # in: 512+256; out: 256
            ConvBnAct(out_c, mid_c, 3, 1, 1, act=True),   # in: 256; out: 512
            ConvBnAct(mid_c, out_c, 1, 1, act=True), 
            ConvBnAct(out_c, mid_c, 3, 1, 1, act=True), 
            ConvBnAct(mid_c, out_c, 1, 1, act=True)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        
        # backbone
        backbone =self.backbone(x)
        x_s = backbone['stage_3']
        x_m = backbone['stage_4']
        x_l = backbone['stage_5']

        # Neck
        x_m_in = self.cba_middle(x_l)
        x_m_x = self.upsample(x_m_in)
        x_m_x = self.cat([x_m_x, x_m])
        x_m = self.branch_middle(x_m_x)

        x_s_in = self.cba_small(x_m)
        x_s_x = self.upsample(x_s_in)
        x_s_x = self.cat([x_s_x, x_s])
        x_s = self.branch_small(x_s_x)

        neck = {'x_s': x_s, 'x_m': x_m, 'x_l': x_l}

        # head
        preds = self.head(neck)

        return preds


if __name__ == "__main__":

    dummy = torch.rand(1, 3, 224, 224)
    yolox = YoloXSmall()
    out = yolox(dummy)
    

