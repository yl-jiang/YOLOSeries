import sys
from pathlib import Path
current_work_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(current_work_dir))
from utils import ConvBnAct, BasicBottleneck, SPP
import torch.nn as nn
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



if __name__ == "__main__":
    dummy = torch.rand(1, 3, 224, 224)
    darknet = DarkNet21()
    out = darknet(dummy)
    a =    1