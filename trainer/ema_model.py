import torch
from copy import deepcopy
import numpy as np

__all__ = ['ExponentialMovingAverageModel']

class ExponentialMovingAverageModel:
    """
    从始至终维持一个model, 并不断更新该model的参数, 但该mdoel仅仅是为了inference。

    随着训练的进行, 越靠后面的模型参数对ema模型的影响越大。
    """
    def __init__(self, model, decay_ratio=0.9999, update_num=0):
        self.ema = deepcopy(model).eval()
        self.update_num = update_num
        self.get_decay_weight = lambda x: decay_ratio * (1 - np.exp(-x / 2000))
        for parm in self.ema.parameters():
            parm.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.update_num += 1
            decay_weight = self.get_decay_weight(self.update_num)
            cur_state_dict = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= decay_weight
                    v += (1 - decay_weight) * cur_state_dict[k].detach()
