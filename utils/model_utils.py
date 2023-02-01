import contextlib
from copy import deepcopy
import warnings
import torch
import torch.nn as nn


__all__ = ["adjust_status", 'summary_model']


@contextlib.contextmanager
def adjust_status(module: nn.Module, training: bool = False) -> nn.Module:
    """Adjust module to training/eval mode temporarily.

    Args:
        module (nn.Module): module to adjust status.
        training (bool): training mode to set. True for train mode, False fro eval mode.

    Examples:
        >>> with adjust_status(model, training=False):
        ...     model(data)
    """
    status = {}

    def backup_status(module):
        for m in module.modules():
            # save prev status to dict
            status[m] = m.training
            m.training = training

    def recover_status(module):
        for m in module.modules():
            # recover prev status from dict
            m.training = status.pop(m)

    backup_status(module)
    yield module
    recover_status(module)


def summary_model(model, input_img_size=[640, 640], verbose=False, prefix=""):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        number_params = sum(x.numel() for x in model.parameters())
        number_gradients = sum(x.numel() for x in model.parameters() if x.requires_grad)
        number_layers = len(list(model.modules()))
        try:
            from thop import profile
            
        except (ImportError, Exception) as err:
            print(f"error occur in summary_model: {err}")
            flops = ""

        else:
            dummy_img = torch.rand(1, 3, input_img_size[0], input_img_size[1], device=next(model.parameters()).device)
            flops, params = profile(deepcopy(model), inputs=(dummy_img, ), verbose=verbose)
            flops /= 1e9 * 2
        
        if verbose:
            msg = f"Model Summary: {prefix} {number_layers} layers; {number_params} parameters; {number_gradients} gradients; {flops} GFLOPs"
            print(msg)
        return {'number_params': number_params, "number_gradients": number_gradients, "flops": flops, "number_layers": number_layers}

