import shutil
from pathlib import Path
import torch
import time
from copy import deepcopy
import warnings
import numba


def catch_warnnings(fn):
    def wrapper(instance):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(instance)
    return wrapper

@numba.njit
def numba_clip(x, amin, amax):
    y = x.flatten()
    for i, a in enumerate(y):
        if a < amin:
            y[i] = 0.5
        if a > amax:
            y[i] = 0.7
    return y.reshape(x.shape)


def maybe_mkdir(dirname):
    if isinstance(dirname, str):
        dirname = Path(dirname)
    if not dirname.exists():
        dirname.mkdir(parents=True)


def time_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def is_exists(path_str):
    return Path(path_str).exists()


def clear_dir(dirname):
    if isinstance(dirname, str):
        dirname = Path(dirname)
    if dirname.exists():
        shutil.rmtree(str(dirname))  # shutil.rmtree会将传入的文件夹整个删除
    dirname.mkdir(parents=True)


def summary_model(model, input_img_size=[640, 640], verbose=False, prefix=""):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        number_params = sum(x.numel() for x in model.parameters())
        number_gradients = sum(x.numel() for x in model.parameters() if x.requires_grad)
        number_layers = len(list(model.modules()))
        try:
            from thop import profile
            dummy_img = torch.rand(1, 3, input_img_size[0], input_img_size[1], device=next(model.parameters()).device)
            flops, params = profile(deepcopy(model), inputs=(dummy_img, ), verbose=verbose)
            flops /= 1e9 * 2
        except (ImportError, Exception) as err:
            print(f"error occur in summary_model: {err}")
            flops = ""
        
        if verbose:
            msg = f"Model Summary: {prefix} {number_layers} layers; {number_params} parameters; {number_gradients} gradients; {flops} GFLOPs"
            print(msg)
        return {'number_params': number_params, "number_gradients": number_gradients, "flops": flops, "number_layers": number_layers}


