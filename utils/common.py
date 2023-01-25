import numbers
import torch
import time
import shutil
import warnings
from pathlib import Path
import numba
import numpy as np


__all__ = ["padding", 'is_parallel', 'catch_warnnings', 'numba_clip', 'maybe_mkdir', 'time_synchronize', 
          'is_exists', 'clear_dir', 'compute_resize_scale', 'compute_featuremap_shape']

def padding(hw, factor=32):
    if isinstance(hw, numbers.Real):
        hw = [hw, hw]
    else:
        assert len(hw) == 2, f"input image size's format should like (h, w)"
    h, w = hw
    h_mod = h % factor
    w_mod = w % factor
    if h_mod > 0:
        h = (h // factor + 1) * factor
    if w_mod > 0:
        w = (w // factor + 1) * factor
    return h, w


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        torch.nn.parallel.DataParallel,
        torch.nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


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


def compute_resize_scale(img, min_side, max_side):
    if min_side is None:
        min_side = 800
    if max_side is None:
        max_side = 1300
    height, width, _ = img.shape

    scale = min(min_side / height, min_side / width)
    if scale * max(height, width) > max_side:
        scale = min(max_side / height, max_side / width)
    return scale


def compute_featuremap_shape(img_shape, pyramid_level):
    """
    compute feature map's shape based on pyramid level.
    :param img_shape: 3 dimension / [h, w, c]
    :param pyramid_level: int
    :return:
    """
    img_shape = np.array(img_shape)
    fm_shape = (img_shape - 1) // (2**pyramid_level) + 1
    return fm_shape

