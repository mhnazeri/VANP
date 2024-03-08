"""General utility functions"""

from time import time
import functools
import torch
from omegaconf import OmegaConf


def get_conf(name: str):
    """Returns yaml config file in DictConfig format

    Args:
        name: (str) name of the yaml file
    """
    name = name if name.split(".")[-1] == "yaml" else f"{name}.yaml"
    return OmegaConf.load(name)


def timeit(fn):
    """Calculate time taken by fn().

    A function decorator to calculate the time a function needed for completion on GPU.
    returns: the function result and the time taken
    """
    # first, check if cuda is available
    cuda = torch.cuda.is_available()
    if cuda:

        @functools.wraps(fn)
        def wrapper_fn(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            # torch.cuda.synchronize()
            # t1 = time()
            start.record()
            result = fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            # t2 = time()
            # take = t2 - t1
            return result, start.elapsed_time(end) / 1000

    else:

        @functools.wraps(fn)
        def wrapper_fn(*args, **kwargs):
            t1 = time()
            result = fn(*args, **kwargs)
            t2 = time()
            take = t2 - t1
            return result, take

    return wrapper_fn
