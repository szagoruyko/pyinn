from collections import namedtuple
from cupy.cuda import device
import torch


Stream = namedtuple('Stream', ['ptr'])


def get_compute_arch(t):
    print 'compute_%s' % device.Device().compute_capability
    return 'compute_%s' % device.Device().compute_capability


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'
