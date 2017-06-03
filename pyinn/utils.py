from collections import namedtuple
from cupy.cuda import device


Stream = namedtuple('Stream', ['ptr'])


def get_compute_arch(t):
    return 'compute_%s' % device.Device().compute_capability
