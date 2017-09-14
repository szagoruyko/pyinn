import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal
from pyinn import conv2d_depthwise
from torchnet.meter import TimeMeter
from torch.backends import cudnn
cudnn.benchmark = True


def mobilenet(depth, width, depthwise_function):
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    cast = lambda x: x.cuda()

    ni = 32
    params = {'conv0': cast(kaiming_normal(torch.Tensor(ni, 3, 3, 3)))}

    for i, x in enumerate(cfg):
        no = x if isinstance(x, int) else x[0]
        params['block%d.conv0' % i] = cast(kaiming_normal(torch.Tensor(ni, 1, 3, 3)))
        params['block%d.conv1' % i] = cast(kaiming_normal(torch.Tensor(no, ni, 1, 1)))
        ni = no

    params = {k: Variable(v, requires_grad=True) for k, v in params.items()}

    def f(input, params):
        o = F.conv2d(input, params['conv0'], padding=1, stride=2)
        o = F.relu(o, inplace=True)
        for i, x in enumerate(cfg):
            stride = 1 if isinstance(x, int) else x[1]
            o = depthwise_function(o, params['block%d.conv0' % i], stride=stride, padding=1)
            o = F.conv2d(o, params['block%d.conv1' % i])
            o = F.relu(o, inplace=True)
        return o

    return f, params


def fconv2d(x, w, stride, padding):
    return F.conv2d(x, w, stride=stride, padding=padding, groups=x.size(1))


x = torch.autograd.Variable(torch.randn(256,3,224,224).cuda())

f_pyinn, params = mobilenet(18, 1, conv2d_depthwise)
f_torch, params = mobilenet(18, 1, fconv2d)

# warmup
f_pyinn(x, params).sum().backward()
f_torch(x, params).sum().backward()

meter = TimeMeter('s')

for i in range(10):
    f_torch(x, params).sum().backward()
    torch.cuda.synchronize()

print(meter.value())

meter.reset()

for i in range(10):
    f_pyinn(x, params).sum().backward()
    torch.cuda.synchronize()

print(meter.value())
