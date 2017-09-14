PyINN
=====

CuPy implementations of fused PyTorch ops.

PyTorch version of [imagine-nn](https://github.com/szagoruyko/imagine-nn)

The purpose of this package is to contain CUDA ops written in Python
with CuPy, which is not a PyTorch dependency.

An alternative to CuPy would be <https://github.com/pytorch/extension-ffi>,
but it requires a lot of wrapping code like <https://github.com/sniklaus/pytorch-extension>,
so doesn't really work with quick prototyping.

Another advantage of CuPy over C code is that dimensions of each op
are known at JIT-ing time, and compiled kernels potentially can be faster.
Also, the first version of the package was in PyCUDA, but it can't work with
PyTorch multi-GPU.

On Maxwell Titan X `pyinn.conv2d_depthwise` MobileNets are ~2.6x faster than `F.conv2d` [benchmark.py](test/benchmark.py)


## Installation

```
pip install git+https://github.com/szagoruyko/pyinn.git@master
```

## Example

```python
import torch
from torch.autograd import Variable
import pyinn as P
x = Variable(torch.randn(1,4,5,5).cuda())
w = Variable(torch.randn(4,1,3,3).cuda())
y = P.conv2d_depthwise(x, w, padding=1)
```

or with modules interface:

```python
from pyinn.modules import Conv2dDepthwise
module = Conv2dDepthwise(channels=4, kernel_size=3, padding=1).cuda()
y = module(x)
```

## Documentation 

### conv2d_depthwise

Implements depthwise convolution as in <https://arxiv.org/abs/1704.04861>
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

CUDA kernels from https://github.com/BVLC/caffe/pull/5665

CPU side is done by `F.conv2d`.

Equivalent to:

```python
F.conv2d(input, weight, groups=input.size(1))
```

Inputs and arguments are the same with `F.conv2d`


### dgmm

Multiplication with a diagonal matrix.

Used CUDA dgmm function, sometimes is faster than expand.

In torch functions does `input.mm(x.diag())`. Both left and right
mutliplications are supported.

Args:
    input: 2D tensor
    x: 1D tensor
    
    
### cdgmm

Complex multiplication with a diagonal matrix.

Does `input.mm(x.diag())` where input and x are complex.

Args:
    input: 3D tensor with last dimension of size 2
    x: 2D tensor with last dimension of size 2
    
    
### NCReLU

Applies NCReLU (negative concatenated ReLU) nonlinearity.

Does `torch.cat([x.clamp(min=0), x.clamp(max=0)], dim=1)` in a single fused op.

Used in <https://arxiv.org/abs/1706.00388>
DiracNets: Training Very Deep Neural Networks Without Skip-Connections

Args:
    input: 4D tensor


### im2col and col2im

Rearrange image blocks into columns.

The representation is used to perform GEMM-based convolution.

Output is 5D (or 6D in case of minibatch) tensor.

Minibatch implementation is inefficient, and could be done in a single CUDA kernel.
