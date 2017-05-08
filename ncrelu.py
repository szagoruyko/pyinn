from collections import namedtuple, defaultdict
from pynvrtc.compiler import Program
import torch
from cupy.cuda.function import Module
from cupy.cuda import device

CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K


Stream = namedtuple('Stream', ['ptr'])


def get_compute_arch(t):
    return 'compute_%s' % device.Device().compute_capability


# k = i_n * 2CHW + i_c * HW + i_h * W + i_w
# i_n * 2CHW + (i_c + C) * HW + i_h * W + i_w =
# k + chw


kernels = '''
extern "C"
__global__ void ncrelu_forward(float *dst, unsigned char* mask, const float *src, int chw, int total)
{
   int tx = blockIdx.x * blockDim.x + threadIdx.x;
   if(tx >= total)
      return;

   float v = src[tx];
   unsigned char flag = v >= 0;
   mask[tx] = flag;
   dst[tx + tx / chw * chw] = flag ? v : 0.f;
   dst[tx + tx / chw * chw + chw] = flag ? 0.f : v;
}

extern "C"
__global__ void ncrelu_backward(float *grad_input, const unsigned char *mask, const float *grad_output, int chw, int bs)
{
   int tx = blockIdx.x * blockDim.x + threadIdx.x;
   int ty = blockIdx.y * blockDim.y + threadIdx.y;
   if(tx >= bs || ty >= chw)
      return;

   int i = tx * chw + ty;
   grad_output += 2*chw*tx + ty;
   bool flag = mask[i];
   grad_input[i] = flag ? grad_output[0] : grad_output[chw];
}
'''

fwd_modules = defaultdict(lambda: None)
bwd_modules = defaultdict(lambda: None)


def compile(modules, input):
    if modules[input.get_device()] is None:
        print 'compiling for dev', input.get_device()
        program = Program(kernels, 'ncrelu.cu')
        ptx = program.compile(['-arch=' + get_compute_arch(input)])

        module = Module()
        module.load(bytes(ptx.encode()))
        modules[input.get_device()] = module
    else:
        module = modules[input.get_device()]
    return module


def ncrelu_forward(input):
    module = compile(fwd_modules, input)

    assert input.dim() == 4
    n, c, h, w = input.size()

    output = input.new(n, 2 * c, h, w)
    mask = torch.cuda.ByteTensor(input.size())

    f = module.get_function('ncrelu_forward')

    f(args=[output.data_ptr(), mask.data_ptr(), input.data_ptr(), c*h*w, input.numel()],
      block=(CUDA_NUM_THREADS,1,1),
      grid=(GET_BLOCKS(input.numel()),1,1),
      stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
    return output, mask


def ncrelu_backward(grad_output, mask):
    module = compile(bwd_modules, grad_output)
    assert grad_output.get_device() == mask.get_device()
    n, c, h, w = mask.size()
    grad_input = grad_output.new(mask.size())

    f = module.get_function('ncrelu_backward')
    f(args=[grad_input.data_ptr(), mask.data_ptr(), grad_output.data_ptr(), c*h*w, n],
      block=(32,32,1),
      grid=(GET_BLOCKS(n, 32), GET_BLOCKS(c*h*w, 32), 1),
      stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
    return grad_input


class NCRELU(torch.autograd.Function):

    def forward(self, input):
        output, self.mask = ncrelu_forward(input)
        return output

    def backward(self, grad_output):
        return ncrelu_backward(grad_output, self.mask)


def ncrelu(input):
    return NCRELU()(input)
