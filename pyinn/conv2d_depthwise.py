from torch.autograd import Function
import torch
from torch.nn.modules.utils import _pair
from pyinn.utils import Dtype, Stream, load_kernel
import torch.nn.functional as F

CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_conv2d_depthwise_kernel = kernel_loop + '''
extern "C"
__global__ void conv2d_dw_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${top_height} / ${top_width};
    const int c = (index / ${top_height} / ${top_width}) % ${channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const ${Dtype}* weight = weight_data + c * ${kernel_h} * ${kernel_w};
    ${Dtype} value = 0;
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
            * ${bottom_width} + w_in;
          value += (*weight) * bottom_data[offset];
        }
        ++weight;
      }
    }
    top_data[index] = value;
  }
}
'''


_conv2d_depthwise_kernel_backward_grad_input = kernel_loop + '''
extern "C"
__global__ void conv2d_dw_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};
    const ${Dtype}* weight = weight_data + c * ${kernel_h} * ${kernel_w};
    ${Dtype} value = 0;
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
        const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
        if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
          const int h_out = h_out_s / ${stride_h};
          const int w_out = w_out_s / ${stride_w};
          if ((h_out >= 0) && (h_out < ${top_height})
                && (w_out >= 0) && (w_out < ${top_width})) {
            const int offset = ((n * ${channels} + c) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            value += (*weight) * top_diff[offset];
          }
        }
        ++weight;
      }
    }
    bottom_diff[index] = value;
  }
}
'''


_conv2d_depthwise_kernel_backward_grad_weight = kernel_loop + '''
extern "C"
__global__ void conv2d_dw_backward_grad_weight_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* const buffer_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int kh = (index / ${kernel_w} / ${num} / ${top_height} / ${top_width})
          % ${kernel_h};
    const int kw = (index / ${num} / ${top_height} / ${top_width}) % ${kernel_w};
    const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
    const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
    if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
      const int c = index / ${kernel_h} / ${kernel_w} / ${num} / ${top_height} / ${top_width};
      const int n = (index / ${top_height} / ${top_width}) % ${num};
      const int top_offset = ((n * ${channels} + c) * ${top_height} + h)
            * ${top_width} + w;
      const int bottom_offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
            * ${bottom_width} + w_in;
      buffer_data[index] = top_diff[top_offset] * bottom_data[bottom_offset];
    } else {
      buffer_data[index] = 0;
    }
  }
}
'''


class Conv2dDepthwise(Function):

    def __init__(self, stride, padding, dilation):
        super(Conv2dDepthwise, self).__init__()
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

    def forward(self, input, weight):
        assert input.dim() == 4 and input.is_cuda and weight.is_cuda
        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:]
        output_h = int((height + 2 * self.padding[0] - (self.dilation[0] * (kernel_h - 1) + 1)) / self.stride[0] + 1)
        output_w = int((width + 2 * self.padding[1] - (self.dilation[1] * (kernel_w - 1) + 1)) / self.stride[1] + 1)

        output = input.new(batch_size, channels, output_h, output_w)
        n = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel('conv2d_dw_forward_kernel', _conv2d_depthwise_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, channels=channels,
                            bottom_height=height, bottom_width=width,
                            top_height=output_h, top_width=output_w,
                            kernel_h=kernel_h, kernel_w=kernel_w,
                            stride_h=self.stride[0], stride_w=self.stride[1],
                            dilation_h=self.dilation[0], dilation_w=self.dilation[1],
                            pad_h=self.padding[0], pad_w=self.padding[1])
            f(block=(CUDA_NUM_THREADS,1,1),
              grid=(GET_BLOCKS(n),1,1),
              args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        self.save_for_backward(input, weight)
        return output

    def backward(self, grad_output):
        assert grad_output.is_cuda and grad_output.is_contiguous()
        input, weight = self.saved_tensors

        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:]
        output_h, output_w = grad_output.size()[2:]

        grad_input, grad_weight = None, None

        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels,
                   bottom_height=height, bottom_width=width,
                   top_height=output_h, top_width=output_w,
                   kernel_h=kernel_h, kernel_w=kernel_w,
                   stride_h=self.stride[0], stride_w=self.stride[1],
                   dilation_h=self.dilation[0], dilation_w=self.dilation[1],
                   pad_h=self.padding[0], pad_w=self.padding[1])

        with torch.cuda.device_of(input):
            if self.needs_input_grad[0]:
                grad_input = input.new(input.size())

                n = grad_input.numel()
                opt['nthreads'] = n

                f = load_kernel('conv2d_dw_backward_grad_input_kernel',
                                _conv2d_depthwise_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            if self.needs_input_grad[1]:
                weight_buffer = weight.new(channels, kernel_h, kernel_w, batch_size, output_h, output_w)

                n = weight_buffer.numel()
                opt['nthreads'] = n

                f = load_kernel('conv2d_dw_backward_grad_weight_kernel',
                                _conv2d_depthwise_kernel_backward_grad_weight, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[grad_output.data_ptr(), input.data_ptr(), weight_buffer.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                grad_weight = weight_buffer.view(weight.size() + (-1,)).sum(-1)

        return grad_input, grad_weight


def conv2d_depthwise(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """Depthwise 2D convolution.

    Implements depthwise convolution as in https://arxiv.org/pdf/1704.04861v1.pdf
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

    CUDA kernels from https://github.com/BVLC/caffe/pull/5665
    CPU side is done by F.conv2d

    Equivalent to:
        `F.conv2d(input, weight, groups=input.size(1))`
    """
    assert input.size(1) == weight.size(0)
    if input.is_cuda:
        out = Conv2dDepthwise(stride, padding, dilation)(input, weight)
        if bias is not None:
            out += bias.view(1,-1,1,1)
    else:
        groups = input.size(1)
        out = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    return out
