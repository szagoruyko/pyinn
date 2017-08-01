from torch.autograd import Function
import torch
from torch.nn.modules.utils import _pair
from pyinn.utils import Dtype, Stream, load_kernel

CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_im2col_kernel = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
extern "C"
__global__ void im2col_kernel(const ${Dtype}* data_im, ${Dtype}* data_col) {
  CUDA_KERNEL_LOOP(index, ${n}) {
    int w_out = index % ${width_col};
    index /= ${width_col};
    int h_out = index % ${height_col};
    int channel_in = index / ${height_col};
    int channel_out = channel_in * ${ksize_h} * ${ksize_w};
    int h_in = h_out * ${stride_h} - ${pad_h};
    int w_in = w_out * ${stride_w} - ${pad_w};
    data_col += (channel_out * ${height_col} + h_out) * ${width_col} + w_out;
    data_im += (channel_in * ${height} + h_in) * ${width} + w_in;
    #pragma unroll
    for (int i = 0; i < ${ksize_h}; ++i) {
      for (int j = 0; j < ${ksize_w}; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < ${height} && w < ${width}) ?
          data_im[i * ${width} + j] : 0;
        data_col += ${height_col} * ${width_col};
      }
    }
  }
}
'''


_col2im_kernel = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)

extern "C"
__global__ void col2im_kernel(const ${Dtype}* data_col, ${Dtype}* data_im) {
  CUDA_KERNEL_LOOP(index, ${n}) {
    ${Dtype} val = 0;
    int w = index % ${width} + ${pad_w};
    int h = (index / ${width}) % ${height} + ${pad_h};
    int c = index / (${width} * ${height});
    // compute the start and end of the output
    int w_col_start = (w < ${ksize_w}) ? 0 : (w - ${ksize_w}) / ${stride_w} + 1;
    int w_col_end = min(w / ${stride_w} + 1, ${width_col});
    int h_col_start = (h < ${ksize_h}) ? 0 : (h - ${ksize_h}) / ${stride_h} + 1;
    int h_col_end = min(h / ${stride_h} + 1, ${height_col});

    // equivalent implementation
    int offset = (c * ${ksize_h} * ${ksize_w} + h * ${ksize_w} + w) * ${height_col} * ${width_col};
    int coeff_h_col = (1 - ${stride_h} * ${ksize_w} * ${height_col}) * ${width_col};
    int coeff_w_col = (1 - ${stride_w} * ${height_col} * ${width_col});
    #pragma unroll
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}
'''


def im2col_shape(size, kernel_size, stride, padding):
    ksize_h, ksize_w = _pair(kernel_size)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    n_input_plane, height, width = size
    height_col = (height + 2 * pad_h - ksize_h) // stride_h + 1
    width_col = (width + 2 * pad_w - ksize_w) // stride_w + 1
    return n_input_plane, ksize_h, ksize_w, height_col, width_col


def _im2col(data, kernel_size, stride, padding, out=None):
    assert data.dim() == 3 and data.is_cuda
    ksize_h, ksize_w = _pair(kernel_size)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    n_input_plane, height, width = data.size()
    height_col = (height + 2 * pad_h - ksize_h) // stride_h + 1
    width_col = (width + 2 * pad_w - ksize_w) // stride_w + 1
    n = n_input_plane * height_col * width_col

    shape = torch.Size((n_input_plane, ksize_h, ksize_w, height_col, width_col))
    if out is not None:
        assert out.size() == shape
        data_col = out
    else:
        data_col = data.new(*shape)

    with torch.cuda.device_of(data):
        f = load_kernel('im2col_kernel', _im2col_kernel, Dtype=Dtype(data), n=n,
                        height_col=height_col,
                        width_col=width_col,
                        height=height, width=width,
                        ksize_h=ksize_h, ksize_w=ksize_w,
                        pad_h=pad_h, pad_w=pad_w,
                        stride_h=stride_h, stride_w=stride_w,
                        channels=n_input_plane)
        f(block=(CUDA_NUM_THREADS,1,1),
          grid=(GET_BLOCKS(n),1,1),
          args=[data.data_ptr(), data_col.data_ptr()],
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
    return data_col


col2im_modules = {}


def col2im_shape(size, kernel_size, stride, padding, input_size=None):
    ksize_h, ksize_w = _pair(kernel_size)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    n_input_plane, ksize_h, ksize_w, height_col, width_col = size
    if input_size is not None:
        height, width = input_size
    else:
        height = (height_col - 1) * stride_h - 2 * pad_h + ksize_h
        width = (width_col - 1) * stride_w - 2 * pad_w + ksize_w
    return n_input_plane, height, width


def _col2im(data_col, kernel_size, stride, padding, out=None, input_size=None):
    assert data_col.dim() == 5
    ksize_h, ksize_w = _pair(kernel_size)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    n_input_plane, ksize_h, ksize_w, height_col, width_col = data_col.size()
    if input_size is not None:
        height, width = input_size
    else:
        height = (height_col - 1) * stride_h - 2 * pad_h + ksize_h
        width = (width_col - 1) * stride_w - 2 * pad_w + ksize_w
    n = n_input_plane * height * width

    if out is not None:
        assert tuple(out.size()) == (n_input_plane, height, width)
        data = out
    else:
        data = data_col.new(n_input_plane, height, width)

    with torch.cuda.device_of(data_col):
        f = load_kernel('col2im_kernel', _col2im_kernel, Dtype=Dtype(data), n=n,
                        height_col=height_col,
                        width_col=width_col,
                        height=height, width=width,
                        ksize_h=ksize_h, ksize_w=ksize_w,
                        pad_h=pad_h, pad_w=pad_w,
                        stride_h=stride_h, stride_w=stride_w,
                        channels=n_input_plane)
        f(block=(CUDA_NUM_THREADS,1,1),
          grid=(GET_BLOCKS(n),1,1),
          args=[data_col.data_ptr(), data.data_ptr()],
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
    return data


def im2col_batch(input, kernel_size, stride, padding):
    if input.dim() == 3:
        return _im2col(input, kernel_size, stride, padding)
    elif input.dim() == 4:
        shape = (input.size(0),) + im2col_shape(input.size()[1:], kernel_size, stride, padding)
        out = input.new(*shape)
        for x, o in zip(input, out):
            _im2col(x, kernel_size, stride, padding, out=o)
        return out


def col2im_batch(grad_output, kernel_size, stride, padding, input_size=None):
    if grad_output.dim() == 5:
        return _col2im(grad_output, kernel_size, stride, padding, input_size)
    elif grad_output.dim() == 6:
        shape = (grad_output.size(0),) + col2im_shape(grad_output.size()[1:], kernel_size, stride, padding, input_size)
        grad_input = grad_output.new(*shape)
        for go, gx in zip(grad_output, grad_input):
            _col2im(go, kernel_size, stride, padding, out=gx, input_size=input_size)
        return grad_input


class Im2Col(Function):
    def __init__(self, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        self.input_size = input.size()[-2:]
        return im2col_batch(input, self.kernel_size, self.stride, self.padding)

    def backward(self, grad_output):
        return col2im_batch(grad_output, self.kernel_size, self.stride, self.padding, self.input_size)


class Col2Im(Function):
    def __init__(self, kernel_size, stride, padding, input_size=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_size = input_size

    def forward(self, input):
        return col2im_batch(input, self.kernel_size, self.stride, self.padding, self.input_size)

    def backward(self, grad_output):
        return im2col_batch(grad_output, self.kernel_size, self.stride, self.padding)


def im2col(input, kernel_size, stride, padding):
    """Rearrange image blocks into columns

    The representation is used in GEMM-based convolution.
    Output is 5D (or 6D in case of minibatch) tensor.

    Minibatch implementation is inefficient, and could be done in a single CUDA kernel.

    TODO: add CPU version (via numpy?)
    """
    return Im2Col(kernel_size, stride, padding)(input)


def col2im(input, kernel_size, stride, padding):
    """Converts columns back to NCHW format.

    This is used in backward wrt inputs in GEMM-based convolution.
    """
    return Col2Im(kernel_size, stride, padding)(input)
