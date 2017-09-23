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

#define UNROLL _Pragma("unroll")

#define ldg __ldg

template <typename T>
inline __device__ T tf_max(const T& x, const T& y) {
  return x < y ? y : x;
}
template <typename T>
inline __device__ T tf_min(const T& x, const T& y) {
  return x > y ? y : x;
}

'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_conv2d_depthwise_kernel = kernel_loop + '''
extern "C"
__global__ void __launch_bounds__(1024)
conv2d_dw_forward_kernel(const ${Dtype}* input, const ${Dtype}* filter, ${Dtype}* output) {
  const int in_rows = ${bottom_width};
  const int in_cols = ${bottom_height};
  const int in_depth = ${channels};
  const int filter_rows = ${kernel_w};
  const int filter_cols = ${kernel_h};
  const int depth_multiplier = 1;
  const int pad_rows = ${pad_w};
  const int pad_cols = ${pad_h};
  const int out_rows = ${top_width};
  const int out_cols = ${top_height};
  const int out_depth = ${channels};

  CUDA_KERNEL_LOOP(thread_id, ${nthreads}) {
    const int OC = thread_id % out_cols;
    const int OR = (thread_id / out_cols) % out_rows;
    const int OD = (thread_id / out_cols / out_rows) % out_depth;
    const int OB = thread_id / out_cols / out_rows / out_depth;

    const int in_d = OD / depth_multiplier;
    const int multiplier = OD % depth_multiplier;

    const int input_offset_temp = (OB * in_depth + in_d) * (in_rows * in_cols);

    const int input_row_start = OR * ${stride_w} - pad_rows;
    const int input_col_start = OC * ${stride_h} - pad_cols;
    const int input_row_end = input_row_start + filter_rows;
    const int input_col_end = input_col_start + filter_cols;

    ${Dtype} sum = 0;
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < in_rows && input_col_end < in_cols) {
      // Loop that doesn't need to check for boundary conditions.
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = input_row_start + f_r;
        const int filter_offset_temp = filter_cols * f_r;
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = input_col_start + f_c;

          const int input_offset = (input_offset_temp) + (in_r * in_cols) + in_c;
          // filters in tensorflow are HWN, in pytorch NHW, so transposed
          // not sure if this breaks/improves coalescing
          //const int filter_offset =
            //  multiplier +
            //  depth_multiplier * (in_d + in_depth * (f_c + filter_offset_temp));
          const int filter_offset = in_d * ${kernel_h} * ${kernel_w} + f_c + filter_offset_temp;
          sum += ldg(input + input_offset) * ldg(filter + filter_offset);
        }
      }
    } else {
      // Loop that needs to check for boundary conditions.
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = input_row_start + f_r;
        const int filter_offset_temp = filter_cols * f_r;
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = input_col_start + f_c;
          // TODO(vrv): the in_r check can be done outside of this loop;
          // benchmark both methods to determine the better decision.
          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
            const int in_c = input_col_start + f_c;

            // input_offset_temp indexes into the start of memory
            // where the spatial data starts.
            const int input_offset =
                (input_offset_temp) + (in_r * in_cols) + in_c;

            //const int filter_offset =
            //    multiplier + depth_multiplier *
            //                     (in_d + in_depth * (f_c + filter_offset_temp));
            const int filter_offset = in_d * ${kernel_h} * ${kernel_w} + f_c + filter_offset_temp;
            sum += ldg(input + input_offset) * ldg(filter + filter_offset);
          }
        }
      }
    }

    output[thread_id] = sum;
  }
}
/*
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
*/
'''


_conv2d_depthwise_kernel_backward_grad_input = kernel_loop + '''
extern "C"
__global__ void __launch_bounds__(1024)
conv2d_dw_backward_grad_input_kernel(const ${Dtype}* const out_backprop,
                    const ${Dtype}* const filter, ${Dtype}* const in_backprop) {
  const int in_rows = ${bottom_width};
  const int in_cols = ${bottom_height};
  const int in_depth = ${channels};
  const int filter_rows = ${kernel_w};
  const int filter_cols = ${kernel_h};
  const int depth_multiplier = 1;
  const int pad_rows = ${pad_w};
  const int pad_cols = ${pad_h};
  const int out_rows = ${top_width};
  const int out_cols = ${top_height};
  const int out_depth = ${channels};

  // TODO(vrv): Consider assigning threads to output and using
  // atomics for accumulation, similar to the filter case.
  CUDA_KERNEL_LOOP(thread_id, ${nthreads}) {
    // Compute the indexes of this thread in the input.
    const int in_c = thread_id % in_cols;
    const int in_r = (thread_id / in_cols) % in_rows;
    const int in_d = (thread_id / in_cols / in_rows) % in_depth;
    const int b = thread_id / in_depth / in_cols / in_rows;

    ${Dtype} sum = 0;
    const int out_d_start = in_d * depth_multiplier;
    const int out_d_end = out_d_start + depth_multiplier;

    const int out_r_start =
        tf_max<int>(0, (in_r - filter_rows + pad_rows + ${stride_w}) / ${stride_w});
    const int out_r_end = tf_min(out_rows - 1, (in_r + pad_rows) / ${stride_w});
    const int out_c_start =
        tf_max(0, (in_c - filter_cols + pad_cols + ${stride_h}) / ${stride_h});
    const int out_c_end = tf_min(out_cols - 1, (in_c + pad_cols) / ${stride_h});

    UNROLL for (int out_d = out_d_start; out_d < out_d_end; ++out_d) {
      UNROLL for (int out_r = out_r_start; out_r <= out_r_end; ++out_r) {
        const int f_r = in_r + pad_rows - out_r * ${stride_w};
        const int filter_dm = out_d - out_d_start;

        const int temp_filter_offset = filter_cols * f_r;
        for (int out_c = out_c_start; out_c <= out_c_end; ++out_c) {
          const int f_c = in_c + pad_cols - out_c * ${stride_h};
          //const int filter_offset =
            //  filter_dm + args.depth_multiplier *
            //                  (in_d + in_depth * (f_c + temp_filter_offset));
          const int filter_offset = in_d * ${kernel_w} * ${kernel_h} + f_c + temp_filter_offset;

          const int out_backprop_offset =
              (b * out_depth * out_rows * out_cols) +
              (out_d * out_rows * out_cols) + (out_r * out_cols) + (out_c);

          sum += ldg(out_backprop + out_backprop_offset) *
                 ldg(filter + filter_offset);
        }
      }
    }
    const int in_backprop_offset = (b * in_rows * in_cols * in_depth) +
                                   (in_d * in_rows * in_cols) +
                                   (in_r * in_cols) + (in_c);
    in_backprop[in_backprop_offset] = sum;
  }
}
/*
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
*/
'''


_conv2d_depthwise_kernel_backward_grad_weight = kernel_loop + '''
extern "C"
__global__ void conv2d_dw_backward_grad_weight_kernel(
    const ${Dtype}* const out_backprop, const ${Dtype}* const input, ${Dtype}* const filter_backprop) {
  const int in_rows = ${bottom_width};
  const int in_cols = ${bottom_height};
  const int in_depth = ${channels};
  const int filter_rows = ${kernel_w};
  const int filter_cols = ${kernel_h};
  const int depth_multiplier = 1;
  const int pad_rows = ${pad_w};
  const int pad_cols = ${pad_h};
  const int out_rows = ${top_width};
  const int out_cols = ${top_height};
  const int out_depth = ${channels};


  CUDA_KERNEL_LOOP(thread_id, ${nthreads}) {
    // Compute the indexes of this thread in the output.
    const int out_c = thread_id % out_cols;
    const int out_r = (thread_id / out_cols) % out_rows;
    const int out_d = (thread_id / out_cols / out_rows) % out_depth;

    const int b = thread_id / out_depth / out_cols / out_rows;
    // Compute the input depth and the index of depth multiplier.
    const int in_d = out_d / depth_multiplier;
    const int dm = out_d % depth_multiplier;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_r_start = out_r * ${stride_w} - pad_rows;
    const int in_c_start = out_c * ${stride_h} - pad_cols;
    const int in_r_end = in_r_start + filter_rows;
    const int in_c_end = in_c_start + filter_cols;

    const int out_backprop_offset = (b * out_depth * out_rows * out_cols) +
                                    (out_d * out_rows * out_cols) +
                                    (out_r * out_cols) + (out_c);

    const ${Dtype} out_bp = ldg(out_backprop + out_backprop_offset);
    if (in_r_start >= 0 && in_c_start >= 0 && in_r_end < in_rows &&
        in_c_end < in_cols) {
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = in_r_start + f_r;
        // Avoid repeated computation.
        const int input_offset_temp = (b * in_depth * in_rows * in_cols) +
                                      (in_d * in_rows * in_cols) +
                                      (in_r * in_cols);

        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = in_c_start + f_c;
          const int input_offset = input_offset_temp + in_c;
          ${Dtype} partial_sum = ldg(input + input_offset) * out_bp;
          //${Dtype}* addr = filter_backprop +
            //        (dm + depth_multiplier *
            //                  (in_d + in_depth * (f_c + filter_cols * f_r)));
          ${Dtype}* addr = filter_backprop + (in_d * ${kernel_w} * ${kernel_h} + f_c + filter_cols * f_r);
          atomicAdd(addr, partial_sum);
        }
      }
    } else {
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = in_r_start + f_r;
        // Avoid repeated computation.
        const int input_offset_temp = (b * in_depth * in_rows * in_cols) +
                                      (in_d * in_rows * in_cols) +
                                      (in_r * in_cols);
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = in_c_start + f_c;
          const int addr_temp = filter_cols * f_r;

          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
            const int input_offset = input_offset_temp + in_c;
            ${Dtype} partial_sum = ldg(input + input_offset) * out_bp;
            //${Dtype}* addr =
            //    filter_backprop +
            //    (dm + depth_multiplier * (in_d + in_depth * (f_c + addr_temp)));
            ${Dtype}* addr = filter_backprop + in_d * ${kernel_w} * ${kernel_h}  + f_c + addr_temp;
            // Potentially many threads can add to the same address so we have
            // to use atomic add here.
            // TODO(jmchen): If atomic add turns out to be slow, we can:
            // 1. allocate multiple buffers for the gradients (one for each
            // example in a batch, for example). This can reduce the
            // contention on the destination; 2. Have each thread compute one
            // gradient for an element in the filters. This should work well
            // when the input depth is big and filter size is not too small.
            atomicAdd(addr, partial_sum);
          }
        }
      }
    }
  }
}
/*
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
*/
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
                n = grad_output.numel()
                grad_weight = input.new(weight.shape).zero_()
                opt['nthreads'] = n

                f = load_kernel('conv2d_dw_backward_grad_weight_kernel',
                                _conv2d_depthwise_kernel_backward_grad_weight, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

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
