import torch
from pyinn.utils import Stream, load_kernel


kernel = """
extern "C"
__global__ void swap(float2 *x, int total)
{
   int tx = blockIdx.x * blockDim.x + threadIdx.x;
   if(tx >= total)
      return;

   float2 v = x[tx];
   //x[tx] = make_float2(v.y, v.x);
   x[tx] = make_float2(v.x, -v.y);
}
"""

CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K


def swap(x):
    assert x.size(-1) == 2
    total = x.numel() // 2
    with torch.cuda.device_of(x):
        f = load_kernel('swap', kernel)
        f(args=[x.data_ptr(), total],
          block=(CUDA_NUM_THREADS,1,1),
          grid=(GET_BLOCKS(total),1,1),
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))


def cublas_cdgmm(A, x, out=None):
    if out is not None:
        assert out.is_contiguous() and out.size() == A.size()
    else:
        out = A.new(A.size())
    assert x.dim() == 2 and x.size(-1) == 2 and A.size(-1) == 2
    assert A.dim() == 3
    assert x.size(0) == A.size(1) or x.size(0) == A.size(0)
    assert A.type() == x.type() == out.type()
    assert A.is_contiguous()

    if not isinstance(A, (torch.cuda.FloatTensor, torch.cuda.DoubleTensor)):
        raise NotImplementedError
    else:
        m, n = A.size(1), A.size(0)
        if x.size(0) == A.size(1):
            mode = 'l'
        elif x.size(0) == A.size(0):
            mode = 'r'
        lda, ldc = m, m
        incx = 1
        handle = torch.cuda.current_blas_handle()
        stream = torch.cuda.current_stream()._as_parameter_
        from skcuda import cublas
        cublas.cublasSetStream(handle, stream)
        args = [handle, mode, m, n, A.data_ptr(), lda, x.data_ptr(), incx, out.data_ptr(), ldc]
        if isinstance(A, torch.cuda.FloatTensor):
            cublas.cublasCdgmm(*args)
        elif isinstance(A, torch.cuda.DoubleTensor):
            cublas.cublasZdgmm(*args)
        return out


class CDGMM(torch.autograd.Function):
    def forward(self, input, x):
        self.save_for_backward(input, x)
        return cublas_cdgmm(input, x)

    def backward(self, grad_output):
        input, x = self.saved_tensors
        grad_input = grad_x = None
        if self.needs_input_grad[0]:
            grad_output = grad_output.contiguous()
            swap(x)
            grad_input = cublas_cdgmm(grad_output.contiguous(), x)
            swap(x)
            
            assert grad_input.size() == input.size()
        if self.needs_input_grad[1]:
            raise NotImplementedError
            # dim = 0 if x.size(0) == input.size(1) else 1
            # grad_x = (grad_output * input).sum(dim).squeeze(dim)
            # assert grad_x.size() == x.size()
        return grad_input, grad_x


def cdgmm(input, x):
    """Complex multiplication with a diagonal matrix.

    Does `input.mm(x.diag())` where input and x are complex.

    Args:
        input: 3D tensor with last dimension of size 2
        x: 2D tensor with last dimension of size 2
    """
    return CDGMM()(input, x)
