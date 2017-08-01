import torch


def cublas_dgmm(A, x, out=None):
    if out is not None:
        assert out.is_contiguous() and out.size() == A.size()
    else:
        out = A.new(A.size())
    assert x.dim() == 1
    assert x.numel() == A.size(-1) or x.numel() == A.size(0)
    assert A.type() == x.type() == out.type()
    assert A.is_contiguous()

    if not isinstance(A, (torch.cuda.FloatTensor, torch.cuda.DoubleTensor)):
        if x.numel() == A.size(-1):
            return A.mm(torch.diag(x), out=out.view_as(A))
        else:
            return torch.diag(x).mm(A, out=out.view_as(A))
    else:
        if x.numel() == A.size(-1):
            m, n =  A.size(-1), A.numel() // A.size(-1)
            mode = 'l'
            # A.mm(x.diag(), out=out)
            # return out
        elif x.numel() == A.size(0):
            n, m = A.size(0), A.numel() // A.size(0)
            mode = 'r'
            # if A.stride(0) == 1:
            #     mode = 'l'
            #     n, m = m, n
            # x.diag().mm(A, out=out)
            # return out
        lda, ldc = m, m
        incx = 1
        handle = torch.cuda.current_blas_handle()
        stream = torch.cuda.current_stream()._as_parameter_
        from skcuda import cublas
        cublas.cublasSetStream(handle, stream)
        args = [handle, mode, m, n, A.data_ptr(), lda, x.data_ptr(), incx, out.data_ptr(), ldc]
        if isinstance(A, torch.cuda.FloatTensor):
            cublas.cublasSdgmm(*args)
        elif isinstance(A, torch.cuda.DoubleTensor):
            cublas.cublasDdgmm(*args)
        return out


class DGMM(torch.autograd.Function):
    def forward(self, input, x):
        self.save_for_backward(input, x)
        return cublas_dgmm(input, x)

    def backward(self, grad_output):
        input, x = self.saved_tensors
        grad_input = grad_x = None
        if self.needs_input_grad[0]:
            grad_input = cublas_dgmm(grad_output.contiguous(), x)
            assert grad_input.size() == input.size()
        if self.needs_input_grad[1]:
            dim = 0 if x.numel() == input.size(-1) else 1
            grad_x = (grad_output * input).sum(dim).squeeze(dim)
            # grad_x = grad_output.t().mm(input).diag()
            assert grad_x.size() == x.size()
        return grad_input, grad_x


def dgmm(input, x):
    """Multiplication with a diagonal matrix.

    Used CUDA dgmm function, sometimes is faster than expand.

    In torch functions does `input.mm(x.diag())`. Both left and right
    mutliplications are supported.

    Args:
        input: 2D tensor
        x: 1D tensor
    """
    return DGMM()(input, x)
