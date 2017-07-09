import unittest
import torch
from torch.autograd import gradcheck, Variable
from pyinn import ncrelu, dgmm, cdgmm, im2col, col2im
import torch.nn.functional as F


def ncrelu_ref(input):
    return torch.cat([F.relu(input), -F.relu(-input)], 1)


def cdgmm_ref(A, B):
    C = Variable(A.data.new(A.size()))

    A_r = A[..., 0].contiguous().view(-1, A.size(-2))
    A_i = A[..., 1].contiguous().view(-1, A.size(-2))

    B_r = B[..., 0].contiguous().view(-1).unsqueeze(0).expand_as(A_i)
    B_i = B[..., 1].contiguous().view(-1).unsqueeze(0).expand_as(A_r)

    C[..., 0] = A_r * B_r - A_i * B_i
    C[..., 1] = A_r * B_i + A_i * B_r
    return C


class TestPYINN(unittest.TestCase):

    def testNCReLU(self):
        for dtype in [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]:
            x = Variable(torch.randn(2,5,3,1).type(dtype), requires_grad=True)
            #go = Variable(torch.randn(2,10,3,1).cuda(), requires_grad=False)
            go = torch.randn(2,10,3,1).type(dtype)

            self.assertEqual((ncrelu_ref(x).data - ncrelu(x).data).abs().sum(), 0)

            ncrelu_ref(x).backward(go)
            gref = x.grad.data.clone()
            x.grad.data.zero_()
            ncrelu(x).backward(go)
            g = x.grad.data.clone()
            self.assertLess((g - gref).abs().sum(), 1e-8)

    def testDGMM(self):
        inputs = Variable(torch.randn(16, 8).cuda())
        x = Variable(torch.randn(8).cuda())

        c_ref = inputs.mm(torch.diag(x))
        c_out = dgmm(inputs, x)
        self.assertEqual((c_ref.data - c_out.data).abs().max(), 0, 'DGMM left')

        # transposed
        c_ref = torch.diag(x).mm(inputs.t())
        c_out = dgmm(inputs.t().contiguous(), x)
        self.assertEqual((c_ref.data - c_out.data).abs().max(), 0, 'DGMM right')

        # grad wrt inputs
        inputs.requires_grad, x.requires_grad = True, False
        dgmm(inputs, x).sum().backward()
        g_out = inputs.grad.data.clone()

        inputs.grad.data.zero_()
        inputs.mm(torch.diag(x)).sum().backward()
        g_ref = inputs.grad.data.clone()

        self.assertEqual((g_ref - g_out).abs().max(), 0)

        # grad wrt x
        inputs.requires_grad, x.requires_grad = False, True
        dgmm(inputs, x).sum().backward()
        g_out = x.grad.data.clone()

        x.grad.data.zero_()
        inputs.mm(torch.diag(x)).sum().backward()
        g_ref = x.grad.data.clone()

        self.assertEqual((g_ref - g_out).abs().max(), 0)
        
        # grad wrt inputs and x
        inputs.requires_grad, x.requires_grad = True, True
        x.grad.data.zero_()
        inputs.grad.data.zero_()
        dgmm(inputs, x).sum().backward()
        g_x_out = x.grad.data.clone()
        g_inputs_out = inputs.grad.data.clone()

        x.grad.data.zero_()
        inputs.grad.data.zero_()
        inputs.mm(torch.diag(x)).sum().backward()
        g_x_ref = x.grad.data.clone()
        g_x_inputs_out = inputs.grad.data.clone()

        self.assertEqual((g_ref - g_out).abs().max(), 0)
        self.assertEqual((g_x_ref - g_x_out).abs().max(), 0)

    def testCDGMM(self):

        inputs = Variable(torch.randn(16, 8, 2).cuda())
        x = Variable(torch.randn(8, 2).cuda())

        c_ref = cdgmm_ref(inputs, x)
        c_out = cdgmm(inputs, x)
        self.assertLess((c_ref.data - c_out.data).abs().max(), 1e-6, 'CDGMM left')

        # grad wrt inputs
        inputs.requires_grad, x.requires_grad = True, False
        cdgmm(inputs, x).sum().backward()
        g_out = inputs.grad.data.clone()

        inputs.grad.data.zero_()
        cdgmm_ref(inputs, x).sum().backward()
        g_ref = inputs.grad.data.clone()

        self.assertLess((g_out - g_ref).abs().max(), 1e-6, 'CDGMM grad wrt A')

        # grad wrt x
        # inputs.requires_grad, x.requires_grad = False, True
        # cdgmm(inputs, x).sum().backward()
        # g_out = x.grad.data.clone()

        # x.grad.data.zero_()
        # cdgmm_ref(inputs, x).sum().backward()
        # g_ref = x.grad.data.clone()

        # self.assertEqual((g_ref - g_out).abs().max(), 0)

    def testCDGMMscat(self):
        shapes = [((1, 3, 40, 40, 2), (40, 40, 2)),
                  ((1, 3, 20, 20, 2), (20, 20, 2))]

        def cdgmm_ref(A, B):
            C = Variable(A.data.new(A.size()))

            A_r = A[..., 0].contiguous().view(-1, A.size(-2)*A.size(-3))
            A_i = A[..., 1].contiguous().view(-1, A.size(-2)*A.size(-3))

            B_r = B[...,0].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_i)
            B_i = B[..., 1].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_r)

            C[..., 0] = A_r * B_r - A_i * B_i
            C[..., 1] = A_r * B_i + A_i * B_r
            return C

        def cdgmm_scat(A, B):
            A_ = A.view(-1, A.size(-2)*A.size(-3), 2)
            B_ = B.view(-1, 2)
            return cdgmm(A_, B_).view_as(A)

        for shape in shapes:
            inputs = Variable(torch.randn(*shape[0]).cuda())
            x = Variable(torch.randn(*shape[1]).cuda())

            c_ref = cdgmm_ref(inputs, x)

            c = cdgmm_scat(inputs, x)

            self.assertLess((c_ref.data - c.data).abs().max(), 1e-6, 'CDGMM left')

            inputs.requires_grad, x.requires_grad = True, False
            cdgmm_scat(inputs, x).sum().backward()
            g_out = inputs.grad.data.clone()

            inputs.grad.data.zero_()
            cdgmm_ref(inputs, x).sum().backward()
            g_ref = inputs.grad.data.clone()

            self.assertLess((g_out - g_ref).abs().max(), 1e-6, 'CDGMM grad wrt A')


    def test_im2col(self):
        src = Variable(torch.randn(8,7,7).cuda())
        k = 1
        pad = 0
        s = (1,1)
        dst = im2col(src, k, s, pad)
        back = col2im(dst, k, s, pad)
        self.assertEqual((src - back).data.abs().max(), 0)

    def test_im2col_batch(self):
        src = Variable(torch.randn(4,8,7,7).cuda())
        k = 1
        pad = 0
        s = (1,1)
        dst = im2col(src, k, s, pad)
        back = col2im(dst, k, s, pad)
        self.assertEqual((src - back).data.abs().max(), 0)


if __name__ == '__main__':
    unittest.main()
