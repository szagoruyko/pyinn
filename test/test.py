import unittest
from functools import partial
import torch
from torch.autograd import gradcheck, Variable
import pyinn as P
from pyinn.modules import Conv2dDepthwise
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

            self.assertEqual((ncrelu_ref(x).data - P.ncrelu(x).data).abs().sum(), 0)

            ncrelu_ref(x).backward(go)
            gref = x.grad.data.clone()
            x.grad.data.zero_()
            P.ncrelu(x).backward(go)
            g = x.grad.data.clone()
            self.assertLess((g - gref).abs().sum(), 1e-8)

    def testDGMM(self):
        inputs = Variable(torch.randn(16, 8).cuda())
        x = Variable(torch.randn(8).cuda())

        c_ref = inputs.mm(torch.diag(x))
        c_out = P.dgmm(inputs, x)
        self.assertEqual((c_ref.data - c_out.data).abs().max(), 0, 'DGMM left')

        # transposed
        c_ref = torch.diag(x).mm(inputs.t())
        c_out = P.dgmm(inputs.t().contiguous(), x)
        self.assertEqual((c_ref.data - c_out.data).abs().max(), 0, 'DGMM right')

        # grad wrt inputs
        inputs.requires_grad, x.requires_grad = True, False
        P.dgmm(inputs, x).sum().backward()
        g_out = inputs.grad.data.clone()

        inputs.grad.data.zero_()
        inputs.mm(torch.diag(x)).sum().backward()
        g_ref = inputs.grad.data.clone()

        self.assertEqual((g_ref - g_out).abs().max(), 0)

        # grad wrt x
        inputs.requires_grad, x.requires_grad = False, True
        P.dgmm(inputs, x).sum().backward()
        g_out = x.grad.data.clone()

        x.grad.data.zero_()
        inputs.mm(torch.diag(x)).sum().backward()
        g_ref = x.grad.data.clone()

        self.assertLess((g_ref - g_out).abs().max(), 1e-6)
        
        # grad wrt inputs and x
        inputs.requires_grad, x.requires_grad = True, True
        x.grad.data.zero_()
        inputs.grad.data.zero_()
        P.dgmm(inputs, x).sum().backward()
        g_x_out = x.grad.data.clone()
        g_inputs_out = inputs.grad.data.clone()

        x.grad.data.zero_()
        inputs.grad.data.zero_()
        inputs.mm(torch.diag(x)).sum().backward()
        g_x_ref = x.grad.data.clone()
        g_x_inputs_out = inputs.grad.data.clone()

        self.assertLess((g_ref - g_out).abs().max(), 1e-6)
        self.assertLess((g_x_ref - g_x_out).abs().max(), 1e-6)

    def testCDGMM(self):

        inputs = Variable(torch.randn(16, 8, 2).cuda())
        x = Variable(torch.randn(8, 2).cuda())

        c_ref = cdgmm_ref(inputs, x)
        c_out = P.cdgmm(inputs, x)
        self.assertLess((c_ref.data - c_out.data).abs().max(), 1e-6, 'CDGMM left')

        # grad wrt inputs
        inputs.requires_grad, x.requires_grad = True, False
        P.cdgmm(inputs, x).sum().backward()
        g_out = inputs.grad.data.clone()

        inputs.grad.data.zero_()
        cdgmm_ref(inputs, x).sum().backward()
        g_ref = inputs.grad.data.clone()

        self.assertLess((g_out - g_ref).abs().max(), 1e-6, 'CDGMM grad wrt A')

        # grad wrt x
        # inputs.requires_grad, x.requires_grad = False, True
        # P.cdgmm(inputs, x).sum().backward()
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
            return P.cdgmm(A_, B_).view_as(A)

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
        dst = P.im2col(src, k, s, pad)
        back = P.col2im(dst, k, s, pad)
        self.assertEqual((src - back).data.abs().max(), 0)

    def test_im2col_batch(self):
        src = Variable(torch.randn(4,8,7,7).cuda())
        k = 1
        pad = 0
        s = (1,1)
        dst = P.im2col(src, k, s, pad)
        back = P.col2im(dst, k, s, pad)
        self.assertEqual((src - back).data.abs().max(), 0)

    def test_conv2d_depthwise(self):
        n = 6
        x = Variable(torch.randn(1,n,5,5).double().cuda(), requires_grad=True)
        w = Variable(torch.randn(n,1,3,3).double().cuda(), requires_grad=True)
        y_fast = P.conv2d_depthwise(x, w, padding=1)
        y_ref = F.conv2d(x, w, padding=1, groups=n)
        go = torch.randn(y_fast.size()).double().cuda()

        self.assertLess((y_fast - y_ref).data.abs().max(), 1e-9)

        x.requires_grad = True
        w.requires_grad = True
        y_fast.backward(go)
        gx_fast = x.grad.data.clone()
        gw_fast = w.grad.data.clone()

        x.grad.data.zero_()
        w.grad.data.zero_()
        y_ref.backward(go)
        gx_ref = x.grad.data.clone()
        gw_ref = w.grad.data.clone()

        self.assertTrue(gradcheck(partial(P.conv2d_depthwise, padding=1), (x, w,)))

    def test_conv2d_depthwise_multigpu(self):
        n = 6
        a0 = Variable(torch.randn(1,n,5,5).cuda(0), requires_grad=True)
        a1 = Variable(torch.randn(1,n,5,5).cuda(1), requires_grad=True)
        w0 = Variable(torch.randn(n,1,3,3).double().cuda(0), requires_grad=True)
        w1 = Variable(torch.randn(n,1,3,3).double().cuda(1), requires_grad=True)
        y0 = P.conv2d_depthwise(a0, w0, padding=1)
        go = torch.randn(y0.size()).double().cuda()
        y0.backward(go)
        y1 = P.conv2d_depthwise(a1, w1, padding=1)
        y1.backward(go.cuda(1))

    def test_modules(self):
        module = Conv2dDepthwise(channels=8, kernel_size=3)
        x = Variable(torch.randn(1,8,5,5))
        y = module(x)
        y_cuda = module.cuda()(x.cuda())
        self.assertLess((y - y_cuda.cpu()).data.abs().max(), 1e-6)


if __name__ == '__main__':
    unittest.main()
