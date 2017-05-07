import unittest
import torch
from torch.autograd import gradcheck, Variable
from ncrelu import ncrelu
import torch.nn.functional as F


def ncrelu_ref(input):
    return torch.cat([F.relu(input), -F.relu(-input)], 1)


class TestPYINN(unittest.TestCase):

    def testNCReLU(self):
        x = Variable(torch.randn(2,5,3,1).cuda(), requires_grad=True)
        go = Variable(torch.randn(2,10,3,1).cuda(), requires_grad=False)

        self.assertEqual((ncrelu_ref(x).data - ncrelu(x).data).abs().sum(), 0)

        ncrelu_ref(x).backward(go)
        gref = x.grad.data.clone()
        x.grad.data.zero_()
        ncrelu(x).backward(go)
        g = x.grad.data.clone()
        self.assertLess((g - gref).abs().sum(), 1e-8)


if __name__ == '__main__':
    unittest.main()
