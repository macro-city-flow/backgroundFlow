import argparse
import torch
import torch.nn as nn
from torch.nn import Tanh, Sigmoid, Conv2d
from torch.nn.init import kaiming_normal_
from torch.nn.parameter import Parameter
from torch import FloatTensor


class ConvLSTM(nn.Module):
    def __init__(self, feature_dim: int, input_dim: int, kernel_size: int, **kwargs):
        super(ConvLSTM, self).__init__()
        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._kernel_size = kernel_size
        self._tanh = Tanh()
        self._sigmoid = Sigmoid()
        self._Wxi = Conv2d(1, 1, [kernel_size, kernel_size], padding=[1])
        self._Whi = Conv2d(1, 1, [kernel_size, kernel_size], padding=[1])
        self._Wci = Conv2d(1, 1, [kernel_size, kernel_size], padding=[1])
        self._Wxf = Conv2d(1, 1, [kernel_size, kernel_size], padding=[1])
        self._Whf = Conv2d(1, 1, [kernel_size, kernel_size], padding=[1])
        self._Wcf = Conv2d(1, 1, [kernel_size, kernel_size], padding=[1])
        self._Wxc = Conv2d(1, 1, [kernel_size, kernel_size], padding=[1])
        self._Whc = Conv2d(1, 1, [kernel_size, kernel_size], padding=[1])
        self._Wxo = Conv2d(1, 1, [kernel_size, kernel_size], padding=[1])
        self._Who = Conv2d(1, 1, [kernel_size, kernel_size], padding=[1])
        self._Wco = Conv2d(1, 1, [kernel_size, kernel_size], padding=[1])
        self._bi = Parameter(FloatTensor(self._input_dim, self._feature_dim))
        self._bf = Parameter(FloatTensor(self._input_dim, self._feature_dim))
        self._bc = Parameter(FloatTensor(self._input_dim, self._feature_dim))
        self._bo = Parameter(FloatTensor(self._input_dim, self._feature_dim))

        self.register_buffer('_H', torch.FloatTensor(
            self._input_dim, self._feature_dim))
        self.register_buffer('_C', torch.FloatTensor(
            self._input_dim, self._feature_dim))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_normal_(self._H, nonlinearity='sigmoid')
        kaiming_normal_(self._C, nonlinearity='sigmoid')
        kaiming_normal_(self._bi, nonlinearity='sigmoid')
        kaiming_normal_(self._bf, nonlinearity='sigmoid')
        kaiming_normal_(self._bc, nonlinearity='tanh')
        kaiming_normal_(self._bo, nonlinearity='sigmoid')
        kaiming_normal_(self._Wxi.weight, nonlinearity='sigmoid')
        kaiming_normal_(self._Whi.weight, nonlinearity='sigmoid')
        kaiming_normal_(self._Wci.weight, nonlinearity='sigmoid')
        kaiming_normal_(self._Wxf.weight, nonlinearity='sigmoid')
        kaiming_normal_(self._Whf.weight, nonlinearity='sigmoid')
        kaiming_normal_(self._Wcf.weight, nonlinearity='sigmoid')
        kaiming_normal_(self._Wxc.weight, nonlinearity='tanh')
        kaiming_normal_(self._Whc.weight, nonlinearity='tanh')
        kaiming_normal_(self._Wxo.weight, nonlinearity='sigmoid')
        kaiming_normal_(self._Wco.weight, nonlinearity='sigmoid')
        kaiming_normal_(self._Who.weight, nonlinearity='sigmoid')

        return

    def forward(self, inputs):
        print(inputs.shape)
        input()
        it = self._sigmoid(self._Wxi(inputs) +
                           self._Whi(self._H)+self._Wci(self._C)+self._bi)
        print(it.shape)
        input()
        ft = self._sigmoid(self._Wxf(inputs) +
                           self._Whf(self._H)+self._Wcf(self._C)+self._bf)
        print(ft.shape)
        input()
        Ct = ft * self._C + it * \
            self._tanh(self._Wxc(inputs)+self._Whc(self._H)+self._bc)
        print(Ct.shape)
        input()
        ot = self._sigmoid(self._Wxo(inputs) +
                           self._Who(self._H)+self._Wco(self._C)+self._bo)
        print(ot.shape)
        input()
        self._H = (ot * self._tanh(Ct)).detach()
        self._C = Ct.detach()
        return ot

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--input-dim', type=int)
        parser.add_argument('--kernel-size', type=int)
        parser.add_argument('--feature-dim', type=int)
        return parser

    @property
    def hyperparameters(self):
        return {
            'input_dim': self._input_dim,
            'kernel_size': self._kernel_size,
            'feature_dim': self._feature_dim
        }
