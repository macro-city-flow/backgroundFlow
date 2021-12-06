# A simple GCN with laplacian transform and corresspoding Linear layer

import argparse
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop
import torch
from torch.nn import Tanh, Sigmoid, Conv2d
from torch.nn.init import kaiming_normal_
from torch.nn.parameter import Parameter
from torch import FloatTensor
from pickle import load


class STGCN(nn.Module):
    def __init__(self, adj_path, feature_dim: int, input_dim: int, output_dim: int, hidden_dim: int, **kwargs):
        super(STGCN, self).__init__()
        self._feature_dim = feature_dim
        self._input_dim = input_dim
        self._output_dim = output_dim
        
        with open(adj_path,'rb') as f:
            adj = load(f)
        
        self._st_conv1 =  STConv()
        # self._st_conv2 = STConv()
        self._weight = Parameter(FloatTensor())
        self._bias = Parameter(FloatTensor())
        self._sigmoid = Sigmoid()
        # TODO
        self.register_buffer(
            '_laplacian', calculate_laplacian_with_self_loop(FloatTensor(adj)))
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(
            self._weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(
            self._bias, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, inputs):
        output = self._st_conv1(inputs)+inputs
        # output = self._st_conv2(output)+inputs
        output = self._sigmoid(self._weight@ output +self._bias)
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--input-dim', type=int)
        parser.add_argument('--hidden-dim', type=int, default=64)
        parser.add_argument('--output-dim', type=int)
        parser.add_argument('--feature-dim', type=int)

        return parser

    @property
    def hyperparameters(self):
        return {
            'input_dim': self._input_dim,
            'output_dim': self._output_dim,
            'feature_dim': self._feature_dim,
            'hidden_dim': self._hidden_dim
        }

class STConv(nn.Module):
    def __init__(self):
        super(STConv).__init__()
    
    def reset_parameters(self):
    
        return

    def forward(self,input):
        return
    
    @property
    def hyperparameters(self):
        return {}