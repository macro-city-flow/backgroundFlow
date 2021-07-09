# GCN-2

import argparse
from torch import functional
import torch.nn as nn
import torch
from utils.graph_conv import calculate_cheby_ploy
from torch.nn.functional import normalize

# gamma: defines the grade of neighbourhod
class ChebyNet(nn.Module):
    def __init__(self, adj, feature_dim: int, input_dim: int, output_dim: int, hidden_dim: int, gamma: int, **kwargs):
        super(ChebyNet, self).__init__()
        self._feature_dim = feature_dim
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        if gamma > self._output_dim:
            raise ValueError("Illegal hyperparameter")
        self._gamma = gamma
        self.register_buffer('_cheby_ploy', calculate_cheby_ploy(adj))
        self._ploy_weight = nn.Parameter(torch.FloatTensor(self._gamma+1))
        self._gc_weight = nn.Parameter(
            torch.FloatTensor(self._input_dim, self._feature_dim))
        self._gc_bias = nn.Parameter(torch.FloatTensor(
            self._input_dim, self._feature_dim))
        self._l1_weight = nn.Parameter(
            torch.FloatTensor(self._feature_dim, self._hidden_dim))
        self._l2_weight = nn.Parameter(
            torch.FloatTensor(self._hidden_dim, self._feature_dim))
        self._l1_bias = nn.Parameter(torch.FloatTensor(
            self._feature_dim, self._hidden_dim))
        self._l2_bias = nn.Parameter(torch.FloatTensor(
            self._feature_dim, self._feature_dim))
        self._gc_activate = nn.Tanh()
        self._l_activate = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self._gc_weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(
            self._gc_bias, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(
            self._l1_weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(
            self._l2_weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(
            self._l1_bias, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(
            self._l2_bias, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, inputs):

        Lambda = sum([self._ploy_weight[_]*self._cheby_ploy[_]
                     for _ in range(self._gamma+1)])
        Lambda = normalize(Lambda)
        
        outputs = inputs @ self._gc_weight + self._gc_bias
        outputs = Lambda @ outputs
        outputs = self._gc_activate(outputs)

        outputs = outputs @ self._l1_weight + self._l1_bias
        outputs = self._l_activate(outputs)

        outputs = outputs@self._l2_weight + self._l2_bias
        outputs = self._l_activate(outputs)

        return outputs

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden-dim', type=int, default=64)
        parser.add_argument('--output-dim', type=int)
        return parser

    @property
    def hyperparameters(self):
        return {
            'input_dim': self._input_dim,
            'output_dim': self._output_dim,
            'feature_dim': self._feature_dim,
            'hidden_dim': self._hidden_dim
        }
