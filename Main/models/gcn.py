# A simple GCN with laplacian transform and corresspoding Linear layer

import argparse
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop
import torch
import pickle

class GCN(nn.Module):
    def __init__(self, adj_path, feature_dim: int, input_dim: int, output_dim: int, hidden_dim: int, **kwargs):
        super(GCN, self).__init__()
        self._feature_dim = feature_dim
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        with open(adj_path,'rb') as f:
            adj = pickle.load(f)
        self.register_buffer(
            '_laplacian', calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self._gc_weight = nn.Parameter(
            torch.FloatTensor(self._feature_dim, self._feature_dim))
        self._gc_bias = nn.Parameter(torch.FloatTensor(
            self._feature_dim, self._feature_dim))
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
        
        prediction = inputs @ self._gc_weight + self._gc_bias
        prediction = self._laplacian @ prediction
        prediction = self._gc_activate(prediction)

        prediction = prediction @ self._l1_weight + self._l1_bias
        prediction = self._l_activate(prediction)
        
        prediction = prediction@self._l2_weight + self._l2_bias
        prediction = self._l_activate(prediction)
        
        return prediction

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--input-dim',type=int)
        parser.add_argument('--hidden-dim',type=int,default=64)
        parser.add_argument('--output-dim',type=int)    
        parser.add_argument('--feature-dim',type=int)    

        return parser

    @property
    def hyperparameters(self):
        return {
            'input_dim': self._input_dim,
            'output_dim': self._output_dim,
            'feature_dim': self._feature_dim,
            'hidden_dim': self._hidden_dim
        }
