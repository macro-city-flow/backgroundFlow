# TGCN based on self realize

import argparse
import torch.nn as nn
from models import GCN
from models import GRU

# fr : forget rate that control how many information forget in a step.
class TGCN(nn.Module):
    def __init__(self,adj_path, feature_dim: int, input_dim: int, hidden_dim: int, output_dim: int, fr: float, **kwargs):
        super(TGCN, self).__init__()
        self._adj_path = adj_path
        self._feature_dim = feature_dim
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        assert(0 <= fr <= 1)
        self._fr = fr
        self._gcn_layer = GCN(self._adj_path,self._feature_dim,self._input_dim,self._hidden_dim,self._hidden_dim)
        self._gru_layer = GRU(self._feature_dim,self._input_dim,self._hidden_dim,self._output_dim,fr=self._fr)
        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, inputs):
        prediction = self._gcn_layer(inputs)
        prediction = self._gru_layer(prediction)
        return prediction

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--output-dim', type=int)
        parser.add_argument('--hidden-dim', type=int, default=64)
        parser.add_argument('--gamma', type=float, default=0.5)
        parser.add_argument('--gradient-clip-val',type=float,default=5)
        return parser

    @property
    def hyperparameters(self):
        return {
            'input_dim': self._input_dim,
            'hidden_dim': self._hidden_dim,
            'output_dim': self._output_dim,
            'feature_dim': self._feature_dim,
        }
