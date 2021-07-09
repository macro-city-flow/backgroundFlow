# A simple GRU realize
import argparse

import torch
import torch.nn as nn


# fr : forget rate that control how many information forget in a step.
class GRU(nn.Module):
    def __init__(self, feature_dim: int, input_dim: int, hidden_dim: int, output_dim: int, fr: float, **kwargs):
        super(GRU, self).__init__()
        self._feature_dim = feature_dim
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        assert(0 <= fr <= 1)
        self._fr = fr
        self._tanh = nn.Tanh()
        self._sigmoid = nn.Sigmoid()
        # self._W_z = nn.Parameter(torch.FloatTensor(
        #     self._hidden_dim, self._input_dim))
        # self._U_z = nn.Parameter(torch.FloatTensor(
        #     self._hidden_dim, self._hidden_dim))
        self._W_z1 = nn.Parameter(torch.FloatTensor(
            1, self._input_dim))
        self._U_z1 = nn.Parameter(torch.FloatTensor(
            1, self._hidden_dim))
        self._W_z2 = nn.Parameter(torch.FloatTensor(
            self._feature_dim,1))
        self._U_z2 = nn.Parameter(torch.FloatTensor(
            self._feature_dim,1))
        self._W_r = nn.Parameter(torch.FloatTensor(
            self._hidden_dim, self._input_dim))
        self._U_r = nn.Parameter(torch.FloatTensor(
            self._hidden_dim, self._hidden_dim))
        self._W_h = nn.Parameter(torch.FloatTensor(
            self._hidden_dim, self._input_dim))
        self._U_h = nn.Parameter(torch.FloatTensor(
            self._hidden_dim, self._hidden_dim))
        self._W = nn.Parameter(torch.FloatTensor(
            self._output_dim, self._hidden_dim))
        self.register_buffer('_H', torch.FloatTensor(
            self._hidden_dim, self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self._W_z1, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(
            self._U_z1, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(
            self._W_z2, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(
            self._U_z2, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(
            self._W_r, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(
            self._U_r, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self._W_h, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._U_h, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(
            self._W, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.kaiming_uniform_(self._H)
        return

    def forward(self, inputs):
        z_t = self._sigmoid(self._W_z1 @ inputs @ self._W_z2 + self._U_z1 @ self._H @ self._U_z2)
        r_t = self._sigmoid(self._W_r @ inputs + self._U_r @ self._H)
        _ = self._tanh(self._W_h @ inputs + r_t * (self._U_h @ self._H))
        #H = self._fr*self._H+(1-self._fr)*_
        H = z_t*self._H+(1-z_t)*_
        self._H = H.detach()
        prediction = self._sigmoid(self._W @ H)
        return prediction
        #Seems that the forget gate z_t is not very important here

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
