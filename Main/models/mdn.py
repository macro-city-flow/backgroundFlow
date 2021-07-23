# MDN layer which can give a set of parameter of Gaussain mixture distribution

import argparse
import torch.nn as nn
import torch
from torch import nn as nn
from torch import distributions as distributions

class MDN(nn.Module):
    def __init__(self,input_dim:int,output_dim:int,feature_dim:int,gamma:int, **kwargs):
        super(MDN, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._feature_dim = feature_dim
        self._gamma = gamma
        self._mu_weights = nn.Parameter(torch.FloatTensor(self._gamma,self._output_dim,self._input_dim))
        self._mu_bias = nn.Parameter(torch.FloatTensor(self._gamma,self._output_dim,self._feature_dim))
        self._sigma_weights = nn.Parameter(torch.FloatTensor(self._gamma,self._output_dim,self._input_dim))
        self._sigma_bias = nn.Parameter(torch.FloatTensor(self._gamma,self._output_dim,self._feature_dim))
        self._distri_weights = nn.Parameter(torch.FloatTensor(self._gamma,self._output_dim,self._output_dim))
        self._regularize = nn.Softmax(dim = 0)
        self._activate = nn.Sigmoid()
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self._mu_weights)
        nn.init.xavier_normal_(self._mu_bias)
        nn.init.xavier_normal_(self._sigma_weights)
        nn.init.xavier_normal_(self._sigma_bias)
        return

    def forward(self, inputs):
        
        mu = self._activate(self._mu_weights @ inputs + self._mu_bias)
        sigma = self._activate(self._sigma_weights @ inputs + self._sigma_bias)
        weights = self._regularize(self._distri_weights)

        return mu,sigma,weights

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--gamma',type=int,default=5)
        parser.add_argument('--input-dim',type=int)
        parser.add_argument('--output-dim',type=int)
        parser.add_argument('--gradient-clip-val',type=float,default=5)    
        return parser

    @property
    def hyperparameters(self):
        return {
            "input_dimension":self._input_dim,
            "output_dimension":self._output_dim,
            "gamma":self._gamma
        }
