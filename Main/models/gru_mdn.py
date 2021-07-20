import argparse

import torch
import torch.nn as nn
from models import GRU,MDN

class GRU_MDN(nn.Module):
    def __init__(self,input_dim:int,output_dim:int,feature_dim:int,gamma:int, **kwargs):
        super(GRU_MDN, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._feature_dim = feature_dim
        self._gamma = gamma
        self._gru_layer = GRU(self._feature_dim,self._input_dim,64,self._output_dim,0.6)
        self._mdn_layer = MDN(self._output_dim,self._output_dim,self._feature_dim,self._gamma)
        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, inputs):
        
        _ = self._gru_layer(inputs)
        mu,sigma,weights = self._mdn_layer(_)

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
            #TODO add learning rate
        }
