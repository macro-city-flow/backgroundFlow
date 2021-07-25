import torch
from torch import Tensor
from utils.probability import sample
import numpy as np

def accuracy(pred, y):

    return 1 - torch.linalg.norm(y - pred, 'fro') / torch.linalg.norm(y, 'fro')


def multiFeatureAccuracy(pred, y):

    return 1 - sum([torch.linalg.norm(y[i]-pred[i]) for i in range(len(pred))])/torch.linalg.norm(y)


def RMCI(mu:Tensor,sigma:Tensor,weights:torch.Tensor,y)-> Tensor:
    
    confidence_level = 0.90#F**k.... I have no idea whether this can be used this way
    #And this can be replaced with hyperparameters 

    assert(mu.shape == sigma.shape)
    assert(mu.shape[0]== y.shape[0])
    assert(len(mu.shape) <=2)#no support for over 2-dimension data
    
    samples = sample(mu,sigma,weights,20).transpose(0,1)#20 is a number that can be further replaced with hyperparameters
    result = [[abs(_-y[__]) for _ in samples[__]].sort() for __ in range(samples.shape[0]) ]
    result = [result[_][int(confidence_level*20)] for _ in range(len(result))]
    result = sum(result)/len(result)

    return result
