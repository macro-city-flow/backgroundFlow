import torch
from torch import distributions

def accuracy(pred, y):

    return 1 - torch.linalg.norm(y - pred, 'fro') / torch.linalg.norm(y, 'fro')

def multiFeatureAccuracy(pred, y):
    return 1 - sum([torch.linalg.norm(y[i]-pred[i]) for i in range(len(pred))])/torch.linalg.norm(y)

def confidence(distributions:torch.distributions.Normal,weights:torch.Tensor,y):
    #TODO realize confidence calculation based on sampling
    return

