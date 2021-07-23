import torch
from torch import distributions

def accuracy(pred, y):

    return 1 - torch.linalg.norm(y - pred, 'fro') / torch.linalg.norm(y, 'fro')

def multiFeatureAccuracy(pred, y):

    return 1 - sum([torch.linalg.norm(y[i]-pred[i]) for i in range(len(pred))])/torch.linalg.norm(y)

def sample(distributions:torch.distributions.Normal,weights:torch.Tensor,times:int)->list:
    assert(len(distributions.lco.shape) <=2)#no support for over 2-dimension data
    result = []
    for _ in range(times):
        sample = None#Not implied yet
        result.append(sample)
    return  result

def confidence(distributions:torch.distributions.Normal,weights:torch.Tensor,y)-> float:
    
    assert(distributions.loc.shape == y.shape)
    assert(len(distributions.lco.shape) <=2)#no support for over 2-dimension data
    

    #TODO realize confidence calculation based on sampling
    #TODO might have hyperparameters for this function
    return

