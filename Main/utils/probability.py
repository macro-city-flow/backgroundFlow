import torch

def sample(distributions:torch.distributions.Normal,weights:torch.Tensor,times:int)->list:
    assert(len(distributions.loc.shape) <=2)#no support for over 2-dimension data
    result = []
    for _ in range(times):
        sample = None#Not implied yet
        result.append(sample)
    return  result