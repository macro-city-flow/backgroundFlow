import torch

def accuracy(pred, y):

    return 1 - torch.linalg.norm(y - pred, 'fro') / torch.linalg.norm(y, 'fro')

def multiFeatureAccuracy(pred, y):
    return 1 - sum([torch.linalg.norm(y[i]-pred[i]) for i in range(len(pred))])/torch.linalg.norm(y)



