import numpy as np
import scipy.sparse as sp
import torch
from torch.nn.functional import normalize


def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    #if here need a step to use the normalize?
    return normalized_laplacian

def calculate_cheby_ploy(X:torch.Tensor,gamma:int):
    output = torch.FloatTensor(gamma+1,X.shape[0],X.shape[1])
    output[0] = torch.eye(X.shape[0])
    if gamma >= 1:
        output[1] = X
    else:
        return output
    for _ in range(gamma-2):
        output[_+2]  = 2*X*output[_+1]-output[_]
    return output 