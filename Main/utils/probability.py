import torch
import numpy as np
from torch import Tensor


def sample(mu: Tensor, sigma: Tensor, weights: Tensor, times: int) -> Tensor:

    assert(mu.shape == sigma.shape == weights.shape)
    assert(len(weights.shape) <= 2)  # no support for over 2-dimension data
    result = torch.FloatTensor(times, mu.shape[0])
    for _ in range(times):
        k = torch.multinomial(weights, number_samples=1, replacement=True)
        result[_] = torch.normal(mu, sigma)[np.arange(mu.shape[0]), k].data

    return result