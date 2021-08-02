import numpy as np


def cheby_poly(X, max):

    result = np.zeros(X.shape)
    pre = None

    for _ in range(X.shape[0]):
        result[_][_] = 1

    for i in range(max):
        yield result
        if pre is None:
            result, pre = 2*X, result
        else:
            result, pre = 2*X@result-pre, result
