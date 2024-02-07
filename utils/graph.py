import numpy as np
import bct
from math import log2


def binarize_matrix(matrix, edge_density):

    """binarize_matrix: binarize matrix following given edge density

    :param matrix: matrix to binarize
    :param edge_density: edge density
    :return binarized_matrix: binarized_matrix
    """

    if len(matrix.shape) == 2:
        binarized_matrix = bct.utils.binarize(bct.threshold_proportional(matrix, edge_density))

    elif len(matrix.shape) == 3:
        n = matrix.shape[0]
        p = matrix.shape[-1]
        binarized_matrix = np.zeros((n, p, p))
        for i in range(n):
            bm = bct.utils.binarize(bct.threshold_proportional(matrix[i, :, :], edge_density))
            binarized_matrix[i, :, :] = bm

    else:
        print('You shoud input 2 or 3 dimensional array!')
        return None

    return binarized_matrix


def kl_div(p, q):

    """

    :param p: distribution p
    :param q:  distribution q
    :return: kl divergence
    """
    eps = 1e-10
    return sum(p[i] * log2((p[i]+eps)/(q[i]+eps)) for i in range(len(p))) / len(p)

