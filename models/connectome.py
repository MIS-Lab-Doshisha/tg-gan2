import torch
from math import sqrt, floor


def sym_matrix_to_vec_torch(symmetric):
    """sym_matrix_to_vec: Return the flattened lower triangular part of an array

    :param symmetric: adjacency matrix
    :return: lower triangular part of an array
    """
    tril_mask = torch.tril(torch.ones(symmetric.shape[-2:]), diagonal=-1).to(dtype=bool).to(device=symmetric.device)

    return symmetric[..., tril_mask]


def vec_to_sym_matrix_torch(vec, diagonal=None):
    """vec_to_sym_matrix_torch

    :param vec: lower triangular part of an array
    :param diagonal: diagonal
    :return: sym: adjacency matrix
    """
    n = vec.shape[-1]

    # Compute the number of the symmetric matrix columns
    # solve n_columns * (n_columns + 1) / 2 = n
    # subject to n_columns > 0
    n_columns = (sqrt(8 * n + 1) - 1.) / 2
    if diagonal is not None:
        n_columns += 1

    if n_columns > floor(n_columns):
        raise ValueError(
            'Vector of unsuitable shape {0} can not be transformes to'
            'a symmetric matrix'.format(vec.shape)
        )

    n_columns = int(n_columns)
    first_shape = vec.shape[:-1]

    if diagonal is not None:
        if diagonal.shape[:-1] != first_shape or diagonal.shape[-1] != n_columns:
            raise ValueError(
                'diagonal of shape {0} incompatible with vector'
                ' of shape {1}'.format(diagonal.shape, vec.shape)
            )

    sym = torch.zeros(first_shape + (n_columns, n_columns)).to(vec.device)

    # Fill lower triangular part
    skip_diagonal = (diagonal is not None)
    mask = torch.tril(torch.ones((n_columns, n_columns)), diagonal=-skip_diagonal).to(bool).to(vec.device)
    sym[..., mask] = vec

    # Fill upper triangular part
    sym.swapaxes(-1, -2)[..., mask] = vec

    # (Fill and) rescale diagonal terms
    mask = torch.full_like(mask, fill_value=False)
    mask.fill_diagonal_(fill_value=True)

    if diagonal is not None:
        diagonal = diagonal.to(vec.device)
        sym[..., mask] = diagonal

    sym[..., mask] *= sqrt(2)

    return sym
