import torch
import numpy as np


def vector_ratio_interpolation(vec1: torch.Tensor, vec2: torch.Tensor, n_splits: int):
    """vector_ratio_interpolation: interpolate vector(s) between vec1 and vec2

    :param vec1: vector 1
    :param vec2: vector 2
    :param n_splits: number of splits
    :return: interpolate_z: interpolated vectors
    """

    interpolated_z = torch.zeros(size=(n_splits - 1, vec1.shape[1]))
    for i in range(n_splits - 1):
        # Interpolation
        # print('vec1: ', n_splits - 1 - i)
        # print('vec2: ', 1 + i)
        z = (vec1 * (n_splits - 1 - i) + vec2 * (1 + i)) / n_splits
        interpolated_z[i, :] = z

    return interpolated_z


def scalar_ratio_interpolation(s1, s2, n_splits: int):
    """scalar_ratio_interpolation: interpolate scalar(s) between s1 and s2

    :param s1: scalar 1
    :param s2: scalar 2
    :param n_splits: number of splits
    :return: interpolated_s: interpolated scalars
    """

    interpolated_s = []
    for i in range(n_splits - 1):
        # Interpolation
        s = (s1 * (n_splits - 1 - i) + s2 * (1 + i)) / n_splits
        interpolated_s.append(s)

    return interpolated_s


def matrices_targets_synthesizer(encoder, decoder, matrices, targets, n_splits: int):
    """matrices_targets_synthesizer: interpolate matrix (matrices) and target(s)

    :param encoder: encoder of tg-gan
    :param decoder: decoder of tg-gan
    :param matrices: matrices to interpolate
    :param targets: targets to interpolate
    :param n_splits: number of splits
    :return: synthesized_matrices: synthesized (interpolated) matrices
    :return: synthesized_targets: synthesized (interpolated) targets
    """

    synthesized_matrices = []
    synthesized_targets = []
    for i in range(len(matrices) - 1):
        # ------ Step 1: Get latent vectors ------ #
        # Matrix and generate latent vectors
        mat1 = matrices[i].to(dtype=torch.float32)
        mat2 = matrices[i + 1].to(dtype=torch.float32)
        vec1 = encoder(mat1)
        vec2 = encoder(mat2)

        # ------ Step 2: Generate interpolated latent vectors ------ #
        z = vector_ratio_interpolation(vec1=vec1, vec2=vec2, n_splits=n_splits)

        # ------ Step 3: Synthesize matrices ------ #
        mat = decoder(z)
        mat = mat.detach().numpy().reshape(mat.shape[0], mat.shape[-1], mat.shape[-1])
        synthesized_matrices.extend(mat)

        # ------ Step 4: Generated objective variables ------ #
        s = scalar_ratio_interpolation(s1=targets[i], s2=targets[i + 1], n_splits=n_splits)
        synthesized_targets.append(s)

    # list -> numpy.array
    synthesized_matrices = np.array(synthesized_matrices)
    synthesized_targets = np.array(synthesized_targets).reshape(-1)

    return synthesized_matrices, synthesized_targets
