import warnings

import numpy as np

from champollion._data import as_float_tensor


def process_prior_data(prior_data):
    """Center and normalize prior features cell-wise.

    The prior representation is row-centered, then nonzero rows are normalized
    to unit Euclidean norm. After this transformation, dot products between
    rows correspond to centered cosine similarities. Rows with zero norm are
    left as zeros to avoid NaNs.

    Parameters
    ----------
    prior_data
        Array-like prior representation with cells in rows and prior features
        in columns.

    Returns
    -------
    numpy.ndarray
        Processed prior representation.
    """
    prior_data = np.asarray(prior_data).copy()
    prior_data[np.isnan(prior_data)] = 0
    prior_data -= prior_data.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(prior_data, axis=1, keepdims=True)
    nonzero_norm = norm[:, 0] > 0
    prior_data[nonzero_norm] /= norm[nonzero_norm]
    return prior_data


def compute_prior_cost(y_prior_1, y_prior_2, device, use_keops=False):
    """Compute the normalized prior cost between two modalities.

    This function is primarily used internally by Champollion. It preprocesses
    both prior matrices with :func:`process_prior_data`, computes
    ``1 - similarity``, and normalizes by the mean cost. If the mean cost is
    numerically zero, a warning is emitted and a zero prior cost is returned.

    Parameters
    ----------
    y_prior_1, y_prior_2
        Prior representations with the same number of columns.
    device
        Torch device on which to place the result.
    use_keops
        If ``True``, return a symbolic KeOps cost object.

    Returns
    -------
    torch.Tensor or pykeops.torch.LazyTensor
        Normalized pairwise prior cost.
    """
    y_prior_1 = process_prior_data(y_prior_1)
    y_prior_2 = process_prior_data(y_prior_2)
    if use_keops:
        from pykeops.torch import LazyTensor

        y_1 = as_float_tensor(y_prior_1, device=device)
        y_2 = as_float_tensor(y_prior_2, device=device)
        y_1_i = LazyTensor(y_1[:, None, :])
        y_2_j = LazyTensor(y_2[None, :, :])
        cost = 1 - (y_1_i | y_2_j)
        cost_mean = cost.sum(dim=1).sum() / (y_prior_1.shape[0] * y_prior_2.shape[0])
    else:
        cost = as_float_tensor(1 - np.dot(y_prior_1, y_prior_2.T), device=device)
        cost_mean = cost.mean()
    if np.isclose(float(cost_mean.detach().cpu()), 0.0, atol=1e-6):
        warnings.warn(
            "Prior cost mean is numerically zero, so the prior cost cannot be "
            "normalized. Returning a zero prior cost; this prior will not affect "
            "the transport problem.",
            UserWarning,
            stacklevel=2,
        )
        return cost * 0
    return cost / cost_mean
