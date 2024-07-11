from typing import Callable, Optional

import logging
logger = logging.getLogger(__name__)

import numpy as np
from stein_thinning.kernel import make_imq


def _greedy_search(
        n_points: int,
        kernel_0: Callable[[], np.ndarray],
        kernel_1: Callable[[int], np.ndarray]
) -> np.ndarray:
    """Select points minimising total kernel Stein distance

    Parameters
    ----------
    n_points: int
        number of points to select.
    kernel_0: Callable[[], np.ndarray]
        function returning values of Stein kernel evaluated between
        each point and itself, i.e. `k(x, x)`.
    kernel_1: Callable[[int], np.ndarray]
        function returning values of Stein kernel evaluated between
        a point with the given index and all points in the sample.

    Returns
    -------
    np.ndarray
        indices of selected points
    """
    # Pre-allocate arrays
    idx = np.empty(n_points, dtype=np.uint32)

    # Populate columns of k0 as new points are selected
    k0 = kernel_0()
    idx[0] = np.argmin(k0)
    logger.debug('THIN: %d of %d', 1, n_points)
    for i in range(1, n_points):
        k0 += 2 * kernel_1(idx[i - 1])
        idx[i] = np.argmin(k0)
        logger.debug('THIN: %d of %d', i + 1, n_points)

    return idx


def thin_gf(
        sample: np.ndarray,
        log_p: np.ndarray,
        log_q: np.ndarray,
        gradient_q: np.ndarray,
        n_points: int,
        standardize: bool = True,
        preconditioner: str = 'id',
        vfk0: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Optimally select m points from n > m samples generated from a target distribution of d dimensions.

    This function is based on the gradient-free kernel Stein discrepancy,
    which uses an auxiliary distribution q as a proxy for the target distribution.
    This is useful when the gradient of the target distribution is difficult to obtain,
    so instead the gradient of the proxy distribution is used.

    Parameters
    ----------
    sample: np.ndarray
        n x d array where each row is a sample point.
    log_p: np.ndarray
        n x 1 array of log-pdf values for the target distribution corresponding
        to points in `sample`.
    log_q: np.ndarray
        n x 1 array of log-pdf values for the proxy distribution corresponding
        to points in `sample`.
    gradient_q: np.ndarray
        n x d array of gradient of the proxy distribution corresponding to points
        in `sample`.
    n_points: int
        integer specifying the desired number of points.
    standardize: bool
        optional logical, either 'True' (default) or 'False', indicating
        whether or not to standardise the columns of `sample` around means
        using the mean absolute deviation from the mean as the scale.
    preconditioner: str
        optional string, either 'id' (default), 'med', 'sclmed', or
        'smpcov', specifying the preconditioner to be used. Alternatively,
        a numeric string can be passed as the single length-scale parameter
        of an isotropic kernel.
    vfk0: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]]
        Stein kernel to use for calculating discrepancies

    Returns
    -------
    np.ndarray
        array shaped (m,) containing the row indices in `sample` (and `gradient`) of the
        selected points.
    """
    # Argument checks
    assert sample.ndim == 2, 'sample is not two-dimensional.'
    n, d = sample.shape
    assert n > 0 and d > 0, 'sample is empty.'
    assert not np.any(np.isnan(sample)), 'sample contains NaNs.'
    assert not np.any(np.isinf(sample)), 'sample contains infs.'

    def validate_log_prob(vals, var_name):
        assert vals.ndim == 1 or vals.ndim == 2 and vals.shape[1] == 1, f'{var_name} must be a vector.'
        assert vals.shape[0] == n, f'Dimensions of sample and {var_name} are inconsistent.'
        assert not np.any(np.isnan(vals)), f'{var_name} contains NaNs.'
        assert not np.any(np.isinf(vals)), f'{var_name} contains infs.'
        return vals.squeeze()

    log_p = validate_log_prob(log_p, 'log_p')
    log_q = validate_log_prob(log_q, 'log_q')

    assert gradient_q.shape == sample.shape, 'Dimensions of sample and gradient_q are inconsistent.'

    # Standardisation
    if standardize:
        loc = np.mean(sample, axis=0)
        scl = np.mean(np.abs(sample - loc), axis=0)
        assert np.min(scl) > 0, 'Too few unique samples in smp.'
        sample = sample / scl
        gradient_q = gradient_q * scl

    # Vectorised Stein kernel function
    if vfk0 is None:
        vfk0 = make_imq(sample, preconditioner)

    def kernel_0():
        return np.exp(log_q - log_p) ** 2 * vfk0(sample, sample, gradient_q, gradient_q)

    def kernel_1(i):
        return np.exp(log_q - log_p + log_q[i] - log_p[i]) * vfk0(sample, sample[[i]], gradient_q, gradient_q[[i]])

    return _greedy_search(n_points, kernel_0, kernel_1)


def thin(
        sample: np.ndarray,
        gradient: np.ndarray,
        n_points: int,
        standardize: bool = True,
        preconditioner: str = 'id',
        vfk0: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Optimally select m points from n > m samples generated from a target distribution of d dimensions.

    Parameters
    ----------
    sample: np.ndarray
        n x d array where each row is a sample point.
    gradient: np.ndarray
        n x d array where each row is a gradient of the log target.
    n_points: int
        integer specifying the desired number of points.
    standardize: bool
        optional logical, either 'True' (default) or 'False', indicating
        whether or not to standardise the columns of `sample` around means
        using the mean absolute deviation from the mean as the scale.
    preconditioner: str
        optional string, either 'id' (default), 'med', 'sclmed', or
        'smpcov', specifying the preconditioner to be used. Alternatively,
        a numeric string can be passed as the single length-scale parameter
        of an isotropic kernel.
    vfk0: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]]
        Stein kernel to use for calculating discrepancies

    Returns
    -------
    np.ndarray
        array shaped (m,) containing the row indices in `sample` (and `gradient`) of the
        selected points.
    """
    n, _ = sample.shape
    z = np.zeros(n)
    return thin_gf(
        sample=sample,
        log_p=z,
        log_q=z,
        gradient_q=gradient,
        n_points=n_points,
        standardize=standardize,
        preconditioner=preconditioner,
        vfk0=vfk0,
    )
