from typing import Any, Callable, Optional, Tuple

import logging
logger = logging.getLogger(__name__)

import numpy as np
from stein_thinning.kernel import make_imq


IndexerT = Any


def _greedy_search(
        n_points: int,
        integrand: Callable[[IndexerT, IndexerT], np.ndarray],
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Select points minimising total kernel Stein distance

    Parameters
    ----------
    n_points: int
        number of points to select.
    integrand: Callable[[IndexerT, IndexerT], np.ndarray]
        function returning values of the integrand in the KSD integral
        for points identified by two indices (row and column).

    Returns
    -------
    np.ndarray
        indices of selected points
    """
    # Pre-allocate arrays
    idx = np.empty(n_points, dtype=np.uint32)

    # Populate columns of k0 as new points are selected
    k0 = integrand(slice(None), slice(None))
    idx[0] = np.argmin(k0)
    logger.debug('THIN: %d of %d', 1, n_points)
    for i in range(1, n_points):
        k0 += 2 * integrand(slice(None), [idx[i - 1]])
        idx[i] = np.argmin(k0)
        logger.debug('THIN: %d of %d', i + 1, n_points)

    return idx


def _validate_sample_and_gradient(sample, gradient, standardize):
    assert sample.ndim == 2, 'sample is not two-dimensional.'
    n, d = sample.shape
    assert n > 0 and d > 0, 'sample is empty.'
    assert not np.any(np.isnan(sample)), 'sample contains NaNs.'
    assert not np.any(np.isinf(sample)), 'sample contains infs.'

    assert gradient.shape == sample.shape, 'Dimensions of sample and gradient_q are inconsistent.'
    assert not np.any(np.isnan(gradient)), 'sample contains NaNs.'
    assert not np.any(np.isinf(gradient)), 'sample contains infs.'

    # Standardisation
    if standardize:
        loc = np.mean(sample, axis=0)
        scl = np.mean(np.abs(sample - loc), axis=0)
        assert np.min(scl) > 0, 'Too few unique samples in smp.'
        sample = sample / scl
        gradient = gradient * scl

    return sample, gradient


def _make_stein_integrand(
        sample: np.ndarray,
        gradient: np.ndarray,
        standardize: bool = True,
        preconditioner: str = 'id',
        vfk0: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] = None,
):
    # Argument checks
    sample, gradient = _validate_sample_and_gradient(sample, gradient, standardize)

    # Vectorised Stein kernel function
    if vfk0 is None:
        vfk0 = make_imq(sample, preconditioner)

    def integrand(ind1, ind2):
        return vfk0(sample[ind1], sample[ind2], gradient[ind1], gradient[ind2])

    return integrand


def _make_stein_gf_integrand(
        sample: np.ndarray,
        log_p: np.ndarray,
        log_q: np.ndarray,
        gradient_q: np.ndarray,
        standardize: bool = True,
        preconditioner: str = 'id',
        vfk0: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] = None,
):
    # Argument checks
    sample, gradient_q = _validate_sample_and_gradient(sample, gradient_q, standardize)
    n, _ = sample.shape

    def validate_log_prob(vals, var_name):
        assert vals.ndim == 1 or vals.ndim == 2 and vals.shape[1] == 1, f'{var_name} must be a vector.'
        assert vals.shape[0] == n, f'Dimensions of sample and {var_name} are inconsistent.'
        assert not np.any(np.isnan(vals)), f'{var_name} contains NaNs.'
        assert not np.any(np.isinf(vals)), f'{var_name} contains infs.'
        return vals.squeeze()

    log_p = validate_log_prob(log_p, 'log_p')
    log_q = validate_log_prob(log_q, 'log_q')

    # Vectorised Stein kernel function
    if vfk0 is None:
        vfk0 = make_imq(sample, preconditioner)

    def integrand(ind1, ind2):
        return (
            np.exp(log_q[ind1] - log_p[ind1] + log_q[ind2] - log_p[ind2]) *
            vfk0(sample[ind1], sample[ind2], gradient_q[ind1], gradient_q[ind2])
        )

    return integrand


def thin_gf(
        sample: np.ndarray,
        log_p: np.ndarray,
        log_q: np.ndarray,
        gradient_q: np.ndarray,
        n_points: int,
        standardize: bool = True,
        preconditioner: str = 'id',
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

    Returns
    -------
    np.ndarray
        array shaped (m,) containing the row indices in `sample` (and `gradient`) of the
        selected points.
    """
    integrand = _make_stein_gf_integrand(sample, log_p, log_q, gradient_q, standardize, preconditioner)
    return _greedy_search(n_points, integrand)


def thin(
        sample: np.ndarray,
        gradient: np.ndarray,
        n_points: int,
        standardize: bool = True,
        preconditioner: str = 'id',
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

    Returns
    -------
    np.ndarray
        array shaped (m,) containing the row indices in `sample` (and `gradient`) of the
        selected points.
    """
    integrand = _make_stein_integrand(sample, gradient, standardize, preconditioner)
    return _greedy_search(n_points, integrand)