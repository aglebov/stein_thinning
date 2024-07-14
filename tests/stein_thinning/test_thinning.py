from pathlib import Path

import numpy as np
import pytest

from stein_thinning.kernel import vfk0_imq, make_precon, make_imq
from stein_thinning.thinning import thin, _make_stein_integrand, _greedy_search


@pytest.fixture
def demo_data_dir():
    return Path('stein_thinning') / 'demo' / 'data' / 'gmm'


@pytest.fixture
def demo_smp(demo_data_dir):
    return np.genfromtxt(demo_data_dir / 'smp.csv', delimiter=',')


@pytest.fixture
def demo_scr(demo_data_dir):
    return np.genfromtxt(demo_data_dir / 'scr.csv', delimiter=',')


def test_thin(demo_smp, demo_scr):
    idx = thin(demo_smp, demo_scr, 40)
    expected = np.array([
        68, 322, 268, 234, 161, 292, 229, 275, 259, 131, 400, 486, 207,
        120, 443, 430, 376, 411,  98, 293, 111, 372, 285, 427, 406, 246,
        148, 260, 296, 208,  79, 430, 369, 363, 462, 393, 321, 460, 373,
        114
    ])
    np.testing.assert_array_equal(idx, expected)

    preconditioner = make_precon(demo_smp, 'id')
    def kernel1(sample1, sample2, gradient1, gradient2):
        return vfk0_imq(sample1, sample2, gradient1, gradient2, preconditioner)
    integrand = _make_stein_integrand(demo_smp, demo_scr, vfk0=kernel1)
    idx = _greedy_search(40, integrand)
    np.testing.assert_array_equal(idx, expected)

    def kernel2(sample1, sample2, gradient1, gradient2):
        return vfk0_imq(sample1, sample2, gradient1, gradient2, preconditioner, beta=-0.75)
    integrand = _make_stein_integrand(demo_smp, demo_scr, vfk0=kernel2)
    idx = _greedy_search(40, integrand)
    expected = np.array([
        68, 322, 268, 234, 161, 292, 229, 276, 259, 131, 207, 431, 486,
        120, 457, 430, 412, 376, 111, 101,  97, 332, 394, 123, 429, 109,
        349,  79, 466, 114, 458, 371, 296, 284,  89, 317, 485, 392, 261,
        246
    ])
    np.testing.assert_array_equal(idx, expected)
