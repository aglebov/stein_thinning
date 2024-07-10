"""Stein thinning for a Gaussian mixture model using the gradient-free approach"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

from jax import grad
import jax.numpy as jnp
import jax.scipy.stats.multivariate_normal as jmvn

from stein_thinning.thinning import thin, thin_gf


# Generate from a mixture model
def make_mvn_mixture(weights, means, covs):
    # invert covariances
    covs_inv = np.linalg.inv(covs)

    k, d = means.shape
    assert weights.shape == (k,)
    assert covs.shape == (k, d, d)

    def rvs(size, random_state):
        component_samples = [
            mvn.rvs(mean=means[i], cov=covs[i], size=size, random_state=random_state) for i in range(len(weights))
        ]
        indices = rng.choice(len(weights), size=size, p=weights)
        return np.take_along_axis(np.stack(component_samples, axis=1), indices.reshape(size, 1, 1), axis=1).squeeze()

    def logpdf_jax(x):
        probs = jmvn.pdf(x.reshape(-1, 1, d), mean=means.reshape(1, k, d), cov=covs.reshape(1, k, d, d))
        return jnp.squeeze(jnp.log(jnp.sum(weights * probs, axis=1)))
    
    return rvs, logpdf_jax

w = np.array([0.3, 0.7])
means = np.array([
    [-1., -1.],
    [1., 1.],
])
covs = np.array([
    [
        [0.5, 0.25],
        [0.25, 1.],
    ],
    [
        [2.0, -np.sqrt(3.) * 0.8],
        [-np.sqrt(3.) * 0.8, 1.5],
    ]
])

rvs, logpdf_jax = make_mvn_mixture(w, means, covs)

rng = np.random.default_rng(12345)
sample_size = 1000
sample = rvs(sample_size, random_state=rng)

# Numerically calculate the gradient
gradient = np.array(jnp.apply_along_axis(grad(logpdf_jax), 1, sample))

# Apply Stein thinning
thinned_size = 40
idx = thin(sample, gradient, thinned_size)

# For the proxy distribution, use a simple Gaussian with sample mean and covariance
sample_mean = np.mean(sample, axis=0)
sample_cov = np.cov(sample, rowvar=False, ddof=1)

# Gradient-free Stein thinning requires us to provide the log-pdf of the proxy
# distribution and its score function:
log_q = mvn.logpdf(sample, mean=sample_mean, cov=sample_cov)
gradient_q = -np.einsum('ij,kj->ki', np.linalg.inv(sample_cov), sample - sample_mean)

# We also need the log-pdf of the target distribution
log_p = np.array(logpdf_jax(sample))

# Apply gradient-free Stein thinning
idx_gf = thin_gf(sample, log_p, log_q, gradient_q, thinned_size)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].scatter(sample[:, 0], sample[:, 1], alpha=0.3, color='gray')
axs[0].scatter(sample[idx, 0], sample[idx, 1], color='red')
axs[0].set_title('Stein thinning')
axs[1].scatter(sample[:, 0], sample[:, 1], alpha=0.3, color='gray')
axs[1].scatter(sample[idx_gf, 0], sample[idx_gf, 1], color='red')
axs[1].set_title('Gradient-free Stein thinning')

plt.show()