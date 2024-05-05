from functools import partial
from typing import NamedTuple

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from spamr.mixture_regression import make_mixture_regression


class Priors(NamedTuple):
    """
    Priors for SpaMR.

    Args:
        alpha: Hyperprior on the stick-breaking process concentration parameter.
            Higher values of alpha will result in fewer effective clusters.
        sigma: response variable noise scale
        tau: Global shrinkage prior on beta. Lower values of tau will collectively induce
            sparsity across all betas, for all clusters, for all response variables.
        lam: Local shrinkage prior on beta. Lower values of lam will individually induce sparsity
            for each beta, for each cluster, for each response variable.
    """

    alpha: dist.Distribution
    sigma: dist.Distribution
    tau: dist.Distribution
    lam: dist.Distribution


DEFAULT_PRIORS = Priors(
    alpha=dist.Gamma(1, 1),
    sigma=dist.HalfCauchy(1),
    tau=dist.HalfCauchy(1),
    lam=dist.HalfCauchy(1),
)


def _validate_prior_dist_shapes(
    priors: Priors, n_clust: int, n_dim_x: int, n_dim_y: int
):
    assert priors.alpha.batch_shape == ()
    assert jnp.broadcast_shapes(priors.sigma.batch_shape, (n_dim_y,)) == (n_dim_y,)
    assert priors.tau.batch_shape == ()
    assert jnp.broadcast_shapes(
        priors.lam.batch_shape, (n_clust, n_dim_x, n_dim_y)
    ) == (n_clust, n_dim_x, n_dim_y)


def stickbreak(v):
    cumprod_one_minus_v = jnp.cumprod(1 - v)
    v_one = jnp.pad(v, (0, 1), constant_values=1)
    one_c = jnp.pad(cumprod_one_minus_v, (1, 0), constant_values=1)

    return v_one * one_c


def sample_stickbreaking_mixture_weights(n_clust, alpha_prior_dist):
    alpha = numpyro.sample("alpha", alpha_prior_dist)

    with numpyro.plate("n_clust_sticks", n_clust - 1):
        clust_sticks = numpyro.sample("clust_sticks", dist.Beta(1, alpha))

    return numpyro.deterministic("mixture_weights", stickbreak(clust_sticks))


def sample_unordered_beta(n_clust, n_dim_x, n_dim_y, tau_prior_dist, lambda_prior_dist):
    tau = numpyro.sample("tau", tau_prior_dist)

    with numpyro.plate("n_clust", n_clust, dim=-3):
        with numpyro.plate("n_dim_x", n_dim_x, dim=-2):
            with numpyro.plate("n_dim_y", n_dim_y, dim=-1):
                lam = numpyro.sample("lambda", lambda_prior_dist)
                unscaled_beta = numpyro.sample("unscaled_beta", dist.Normal(0, 1))

    return numpyro.deterministic("beta", tau * lam * unscaled_beta)


def sample_ordered_beta(
    n_clust, n_dim_x, n_dim_y, *, tau_prior_dist, lambda_prior_dist
):
    tau = numpyro.sample("tau", tau_prior_dist)

    with numpyro.plate("n_clust", n_clust, dim=-3):
        with numpyro.plate("n_dim_x", n_dim_x, dim=-2):
            with numpyro.plate("n_dim_y", n_dim_y, dim=-1):
                lam = numpyro.sample("lambda", lambda_prior_dist)

    with numpyro.plate("n_dim_x", n_dim_x, dim=-2):
        with numpyro.plate("n_dim_y", n_dim_y, dim=-1):
            ordered_dist = dist.TransformedDistribution(
                dist.Normal(0, jnp.ones(n_clust)), dist.transforms.OrderedTransform()
            )
            unscaled_beta = numpyro.sample("unscaled_beta", ordered_dist)
            unscaled_beta = jnp.moveaxis(unscaled_beta, 2, 0)

    return numpyro.deterministic("beta", tau * lam * unscaled_beta)


def sample_continous_Y(n_obs, n_dim_y, X_beta, Y, sigma_prior_dist):
    with numpyro.plate("n_dim_y", n_dim_y, dim=-1):
        sigma = numpyro.sample("sigma", sigma_prior_dist)

    with numpyro.plate("n_obs", n_obs, dim=-2):
        with numpyro.plate("n_dim_y", n_dim_y, dim=-1):
            Y_obs = numpyro.sample("Y", dist.Normal(X_beta, sigma), obs=Y)

    return Y_obs


def make_spamr(
    X, Y, *, n_clust=None, ordered_beta=True, priors: Priors = DEFAULT_PRIORS
):
    """
    Create a SpaMR model.

    Args:
        X (Array): `(n_obs, n_dim_x)` array of covariates
        Y (Array): `(n_obs, n_dim_y)` array of response variables
        n_clust (int): number of clusters to fit. Defaults to `int(2 * log(n_obs))`.
        ordered_beta (bool): Whether or not to fit the betas with an ordered transform. Selecting True
            can help resolve label switching but may degenerate the posterior geometry. Defaults to True.
        priors (Priors): named tuple of prior distributions to use in the model. Defaults
            to `model.DEFAULT_PRIORS`.
    Returns:
        Callable SpaMR model
    """
    assert X.shape[0] == Y.shape[0]
    assert jnp.ndim(X) == jnp.ndim(Y) == 2

    n_clust = int(2 * jnp.log(X.shape[0])) if n_clust is None else n_clust

    _validate_prior_dist_shapes(priors, n_clust, X.shape[1], Y.shape[1])

    sample_mixture_weights = partial(
        sample_stickbreaking_mixture_weights, alpha_prior_dist=priors.alpha
    )

    sample_beta = sample_ordered_beta if ordered_beta else sample_unordered_beta
    sample_beta = partial(
        sample_beta, tau_prior_dist=priors.tau, lambda_prior_dist=priors.lam
    )

    # XXX we should allow for arbitrary response values
    sample_Y = partial(sample_continous_Y, sigma_prior_dist=priors.sigma)

    return make_mixture_regression(
        X,
        Y,
        n_clust,
        sample_mixture_weights=sample_mixture_weights,
        sample_beta=sample_beta,
        sample_Y=sample_Y,
    )
