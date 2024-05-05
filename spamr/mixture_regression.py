from typing import Callable
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def make_mixture_regression(
    X,
    Y,
    n_clust: int,
    sample_mixture_weights: Callable,
    sample_beta: Callable,
    sample_Y: Callable,
):
    """
    Low-level API for building a mixture regression model.

    Args:
        X (Array): `(n_obs, n_dim_x)` array of covariates
        Y (Array): `(n_obs, n_dim_y)` array of response variables
        n_clust (int): Number of clusters to fit.
        sample_mixture_weights (Callable): Callable of the following form:
            `fn(n_clust: int)` -> array of shape `(n_clust,)`
        sample_beta (Callable): Callable of the following form:
            `fn(n_clust: int, n_dim_x: int, n_dim_y: int)` -> array of shape `(n_clust, n_dim_x, n_dim_y)`
        sample_Y (Callable): Callable of the following form:
            `fn(n_obs: int, n_dim_y: int, X_beta: Array, Y: Array)` -> array of shape `(n_obs, n_dim_y)`
    Returns:
        Callable mixture regression model
    """
    assert X.shape[0] == Y.shape[0]
    assert jnp.ndim(X) == jnp.ndim(Y) == 2

    N_OBS, N_DIM_X = X.shape
    N_OBS, N_DIM_Y = Y.shape
    N_CLUST = n_clust

    def mixture_regression(X, *, Y=None):
        mixture_weights = sample_mixture_weights(N_CLUST)
        beta = sample_beta(N_CLUST, N_DIM_X, N_DIM_Y)

        with numpyro.plate("n_obs", N_OBS, dim=-2):
            clust_idx = numpyro.sample(
                "clust_idx",
                dist.Categorical(probs=mixture_weights),
                infer={"enumerate": "parallel"},
            )

        # (n_obs, 1, n_dim_x, n_dim_y) * (n_obs, 1, n_dim_x, 1) -> (n_obs, n_dim_y)
        X_beta = jnp.sum(beta[clust_idx] * X[:, None, :, None], axis=(-3, -2))
        Y_obs = sample_Y(N_OBS, N_DIM_Y, X_beta, Y)

        return Y_obs

    return mixture_regression
