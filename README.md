# spamr

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11116468.svg)](https://doi.org/10.5281/zenodo.11116468)

This repository contains the Sparse Bayesian Mixture Regression model (`spamr.py`) and a modular interface for experimenting with custom mixture regressions (`mixture_regression.py`). 

Cluster assignments for each (x, y) pair are marginalized out during inference with NumPyro's enumeration machinery.

###  Fitting a spamr model
```python
import jax
from numpyro.infer import MCMC, NUTS
from spamr import make_spamr

X = ...
Y = ...

spamr = make_spamr(X, Y)
mcmc = MCMC(NUTS(spamr), num_warmup=1_000, num_samples=1_000)
mcmc.run(jax.random.PRNGKey(0), X, Y=Y)
mcmc.get_samples()
```

For more info, see the demo notebook.

### Installation

You can install the latest version of spamr directly from GitHub using the following command:

```bash
pip install git+https://github.com/compmem/spamr
```

### Citation

```bibtex
@software{falk_2024_11116468,
  author       = {Falk, Ami and
                  Sederberg, Per B.},
  title        = {compmem/spamr: spamr 0.1.0 - Initial Prerelease},
  month        = may,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.11116468},
  url          = {https://doi.org/10.5281/zenodo.11116468}
}
```
