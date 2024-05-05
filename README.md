<h1 align='center'>spamr</h1>

XXX (register on Zenodo) (and make a prelease)

This repository contains the Sparse Bayesian Mixture Regression model (`spamr.py`) and a modular interface for experimenting with custom mixture regressions (`mixture_regression.py`). 

Cluster assignments for each (x, y) pair are marginalized out during inference with NumPyro's enumeration machinery.

###  Fitting a spamr model
```python
from numpyro.infer import MCMC, NUTS
from spamr import make_spamr

X = ...
Y = ...

mcmc = MCMC(NUTS(spamr), num_warmup=1_000, num_samples=1_000)
mcmc.run(mcmc_key, X, Y=Y)
mcmc.get_samples()
```

For more info, see the demo notebook.

### Installation

You can install the latest development version of spamr directly from GitHub using the following command:

```bash
pip install git+https://github.com/compmem/spamr
```

If you found this helpful for your research, please cite us with XXX

