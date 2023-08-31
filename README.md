# 2-step ICA for Causal Discovery

This Python implementation is mainly translated from Kun's [Matlab codes](two_step_CD_regu_matlab_by_kun/).

### 0. Requirements

+ python 3 (>=3.7)
+ numpy
+ sklearn
+ numba (this is for matrix calculations speedup; ~4x faster than matlab codes)
+ [lingam](https://pypi.org/project/lingam/) (this is only for plotting)

### 1. Quick Start

Check `main.py` and run the function

```python
estimate(data, allowed_directed_edges, forbidden_directed_edges, init_mask_by_lasso)
```

where:

+ `data` : array, shape (n_features, n_samples). The data to estimate the graph from. The node names are [0, 1, ..., n_features-1].
+ `allowed_directed_edges` : List of tuples, optional. The allowed **directed** edges. Default as None.
+ `forbidden_directed_edges` : List of tuples, optional. The forbidden **directed** edges. Default as None.
+ `init_mask_by_lasso` : bool, optional. Determines how to set Mask, i.e., the allowed entries for gradient updates. If False (default), Mask is set to all ones except for diagonals. If True (default), use Lasso, i.e., an edge X-Y is allowed only if X and Y are conditionally dependent given all other variables. Setting `init_mask_by_lasso=True` can be faster.

The returned `adjacency_matrix` has a nonzero entry at `[j, i]` iff there is an estimated edge from `i` to `j`, and the value is the corresponding causal weight.

Run `python main.py` to see a toy example, including the data simulation and the plotted resulting graphs (the pdf files in this folder) with different prior constraints. Note that currently the prior constraints on indirect influences are not supported yet.