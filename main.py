#!/usr/bin/python3
# -*- coding: utf-8 -*-
from itertools import combinations
import numpy as np
from sklearn.covariance import GraphicalLasso
from ISA import sparseica_W_adasize_Alasso_mask_regu, from_W_to_B

def estimate(data, allowed_directed_edges=None, forbidden_directed_edges=None, init_mask_by_lasso=False, assume_acyclic_to_prevent_permutations=False):
    """
    Estimate the directed LiNGAM graph from data.
    Parameters
    ----------
    data : array, shape (n_features, n_samples). The data to estimate the graph from.
    allowed_directed_edges : List<Tuple>, optional. The allowed directed edges. Default as None.
    forbidden_directed_edges : List<Tuple>, optional. The forbidden directed edges. Default as None.
    init_mask_by_lasso : bool, optional. If False (default), Mask is set to all ones except for diagonals. If True (default),
            use Lasso, i.e., an edge X-Y is allowed only if X and Y are conditionally dependent given all other variables.
    assume_acyclic_to_prevent_permutations : bool, optional. If False (default), the final adjacency matrix is found by
            finding the stablest permutation on the estimated demixing matrix W. If True, the adjacency matrix is assumed to be
            permutable to a lower triangular matrix, and tricks from https://github.com/cdt15/lingam/blob/master/lingam/ica_lingam.py
            are used to prevent slow permutations (needed when the number of nodes is large, e.g., >8).
    Returns
    -------
    B : array, shape (n_features, n_features). The estimated directed graph with edge weights.
    """
    # 1. Hyper-parameters setting for sparseica_W_adasize_Alasso_mask_regu.
    num_nodes, samplesize = data.shape
    ICA_lambda = np.log(samplesize) * 4  # usually lamda is set to a constant times log(T), where T is sample size.
    ICA_regu = 0.05
    stablize_tol = 0.02
    stablize_sparsify = True

    # 2. Set init Mask.
    if init_mask_by_lasso:  # TODO: use the alasso in matlab code
        gl = GraphicalLasso()
        gl.fit(data.T)
        ICA_Mask = np.abs(gl.precision_) > 0.05 * np.max(np.abs(gl.precision_))
    else: ICA_Mask = np.ones((num_nodes, num_nodes))
    ICA_Mask[np.diag_indices(num_nodes)] = 0

    if allowed_directed_edges:
        for pa, ch in allowed_directed_edges: ICA_Mask[ch, pa] = ICA_Mask[pa, ch] = 1
    if forbidden_directed_edges:
        for pa, ch in forbidden_directed_edges:
            if (ch, pa) in forbidden_directed_edges: # only disable at Mask when it is 2-way forbidden.
                ICA_Mask[ch, pa] = ICA_Mask[pa, ch] = 0

    # 3. Run 2-step ICA and get the estimated demixing matrix W.
    _, W, _, _, _, _, _, _ = sparseica_W_adasize_Alasso_mask_regu(ICA_lambda, ICA_Mask, data, ICA_regu)

    # 4. Further process the estimated demixing matrix W so that the corresponding causal system is stable.
    adjacency_matrix, nodes_permutation = from_W_to_B(W, tol=stablize_tol, sparsify=stablize_sparsify,
                                                      assume_acyclic_to_prevent_permutations=assume_acyclic_to_prevent_permutations)

    # 5. Check if the forbidden directed edges are present.
    forbidden_edge_presented = forbidden_directed_edges is not None and \
                               any([adjacency_matrix[ch, pa] != 0 for pa, ch in forbidden_directed_edges])
    if forbidden_edge_presented:
        new_Mask = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
        # note: we cannot use lasso mask now. Need other possible edges (even cond.inds are violated) to explain the data.
        for pa, ch in forbidden_directed_edges: new_Mask[ch, pa] = 0    # the inverse direction (ch->pa) is allowed.
        init_W = np.eye(num_nodes) - adjacency_matrix * new_Mask    # trust the estimated causal system except for the forbidden edges.
        _, W, _, _, _, _, _, _ = sparseica_W_adasize_Alasso_mask_regu(ICA_lambda, new_Mask, data, ICA_regu, init_W)
        adjacency_matrix, nodes_permutation = from_W_to_B(W, tol=stablize_tol, sparsify=stablize_sparsify,
                                                      assume_acyclic_to_prevent_permutations=assume_acyclic_to_prevent_permutations)

    return adjacency_matrix



if __name__ == '__main__':
    import os
    from lingam.utils import make_dot

    nodenum = 5
    samplesize = 2000
    edgelist = [(3, 1), (1, 0), (0, 4), (4, 2)]
    adjmat = np.zeros((nodenum, nodenum))
    for pa, ch in edgelist: adjmat[ch, pa] = np.random.uniform(0.25, 1) * np.random.choice([-1, 1])
    mixingmat = np.linalg.pinv(np.eye(nodenum) - adjmat)
    E = np.random.uniform(low=np.random.uniform(-2, -1), high=np.random.uniform(1, 2), size=(nodenum, samplesize))
    data = mixingmat @ E

    settings = {
        'ground_truth': {'truth': adjmat},
        'no_knowledge': {'allowed': None, 'forbidden': None, 'init_lasso': True},
        'forbid_edge': {'allowed': None, 'forbidden': [(1, 0)], 'init_lasso': False},
    }

    for sname, setting in settings.items():
        adjacency_matrix = setting['truth'] if 'truth' in setting else estimate(data, setting['allowed'], setting['forbidden'], setting['init_lasso'])
        print(f'=== {sname} ==='); print(np.round(adjacency_matrix, 2), end='\n\n')
        d = make_dot(adjacency_matrix, labels=list(map(str, range(nodenum))))
        d.render(filename=sname, directory='./')
        os.system(f'rm -rf {os.path.join("./", sname)}')

    # update 07/01/2025: fast causal order search, for large acyclic graphs, prevent from permutations traversing
    nodenum = 10
    dag_sparsity = 0.5
    samplesize = 3000
    edges = [(i, j) for i, j in combinations(range(nodenum), 2) if np.random.uniform(0, 1) > dag_sparsity]
    name_perm = np.random.permutation(nodenum)
    edges = [(name_perm[i], name_perm[j]) for i, j in edges]
    adjmat = np.zeros((nodenum, nodenum), dtype=float)
    for i, j in edges: adjmat[j, i] = np.random.uniform(0.5, 2.5) * np.random.choice([-1, 1])
    mixingmat = np.linalg.inv(np.eye(nodenum) - adjmat)
    E = np.random.rand(nodenum, samplesize) # uniform
    data = mixingmat @ E
    adjmat_estimated = estimate(data, init_mask_by_lasso=False, assume_acyclic_to_prevent_permutations=True)
