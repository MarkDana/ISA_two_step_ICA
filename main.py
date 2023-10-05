#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.covariance import GraphicalLasso
from ISA import sparseica_W_adasize_Alasso_mask_regu, from_W_to_B

def estimate(data, allowed_directed_edges=None, forbidden_directed_edges=None, init_mask_by_lasso=False):
    """
    Estimate the directed LiNGAM graph from data.
    Parameters
    ----------
    data : array, shape (n_features, n_samples). The data to estimate the graph from.
    allowed_directed_edges : List<Tuple>, optional. The allowed directed edges. Default as None.
    forbidden_directed_edges : List<Tuple>, optional. The forbidden directed edges. Default as None.
    init_mask_by_lasso : bool, optional. If False (default), Mask is set to all ones except for diagonals. If True (default),
            use Lasso, i.e., an edge X-Y is allowed only if X and Y are conditionally dependent given all other variables.
    Returns
    -------
    B : array, shape (n_features, n_features). The estimated directed graph with edge weights.
    """

    # 1. Hyper-parameters setting for sparseica_W_adasize_Alasso_mask_regu.
    num_nodes, samplesize = data.shape
    ICA_lambda = 4  # usually lamda is set to a constant times log(T), where T is sample size.
    ICA_regu = 0. # 0.05
    stablize_tol = 0.25 # 0.02
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
    _, W, _, _, _, _, _, _ = sparseica_W_adasize_Alasso_mask_regu(ICA_lambda * np.log(samplesize), ICA_Mask, data, ICA_regu)

    # 4. Further process the estimated demixing matrix W so that the corresponding causal system is stable.
    adjacency_matrix, nodes_permutation = from_W_to_B(W, tol=stablize_tol, sparsify=stablize_sparsify)

    # 5. Check if the forbidden directed edges are present.
    forbidden_edge_presented = forbidden_directed_edges is not None and \
                               any([adjacency_matrix[ch, pa] != 0 for pa, ch in forbidden_directed_edges])
    if forbidden_edge_presented:
        new_Mask = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
        # note: we cannot use lasso mask now. Need other possible edges (even cond.inds are violated) to explain the data.
        for pa, ch in forbidden_directed_edges: new_Mask[ch, pa] = 0    # the inverse direction (ch->pa) is allowed.
        init_W = np.eye(num_nodes) - adjacency_matrix * new_Mask    # trust the estimated causal system except for the forbidden edges.
        _, W, _, _, _, _, _, _ = sparseica_W_adasize_Alasso_mask_regu(ICA_lambda, new_Mask, data, ICA_regu, init_W)
        adjacency_matrix, nodes_permutation = from_W_to_B(W, tol=stablize_tol, sparsify=stablize_sparsify)

    return adjacency_matrix


if __name__ == '__main__':
    import os
    from lingam.utils import make_dot

    sname = 'f18test'
    data = np.loadtxt('./f18test.csv', delimiter=',').T
    nodenum = data.shape[0]

    adjacency_matrix = estimate(data)
    print(np.round(adjacency_matrix, 2), end='\n\n')
    d = make_dot(adjacency_matrix, labels=list(map(str, range(nodenum))))
    d.render(filename=sname, directory='./')
    os.system(f'rm -rf {os.path.join("./", sname)}')

