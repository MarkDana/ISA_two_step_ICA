#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: Kun's 2-step-CD matlab code, translated by haoyue@04/30/2022
# note that every step is fully deterministic, so all results and intermediate variables
#   are suppose to be identical across matlab and python code.
# however, there remains some inaccuracy due to data type precision used by matlab and python,
#   so the np.isclose is not always true (especially when epochs goes large, with propogated error).

from scipy import linalg as scipylinalg
from scipy import sparse
import numpy as np

def orthogonalize(W):
    M = np.dot(W, W.T)
    ret = np.real(np.linalg.inv(scipylinalg.sqrtm(M))).dot(W)
    return ret

def GETSPARSETEST(aa, bb, cc, dd):
    return sparse.csr_matrix((aa, (bb, cc)), shape=(dd, 1))


def scorecond(data, q=0, bdwidth=None):
    '''
    % Syntaxe [psi, entropy] = scorecond(data, bdwidth, cova)
    %
    % Estimate the conditional score function defined as minus the
    % gradient of the conditional density of of a random variable X_p
    % given x_{p-1}, \dots, x_{q+1}. The values of these random variables are
    % provided in the n x p array data.
    %
    % The estimator is based on the partial derivatives of the
    % conditional entropy with respect to the data values, the entropy
    % being estimated through the kernel density estimate _after a
    % prewhitening operation_, with the kernel being the density of the
    % sum of 3 independent uniform random variables in [.5,.5]. The kernel
    % bandwidth is set to bdwidth*(standard deviation) and the density is
    % evaluated at bandwidth apart. bdwidth defaults to
    %   2*(11*sqrt(pi)/20)^((p-q)/(p-q+4))*(4/(3*n))^(1/(p-q+4)
    % (n = sample size), which is optimal _for estimating a normal density_
    %
    % If cova (a p x p matrix) is present, it is used as the covariance
    % matrix of the data and the mean is assume 0. This prevents
    % recomputation of the mean and covariance if the data is centered
    % and/or prewhitenned.
    %
    % The score function is computed at the data points and returned in
    % psi.
    @param data: in shape (n, p), note that n is sample size, and p is #vars (different from `X' later)
    @param q:
    @param bdwidth:
    @return: psi, entropy
    '''
    n, p = data.shape
    if p < q + 1: raise ValueError('Sorry: not enough variables')

    data = data - data.mean(axis=0)
    cova = data.T @ data / n

    # prewhitening
    T = np.linalg.cholesky(cova).T # note that numpy cholesky returns lower triangular
    data = data @ np.linalg.inv(T)

    if q > 0:
        data = data[:, q:] # delete first q columns
        p = p - q

    if bdwidth is None:
        bdwidth = 2 * (11 * np.sqrt(np.pi) / 20) ** (p / (p + 4)) * (4 / (3 * n)) ** (1 / (p + 4))

    # # Grouping the data into cells, idx gives the index of the cell
    # # % containing a datum, r gives its relative distance to the leftmost
    # # % border of the cell
    r = data / bdwidth
    idx = np.floor(r).astype(int)
    r = r - idx
    idx = idx - idx.min(axis=0) + 1 # 0 <= idx-1

    # % Compute the probabilities at grid cells
    # % The kernel function is
    # %        1/2 + (1/2+u)(1/2-u) |u| <= 1/2
    # % k(u) = (3/2 - |u|)^2/2 1/2 <= |u| <= 3/2
    # %        0 otherwise
    # %
    # % The contribution to the probability at i-1, i, i+1 by a
    # % data point at distance r (0 <= r < 1) to i, are respectively:
    # % (1-r)^2/2, 1/2 + r*(1-r), r^2/2
    # % The derivative of the contribution to the probability at i-1, i, i+1
    # % by a data point at distance r to i are respectively: r-1, 1-2r, r
    #
    # % The array ker contains the contributions to the probability of cells
    # % The array kerp contains the gradient of these contributions
    # % The array ix contains the indexes of the cells, arranged in
    # % _lexicographic order_

    ker = np.vstack([
        (1 - r[:, 0]) ** 2 / 2,
        0.5 + r[:, 0] * (1 - r[:, 0]),
        r[:, 0] ** 2 / 2
    ]).T
    ix = np.vstack([idx[:, 0], idx[:, 0] + 1, idx[:, 0] + 2]).T
    kerp = np.vstack([1 - r[:, 0], 2 * r[:, 0] - 1, -r[:, 0]]).T
    mx = idx.max(axis=0) + 2
    M = np.cumprod(mx)
    nr = np.arange(n)

    for i in range(1, p):
        ii = np.ones((3 ** i,), dtype=int) * i
        kerp = np.block(
            [[kerp * (1 - r[nr][:, ii]) ** 2 / 2, kerp * (0.5 + r[nr][:, ii] * (1 - r[nr][:, ii])), kerp * r[nr][:, ii] ** 2 / 2],
             [ker * (1 - r[:, ii]), ker * (2 * r[:, ii] - 1), -ker * r[:, ii]]])

        nr = np.hstack([nr, np.arange(n)])
        ker = np.block([ker * (1 - r[:, ii]) ** 2 / 2, ker * (0.5 + r[:, ii] * (1 - r[:, ii])), ker * r[:, ii] ** 2 / 2])

        Mi = M[i - 1]
        ix = np.block([ix + Mi * (idx[:, ii] - 1), ix + Mi * idx[:, ii], ix + Mi * (idx[:, ii] + 1)])

    # now ix, ker are in shape (n, 3**p)
    # pr is joint prob. of cells

    pr = sparse.csr_matrix((ker.flatten() / float(n), (ix.flatten() - 1, np.zeros((ix.size,), dtype=int))), shape=(M[-1], 1))

    # note that the sparse indexing value: when there's repeated index, value is summed over, not refreshed
    # logp = sparse.csr_matrix((M[-1], 1)) # to contain log(cond. prob.)
    logp = np.zeros((M[-1], 1)) # to contain log(cond. prob.)

    # print(pr.toarray()[:, 0])

    pr_nonzero_inds = np.where(pr.toarray() != 0)

    if p > 1:
        pm = pr.reshape((M[-2], mx[-1]), order='F').sum(axis=1)
        pm = pm[:, [0] * mx[-1]].reshape((M[-1], 1), order='F')
        logp[pr_nonzero_inds] = np.log(pr[pr_nonzero_inds] / pm[pr_nonzero_inds])
    else:
        logp[pr_nonzero_inds] = np.log(pr[pr_nonzero_inds])

    # compute the conditional entropy (if asked)
    entropy = np.log(bdwidth * T[-1, -1]) - (pr.T @ logp)[0, 0]
    psi = np.sum(logp[ix[nr] - 1, [0]] * kerp, axis=1)
    psi = psi.reshape((n, p), order='F') / bdwidth

    psi = psi - psi.mean(axis=0) # center psi

    lam = psi.T @ data / float(n) # correction
    lam = np.tril(lam) + np.tril(lam, -1).T
    lam[-1, -1] = lam[-1, -1] - 1

    if q > 0:
        psi = np.block([np.zeros((n, q)), psi - data @ lam]) @ np.linalg.inv(T.T)
    else:
        psi = (psi - data @ lam) @ np.linalg.inv(T.T)

    return psi, entropy

def estim_beta_pham(x):
    '''
    @param x: data rows (k, T), usually in shape (1, T)
        then psi returned by scorecond(x.T) is in shape (T, k)
        row beta[0] = -scorecond(x.T)[:, 0], only take the first column of psi
        row beta[1] = -scorecond(flipud(x).T)[:, 0], only take the first column of psi
        and when k > 2, the rest rows of beta are just 0.
            when k = 1 (only one row of data), the two rows beta[0] and beta[1] are identical
    @return: beta in shape (max(2, k), T)
    '''
    t1, t2 = x.shape
    if t1 > t2:
        raise ValueError('error in eaastim_beta_pham(x): data must be organized in x in a row fashion')
    beta = np.zeros((max(2, t1), t2))
    psi, _ = scorecond(x.T)
    beta[0] = -1. * psi[:, 0]
    psi, _ = scorecond(np.flipud(x).T)
    beta[1] = -1. * psi[:, 0]
    return beta

def adaptive_size(grad_new, grad_old, eta_old, z_old):
    alpha = 0 # 0.7
    up = 1.05 # 1.1 1.05
    down = 0.8 # 0.4 0.5 0.34 0.5
    z = grad_new + alpha * z_old
    etaup = (grad_new * grad_old) >= 0
    eta = eta_old * (up * etaup + down * (1 - etaup))
    eta[eta >= 0.03] = 0.03 # min(eta, 0.03)
    return eta, z

def natural_grad_Adasize_Mask_regu(X, Mask, regu):
    N, T = X.shape
    mu = 3e-3
    itmax = 18000 # 18000
    Tol = 1e-4
    num_edges = Mask.sum()
    # [icasig, AA, W] = fastica(x, 'approach', 'symm', 'g', 'tanh');

    # initilization of W
    WW = np.eye(N, N)
    for i in range(N):
        Ind_i = np.where(Mask[i] != 0)[0]
        X_Ind_i = X[Ind_i]
        WW[i, Ind_i] = -0.5 * (X[i] @ X_Ind_i.T) @ np.linalg.pinv(X_Ind_i @ X_Ind_i.T)
    W = 0.5 * (WW + WW.T)
    W = W + np.diag(1 - np.diag(W))

    z = np.zeros((N, N))
    eta = mu * np.ones_like(W)
    y_psi = np.zeros_like(X)
    y_psi0 = np.zeros_like(X)
    Grad_W_o = None

    init_avg_gradient_curve = []
    init_loss_curve = []
    for iter in range(itmax):
        # if iter % 100 == 0: print(f'natural_grad_Adasize_Mask_regu: ======== {iter}/{itmax} ========')
        y = W @ X

        # update W: linear ICA with marginal score function estimated from data...
        if iter % 12 == 0:
            for i in range(N):
                tem = estim_beta_pham(y[[i]])
                y_psi[i] = np.copy(tem[0])
                y_psi0[i] = np.copy(y_psi[i, np.argsort(y[i])])  # y0 is not used
        else:
            for i in range(N):
                y_psi[i, np.argsort(y[i])] = np.copy(y_psi0[i])

        # with regularization to make W small
        Grad_W_n = y_psi @ X.T / float(T) + np.linalg.inv(W.T) - 2 * regu * W
        if iter == 0: Grad_W_o = np.copy(Grad_W_n)
        eta, z = adaptive_size(Grad_W_n, Grad_W_o, eta, z)
        delta_W = eta * z
        W = W + delta_W * Mask

        avg_gradient = np.abs(Grad_W_n * Mask).sum() / num_edges
        init_avg_gradient_curve.append(avg_gradient)
        if avg_gradient < Tol: break

        Grad_W_o = np.copy(Grad_W_n)

    return W, np.array(init_avg_gradient_curve), np.array(init_loss_curve)

def sparseica_W_adasize_Alasso_mask_regu(lamda, Mask, X, regu):
    ''' ICA with SCAD penalized entries of the de-mixing matrix
    @param lamda: float, usually lamda is set to a constant times log(T), where T is sample size
    @param Mask: N*N 0 1 matrix, only updates the 1 entries on gradient
        in 2-step CD, if no mask, it's set to ones(N,N) - eye(N)
    @param X: data matrix in shape N*T, where N is the number of nodes, T is sample size. don't need to be whitened
    @param regu: float, e.g., 0.00, 0.002, 0.01, 0.05
    @return:
    '''
    N, T = X.shape
    XX = X - X.mean(axis=1)[:, None]
    # To avoid instability
    std_XX = XX.std(axis=1, ddof=1) # note that the std function in matlab is sample stddev, so ddof=1 here
    XX = np.diag(1. / std_XX) @ XX # it should be @XX here, not @X, in case X is not zero-meaned. (bug in matlab code)
    Refine = True
    num_edges = Mask.sum()

    # learning rate
    mu = 1e-6 # 1e-6
    beta = 0 # 1
    m = 60 # for approximate the derivative of |.|
    itmax = 15000 # 15000 # 10000
    Tol = 1e-6

    # initiliazation
    # print('Initialization....')
    WW, init_avg_gradient_curve, init_loss_curve = natural_grad_Adasize_Mask_regu(XX, Mask, regu)

    omega1 = 1. / np.abs(WW[Mask != 0])
    # to avoid instability
    Upper = 3 * omega1.mean()
    omega1[omega1 > Upper] = Upper
    omega = np.zeros((N, N))
    omega[Mask != 0] = omega1
    W = np.copy(WW)

    z = np.zeros((N, N))
    eta = mu * np.ones_like(W)
    W_old = W + np.eye(N)
    grad_new = np.copy(W_old)
    y_psi = np.zeros_like(XX)
    y_psi0 = np.zeros_like(XX)
    grad_old = None
    y = None

    penal_avg_gradient_curve = []
    penal_loss_curve = []

    for iter in range(itmax):
        # if iter % 100 == 0: print(f'sparseica_W_adasize_Alasso_mask_regu: ======== {iter}/{itmax} ========')
        y = W @ XX
        avg_gradient = np.abs(grad_new * Mask).sum() / num_edges
        penal_avg_gradient_curve.append(avg_gradient)
        if avg_gradient < Tol:
            if Refine:
                Mask = np.abs(W) > 0.01
                Mask[np.diag_indices(N)] = 0
                lamda = 0.
                Refine = False
            else:
                break

        # update W: linear ICA with marginal score function estimated from data...
        if iter % 8 == 0:
            for i in range(N):
                tem = estim_beta_pham(y[[i]])
                y_psi[i] = np.copy(tem[0])
                y_psi0[i] = np.copy(y_psi[i, np.argsort(y[i])]) # y0 is not used
        else:
            for i in range(N):
                y_psi[i, np.argsort(y[i])] = np.copy(y_psi0[i])

        dev = omega * np.tanh(m * W)

        # with additional regularization
        regu_l1 = regu / 2.
        grad_new = y_psi @ XX.T / T + np.linalg.inv(W.T) - \
                     4 * beta * (np.diag(np.diag(y @ y.T / T)) - np.eye(N)) * (y @ XX.T / T) - \
                        dev * lamda / T - \
                            2 * regu_l1 * W # seems that it should be 0-mean-1-std XX here. in original code, it's X here?
        if iter == 0: grad_old = np.copy(grad_new)

        # adaptive size
        eta, z = adaptive_size(grad_new, grad_old, eta, z)
        delta_W = eta * z
        W = W + 0.9 * delta_W * Mask
        grad_old = np.copy(grad_new)

    # re-scaling
    W = np.diag(std_XX) @ W @ np.diag(1. / std_XX)
    WW = np.diag(std_XX) @ WW @ np.diag(1. / std_XX)
    y = np.diag(std_XX) @ y
    Score = omega * np.abs(W)
    return y, W, WW, Score, \
           init_avg_gradient_curve, init_loss_curve, \
           np.array(penal_avg_gradient_curve), np.array(penal_loss_curve)

