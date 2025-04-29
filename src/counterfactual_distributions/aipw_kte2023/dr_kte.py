from __future__ import division
import numpy as np
from sys import stdout
from sklearn.metrics import pairwise_kernels
import scipy.stats as st
from sklearn.metrics import pairwise_distances

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer

# Code for the DR-xMMD test proposed in Fawkes et al.

def MMD2dru(XY, w, Xcov, T, idx, idx2, kernel_function, **kwargs):
    
    """The DR-xMMD^2 statistic.
    """
    
    N = len(XY)
    N2 = N // 2
    
    Yi = XY[:]
    wi = w.squeeze()[:]
    Xi = Xcov[:, :]
    Ti = T[idx]
    
    Ya = Yi[:N2]
    Yb = Yi[N2:]
    Ya0 = Ya[Ti[:N2]==0]
    Ya1 = Ya[Ti[:N2]==1]
    Yb0 = Yb[Ti[N2:]==0]
    Yb1 = Yb[Ti[N2:]==1]
    Y = np.vstack((Ya0, Ya1, Yb0, Yb1))
    m1 = len(Ya0)
    m = m1 + len(Yb0)
    n1 = len(Ya1)
    n = n1 + len(Yb1)
    Yia = Y[:m1+n1]
    Yib = Y[m1+n1:]
    
    Xa = Xi[:N2, :]
    Xb = Xi[N2:, :]
    Xa0 = Xa[Ti[:N2]==0, :]
    Xa1 = Xa[Ti[:N2]==1, :]
    Xb0 = Xb[Ti[N2:]==0, :]
    Xb1 = Xb[Ti[N2:]==1, :]
    X = np.vstack((Xa0, Xa1, Xb0, Xb1))
       
    wa = wi[:N2]
    wb = wi[N2:]
    wa0 = -1.0/np.array(1 - wa[Ti[:N2]==0])
    wa1 = 1.0/np.array(wa[Ti[:N2]==1])
    wb0 = -1.0/np.array(1 - wb[Ti[N2:]==0])
    wb1 = 1.0/np.array(wb[Ti[N2:]==1])
    w = np.concatenate((wa0, wa1, wb0, wb1))
    
    
    Yj = XY[:]
    wj = w.squeeze()[:]
    Xj = Xcov[:, :]
    Tj = T[idx2]
    
    Yja = Yj[:N2]
    Yjb = Yj[N2:]
    Yja0 = Yja[Tj[:N2]==0]
    Yja1 = Yja[Tj[:N2]==1]
    Yjb0 = Yjb[Tj[N2:]==0]
    Yjb1 = Yjb[Tj[N2:]==1]
    YY = np.vstack((Yja0, Yja1, Yjb0, Yjb1))
    mj1 = len(Yja0)
    mj = mj1 + len(Yjb0)
    nj1 = len(Yja1)
    nj = nj1 + len(Yjb1)
    YYa = YY[:mj1+nj1]
    YYb = YY[mj1+nj1:]
    
    Xja = Xj[:N2, :]
    Xjb = Xj[N2:, :]
    Xja0 = Xja[Tj[:N2]==0, :]
    Xja1 = Xja[Tj[:N2]==1, :]
    Xjb0 = Xjb[Tj[N2:]==0, :]
    Xjb1 = Xjb[Tj[N2:]==1, :]
    XX = np.vstack((Xja0, Xja1, Xjb0, Xjb1))
    
    wja = wj[:N2]
    wjb = wj[N2:]
    wja0 = -1.0/np.array(1 - wja[Tj[:N2]==0])
    wja1 = 1.0/np.array(wja[Tj[:N2]==1])
    wjb0 = -1.0/np.array(1 - wjb[Tj[N2:]==0])
    wjb1 = 1.0/np.array(wjb[Tj[N2:]==1])
    ww = np.concatenate((wja0, wja1, wjb0, wjb1))
    wwa = ww[:mj1+nj1]
    wwb = ww[mj1+nj1:]
    
    
    sigmaKX = np.median(pairwise_distances(Xb, Xb, metric='euclidean'))**2
    KX = pairwise_kernels(X, metric='rbf', gamma=1.0/sigmaKX)
    KXx = pairwise_kernels(X, XX, metric='rbf', gamma=1.0/sigmaKX)
    gamma = sigmaKX
    
    mu0a = np.linalg.solve(KX[:m1, :m1] + gamma * np.eye(m1), KXx[:m1, mj1+nj1:])
    zeroed_mu0a = np.vstack(( mu0a, np.zeros((n1, mj+nj-mj1-nj1)) ))
    mu1a = np.linalg.solve(KX[m1:m1+n1, m1:m1+n1] + gamma * np.eye(n1), KXx[m1:m1+n1, mj1+nj1:])
    zeroed_mu1a = np.vstack(( np.zeros((m1, mj+nj-mj1-nj1)), mu1a ))    
    muAa = np.hstack(( zeroed_mu0a[:, :mj-mj1], zeroed_mu1a[:, mj-mj1:] ))
    
    mu0b = np.linalg.solve(KX[m1+n1:m+n1, m1+n1:m+n1] + gamma * np.eye(m-m1), KXx[m1+n1:m+n1, :mj1+nj1])
    zeroed_mu0b = np.vstack((mu0b, np.zeros((n-n1,mj1+nj1))))
    mu1b = np.linalg.solve(KX[m+n1:, m+n1:] + gamma * np.eye(n-n1), KXx[m+n1:, :mj1+nj1])
    zeroed_mu1b = np.vstack((np.zeros((m-m1,mj1+nj1)), mu1b))
    muAb = np.hstack((zeroed_mu0b[:, :mj1], zeroed_mu1b[:, mj1:]))
    
    #DR
    part1 = np.vstack(( zeroed_mu1a - zeroed_mu0a - wwb*muAa, wwb*np.eye(mj+nj-(mj1+nj1)) ))
    part2 = np.vstack(( zeroed_mu1b - zeroed_mu0b - wwa*muAb, wwa*np.eye(mj1+nj1) ))
    
    
    sigmaKY = np.median(pairwise_distances(Ya, Ya, metric='euclidean'))**2
    KY1 = pairwise_kernels(np.vstack(( Yia, YYb )), metric=kernel_function, **kwargs)
    KY2 = pairwise_kernels(np.vstack(( Yib, YYa )), metric=kernel_function, **kwargs)
    prod1 = part1.T @ KY1 @ part1
    prod2 = part2.T @ KY2 @ part2
    
    return (prod1.mean() + prod2.mean()) / 2


def permute(n_bins, og_indices, clusters):
    """Permutation function.
    """
    permutation = og_indices.copy()
    for i in range(n_bins):
        mask = i==clusters
        group = og_indices[mask]
        permuted_group=np.random.permutation(group)
        permutation[mask]=permuted_group
    return permutation


def compute_null_distribution(Y, Xcov, w, T, experiment, iterations=10000, verbose=False,
                              random_state=None, marker_interval=1000, kernel_function='rbf',
                             **kwargs):
    """Compute DR-MMD^2 null distribution based on conditional permutation based approach.
    """
    
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    N2 = Xcov.shape[0]//2
    og_indices=np.arange(N2)
    if experiment:
        idx10 = np.random.permutation(og_indices)
        idx11 = np.random.permutation(og_indices) + N2
        idx1 = np.hstack((idx10, idx11))
    else:
        
        n_bins=N2//20
        binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        clusters1 = binner.fit_transform(w[:N2][:,np.newaxis]).squeeze()
        clusters2 = binner.fit_transform(w[N2:][:,np.newaxis]).squeeze()
        idx10 = permute(n_bins, og_indices, clusters1)
        idx11 = permute(n_bins, og_indices, clusters2) + N2
        idx1 = np.hstack((idx10, idx11))
    

    if experiment:
        w = np.zeros(len(T)) + 0.5         
    else:
        N = len(T)
        N2 = N // 2
        w1 = LogisticRegression(C=1e6, max_iter=1000).fit(Xcov[N2:,:], T[N2:]).predict_proba(Xcov[:N2,:])[:, 1]
        w2 = LogisticRegression(C=1e6, max_iter=1000).fit(Xcov[:N2,:], T[:N2]).predict_proba(Xcov[N2:,:])[:, 1]
        w = np.hstack((w1, w2))

    mmd2u_null = np.zeros(iterations)
    mmd2u_null[-1] = MMD2dru(Y, w, Xcov, T, idx1, idx1, kernel_function, **kwargs)
    for i in range(iterations-1):
        if verbose and (i % marker_interval) == 0:
            print(i),
            stdout.flush()
        
        if experiment: 
            idx20 = np.random.permutation(og_indices)
            idx21 = np.random.permutation(og_indices) + N2
            idx2 = np.hstack((idx20, idx21))
        else:
            idx20 = permute(n_bins, og_indices, clusters1)
            idx21 = permute(n_bins, og_indices, clusters2) + N2
            idx2 = np.hstack((idx20, idx21))
        
        # Interestingly, the test is well calibrated when we input idx2 two times i.e. the conditional mean embeddings
        # are estimated in each iteration. It makes me think that the proposped test is not well calibrated.  
        mmd2u_null[i] = MMD2dru(Y, w, Xcov, T, idx1, idx2, kernel_function, **kwargs)

    if verbose:
        print("")

    return mmd2u_null



def kernel_dr_nonuniform(Y, Xcov, T, w, experiment,
                         iterations=1000,
                        verbose=False, random_state=None,
                         kernel_function='rbf',
                         **kwargs):
    """Compute DR-MMD^2, its null distribution and the p-value of the
    kernel two-sample test.
    """
    
    
    og_indices=np.arange(Xcov.shape[0])
    mmd2dru = MMD2dru(Y, w, Xcov, T, og_indices, og_indices, kernel_function, **kwargs)
    
    mmd2dru_null = compute_null_distribution(Y, Xcov, w, T, experiment, kernel_function=kernel_function, iterations=iterations, verbose=verbose, random_state=random_state, **kwargs)
    p_value = np.mean(mmd2dru_null > mmd2dru)
    
    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0/iterations))

    return mmd2dru, mmd2dru_null, p_value
