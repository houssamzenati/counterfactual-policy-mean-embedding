from __future__ import division
import numpy as np
from sys import stdout
from sklearn.metrics import pairwise_kernels
import scipy.stats as st
from sklearn.metrics import pairwise_distances


def xMMD2(XY, T, w, kernel_function, **kwargs):
    """The IPW-xKTE^2 statistic.
    """
    
    N = len(XY)
    N2 = N//2
    
    Ta = T[:N2]
    Tb = T[N2:]
    
    Ya = XY[:N2]
    Yb = XY[N2:]
    Y0a = Ya[T[:N2]==0]
    Y1a = Ya[T[:N2]==1]
    Y0b = Yb[T[N2:]==0]
    Y1b = Yb[T[N2:]==1]
    Y = np.vstack((Y0a, Y1a, Y0b, Y1b))
       
    wa = w.squeeze()[:N2]
    wb = w.squeeze()[N2:]
    w0a = 1.0/np.array(1 - wa[Ta==0])
    w0b = 1.0/np.array(1 - wb[Tb==0])
    w1a = 1.0/np.array(wa[Ta==1])
    w1b = 1.0/np.array(wb[Tb==1])
    ww = np.concatenate((-w0a, w1a, -w0b, w1b))

    
    # IPW
    left_side = np.diag(ww[:N2])
    right_side = np.diag(ww[N2:])
    
    KY = pairwise_kernels(Y[:N2], Y[N2:], metric=kernel_function, **kwargs)
    prod = left_side.T @ KY @ right_side
    
    U = prod.mean(1)
    return np.sqrt(len(U)) * U.mean() / U.std()


def xMMD2dr(XY, w, Xcov, T, kernel_function, **kwargs):
    
    """The DR-xKTE^2 statistic.
    """
    
    
    N = len(XY)
    N2 = N//2
    
    Ta = T[:N2]
    Tb = T[N2:]
    
    Ya = XY[:N2]
    Yb = XY[N2:]
    Y0a = Ya[T[:N2]==0]
    Y1a = Ya[T[:N2]==1]
    Y0b = Yb[T[N2:]==0]
    Y1b = Yb[T[N2:]==1]
    Y = np.vstack((Y0a, Y1a, Y0b, Y1b))
    m1 = len(Y0a)
    m = m1 + len(Y0b)
    n1 = len(Y1a)
    n = n1 + len(Y1b)
    
    Xa = Xcov[:N2,:]
    Xb = Xcov[N2:,:]
    X0a = Xa[Ta==0, :]
    X0b = Xb[Tb==0, :]
    X1a = Xa[Ta==1, :]
    X1b = Xb[Tb==1, :]
    X = np.vstack((X0a, X1a, X0b, X1b))
    
       
    wa = w.squeeze()[:N2]
    wb = w.squeeze()[N2:]
    w0a = 1.0/np.array(1 - wa[Ta==0])
    w0b = 1.0/np.array(1 - wb[Tb==0])
    w1a = 1.0/np.array(wa[Ta==1])
    w1b = 1.0/np.array(wb[Tb==1])
    ww = np.concatenate((-w0a, w1a, -w0b, w1b))
    
    
    sigmaKX = np.median(pairwise_distances(X[N2:, :], X[N2:, :], metric='euclidean'))**2
    KX = pairwise_kernels(X, metric='rbf', gamma=1.0/sigmaKX) 
    gamma = sigmaKX
    
    mu0a = np.linalg.solve(KX[:m1, :m1] + gamma * np.eye(m1), KX[:m1, :m1+n1])
    zeroed_mu0a = np.vstack((mu0a, np.zeros((n1, m1+n1))))
    mu1a = np.linalg.solve(KX[m1:m1+n1, m1:m1+n1] + gamma * np.eye(n1), KX[m1:m1+n1, :m1+n1])
    zeroed_mu1a = np.vstack(( np.zeros((m1, m1+n1)), mu1a ))    
    muAa = np.hstack((zeroed_mu0a[:, :m1], zeroed_mu1a[:, m1:m1+n1]))
    
    mu0b = np.linalg.solve(KX[m1+n1:m+n1, m1+n1:m+n1] + gamma * np.eye(m-m1), KX[m1+n1:m+n1, m1+n1:])
    zeroed_mu0b = np.vstack((mu0b, np.zeros((n-n1,m+n-m1-n1))))
    mu1b = np.linalg.solve(KX[m+n1:, m+n1:] + gamma * np.eye(n-n1), KX[m+n1:, m1+n1:])
    zeroed_mu1b = np.vstack((np.zeros((m-m1,m+n-m1-n1)), mu1b))
    muAb = np.hstack((zeroed_mu0b[:, :m-m1], zeroed_mu1b[:, m-m1:]))
    
    
    #DR
    left_side = zeroed_mu1a - zeroed_mu0a + ww[:N2]*(np.eye(m1+n1) - muAa)
    right_side = zeroed_mu1b - zeroed_mu0b + ww[N2:]*(np.eye(m+n-m1-n1) - muAb)
    
    
    KY = pairwise_kernels(Y[:N2], Y[N2:], metric=kernel_function, **kwargs)
    prod = left_side.T @ KY @ right_side
    
    U = prod.mean(1)
    return np.sqrt(len(U)) * U.mean() / U.std()




def kernel_two_sample_test_agnostic(Y, T, w, kernel_function='rbf', p=0.5,
                           verbose=False, random_state=None, **kwargs):
    """Compute the statistic IPW-xKTE and its p-value given normal distribution.
    """
    xmmd2 = xMMD2(Y, T, w, kernel_function, **kwargs)
    if verbose:
        print("xMMD^2 = %s" % xmmd2)
    p_value = 1 - st.norm.cdf(xmmd2)
    return xmmd2, p_value


def kernel_dr_two_sample_test_agnostic(Y, Xcov, T, w, kernel_function='rbf', p=0.5,
                           verbose=False, random_state=None, **kwargs): 
    """Compute the statistic AIPW-xKTE and its p-value given normal distribution.
    """ 
    xmmd2dr = xMMD2dr(Y, w, Xcov, T, kernel_function, **kwargs)
    if verbose:
        print("DR xMMD^2 = %s" % xmmd2dr)
    p_value = 1 - st.norm.cdf(xmmd2dr)
    return xmmd2dr, p_value


