from cProfile import label
import string
import numpy as np
from sklearn.model_selection import KFold


def vrow(v):
    return v.reshape((1, v.size))

def mcol(v):
    return v.reshape((v.size, 1))

def empirical_mean(X: np.array):
    return mcol(X.mean(1))


def cov(D, L=None, type: str=None):
    """
    Parameters
    ----------

    D: numpy array
    
    type: optional
          "None" full covariance matrix is computed
          "naive" diagonal of covariance matrix
          "tied" 
    """
    # calculate and reshape the mean
    mu = empirical_mean(D)
    # center the data
    DC = D - mu
    # calculate covariance matrix
    C = (DC @ DC.T)/DC.shape[1]
    if type == None:
        return C
    elif type == "naive":
        return np.diag(np.diag(C))

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def within_class_cov_matrix(D, L):
    SW = 0
    for i in [0, 1]:
        SW += (L == i).sum() * cov(D[:, L == i])
    return SW/D.shape[1]

def k_fold(K):
    return KFold(n_splits=K, random_state=1, shuffle=True)