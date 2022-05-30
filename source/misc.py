import numpy as np

def vrow(v):
    return v.reshape((1, v.size))


def empirical_mean(X: np.array):
    return mcol(X.mean(1))


def cov(D):
    # calculate and reshape the mean
    mu = empirical_mean(D)
    # center the data
    DC = D - mu
    # calculate covariance matrix
    C = (DC @ DC.T)/DC.shape[1]
    return C

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

def mcol(v):
    return v.reshape((v.size, 1))