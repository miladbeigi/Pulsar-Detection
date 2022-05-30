import numpy as np
import misc
from pca_lda import calculate_pca, normalize_data

from scipy import special
from load import load_data

def logpdf_GAU_ND(X, mu, C):
    Y = [logpdf_one_sample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return np.array(Y).ravel()

def logpdf_GAU_ND_simplified(X, mu, C):
    P = np.linalg.inv(C)
    return -0.5 * X.shape[0] * np.log(np.pi * 2) + 0.5 * np.linalg.slogdet(P)[1] - 0.5 * (np.dot(P, (X - mu)) * (X - mu)).sum(0)

def logpdf_one_sample(x, mu, C):
    P = np.linalg.inv(C)
    res = -0.5 * x.shape[0] * np.log(2*np.pi)
    res += -0.5 * np.linalg.slogdet(C)[1]
    res += -0.5 * np.dot((x - mu).T , np.dot( P,  (x - mu)))
    return res.ravel()
 
def ML_GAU(D):
    mu = misc.empirical_mean(D)
    C = misc.cov(D)
    return mu, C

def mvg_model():
    D, L = load_data()
    # Using Normalization and PCA before 
    # D = normalize_data(D)
    # D = calculate_pca(D)

    (DTR, LTR), (DTE, LTE) = misc.split_db_2to1(D, L, 0)
    h = {}
    
    SJoint = np.zeros((2, DTE.shape[1]))
    LogSJoint = np.zeros((2, DTE.shape[1]))
    ClassPriors = [1.0/2.0, 1.0/2.0]
    
    for label in [0, 1]:
        # Compute model parameters for each class
        h[label] = ML_GAU(DTR[:, LTR==label])
    
    for label in [0, 1]:
        mu, C = h[label]
        SJoint[label, :] = np.exp(logpdf_GAU_ND_simplified(DTE, mu, C).ravel() * ClassPriors[label])
        LogSJoint[label, :] = logpdf_GAU_ND_simplified(DTE, mu, C).ravel() + np.log(ClassPriors[label])
    
    SMarginal = SJoint.sum(0)
    LogSMarginal = special.logsumexp(LogSJoint, axis=0)
    Post1 = SJoint / SMarginal
    Post2 = np.exp(LogSJoint - misc.vrow(LogSMarginal))
    
    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)

    print((LTE==LPred1).sum()/LTE.shape[0])
    print((LTE==LPred2).sum()/LTE.shape[0])


if __name__ == "__main__":
    mvg_model()