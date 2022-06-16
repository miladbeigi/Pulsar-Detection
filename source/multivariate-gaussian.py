from distutils.log import Log
from doctest import testfile
from re import S
import numpy as np
import misc
from pca_lda import calculate_pca, normalize_data
import pylab
from gaussianize import gaussianization
from scipy import special
from load import load_data
from model_evaluation import compute_min_DCF, compute_act_DCF, bayes_error_plot

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
    
    # Using gaussianized data
    # D = gaussianization(D.T)
    # D = D.T
    
    # Using Normalization and PCA before 
    # D = normalize_data(D)
    # D = calculate_pca(D)
    
    # Split into two 
    # (DTR, LTR), (DTE, LTE) = misc.split_db_2to1(D, L, 0)

    # K-Fold
    K = 5
    kf = misc.k_fold(K)
    min_minDCF = []
    avg_actDCF = 0

    for train_index , test_index in kf.split(D.T):

        DTR = D[:, train_index]
        LTR = L[train_index]
        
        DTE = D[:, test_index]
        LTE = L[test_index]
        
        llr = train_mvg(DTR, LTR, DTE, LTE)
        
        min_minDCF.append(compute_min_DCF(llr, LTE, 0.5, 1, 1))        
        avg_actDCF += compute_act_DCF(llr, LTE, 0.5, 1, 1) / K
    
    print(min(min_minDCF))
    print(avg_actDCF)

def train_mvg(DTR, LTR, DTE, LTE):
    h = {}
    
    SJoint = np.zeros((2, DTE.shape[1]))
    LogSJoint = np.zeros((2, DTE.shape[1]))
    ll = np.zeros((2, DTE.shape[1]))
    ClassPriors = [1.0/2.0, 1.0/2.0]
    
    # Compute model parameters for each class
    for label in [0, 1]:
        h[label] = ML_GAU(DTR[:, LTR==label])
    
    # Compute joint probabilities
    for label in [0, 1]:
        mu, C = h[label]
        ll[label, :] = logpdf_GAU_ND_simplified(DTE, mu, C).ravel()
        SJoint[label, :] = np.exp(ll[label, :] * ClassPriors[label])
        LogSJoint[label, :] = ll[label, :] + np.log(ClassPriors[label])
    
    SMarginal = SJoint.sum(0)
    Post1 = SJoint / SMarginal
    
    # LogSMarginal = special.logsumexp(LogSJoint, axis=0)
    # Post2 = np.exp(LogSJoint - misc.vrow(LogSMarginal))
    # LPred2 = Post2.argmax(0)
    # print((LTE==LPred2).sum()/LTE.shape[0])
    
    LPred1 = Post1.argmax(0)
    
    # Compute log-likelihood ratio
    llr = ll[1, :] - ll[0, :]

    # print("Accuracy: ", (LTE==LPred1).sum()/LTE.shape[0])
    
    return llr

if __name__ == "__main__":
    mvg_model()

    # P = np.linspace(-3, 3, 21)
    # pylab.plot(P, bayes_error_plot(P, llr, LTE, minCost=False), color='r')
    # pylab.plot(P, bayes_error_plot(P, llr, LTE, minCost=True), color='b')
    # pylab.ylim(0, 1,1)
    # pylab.show()