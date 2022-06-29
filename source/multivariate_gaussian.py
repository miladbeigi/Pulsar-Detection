from cProfile import label
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


def logpdf_GAU_ND_simplified(X, mu, C):
    P = np.linalg.inv(C)
    return -0.5 * X.shape[0] * np.log(np.pi * 2) + 0.5 * np.linalg.slogdet(P)[1] - 0.5 * (np.dot(P, (X - mu)) * (X - mu)).sum(0)


def ML_GAU(D):
    mu = misc.empirical_mean(D)
    C = misc.cov(D)
    return mu, C


def ML_GAU_NAIVE(D):
    mu = misc.empirical_mean(D)
    C = misc.cov(D, type="naive")
    return mu, C


def ML_GAU_TIED(D, L, class_label):
    mu = misc.empirical_mean(D[:, L == class_label])
    C = misc.within_class_cov_matrix(D, L)
    return mu, C


def ML_GAU_TIED_NAIVE(D, L, class_label):
    mu = misc.empirical_mean(D[:, L == class_label])
    C = misc.within_class_cov_matrix(D, L)
    return mu, np.diag(np.diag(C))


def compute_densities(model_parameters, DTE):
    SJoint = np.zeros((2, DTE.shape[1]))
    LogSJoint = np.zeros((2, DTE.shape[1]))
    ll = np.zeros((2, DTE.shape[1]))
    ClassPriors = [1.0/2.0, 1.0/2.0]

    for label in [0, 1]:
        mu, C = model_parameters[label]
        ll[label, :] = logpdf_GAU_ND_simplified(DTE, mu, C).ravel()
        SJoint[label, :] = np.exp(ll[label, :] * ClassPriors[label])
        LogSJoint[label, :] = ll[label, :] + np.log(ClassPriors[label])

    return ll, SJoint


def mvg_models(D, L):

    # Split into two
    # (DTR, LTR), (DTE, LTE) = misc.split_db_2to1(D, L, 0)

    # K-Fold
    K = 5
    kf = misc.k_fold(K)

    llr_mvg = []
    llr_mvg_naive = []
    llr_mvg_tied = []
    llr_mvg_tied_naive = []

    labels = []

    for train_index, test_index in kf.split(D.T):
        DTR = D[:, train_index]
        LTR = L[train_index]
        DTE = D[:, test_index]
        LTE = L[test_index]
        
        llr_mvg.append(train_mvg_models(DTR, LTR, DTE, LTE, "Full"))
        llr_mvg_naive.append(train_mvg_models(DTR, LTR, DTE, LTE, "Naive"))
        llr_mvg_tied.append(train_mvg_models(DTR, LTR, DTE, LTE, "Tied"))
        llr_mvg_tied_naive.append(train_mvg_models(DTR, LTR, DTE, LTE, "Tied-Naive"))
        
        labels.append(LTE)

    for app in [0.5, 0.1, 0.9]:
        print(f"{app} Full :", compute_min_DCF(np.hstack(llr_mvg), np.hstack(labels), app, 1, 1))
        print(f"{app} Naive :", compute_min_DCF(np.hstack(llr_mvg_naive), np.hstack(labels), app, 1, 1))
        print(f"{app} Tied :", compute_min_DCF(np.hstack(llr_mvg_tied), np.hstack(labels), app, 1, 1))
        print(f"{app} Tied-Naive :", compute_min_DCF(np.hstack(llr_mvg_tied_naive), np.hstack(labels), app, 1, 1))


def train_mvg_models(DTR, LTR, DTE, LTE, model_type):
    model_parameters = {}

    # Compute model parameters for each class
    for label in [0, 1]:
        if model_type == "Full":
            model_parameters[label] = ML_GAU(DTR[:, LTR == label])
        elif model_type == "Naive":
            model_parameters[label] = ML_GAU_NAIVE(DTR[:, LTR == label])
        elif model_type == "Tied":
            model_parameters[label] = ML_GAU_TIED(DTR, LTR, label)
        elif model_type == "Tied-Naive":
            model_parameters[label] = ML_GAU_TIED_NAIVE(DTR, LTR, label)
        else:
            model_parameters[label] = ML_GAU(DTR[:, LTR == label])

    ll, SJoint = compute_densities(model_parameters, DTE)

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
    D, L = load_data()

    # Using gaussianized data
    D = gaussianization(D.T)
    D = D.T

    # Using Normalization and PCA before
    # D = normalize_data(D)
    # D = calculate_pca(D, 6)

    mvg_models(D, L)

    # P = np.linspace(-3, 3, 21)
    # pylab.plot(P, bayes_error_plot(P, llr, LTE, minCost=False), color='r')
    # pylab.plot(P, bayes_error_plot(P, llr, LTE, minCost=True), color='b')
    # pylab.ylim(0, 1,1)
    # pylab.show()