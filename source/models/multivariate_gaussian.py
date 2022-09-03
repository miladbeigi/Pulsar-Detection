import numpy as np
import misc.misc as misc
from pre_processing.gaussianize import features_gaussianization
from pre_processing.pca_lda import calculate_pca
from utils.model_evaluation import compute_min_DCF, compute_act_DCF
from models.logistic_regression import train_logistic_regression
import misc.constants as constants


def logpdf_GAU_ND_simplified(X, mu, C):
    P = np.linalg.inv(C)
    return -0.5 * X.shape[0] * np.log(np.pi * 2) + 0.5 * np.linalg.slogdet(P)[1] - 0.5 * (np.dot(P, (X - mu)) * (X - mu)).sum(0)


def ML_GAU(D):
    mu = misc.compute_mean(D)
    C = misc.compute_covariance(D)
    return mu, C


def ML_GAU_NAIVE(D):
    mu = misc.compute_mean(D)
    C = misc.compute_covariance(D, type="naive")
    return mu, C


def ML_GAU_TIED(D, L, class_label):
    mu = misc.compute_mean(D[:, L == class_label])
    C = misc.compute_within_class_covariance(D, L)
    return mu, C


def ML_GAU_TIED_NAIVE(D, L, class_label):
    mu = misc.compute_mean(D[:, L == class_label])
    C = misc.compute_within_class_covariance(D, L)
    return mu, np.diag(np.diag(C))


def compute_densities(model_parameters, DTE):
    SJoint = np.zeros((2, DTE.shape[1]))
    LogSJoint = np.zeros((2, DTE.shape[1]))
    ll = np.zeros((2, DTE.shape[1]))
    ClassPriors = constants.MVG_CLASS_PRIORS

    for label in constants.CLASS_LABELS:
        mu, C = model_parameters[label]
        ll[label, :] = logpdf_GAU_ND_simplified(DTE, mu, C).ravel()
        SJoint[label, :] = np.exp(ll[label, :] * ClassPriors[label])
        LogSJoint[label, :] = ll[label, :] + np.log(ClassPriors[label])

    return ll


def mvg_models(D, L, applications, K, options={"m_pca": None, "gaussianize": False}):

    # Shuffle data
    Random_Data, Random_Labels = misc.shuffle_data(D, L)

    llr_mvg = []
    llr_mvg_naive = []
    llr_mvg_tied = []
    llr_mvg_tied_naive = []

    labels = []

    for train_index, test_index in misc.k_fold(K, D.shape[1]):
        DTR = Random_Data[:, train_index]
        LTR = Random_Labels[train_index]
        DTE = Random_Data[:, test_index]
        LTE = Random_Labels[test_index]

        if options["gaussianize"]:
            DTR, DTE = features_gaussianization(DTR, DTE)

        if options["m_pca"]:
            DTR, P = calculate_pca(DTR, options["m_pca"])
            DTE = np.dot(P.T, DTE)

        llr_mvg.append(train_mvg_models(DTR, LTR, DTE, "Full"))
        llr_mvg_naive.append(train_mvg_models(DTR, LTR, DTE, "Naive"))
        llr_mvg_tied.append(train_mvg_models(DTR, LTR, DTE, "Tied"))
        llr_mvg_tied_naive.append(
            train_mvg_models(DTR, LTR, DTE, "Tied-Naive"))

        labels.append(LTE)

    for app in applications:
        print(f"{app} Full :", compute_min_DCF(
            np.hstack(llr_mvg), np.hstack(labels), app, 1, 1))
        print(f"{app} Naive :", compute_min_DCF(
            np.hstack(llr_mvg_naive), np.hstack(labels), app, 1, 1))
        print(f"{app} Tied :", compute_min_DCF(
            np.hstack(llr_mvg_tied), np.hstack(labels), app, 1, 1))
        print(f"{app} Tied-Naive :", compute_min_DCF(np.hstack(llr_mvg_tied_naive),
              np.hstack(labels), app, 1, 1))


def train_mvg_models(DTR, LTR, DTE, model_type):

    model_parameters = {}

    # Compute model parameters for each class
    for label in constants.CLASS_LABELS:
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

    ll = compute_densities(model_parameters, DTE)

    # Compute log-likelihood ratio
    llr = ll[1, :] - ll[0, :]

    return llr


def mvg_evaluation(Train_D, Train_L, Evaluation_D, Evaluation_L, model_type, applications):
    llr = train_mvg_models(Train_D, Train_L, Evaluation_D, model_type)
    LP = (llr > 0)*1
    C_MVG_S = train_logistic_regression(misc.make_row_shape(
        llr), LP, misc.make_row_shape(llr), 10**-5, 0.5, True)

    for app in applications:
        print(f"{app} {model_type} minDCF:",
              compute_min_DCF(llr, Evaluation_L, app, 1, 1))
        print(f"{app} {model_type} ACT DCF:",
              compute_act_DCF(llr, Evaluation_L, app, 1, 1))
        print(f"{app} {model_type} ACT DCF (Calibrated):",
              compute_act_DCF(C_MVG_S, Evaluation_L, app, 1, 1))
