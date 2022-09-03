import numpy as np
import misc.misc as misc
from pre_processing.pca_lda import calculate_pca
from pre_processing.gaussianize import features_gaussianization
import scipy.optimize
from utils.model_evaluation import compute_min_DCF, compute_act_DCF
import matplotlib.pyplot as plt


def logreg_obj_wrap_imbalanced(DTR, LTR, l, prior):
    M = DTR.shape[0]
    DTR_L1 = DTR[:, LTR == 1]
    DTR_L0 = DTR[:, LTR == 0]

    def logreg_obj(v):
        w = misc.make_column_shape(v[0:M])
        b = v[-1]
        cxe1 = 0
        cxe0 = 0
        S1 = np.dot(w.T, DTR_L1) + b
        S0 = np.dot(w.T, DTR_L0) + b
        cxe1 = np.logaddexp(0,  -S1*1).mean()
        cxe0 = np.logaddexp(0,  -S0*(-1)).mean()
        return (0.5 * l * np.linalg.norm(w)**2) + (cxe1 * prior) + (cxe0 * (1-prior))
    return logreg_obj


def logreg_obj_wrap_balanced(DTR, LTR, l):
    Z = LTR * 2.0 - 1.0
    M = DTR.shape[0]

    def logreg_obj(v):
        w = misc.make_column_shape(v[0:M])
        b = v[-1]
        cxe = 0

        S = np.dot(w.T, DTR) + b
        cxe = np.logaddexp(0,  -S*Z).mean()
        return cxe + 0.5 * l * np.linalg.norm(w)**2
    return logreg_obj


def train_logistic_regression(DTR, LTR, DTE, l, prior, imbalanced: bool):

    if imbalanced:
        logreg_obj = logreg_obj_wrap_imbalanced(DTR, LTR, l, prior)
    else:
        logreg_obj = logreg_obj_wrap_balanced(DTR, LTR, l)

    _v,  _J, _d = scipy.optimize.fmin_l_bfgs_b(
        logreg_obj, np.zeros(DTR.shape[0]+1), approx_grad=True)

    _w = _v[0:DTR.shape[0]]
    _b = _v[-1]

    STE = np.dot(_w.T, DTE) + _b - \
        np.log(DTR[:, LTR == 1].shape[1]/DTR[:, LTR == 0].shape[1])

    return STE


def logistic_regression(D, L, applications, K, l_list, prior, imbalanced, options={"m_pca": None, "quadratic": False, "gaussianize": False, "figures": False}):

    # Shuffle data
    random_index_list = misc.generate_shuffled_indexes(D.shape[1])
    R_Data = D[:, random_index_list]
    R_Label = L[random_index_list]

    K_scores = {
        'scores': [],
        'labels': []
    }

    minDCF = {}

    for app in applications:
        minDCF[app] = []

    for l in l_list:
        K_scores['labels'] = []
        K_scores['scores'] = []
        for train_index, test_index in misc.k_fold(K, D.shape[1]):

            DTR = R_Data[:, train_index]
            LTR = R_Label[train_index]
            DTE = R_Data[:, test_index]
            LTE = R_Label[test_index]

            if options["gaussianize"]:
                DTR, DTE = features_gaussianization(DTR, DTE)

            if options["m_pca"]:
                DTR, P = calculate_pca(DTR, options["m_pca"])
                DTE = np.dot(P.T, DTE)

            if options["quadratic"]:
                DTR = quadratic(DTR)
                DTE = quadratic(DTE)

            STE = train_logistic_regression(
                DTR, LTR, DTE, l, prior, imbalanced)

            K_scores['labels'].append(LTE)
            K_scores['scores'].append(STE)

        STE = np.hstack(K_scores['scores'])
        LTE = np.hstack(K_scores['labels'])
        for app in applications:
            _minDCF = compute_min_DCF(STE, LTE, app, 1, 1)
            print(f"prior={prior}, app={app}, lambda={l}, MinDCF: ", _minDCF)
            minDCF[app].append(_minDCF)
    
    if options["figures"]:
        plt.figure()
        for app in applications:
            plt.plot(l_list, np.array(minDCF[app]), label=f'minDCF(π={app})')
        plt.xscale('log')
        plt.xlabel('λ')
        plt.ylabel('DCF')
        plt.legend()
        plt.show()


def quadratic(D):
    q_x = []
    for i in range(D.shape[1]):
        x = D[:, i]
        x_xT = misc.make_column_shape(x).dot(misc.make_row_shape(x.T))
        v = np.concatenate((np.ravel(x_xT), np.ravel(x)), axis=0)
        q_x.append(misc.make_column_shape(v))
    return np.hstack(q_x)


def lg_evaluation(Train_D, Train_L, Evaluation_D, Evaluation_L, l, prior, imbalanced):
    STE = train_logistic_regression(
        Train_D, Train_L, Evaluation_D, l, prior, imbalanced)
    LP = (STE > 0) * 1
    lg_scores = misc.make_row_shape(STE)
    C_LG_S = train_logistic_regression(
        lg_scores, LP, lg_scores, 10**-5, 0.9, False)
    for app in [0.5, 0.1, 0.9]:
        print(f"prior={prior}, app={app}, lambda={l}, MinDCF: ",
              compute_min_DCF(STE, Evaluation_L, app, 1, 1))
        print(f"prior={prior}, app={app}, lambda={l}, ACT DCF: ",
              compute_act_DCF(STE, Evaluation_L, app, 1, 1))
        print(f"prior={prior}, app={app}, lambda={l}, ACT DCF (Calibrated): ",
              compute_act_DCF(C_LG_S, Evaluation_L, app, 1, 1))