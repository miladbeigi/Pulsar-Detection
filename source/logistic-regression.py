import numpy as np
import misc
from pca_lda import calculate_pca, normalize_data
from gaussianize import gaussianization
import scipy.optimize
from model_evaluation import compute_min_DCF, compute_act_DCF, bayes_error_plot
from scipy import special
from load import load_data
import pylab


def logreg_obj_wrap_imbalanced(DTR, LTR, l, prior):
    M = DTR.shape[0]
    DTR_L1 = DTR[:, LTR == 1]
    DTR_L0 = DTR[:, LTR == 0]

    def logreg_obj(v):
        # Compute and return the objective function value using DTR, LTR, l
        w = misc.mcol(v[0:M])
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
        w = misc.mcol(v[0:M])
        b = v[-1]
        cxe = 0

        S = np.dot(w.T, DTR) + b
        cxe = np.logaddexp(0,  -S*Z).mean()
        return cxe + 0.5 * l * np.linalg.norm(w)**2
        # Compute and return the objective function value using DTR, LTR, l
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

    STE = np.dot(_w.T, DTE) + _b - np.log(prior/(1-prior))
    LP = STE > 0

    return STE


def logistic_regression(D, L, applications, K, l, prior, imbalanced):

    # Split into two
    # (DTR, LTR), (DTE, LTE) = misc.split_db_2to1(D, L, 0)

    # K-Fold
    kf = misc.k_fold(K)

    K_scores = {
        'scores': [],
        'labels': []
    }

    minDCF = {}

    for app in applications:
        minDCF[app] = []

    for l in l_list:
        for train_index, test_index in kf.split(D.T):

            DTR = D[:, train_index]
            LTR = L[train_index]

            DTE = D[:, test_index]
            LTE = L[test_index]

            STE = train_logistic_regression(DTR, LTR, DTE, l, prior, imbalanced)

            K_scores['labels'].append(LTE)
            K_scores['scores'].append(STE)

        STE = np.hstack(K_scores['scores'])
        LTE = np.hstack(K_scores['labels'])

        for app in applications:
            _minDCF = compute_min_DCF(STE, LTE, app, 1, 1)
            print(f"prior={prior}, app={app}, lambda={l}, MinDCF: ",_minDCF)
            minDCF[app].append(_minDCF)
    
    for app in applications:
        pylab.plot(l_list, np.array(minDCF[app]))
    
    pylab.show()


def bayes_plot():
    pass
    # P = np.linspace(-3, 3, 21)
    # pylab.plot(P, bayes_error_plot(P, STE, LTE, minCost=False), color='r')
    # pylab.plot(P, bayes_error_plot(P, STE, LTE, minCost=True), color='b')
    # pylab.ylim(0, 1.1)
    # pylab.show()

def quadratic(D):
    q_x = []
    for i in range(D.shape[1]):
        x = D[:, i]
        x_xT = misc.mcol(x).dot(misc.vrow(x.T))
        v = np.concatenate((np.ravel(x_xT), np.ravel(x)), axis=0)
        q_x.append(misc.mcol(v))
    return np.hstack(q_x)


def vec(x):
    x = x[:, None]
    x_xT = x.dot(x.T).reshape(x.size**2, order='F')
    return x_xT


if __name__ == "__main__":

    D, L = load_data()

    ## Using gaussianized data
    # D = gaussianization(D.T)
    # D = D.T

    ## Using Normalization and PCA before
    # D = normalize_data(D)
    # D = calculate_pca(D, 6)

    ### Quadratic
    ## First Approach
    # D = quadratic(D)
    ## Second Approach
    # D = np.apply_along_axis(vec, 0, D)

    applications = [0.5, 0.1, 0.9]
    K = 5
    imbalanced = True
    l_list = [10**-5]
    # l_list = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 100, 1000, 10000]
    prior = 0.5

    logistic_regression(D, L, applications, K, l_list, prior, imbalanced)