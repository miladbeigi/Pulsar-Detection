import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from utils.model_evaluation import compute_min_DCF
from pre_processing.gaussianize import features_gaussianization
from pre_processing.pca_lda import calculate_pca
import misc.misc as misc
import misc.constants as constants


def train_SVM(DTR, LTR, DTE, LTE, Weighted_C_list, gamma, C, K=0, kernel_type=None):

    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    # Radial Basis Function
    if kernel_type == "rbf":
        Dist = misc.make_column_shape((DTR**2).sum(0)) + \
            misc.make_row_shape((DTR**2).sum(0)) - 2*np.dot(DTR.T, DTR)
        kernel = np.exp(-gamma*Dist) + (K**2)
        H = misc.make_column_shape(Z)*misc.make_row_shape(Z)*kernel

    else:
        DTREXT = np.vstack([DTR, np.ones((DTR.shape[1])) * K])
        H = np.dot(DTREXT.T, DTREXT)
        H = misc.make_column_shape(Z) * misc.make_row_shape(Z) * H

    def JPrimal(w):
        S = np.dot(misc.make_row_shape(w), DTREXT)
        loss = np.maximum(np.zeros(S.shape), 1 - Z*S).sum()
        return -0.5*np.linalg.norm(w)**2 + C*loss

    def JDual(alpha):
        Ha = np.dot(H, misc.make_column_shape(alpha))
        aHa = np.dot(misc.make_row_shape(alpha), Ha)
        a1 = alpha.sum()

        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    def evaluation_SVM(STE, LTE):
        label = np.int32(STE > 0)
        err = (label != LTE).sum()
        err = err / LTE.shape[0] * 100
        return err

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        np.zeros(DTR.shape[1]),
        bounds=Weighted_C_list,
        factr=constants.FMIN_L_BFGS_B_factr,
        maxiter=constants.FMIN_L_BFGS_B_maxiter,
        maxfun=constants.FMIN_L_BFGS_B_maxfun,
    )

    STE = np.zeros(DTE.shape[1])

    # Radial Basis Function
    if kernel_type == "rbf":
        Dist = misc.make_column_shape((DTR ** 2).sum(0)) + misc.make_row_shape((DTE ** 2).sum(0)) - 2 * np.dot(DTR.T, DTE)
        kernel = np.exp(-gamma * Dist) + (K ** 2)
        STE = np.dot(alphaStar * Z, kernel)
        err = evaluation_SVM(STE, LTE)

    else:
        DTEEx = np.vstack([DTE, np.ones(DTE.shape[1]) * K])
        wStar = np.dot(DTREXT, misc.make_column_shape(
            alphaStar) * misc.make_column_shape(Z))
        STE = np.dot(alphaStar.T, DTEEx)
        err = evaluation_SVM(STE, LTE)
        print("Primal Loss: ", JPrimal(wStar))
        print("Duality Gap: ", JPrimal(wStar) - JDual(alphaStar)[0])

    print("Dual Loss: ", JDual(alphaStar)[0])
    print("Error Rate: ", err)
    print("minDCF: :", compute_min_DCF(STE, LTE, 0.5, 1, 1))

    return STE


def svm_model(D, L, applications, K, C_list, gamma, prior, imbalanced_data, options={"m_pca": None, "gaussianize": False, "kernel_type": None}):

    # Shuffle data
    Random_Data, Random_Labels = misc.shuffle_data(D, L)

    K_scores = {
        'scores': [],
        'labels': []
    }

    minDCF = {}

    for app in applications:
        minDCF[app] = []

    for C in C_list:
        K_scores['labels'] = []
        K_scores['scores'] = []
        for train_index, test_index in misc.k_fold(K, D.shape[1]):

            DTR = Random_Data[:, train_index]
            LTR = Random_Labels[train_index]
            DTE = Random_Data[:, test_index]
            LTE = Random_Labels[test_index]

            Weighted_C_list = compute_weights(LTR, prior, C, imbalanced_data)

            if options["gaussianize"]:
                DTR, DTE = features_gaussianization(DTR, DTE)

            if options["m_pca"]:
                DTR, P = calculate_pca(DTR, options["m_pca"])
                DTE = np.dot(P.T, DTE)

            STE = train_SVM(
                DTR, LTR, DTE, LTE, Weighted_C_list, gamma, C, K=constants.SVM_K, kernel_type=options["kernel_type"])

            K_scores['labels'].append(LTE)
            K_scores['scores'].append(STE)

        STE = np.hstack(K_scores['scores'])
        LTE = np.hstack(K_scores['labels'])

        for app in applications:
            _minDCF = compute_min_DCF(STE, LTE, app, 1, 1)
            print(f"prior={prior}, app={app}, C={C}, MinDCF: ", _minDCF)
            minDCF[app].append(_minDCF)

    if options["figures"]:
        plt.figure()

        for app in applications:
            plt.plot(C_list, np.array(minDCF[app]), label=f'minDCF(π={app})')

        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('DCF')
        plt.legend()
        plt.show()


def compute_weights(LTR, prior, C, imbalanced_data):

    if imbalanced_data == False:
        return [(0, C)] * LTR.shape[0]

    prior_t = (LTR == 1).sum()/LTR.shape[0]
    prior_f = (LTR == 0).sum()/LTR.shape[0]
    C_T = C * (prior/prior_t)
    C_F = C * (prior/prior_f)

    Weighted_C_list = []
    for x in LTR:
        if x == 1:
            Weighted_C_list.append((0, C_T))
        else:
            Weighted_C_list.append((0, C_F))
    return Weighted_C_list


def svm_evaluation(DTR, LTR, DTE, LTE, gamma, prior, imbalanced_data, C=1, K=0, kernel_type=None):

    Weighted_C_list = compute_weights(LTR, prior, C, imbalanced_data)

    STE = train_SVM(DTR, LTR, DTE, LTE, Weighted_C_list,
                    gamma, C=1, K=0, kernel_type=None)

    for app in [0.1, 0.5, 0.9]:
        _minDCF = compute_min_DCF(STE, LTE, app, 1, 1)
        print(f"prior={prior}, app={app}, C={C}, MinDCF: ", _minDCF)