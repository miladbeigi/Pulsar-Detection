from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from load import load_data
from model_evaluation import compute_min_DCF
import misc


def train_SVM(DTR, LTR, DTE, LTE, Weighted_C_list, gamma, C=1, K=0, kernel_type=None):
    """
    @kernel_type: rbf, pol, None
    """

    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    # Radial Basis Function
    if kernel_type == "rbf":
        Dist = misc.make_column_shape((DTR**2).sum(0)) + \
            misc.make_row_shape((DTR**2).sum(0)) - 2*np.dot(DTR.T, DTR)
        kernel = np.exp(-gamma*Dist) + K
        H = misc.make_column_shape(Z)*misc.make_row_shape(Z)*kernel

    else:
        DTREXT = np.vstack([DTR, np.ones((1, DTR.shape[1]))])
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
        factr=0.0,
        maxiter=100000,
        maxfun=100000,
    )

    STE = np.zeros(DTE.shape[1])

    # Radial Basis Function
    if kernel_type == "rbf":
        for i in range(0, DTE.shape[1]):
            for j in range(0, DTR.shape[1]):
                exp = np.linalg.norm(DTR[:, j] - DTE[:, i]) ** 2 * gamma
                kern = np.exp(-exp) + K
                STE[i] += alphaStar[j] * Z[j] * kern
        err = evaluation_SVM(STE, LTE)

    else:
        wStar = np.dot(DTREXT, misc.make_column_shape(alphaStar) * misc.make_column_shape(Z))
        w = wStar[0:DTR.shape[0]]
        b = wStar[-1]
        STE = np.dot(w.T, DTE) + b
        STE = STE.reshape((STE.shape[1]))
        err = evaluation_SVM(STE, LTE)
        print("Primal Loss: ", JPrimal(wStar))
        print("Duality Gap: ", JPrimal(wStar) - JDual(alphaStar)[0])

    print("Dual Loss: ", JDual(alphaStar)[0])
    print("Error Rate: ", err)
    print("minDCF: :", compute_min_DCF(STE, LTE, 0.5, 1, 1))

    return STE


def svm_model(D, L, applications, K, C_list, prior, rebalanced, kernel_type=None):

    np.random.seed(seed=0)
    random_index_list = np.random.permutation(D.shape[1])
    R_D = D[:, random_index_list]
    R_L = L[random_index_list]

    kf = misc.k_fold(K)
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
        for train_index, test_index in kf.split(R_D.T):

            DTR = R_D[:, train_index]
            LTR = R_L[train_index]

            DTE = R_D[:, test_index]
            LTE = R_L[test_index]

            Weighted_C_list = compute_weights(LTR, prior, C, rebalanced)

            STE = train_SVM(
                DTR, LTR, DTE, LTE, Weighted_C_list, gamma=0.001, C=C, K=0, kernel_type=kernel_type)

            K_scores['labels'].append(LTE)
            K_scores['scores'].append(STE)

        STE = np.hstack(K_scores['scores'])
        LTE = np.hstack(K_scores['labels'])

        for app in applications:
            _minDCF = compute_min_DCF(STE, LTE, app, 1, 1)
            print(f"prior={prior}, app={app}, C={C}, MinDCF: ", _minDCF)
            minDCF[app].append(_minDCF)

    plt.figure()

    for app in applications:
        plt.plot(C_list, np.array(minDCF[app]), label=f'minDCF(π={app})')

    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    plt.show()


def compute_weights(LTR, prior, C, rebalanced):

    if rebalanced == False:
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


def svm_evaluation(DTR, LTR, DTE, LTE, gamma, prior, imbalanced, C=1, K=0, kernel_type=None):

    Weighted_C_list = compute_weights(LTR, prior, C, rebalanced)

    STE = train_SVM(DTR, LTR, DTE, LTE, Weighted_C_list,
                    gamma, C=1, K=0, kernel_type=None)

    for app in [0.1, 0.5, 0.9]:
        _minDCF = compute_min_DCF(STE, LTE, app, 1, 1)
        print(f"prior={prior}, app={app}, C={C}, MinDCF: ", _minDCF)


if __name__ == '__main__':

    Train_D, Train_L = load_data('Train')
    Evaluation_D, Evaluation_L = load_data('Test')

    D = np.concatenate([Train_D, Evaluation_D], axis=1)

    # Using gaussianized data
    # D = gaussianization(D)

    # Using PCA
    # D = calculate_pca(D, 7)

    Train_D = D[:, 0:8929]
    Evaluation_D = D[:, 8929:]

    rebalanced = True

    # C_list = [10**-2, 10**-1, 1, 10, 100]
    C_list = [100]

    applications = [0.5, 0.1, 0.9]
    K = 3
    prior = 0.5

    # svm_model(Train_D, Train_L, applications, K, C_list,
    #           prior, rebalanced, kernel_type="rbf")
    C = 0.1
    svm_evaluation(Train_D, Train_L, Evaluation_D,
                   Evaluation_L, 0.001, 0.5, True, C, 0, None)
