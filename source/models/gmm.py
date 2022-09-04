from cmath import rect
import matplotlib.pyplot as plt
import numpy as np
import scipy.special._logsumexp
import misc.misc as misc
from utils.model_evaluation import compute_min_DCF
from models.multivariate_gaussian import logpdf_GAU_ND_simplified
from pre_processing.pca_lda import calculate_pca
from pre_processing.gaussianize import features_gaussianization


def GEM_ll_perSample(X, gmm):
    G = len(gmm)
    N = X.shape[1]
    S = np.zeros((G, N))
    for g in range(G):
        S[g, :] = logpdf_GAU_ND_simplified(
            X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
    return scipy.special._logsumexp.logsumexp(S, axis=0)


def GMM_EM(X, gmm, model_type):
    llNew = None
    llOld = None

    G = len(gmm)
    N = X.shape[1]

    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = np.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND_simplified(
                X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
        SM = scipy.special._logsumexp.logsumexp(SJ, axis=0)
        llNew = SM.sum()/N
        P = np.exp(SJ-SM)
        gmmNew = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (misc.make_row_shape(gamma) * X).sum(1)
            S = np.dot(X, (misc.make_row_shape(gamma)*X).T)
            w = Z/N
            mu = misc.make_column_shape(F/Z)
            Sigma = S/Z - np.dot(mu, mu.T)
            U, s, _ = np.linalg.svd(Sigma)
            
            # check this
            s[s < 0.5] = 0.5
            Sigma = np.dot(U, misc.make_column_shape(s)*U.T)
            
            if model_type == 'naive':
                Sigma = Sigma * np.eye(Sigma.shape[0])
            
            if model_type == 'tied':
                Sigma = Sigma * Z
            
            gmmNew.append((w, mu, Sigma))
        
        if model_type == 'full' or model_type == 'naive':
            gmm = gmmNew
        
        if model_type == 'navie-tied':
            gmm = []
            newSigma = sum([gmmNew[x][2] for x in range(G)])/G
            newSigma = newSigma * Z
            for g in range(G):
                gmm.append((gmmNew[g][0], gmmNew[g][1], newSigma))
        
        if model_type == 'tied':
            gmm = []
            newSigma = sum([gmmNew[x][2] for x in range(G)])/G
            for g in range(G):
                gmm.append((gmmNew[g][0], gmmNew[g][1], newSigma))
        # print(llNew)
    # print(llNew - llOld)
    return gmm


def init_GMM(DTR, N, gmm=[]):
    new_GMM = []

    if gmm == []:
        mu = misc.compute_mean(DTR)
        C = misc.compute_covariance(DTR)
        U, s, Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * 0.1
        new_GMM = [(1/N, mu-d, C), (1/N, mu+d, C)]
        return new_GMM
    else:
        for each_gmm in gmm:
            U, s, Vh = np.linalg.svd(each_gmm[2])
            d = U[:, 0:1] * s[0]**0.5 * 0.1
            temp_gmm = [(1/N, each_gmm[1]+d, each_gmm[2]),
                        (1/N, each_gmm[1]-d, each_gmm[2])]
            new_GMM += temp_gmm
        return new_GMM


def train_gmm(DTR, LTR, DTE, gmm, model_type):

    gmm_classes = {}
    llr_c = np.zeros((2, DTE.shape[1]))

    for label in [0, 1]:
        gmm_classes[label] = GMM_EM(DTR[:, LTR == label], gmm, model_type)
        llr_c[label, :] = GEM_ll_perSample(DTE, gmm_classes[label])

    llr = llr_c[1, :] - llr_c[0, :]

    return llr


def gmm_model(D, L, applications, K, target_number_of_components, options):

   # Shuffle data
    Random_Data, Random_Labels = misc.shuffle_data(D, L)

    K_scores = {
        'scores': [],
        'labels': []
    }

    minDCF = {}

    for app in applications:
        minDCF[app] = []

    number_of_components = 2
    gmm = init_GMM(Random_Data, number_of_components, [])

    while number_of_components <= target_number_of_components:

        K_scores['labels'] = []
        K_scores['scores'] = []

        if number_of_components != 2:
            # check this
            gmm = init_GMM(DTR, number_of_components, gmm)

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

            llr = train_gmm(DTR, LTR, DTE, gmm, options["model_type"])

            K_scores['scores'].append(llr)
            K_scores['labels'].append(LTE)

        STE = np.hstack(K_scores['scores'])
        LTE = np.hstack(K_scores['labels'])

        for app in applications:
            _minDCF = compute_min_DCF(STE, LTE, app, 1, 1)
            print(
                f"app={app}, {options['model_type']} cov, number of components={number_of_components}:", _minDCF)
            minDCF[app].append(_minDCF)

        gmm = GMM_EM(Random_Data, gmm, options["model_type"])
        number_of_components *= 2
        # End of While loop

    if options["figures"]:

        number_of_components = 2
        out = [
            2**j for j in range(1, int(np.log2(target_number_of_components))+1)]

        x = np.arange(len(out))  # the label locations
        width = 0.20  # the width of the bars

        fig, ax = plt.subplots()
    
        rects1 = ax.bar(x - width, minDCF[0.5], width, label=f'minDCF(π=0.5)')
        rects2 = ax.bar(x, minDCF[0.1], width, label=f'minDCF(π=0.1)')
        rects3 = ax.bar(x + width, minDCF[0.9], width, label=f'minDCF(π=0.9)')

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        fig.tight_layout()
        ax.set_ylabel('minDCF')
        ax.set_title('minDCF for different number of components')
        ax.set_xticks(x, out)
        ax.legend()

        plt.show()
