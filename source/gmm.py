import matplotlib.pyplot as plt
import numpy as np
import scipy.special._logsumexp
import misc
import load
from model_evaluation import compute_min_DCF
from multivariate_gaussian import logpdf_GAU_ND_simplified
from pca_lda import calculate_pca, normalize_data
from gaussianize import gaussianization


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
            F = (misc.vrow(gamma) * X).sum(1)
            S = np.dot(X, (misc.vrow(gamma)*X).T)
            w = Z/N
            mu = misc.mcol(F/Z)
            Sigma = S/Z - np.dot(mu, mu.T)
            U, s, _ = np.linalg.svd(Sigma)
            s[s<0.5] = 0.5
            Sigma = np.dot(U, misc.mcol(s)*U.T)
            if model_type == 'naive':
                Sigma = Sigma * np.eye(Sigma.shape[0])
            if model_type == 'tied':
                Sigma = Sigma * Z
            gmmNew.append((w, mu, Sigma))
        if model_type == 'full' or model_type=='naive':
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


def init_GMM(DTR, N, gmm = []):
    new_GMM = []
    
    if gmm == []:
        mu = misc.empirical_mean(DTR)
        C = misc.cov(DTR)
        U, s, Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * 0.1
        new_GMM = [(1/N, mu-d, C) , (1/N, mu+d, C)]
        return new_GMM
    else:
        for each_gmm in gmm:
            U, s, Vh = np.linalg.svd(each_gmm[2])
            d = U[:, 0:1] * s[0]**0.5 * 0.1
            temp_gmm = [(1/N, each_gmm[1]+d, each_gmm[2]) , (1/N, each_gmm[1]-d, each_gmm[2])]
            new_GMM += temp_gmm
        return new_GMM
    
    
    


def train_mmg(DTR, LTR, DTE, gmm, model_type):

    gmm_classes = {}
    llr_c = np.zeros((2, DTE.shape[1]))

    for label in [0, 1]:
        gmm_classes[label] = GMM_EM(DTR[:, LTR == label], gmm, model_type)
        llr_c[label, :] = GEM_ll_perSample(DTE, gmm_classes[label])

    llr = llr_c[1, :] - llr_c[0, :]

    return llr


def mmg_model(D, L, applications, K, N_C_target, model_type):

    # K-Fold
    kf = misc.k_fold(K)
    # Permutation
    np.random.seed(seed=0)
    random_index_list = np.random.permutation(D.shape[1])
    R_D = D[:, random_index_list]
    R_L = L[random_index_list]

    K_scores = {
        'scores': [],
        'labels': []
    }

    minDCF = {}

    for app in applications:
        minDCF[app] = []
    
    N_C = 2
    gmm = init_GMM(R_D, N_C, [])

    while N_C <= N_C_target:
        K_scores['labels'] = []
        K_scores['scores'] = []
        if N_C != 2:
            gmm = init_GMM(DTR, N_C, gmm)
        for train_index, test_index in kf.split(D.T):

            DTR = R_D[:, train_index]
            LTR = R_L[train_index]
            DTE = R_D[:, test_index]
            LTE = R_L[test_index]

            llr = train_mmg(DTR, LTR, DTE, gmm, 'full')

            K_scores['scores'].append(llr)
            K_scores['labels'].append(LTE)
        STE = np.hstack(K_scores['scores'])
        LTE = np.hstack(K_scores['labels'])
        for app in applications:
            _minDCF = compute_min_DCF(STE, LTE, app, 1, 1)
            print(f"app={app}, {model_type} cov, number of components={N_C}:", _minDCF )
            minDCF[app].append(_minDCF)
        
        gmm = GMM_EM(R_D, gmm, model_type)
        N_C *= 2
    
    N_C = 2
    out = [ 2**j for j in range(1, int(np.log2(N_C_target))+1) ]


    x = np.arange(len(out))  # the label locations
    width = 0.20  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, minDCF[0.5], width, label=f'minDCF(π=0.5)')
    rects2 = ax.bar(x, minDCF[0.1], width, label=f'minDCF(π=0.1)')
    rects3 = ax.bar(x + width, minDCF[0.9], width, label=f'minDCF(π=0.9)')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('minDCF')
    ax.set_title('minDCF for different number of components')
    ax.set_xticks(x, out)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    D, L = load.load_data()

    ## Using gaussianized data
    # D = gaussianization(D.T).T

    applications = [0.5, 0.1, 0.9]
    K = 3
    N_C_target = 32
    model_type = "full"
    
    mmg_model(D, L, applications, K, N_C_target, model_type)