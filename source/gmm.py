from re import U
import numpy as np
import scipy.special._logsumexp
import misc
import load
from model_evaluation import compute_min_DCF
from multivariate_gaussian import logpdf_GAU_ND_simplified


def GEM_ll_perSample(X, gmm):
    G = len(gmm)
    N = X.shape[1]
    S = np.zeros((G, N))
    for g in range(G):
        S[g, :] = logpdf_GAU_ND_simplified(
            X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
    return scipy.special._logsumexp.logsumexp(S, axis=0)


def GMM_EM(X, gmm):
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
            gmmNew.append((w, mu, Sigma))
        gmm = gmmNew
        print(llNew)
    print(llNew - llOld)
    return gmm


def init_GMM(DTR, N):
    mu = misc.empirical_mean(DTR)
    C = misc.cov(DTR)
    
    ## Using identity matrix 
    # C = np.identity(8, dtype="float")
    return [(1/N, mu, C) for i in range(N)]


def mmg_model(D, L, applications, K, N_C):

    # K-Fold
    kf = misc.k_fold(K)
    
    llr_mvg_full=[]
    labels = []
    
    for train_index, test_index in kf.split(D.T):

        DTR = D[:, train_index]
        LTR = L[train_index]
        DTE = D[:, test_index]
        LTE = L[test_index]

        gmm = init_GMM(DTR, N_C)

        llr_mvg_full.append(train_mmg(DTR, LTR, DTE, gmm))
        labels.append(LTE)
    
    for app in applications:
        print(f"{app} Full :", compute_min_DCF(np.hstack(llr_mvg_full), np.hstack(labels), app, 1, 1))


def train_mmg(DTR, LTR, DTE, gmm):
    
    gmm_classes = {}
    llr_c = np.zeros((2, DTE.shape[1]))
    
    for label in [0, 1]:
        gmm_classes[label] = GMM_EM(DTR[:, LTR == label], gmm)
        llr_c[label, :] = GEM_ll_perSample(DTE, gmm_classes[label])

        
    llr = llr_c[1, :] - llr_c[0, :]

    return llr


if __name__ == "__main__":
    D, L = load.load_data()

    applications = [0.5, 0.1, 0.9]
    K = 5
    N_C = 8
    
    mmg_model(D, L, applications, K, N_C)