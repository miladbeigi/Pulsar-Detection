import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import misc.misc as misc
from utils.model_evaluation import compute_min_DCF
from pre_processing.pca_lda import calculate_pca
from pre_processing.gaussianize import features_gaussianization

def _logpdf_GAU_ND_Opt(X, mu, C):
    P = np.linalg.inv(C)
    const = -0.5 * X.shape[0] * np.log(2*np.pi)
    const += -0.5 * np.linalg.slogdet(C)[1]
    
    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const + -0.5 * np.dot((x-mu).T, np.dot(P, (x-mu)))
        Y.append(res)
    
    return np.array(Y).ravel()

def _GMM_ll_per_sample(X, gmm):
    G = len(gmm)
    N = X.shape[1]
    S = np.zeros((G, N))
    
    for g in range(G):
        S[g, :] = _logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
    return scipy.special.logsumexp(S, axis=0)

def train_gmm(DTR, LTR, DTE, number_of_components , model_type):

    gmm_classes = {}
    ll_c = np.zeros((2, DTE.shape[1]))

    for label in [0, 1]:
        gmm_classes[label] = GMM_LBG(DTR[:, LTR == label], number_of_components, model_type)
        ll_c[label, :] = _GMM_ll_per_sample(DTE, gmm_classes[label])

    llr = ll_c[1, :] - ll_c[0, :]

    return llr


def GMM_LBG(DTR, number_of_components, model_type):
    initial_mu = misc.make_column_shape(DTR.mean(1))
    initial_sigma = misc.compute_covariance(DTR)
    
    gmm = [(1.0, initial_mu, initial_sigma)]
    for i in range(number_of_components):
        doubled_gmm = []
        for component in gmm:
            w = component[0]
            mu = component[1]
            sigma = component[2]
            
            U, s, Vh = np.linalg.svd(sigma)
            d = U[:, 0:1] * s[0]**0.5 * 0.1
            component1 = (w/2, mu+d, sigma)
            component2 = (w/2, mu-d, sigma)
            doubled_gmm.append(component1)
            doubled_gmm.append(component2)
        gmm = _GMM_EM(DTR, doubled_gmm, model_type)
    return gmm

def _compute_ll_new_P(X, G, N, gmm):
    SJ = np.zeros((G, N))
    for g in range(G):
        SJ[g, :] = _logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
    SM = scipy.special.logsumexp(SJ, axis=0)
    ll_new = SM.sum() / N
    P = np.exp(SJ - SM)
    return ll_new , P



def _GMM_EM(X, gmm, model_type):
    ll_new = None
    ll_old = None
    G = len(gmm)
    N = X.shape[1]
    
    psi = 0.01
    
    while ll_old is None or ll_new-ll_old>1e-6:
        ll_old = ll_new
        ll_new , P = _compute_ll_new_P(X, G, N, gmm)
        
        gmm_new = []
        summatory = np.zeros((X.shape[0], X.shape[0]))
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (misc.make_row_shape(gamma)*X).sum(1)
            S = np.dot(X, (misc.make_row_shape(gamma)*X).T)
            w = Z/N
            mu = misc.make_column_shape(F/Z)
            sigma = S/Z - np.dot(mu, mu.T)

            if model_type == "naive":
                sigma = sigma * np.eye(sigma.shape[0])
            
            if model_type == "tied":
                summatory += Z*sigma
            
            if model_type == "naive-tied":
                sigma = sigma * np.eye(sigma.shape[0])
                summatory += Z*sigma
            
            if model_type == "naive" or "full":
                # Constraint
                U, s, _ = np.linalg.svd(sigma)
                s[s<psi] = psi
                sigma = np.dot(U, misc.make_column_shape(s)*U.T)
            
            gmm_new.append((w, mu, sigma))
        
        if model_type == "tied" or "naive-tied":
            # Tied
            sigma = summatory / G
            # Constraint
            U, s, _ = np.linalg.svd(sigma)
            s[s<psi] = psi
            sigma = np.dot(U, misc.make_column_shape(s)*U.T)
        
        gmm = gmm_new
    return gmm


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

    while number_of_components <= target_number_of_components:

        K_scores['labels'] = []
        K_scores['scores'] = []

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

            llr = train_gmm(DTR, LTR, DTE, number_of_components, options["model_type"])

            K_scores['scores'].append(llr)
            K_scores['labels'].append(LTE)

        STE = np.hstack(K_scores['scores'])
        LTE = np.hstack(K_scores['labels'])

        for app in applications:
            _minDCF = compute_min_DCF(STE, LTE, app, 1, 1)
            print(
                f"app={app}, {options['model_type']} cov, number of components={number_of_components}:", _minDCF)
            minDCF[app].append(_minDCF)

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
