import numpy as np
from scipy.stats import norm

def features_gaussianization(DTR, DTE):
    rankDTR = np.zeros(DTR.shape)
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[1]):
            rankDTR[i][j] = (DTR[i] < DTR[i][j]).sum()
    rankDTR = (rankDTR + 1) / (DTR.shape[1] + 2)
    if(DTE is not None):
        rankDTE = np.zeros(DTE.shape)
        for i in range(DTE.shape[0]):
            for j in range(DTE.shape[1]):
                rankDTE[i][j] = (DTR[i] < DTE[i][j]).sum() + 1
        rankDTE /= (DTR.shape[1] + 2)
        return norm.ppf(rankDTR), norm.ppf(rankDTE)
    return norm.ppf(rankDTR)
