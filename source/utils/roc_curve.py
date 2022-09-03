import matplotlib.pyplot as plt
import numpy as np
import misc
from pca_lda import calculate_pca, normalize_data
import pylab
from gaussianize import gaussianization
from scipy import special
from load import load_data
from model_evaluation import compute_min_DCF, compute_act_DCF, bayes_error_plot
from multivariate_gaussian import train_mvg_models
from logistic_regression import train_logistic_regression, quadratic
from plot import roc_curve


if __name__ == '__main__':

    D, L = load_data()
    
    (DTR, LTR), (DTE, LTE) = misc.split_db_2to1(D, L, 0)
    
    model_type = "Tied"

    scores = train_mvg_models(DTR, LTR, DTE, model_type)
    (MVG_FPR, MVG_FNR) = roc_curve(scores, LTE)
    
    plt.figure()
    plt.plot(MVG_FPR, MVG_FNR, label="MVG")

    STE = train_logistic_regression(DTR, LTR, DTE, 10**-5, 0.5, True)
    (LG_FPR, LG_FNR) = roc_curve(STE, LTE)
    plt.plot(LG_FPR, LG_FNR, label="Logistic Regression")
    
    plt.xlabel('FPR')
    plt.ylabel('FNR')

    plt.legend()
    plt.show()