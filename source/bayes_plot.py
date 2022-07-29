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
    D, L = load_data()
    
    Train_D, Train_L = load_data('Train')
    Evaluation_D, Evaluation_L = load_data('Test')
    
    
    
    fig, ax = plt.subplots()
    P = np.linspace(-3, 3, 1000)
    
    # MVG    
    model_type = "Tied"
    scores, LP = train_mvg_models(Train_D, Train_L, Evaluation_D, model_type)
    # LP = (scores > 0) * 1
    
    C_MVG_S = train_logistic_regression(misc.vrow(scores), LP, misc.vrow(scores), 10**-5, 0.5, False)
    plt.plot(P, bayes_error_plot(P, scores, Evaluation_L, minCost=False), color='r', label='MVG actDCF')
    plt.plot(P, bayes_error_plot(P, scores, Evaluation_L, minCost=True), dashes=[6,2], color='r', label='MVG minDCF')
    plt.plot(P, bayes_error_plot(P, C_MVG_S, Evaluation_L, minCost=False), dashes=[1,2], color='r', label="MVG actDCF (Calibrated scores)")

    # LR    
    # STE = train_logistic_regression(Train_D, Train_L, Evaluation_D, 10**-5, 0.5, True)
    # lg_scores = misc.vrow(STE)
    # LP = (STE > 0) * 1
    # C_LG_S = train_logistic_regression(lg_scores, LP, lg_scores, 10**-5, 0.5, True)

    
    # plt.plot(P, bayes_error_plot(P, STE, Evaluation_L, minCost=False), color='b', label='LG actDCF')
    # plt.plot(P, bayes_error_plot(P, STE, Evaluation_L, minCost=True), dashes=[6,2], color='b', label='LG minDCF')
    # plt.plot(P, bayes_error_plot(P, C_LG_S, Evaluation_L, minCost=False), dashes=[1,2], color='b', label='LG actDCF (Calibrated scores)')

    plt.ylabel('DCF')
    plt.ylim((0, 1.2))
    plt.legend()
    plt.show()