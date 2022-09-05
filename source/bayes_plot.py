import matplotlib.pyplot as plt
import numpy as np
import misc.misc as misc

from utils.load import load_data
from utils.model_evaluation import bayes_error_plot
from models.logistic_regression import train_logistic_regression


if __name__ == '__main__':
    
    Train_D, Train_L = load_data('Train')
    Evaluation_D, Evaluation_L = load_data('Test')
    
    
    
    fig, ax = plt.subplots()
    P = np.linspace(-3, 3, 1000)

    # LR    
    STE = train_logistic_regression(Train_D, Train_L, Evaluation_D, 10**-5, 0.5, True)
    lg_scores = misc.make_row_shape(STE)
    LP = (STE > 0) * 1
    C_LG_S = train_logistic_regression(lg_scores, LP, lg_scores, 10**-5, 0.5, True)

    
    plt.plot(P, bayes_error_plot(P, STE, Evaluation_L, minCost=False), color='b', label='LG actDCF')
    plt.plot(P, bayes_error_plot(P, STE, Evaluation_L, minCost=True), dashes=[6,2], color='b', label='LG minDCF')
    plt.plot(P, bayes_error_plot(P, C_LG_S, Evaluation_L, minCost=False), dashes=[1,2], color='b', label='LG actDCF (Calibrated scores)')

    plt.ylabel('DCF')
    plt.ylim((0, 1.2))
    plt.legend()
    plt.show()