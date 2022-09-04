import matplotlib.pyplot as plt
import misc.misc as misc
from utils.load import load_data
from models.multivariate_gaussian import train_mvg_models
from models.logistic_regression import train_logistic_regression
from utils.plot import compute_roc_curve


if __name__ == '__main__':

    D, L = load_data('Train')
    
    (DTR, LTR), (DTE, LTE) = misc.split_db_2to1(D, L, 0)
    
    model_type = "Tied"

    scores = train_mvg_models(DTR, LTR, DTE, model_type)
    (MVG_FPR, MVG_FNR) = compute_roc_curve(scores, LTE)
    
    plt.figure()
    plt.plot(MVG_FPR, MVG_FNR, label="MVG")

    STE = train_logistic_regression(DTR, LTR, DTE, 10**-5, 0.5, True)
    (LG_FPR, LG_FNR) = compute_roc_curve(STE, LTE)
    plt.plot(LG_FPR, LG_FNR, label="Logistic Regression")
    
    plt.xlabel('FPR')
    plt.ylabel('FNR')

    plt.legend()
    plt.show()