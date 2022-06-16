import numpy as np
import misc
from pca_lda import calculate_pca, normalize_data
import scipy.optimize
from model_evaluation import compute_min_DCF, compute_act_DCF, bayes_error_plot
from scipy import special
from load import load_data
import pylab


def logreg_obj_wrap(DTR, LTR, l):
    Z = LTR * 2.0 - 1.0
    M = DTR.shape[0]

    def logreg_obj(v):
        # Compute and return the objective function value using DTR, LTR, l
        w = misc.mcol(v[0:M])
        b = v[-1]
        cxe = 0
        S = np.dot(w.T, DTR) + b
        cxe = np.logaddexp(0,  -S*Z).mean()
        return cxe + 0.5* l * np.linalg.norm(w)**2
    return logreg_obj

def logistic_regression():
    D, L = load_data()
    
    (DTR, LTR), (DTE, LTE) = misc.split_db_2to1(D, L, 0)
    
    for lamda in [ 0, 1e-6, 1e-3, 0.1 ,  1.0, 10]:
        logreg_obj = logreg_obj_wrap(DTR, LTR, lamda)
        _v ,  _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0]+1), approx_grad=True)
        _w = _v [0:DTR.shape[0]] 
        _b = _v[-1]
        STE = np.dot(_w.T, DTE) + _b
        LP = STE > 0
        print("==========================")
        print("Lambda: ", lamda)
        print("Min DCF: ", compute_min_DCF(STE, LTE, 0.5, 1, 1))
        print("ACT DCF: ", compute_act_DCF(STE, LTE, 0.5, 1, 1))
        print("==========================") 
        P = np.linspace(-3, 3, 21)
        pylab.plot(P, bayes_error_plot(P, STE, LTE, minCost=False), color='r')
        pylab.plot(P, bayes_error_plot(P, STE, LTE, minCost=True), color='b')
        pylab.ylim(0, 1,1)
        pylab.show()



if __name__ == "__main__":
    logistic_regression()