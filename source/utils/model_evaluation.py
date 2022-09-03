import numpy as np


def assign_labels(scores, pi, cfn, cfp, th=None):
    if th is None:
        th = - np.log(pi + cfn) + np.log((1-pi) * cfp)
    P = scores > th
    return np.int32(P)


def compute_conf_matrix_binary(Pred, Labels):
    C = np.zeros((2, 2))
    C[0, 0] = ((Pred == 0) * (Labels == 0)).sum()
    C[0, 1] = ((Pred == 0) * (Labels == 1)).sum()
    C[1, 0] = ((Pred == 1) * (Labels == 0)).sum()
    C[1, 1] = ((Pred == 1) * (Labels == 1)).sum()
    return C


def compute_emp_Bayes_binary(CM, pi, cfn, cfp):
    fnr = CM[0, 1] / (CM[0, 1] + CM[1, 1])
    fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])
    return pi * cfn * fnr + (1-pi) * cfp * fpr


def compute_normalized_emp_Bayes(CM, pi, cfn, cfp):
    empBayes = compute_emp_Bayes_binary(CM, pi, cfn, cfp)
    return empBayes / min(pi * cfn, (1-pi) * cfp)


def compute_act_DCF(scores, labels, pi, cfn, cfp, th=None):
    Pred = assign_labels(scores, pi, cfn, cfp, th=th)
    CM = compute_conf_matrix_binary(Pred, labels)
    return compute_normalized_emp_Bayes(CM, pi, cfn, cfp)


def compute_min_DCF(scores, labels, pi, cfn, cfp):
    t = np.array(scores)
    t.sort()
    np.concatenate([np.array([-np.inf]), t, np.array([np.inf])])
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, cfn, cfp, th=_th))
    return np.array(dcfList).min()


def bayes_error_plot(pArray, scores, labels, minCost=False):
    y = []
    for p in pArray:
        pi = 1.0 / (1.0 + np.exp(-p))
        if minCost:
            y.append(compute_min_DCF(scores, labels, pi, 1, 1))
        else:
            y.append(compute_act_DCF(scores, labels, pi, 1, 1))
    return np.array(y)
