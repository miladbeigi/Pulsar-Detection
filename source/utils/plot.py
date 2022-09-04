import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import misc.constants as constants
from utils.model_evaluation import bayes_error_plot


def plot_correlation(D):
    corr = np.zeros((D.shape[0], D.shape[0]))
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            corr[i, j] = np.cov(D[i, :], D[j, :])[0][1] / \
                (np.std(D[i, :] * np.std(D[j, :])))
    plt.figure(figsize=(12, 7))
    sns.heatmap(corr, annot=True, fmt='.2f',
                cmap='Blues', mask=np.triu(corr, +1))
    plt.show()


def plot_hist(D, L):

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for dIdx in range(D.shape[0]):
        print(dIdx)
        fig, axes = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(hspace=.4, wspace=.2)
        sns.distplot(D0[dIdx, :], ax=axes)  # Non-Pulsar
        sns.distplot(D1[dIdx, :], ax=axes).set_title(
            constants.hFea[dIdx])  # Pulsar
        plt.savefig(f'{constants.PWD}/plots/seaborn_hist_%d.pdf' % dIdx)

    for dIdx in range(D.shape[0]):
        plt.figure()
        plt.xlabel(constants.hFea[dIdx])
        plt.hist(D0[dIdx, :], bins=10, density=True, alpha=0.4, label='False')
        plt.hist(D1[dIdx, :], bins=10, density=True, alpha=0.4, label='True')

        plt.legend()
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        plt.savefig(f'{constants.PWD}/plots/hist_%d.pdf' % dIdx)

    plt.show()


def plot_scatter(D, L):

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for dIdx1 in range(D.shape[0]):
        for dIdx2 in range(D.shape[0]):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(constants.hFea[dIdx1])
            plt.ylabel(constants.hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label='False', alpha=0.3)
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label='True', alpha=0.3)

            plt.legend()
            plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
            plt.savefig(f'{constants.PWD}/plots/scatter_%d_%d.pdf' %
                        (dIdx1, dIdx2))
        plt.show()


def plot_data(D, L):

    # Change default font size - comment to use default values
    plt.rc('font', size=12)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    plot_hist(D, L)
    plot_scatter(D, L)
    plot_correlation(D)


def roc_curve(scores, labels):
    thresholds = np.array(scores)
    thresholds.sort()

    thresholds = thresholds[400:2500]

    FPR = np.zeros(thresholds.size)
    FNR = np.zeros(thresholds.size)

    for idx, t in enumerate(thresholds):
        Pred = np.int32(scores > t)
        Conf = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                Conf[i, j] = ((Pred == i) * (labels == j)).sum()
        FNR[idx] = Conf[0, 1] / Conf[1, 1] + Conf[0, 1]
        FPR[idx] = Conf[1, 1] / Conf[1, 0] + Conf[0, 0]

    return ((FPR/labels.shape[0])*100, (FNR/labels.shape[0])*100)


def bayes_plot(scores, calibrated_scores, Evaluation_Labels):
    
    fig, ax = plt.subplots()
    P = np.linspace(-3, 3, 1000)
    
    plt.plot(P, bayes_error_plot(P, scores, Evaluation_Labels, minCost=False), color='r', label='actDCF')
    plt.plot(P, bayes_error_plot(P, scores, Evaluation_Labels, minCost=True), dashes=[6,2], color='r', label='minDCF')
    plt.plot(P, bayes_error_plot(P, calibrated_scores, Evaluation_Labels, minCost=False), dashes=[1,2], color='r', label="actDCF (Calibrated scores)")

    plt.ylabel('DCF')
    plt.ylim((0, 1.2))
    plt.legend()
    plt.show()