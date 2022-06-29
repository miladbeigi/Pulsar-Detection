import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from load import load_data
import constants
import gaussianize


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
    """This function does 
    @D 
    @L
    """

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

    # plot_hist(D, L)
    # plot_scatter(D, L)
    plot_correlation(D)


def plot():
    D, L = load_data()

    # Using gaussianized data
    # D = gaussianize.gaussianization(D.T)
    # D = D.T

    plot_data(D, L)


if __name__ == '__main__':
    plot()
