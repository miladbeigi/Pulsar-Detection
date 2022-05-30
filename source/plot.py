import numpy
import matplotlib
import matplotlib.pyplot as plt
from load import load_data


def plot_hist(D, L):

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]


    hFea = {
        0: 'Mean of the integrated profile.',
        1: 'Standard deviation of the integrated profile',
        2: 'Excess kurtosis of the integrated profile',
        3: 'Skewness of the integrated profile',
        4: 'Mean of the DM-SNR curve.',
        5: 'Standard deviation of the DM-SNR curve.',
        6: 'Excess kurtosis of the DM-SNR curve.',
        7: 'Skewness of the DM-SNR curve.'
    }

    for dIdx in range(8):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins=10, density=True, alpha=0.4, label='False')
        plt.hist(D1[dIdx, :], bins=10, density=True,
                 alpha=0.4, label='True')

        plt.legend()
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        plt.savefig('plots/hist_%d.pdf' % dIdx)
    plt.show()


def plot_scatter(D, L):

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    hFea = {
        0: 'Mean of the integrated profile.',
        1: 'Standard deviation of the integrated profile',
        2: 'Excess kurtosis of the integrated profile',
        3: 'Skewness of the integrated profile',
        4: 'Mean of the DM-SNR curve.',
        5: 'Standard deviation of the DM-SNR curve.',
        6: 'Excess kurtosis of the DM-SNR curve.',
        7: 'Skewness of the DM-SNR curve.'
    }

    for dIdx1 in range(8):
        for dIdx2 in range(8):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label='False')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label='True')

            plt.legend()
            plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
            plt.savefig('plots/scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()

def plot_data(D, L):
    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    plot_hist(D, L)
    plot_scatter(D, L)


def plot():
    D, L = load_data()
    plot_data(D, L)


if __name__ == '__main__':
    plot()
