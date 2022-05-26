import numpy
import matplotlib
import matplotlib.pyplot as plt


def mcol(v):
    return v.reshape((v.size, 1))


def get_labels(name):
    hLabels = {
        '0': 0,
        '1': 1
    }
    return hLabels[name]


def load(file_name):
    DList = []
    labelsList = []

    with open(file_name) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:8]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = get_labels(name)
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


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
        plt.savefig('hist_%d.pdf' % dIdx)
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
            plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()

def plot_data(D, L):
    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    plot_hist(D, L)
    plot_scatter(D, L)


def calculate_pca(D, L):

    # calculate and reshape the mean
    mu = D.mean(1)
    mu = mcol(mu)
    # center the data
    DC = D - mu
    # calculate covariance matrix
    C = (DC @ DC.T)/DC.shape[1]
    # compute eigen values and eigen vectors
    s, U = numpy.linalg.eigh(C)
    # compute principal components
    P = U[:, ::-1][:, 0:2]
    # project the data
    DP = numpy.dot(P.T, D)
    show_pca(DP, L)


def show_pca(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    plt.scatter(D0[0, :], D0[1, :], label='Setosa')
    plt.scatter(D1[0, :], D1[1, :], label='Versicolor')

    plt.show()


def main():
    D, L = load('Train.txt')
    print(D.shape, L.shape)
    # plot_data(D, L)
    calculate_pca(D, L)


if __name__ == '__main__':
    main()
