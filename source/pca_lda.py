import numpy
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from load import load_data
from misc import mcol


def get_labels(name):
    hLabels = {
        '0': 0,
        '1': 1
    }
    return hLabels[name]

def normalize_data(D):
    return preprocessing.normalize(D, axis=1, norm='l1')

def calculate_pca(D, m):
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
    P = U[:, ::-1][:, 0:m]
    # project the data
    DP = numpy.dot(P.T, D)
    return DP

def show_pca(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    plt.scatter(D0[0, :], D0[1, :], label='PCA-1')
    plt.scatter(D1[0, :], D1[1, :], label='PCA-2')

    plt.show()


def PCA_LDA():
    D, L = load_data()
    D = normalize_data(D)
    DP = calculate_pca(D)
    show_pca(DP, L)

if __name__ == '__main__':
    PCA_LDA()
