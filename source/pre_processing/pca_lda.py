import numpy
from sklearn import preprocessing
from misc.misc import make_column_shape


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
    mu = make_column_shape(mu)
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
    return DP, P
