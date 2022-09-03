import numpy as np


def make_row_shape(x: np.ndarray):
    return x.reshape((1, x.size))


def make_column_shape(x: np.ndarray):
    return x.reshape((x.size, 1))


def compute_mean(x: np.ndarray):
    return make_column_shape(x.mean(1))


def compute_covariance(D: np.ndarray, type: str = None):
    """
    type: optional
          "None" full covariance matrix is computed
          "naive" 
    """
    # calculate and reshape the mean
    mu = compute_mean(D)
    # center the data
    DC = D - mu
    # calculate covariance matrix
    C = (DC @ DC.T)/DC.shape[1]
    if type == None:
        return C
    elif type == "naive":
        return np.diag(np.diag(C))


def compute_within_class_covariance(D: np.ndarray, L: np.ndarray):
    SW = 0
    for i in [0, 1]:
        SW += (L == i).sum() * compute_covariance(D[:, L == i])
    return SW/D.shape[1]


def generate_shuffled_indexes(shape: int, seed: int = 0):
    np.random.seed(seed)
    return np.random.permutation(shape)


def k_fold(K: int, samplesNumber: int):
    N = int(samplesNumber / K)
    indexes = np.arange(samplesNumber)
    list_index = []
    for i in range(K):
        idxTest = indexes[i*N:(i+1)*N]
        idxTrainLeft = indexes[0:i*N]
        idxTrainRight = indexes[(i+1)*N:]
        idxTrain = np.hstack([idxTrainLeft, idxTrainRight])

        list_index.append((idxTrain, idxTest))
    return list_index
