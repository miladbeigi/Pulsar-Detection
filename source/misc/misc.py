import numpy as np


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

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

def shuffle_data(D, L):
    random_index_list = generate_shuffled_indexes(D.shape[1])
    Random_Data = D[:, random_index_list]
    Random_Labels = L[random_index_list]
    return Random_Data, Random_Labels


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
