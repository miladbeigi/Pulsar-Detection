from typing import Text, List, Union
import numpy as np
from scipy.stats import norm, rankdata
from load import load_data
import plot
from misc import vrow

def _update_x(x: Union[np.ndarray, List]) -> np.ndarray:
    x = np.asarray(x)
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    elif len(x.shape) != 2:
        raise ValueError("Data should be a 1-d list of samples to transform or a 2d array with samples as rows.")
    return x

def gaussianization(x: np.array):
    x = _update_x(x)
    return np.array([norm.ppf((rankdata(x_i) - 0.5) / len(x_i)) for x_i in x.T]).T

if __name__ == "__main__":
    pass
    D, L = load_data()
    plot.plot_hist(D, L)

    D = gaussianization(D.T)
    plot.plot_hist(D.T, L)