import numpy as np
from scipy import stats


def normalize(data):

    total = len(data)
    X_total = sum(data)

    X_mean = X_total / total

    X_std = np.std(data)

    data = [np.subtract(x, X_mean) for x in data]
    data = [np.divide(x, X_std) for x in data]

    return data

def zscore(data):
    return stats.zscore(data)