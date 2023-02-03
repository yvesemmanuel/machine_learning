import numpy as np
from scipy import stats


def normalize(data):

    total = len(data)
    X_total = sum(data)

    X_mean = X_total / total
    X_std = np.std(data)

    new_data = []
    for x in data:
        diff = np.subtract(x, X_mean)
        new_x = np.divide(diff, X_std)

        new_data.append(new_x)

    return new_data

def zscore(data):
    return stats.zscore(data)