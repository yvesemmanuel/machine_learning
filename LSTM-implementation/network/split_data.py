import numpy as np

def window_data(data, window_size):
    X = []
    y = []
    
    i = 0
    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
        
        i += 1
    assert len(X) == len(y)
    return X, y


def split_data(data, window_size, percentage_2_train: float =0.8):
    X, y = window_data(data, window_size)

    eight_percent_of_X = int(len(X)*percentage_2_train)

    X_train  = np.array(X[eight_percent_of_X:])
    y_train = np.array(y[eight_percent_of_X:])

    X_test = np.array(X[:eight_percent_of_X])
    y_test = np.array(y[:eight_percent_of_X])

    return X_train, y_train, X_test, y_test
