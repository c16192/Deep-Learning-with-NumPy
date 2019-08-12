import numpy as np

def sigmoid(x):
    if x > 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (np.exp(x) + 1)

def softmax(x):
    if len(x.shape) == 2:
        is_2d = True
    elif len(x.shape) == 1:
        is_2d = False
        x = x.reshape(1, -1)
    else:
        raise Exception("Input has to be of 1 or 2 dimentions.")

    x -= x.max(axis=1)
    out = np.exp(x - np.log(np.sum(np.exp(x), axis=1)))
    return out if is_2d else out.reshape(-1)