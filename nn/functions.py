import numpy as np


def grad_numerical(f, x):
    assert x.dtype in [np.float32, np.float64], "gradient must be evaluated at a numpy array of floats."
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.size):
        idx = np.unravel_index(i, x.shape)
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


def grads_numerical(loss_fn, params):
    grads = {}
    for key in params:
        grads[key] = grad_numerical(loss_fn, params[key])
    return grads


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def categorical_cross_entropy(y, t):
    assert y.shape == t.shape, "Shape of y and t must be the same"
    # input element sums to 1
    delta = 1e-7
    return 0 - np.mean(t * np.log(y + delta))


def binary_cross_entropy(y, t):
    assert y.shape == t.shape, "Shape of y and t must be the same"
    # each input element between 0 and 1
    delta = 1e-7
    return 0 - np.mean(t * np.log(y + delta) + (1 - t) * np.log(1 - y + delta))


def im2col(input_data, filter_h, filter_w, pad=0, stride=1):
    """
    :param input_data: (N, C, H, W)
    :param filter_h:
    :param filter_w:
    :param pad:
    :param stride:
    :return: col: (N * out_h * out_w, C * filter_h * filter_w)
    """
    N, C, H, W = input_data.shape

    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for h in range(filter_h):
        h_max = h + stride * out_h
        for w in range(filter_w):
            w_max = w + stride * out_w
            col[:, :, h, w, :, :] = img[:, :, h:h_max:stride, w:w_max:stride]

    return col.transpose((0, 4, 5, 1, 2, 3)).reshape(N * out_h * out_w, -1)


def col2img(col, input_shape, filter_h, filter_w, pad=0, stride=1):
    """
    :param col: (N * out_h * out_w, C * filter_h * filter_w)
    :param input_shape: (N, C, H, W)
    :param filter_h:
    :param filter_w:
    :param pad:
    :param stride:
    :return: img: (N, C, H, W)
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose((0, 3, 4, 5, 1, 2))
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    for h in range(filter_h):
        h_max = h + stride * out_h
        for w in range(filter_w):
            w_max = w + stride * out_w
            img[:, :, h:h_max:stride, w:w_max:stride] += col[:, :, h, w, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
