import numpy as np
from .functions import sigmoid, softmax, categorical_cross_entropy, binary_cross_entropy, im2col, col2img


class Layer(object):
    def __init__(self):
        self.params = {}
        self.grads = {}


class Affine(Layer):
    def __init__(self, W, b=None):
        super(Affine, self).__init__()
        self.params["W"] = W
        if b is not None:
            self.params["b"] = b
        self.x = None
        self.original_x_shape = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        x_dot_W = np.dot(x, self.params["W"])
        if "b" in self.params:
            return x_dot_W + self.params["b"]
        else:
            return x_dot_W

    def backward(self, dout):
        if dout.ndim != 2:
            raise Exception("Derivative of output should be of shape (batch size, out_dim)")
        # dout: (N, out_dim)
        dx = np.dot(dout, self.params["W"].T)
        dW = np.dot(self.x.T, dout)
        self.grads["W"] = dW
        if "b" in self.params:
            db = np.sum(dout, axis=0)
            self.grads["b"] = db
        dx = dx.reshape(*self.original_x_shape)
        return dx


class Sigmoid(Layer):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.out = None
    
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class ReLU(Layer):
    def __init__(self):
        super(ReLU, self).__init__()
        self.mask = None
    
    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx


class BinaryCategory(Layer):
    def __init__(self, onehot=True):
        super(BinaryCategory, self).__init__()
        self.loss = None
        self.onehot = onehot
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        if x.ndim != 2 or t.ndim != 2:
            print(x.shape, t.shape)
            raise Exception("Input should be of shape (batch size, in_dim)")
        self.t = t
        if self.onehot:
            self.y = softmax(x)
            self.loss = categorical_cross_entropy(self.y, self.t)
        else:
            self.y = sigmoid(x)
            self.loss = binary_cross_entropy(self.y, self.t)
        return self.loss

    def backward(self, dout=1.0):
        dx = dout * (self.y - self.t) / self.t.size
        return dx


class BatchNorm(Layer):
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        super(BatchNorm, self).__init__()
        self.xn = None
        self.xc = None
        self.std = None
        self.params["gamma"] = gamma    # ones(D,) for input of (N,D)
        self.params["beta"] = beta      # zeros(D,) for input of (N,D)

        # at test time
        self.momentum = momentum
        self.running_mean = running_mean
        self.running_var = running_var

    def forward(self, x, train_flag=True):
        if self.running_mean is None:
            self.running_mean = np.zeros(x.shape[1:])
            self.running_var = np.zeros(x.shape[1:])

        if train_flag:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            self.xc = x - mu
            self.std = np.sqrt(var + 1e-7)
            xn = self.xc / self.std
            self.xn = xn
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 1e-7))
        return xn * self.params["gamma"] + self.params["beta"]

    def backward(self, dout):
        batch_size = dout.shape[0]
        self.grads["beta"] = dout.sum(axis=0)
        self.grads["gamma"] = np.sum(dout * self.xn, axis=0)
        dxn = self.params["gamma"] * dout
        dxc = dxn / self.std
        dstd = -np.sum(dxn * self.xc / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / batch_size) * self.xc * dvar
        dx = dxc - dxc.mean(axis=0)
        return dx


class Dropout(Layer):
    def __init__(self, dropout_ratio=0.5):
        super(Dropout, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flag=True, renew_mask=True):
        if train_flag:
            if self.mask is None or renew_mask:
                self.mask = np.random.random(x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class Conv2D(Layer):
    def __init__(self, W, b, pad=0, stride=1):
        """
        :param W: (FN, C, FH, FW)
        :param b: (FN,)
        :param stride:
        :param pad:
        """
        super(Conv2D, self).__init__()
        self.params["W"] = W
        self.params["b"] = b
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None
        self.input_shape = None

    def forward(self, x):
        FN, C, FH, FW = self.params["W"].shape
        N, C, H, W = x.shape

        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        col = im2col(x, FH, FW, self.pad, self.stride)
        col_W = self.params["W"].reshape(FN, -1).T

        out = np.dot(col, col_W) + self.params["b"]
        out = out.reshape(N, out_h, out_w, FN).transpose((0, 3, 1, 2))

        self.x = x
        self.col = col      # (N * out_h * out_w, C * FH * FW)
        self.col_W = col_W
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.params["W"].shape
        dout = dout.transpose((0, 2, 3, 1)).reshape(-1, FN)     # (N * out_h * out_w, FN)

        dcol = np.dot(dout, self.col_W.T)                       # (N * out_h * out_w, C * FH * FW)
        dW = np.dot(dout.T, self.col).reshape(FN, C, FH, FW)    # (FN, C, FH, FW)
        db = np.sum(dout, axis=0)

        self.grads["W"] = dW
        self.grads["b"] = db

        dx = col2img(dcol, self.x.shape, FH, FW, self.pad, self.stride)     # (N, C, H, W)
        return dx


class Pooling(Layer):
    def __init__(self, pool_h, pool_w, pad=0, stride=1):
        super(Pooling, self).__init__()
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.pad = pad
        self.stride = stride

        self.x = None
        self.argmax = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H + 2 * self.pad - self.pool_h) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_w) // self.stride + 1

        col = im2col(x, self.pool_h, self.pool_w, self.pad, self.stride)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        out = col.max(axis=1)
        argmax = col.argmax(axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose((0, 3, 1, 2))

        self.x = x
        self.argmax = argmax

        return out

    def backward(self, dout):
        N, C, H, W = self.x.shape
        dout = dout.transpose((0, 2, 3, 1)).flatten()
        dcol = np.zeros((dout.size, self.pool_h * self.pool_w))
        dcol[np.arange(dout.size), self.argmax] = dout
        dcol = dcol.reshape(-1, C * self.pool_h * self.pool_w)
        dx = col2img(dcol, self.x.shape, self.pool_h, self.pool_w, self.pad, self.stride)

        return dx
