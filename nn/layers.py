from collections import OrderedDict
import numpy as np

class Layer(object):
    def __init__(self):
        self.vars = OrderedDict()
        self.params = OrderedDict()
        self.grads = OrderedDict()

    def init_params(self):
        pass

    def forward(self):
        pass
    
    def backward(self):
        pass
    

class Linear(Layer):
    def __init__(self, in_dim, out_dim, bias=True):
        super(Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._bias = bias
        self.init_params()

    def init_params(self):
        xavier_std = 1 / np.sqrt(self.in_dim)
        self.params["W"] = np.random.randn(self.in_dim, self.out_dim) * xavier_std
        self.params["b"] = np.random.randn(self.out_dim) * xavier_std if self._bias else np.zeros(self.out_dim)
    
    def forward(self, x):
        if len(x.shape) != 2:
            raise Exception("Input should be of shape (batch size, in_dim)")
        self.vars["x"] = x
        W, b = self.params["W"], self.params["b"]
        return np.dot(x, W) + b

    def backward(self, dout):
        if len(dout.shape) != 2:
            raise Exception("Derivative of output should be of shape (batch size, out_dim)")
        # dout: (N, out_dim)
        dx = np.dot(dout, self.params["W"].T)
        dW = np.dot(self.vars["x"].T, dout)
        self.grads["W"] = dW
        if self._bias:
            db = np.sum(dout, axis=0)
            self.grads["b"] = db
        else:
            self.grads["b"] = np.zeros(self.out_dim)
        return dx

class Sigmoid(Layer):
    def __init__(self, in_dim, out_dim, bias=True):
        super(Sigmoid, self).__init__()
