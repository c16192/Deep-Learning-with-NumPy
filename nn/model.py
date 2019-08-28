from collections import OrderedDict

from .functions import grads_numerical
from .layers import *


class Model(object):
    def __init__(self):
        self.params = {}

    def predict(self, x):
        raise NotImplementedError

    def loss(self, x, t):
        raise NotImplementedError

    def grad_numerical(self, x, t):
        loss_fn = lambda w: self.loss(x, t)
        return grads_numerical(loss_fn, self.params), self.loss(x, t)

    def grads(self, x, t):
        raise NotImplementedError


class Vec2Class(Model):
    def __init__(self, dims, onehot=False, batch_norm=False):
        """
        :param dims: dimensions of input, hidden and output layers, given as a list. Example: [2, 16, 16, 1]
        :param onehot: if the target class label is one hot or not.
        """
        super(Vec2Class, self).__init__()

        self.batch_norm = batch_norm
        self.layers = OrderedDict()

        for i in range(len(dims) - 1):
            W = 1 / np.sqrt(dims[i]) * np.random.randn(dims[i], dims[i+1])
            b = np.zeros(dims[i+1])
            self.params["W"+str(i)] = W
            self.params["b"+str(i)] = b

            self.layers["Affine"+str(i)] = Affine(W, b)
            if i != len(dims) - 2:
                if self.batch_norm:
                    gamma, beta = np.ones((dims[i+1],)), np.zeros((dims[i+1],))
                    self.layers["BatchNorm"+str(i)] = BatchNorm(gamma, beta)
                    self.params["gamma" + str(i)] = gamma
                    self.params["beta" + str(i)] = beta
                self.layers["Relu"+str(i)] = ReLU()

        self.onehot = onehot

        self.loss_layer = BinaryCategory(onehot=onehot)

    def predict(self, x, train_flag=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flag=True):
        y = self.predict(x, train_flag)
        return self.loss_layer.forward(y, t)

    def accuracy(self, x, t, train_flag=False):
        y = self.predict(x, train_flag)
        if self.onehot:
            return np.mean(y.argmax(axis=1) == t.argmax(axis=1))
        else:
            y = np.where(y > 0.5, 1, 0)
            return np.mean(y == t)

    def grads(self, x, t):
        # forward
        loss = self.loss(x, t, train_flag=True)

        # backward
        dout = 1
        dout = self.loss_layer.backward(dout)
        layers = list(self.layers.values())
        layers = reversed(layers)
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}

        for key in self.layers:
            if "Affine" in key:
                i = key.replace("Affine", "")
                grads["W"+i] = self.layers[key].grads["W"]
                grads["b"+i] = self.layers[key].grads["b"]
            if "BatchNorm" in key:
                i = key.replace("BatchNorm", "")
                grads["gamma"+i] = self.layers[key].grads["gamma"]
                grads["beta"+i] = self.layers[key].grads["beta"]

        return grads, loss

        
class ConvNet(Model):
    def __init__(self, input_dims, n_channels=32, filter_size=5, pad=0, stride=1, hidden_size=128, output_size=10, weight_init_std=0.01, onehot=True):

        super(ConvNet, self).__init__()

        input_size = input_dims[1]
        conv_output_size = (input_size + 2 * pad - filter_size) // stride + 1
        pool_output_size = n_channels * (conv_output_size // 2) ** 2

        W_shapes = [(n_channels, input_dims[0], filter_size, filter_size), (pool_output_size, hidden_size), (hidden_size, output_size)]
        b_shapes = [n_channels, hidden_size, output_size]

        for i in range(len(W_shapes)):
            W = weight_init_std * np.random.randn(*W_shapes[i])
            b = np.zeros(b_shapes[i])
            self.params["W"+str(i)] = W
            self.params["b"+str(i)] = b

        self.layers = OrderedDict(
            Conv1=Conv2D(self.params["W0"], self.params["b0"], pad, stride),
            ReLU1=ReLU(),
            Pool1=Pooling(2, 2, stride=2),
            Affine1=Affine(self.params["W1"], self.params["b1"]),
            ReLU2=ReLU(),
            Affine2=Affine(self.params["W2"], self.params["b2"])
        )

        self.loss_layer = BinaryCategory(onehot=onehot)
        self.onehot = onehot

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.loss_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        if self.onehot:
            return np.mean(y.argmax(axis=1) == t.argmax(axis=1))
        else:
            y = np.where(y > 0.5, 1, 0)
            return np.mean(y == t)

    def grads(self, x, t):
        # forward
        loss = self.loss(x, t)

        # backward
        dout = 1
        dout = self.loss_layer.backward(dout)
        layers = list(self.layers.values())
        layers = reversed(layers)
        for layer in layers:
            dout = layer.backward(dout)

        grads = {
            "W0": self.layers["Conv1"].grads["W"],
            "b0": self.layers["Conv1"].grads["b"],
            "W1": self.layers["Affine1"].grads["W"],
            "b1": self.layers["Affine1"].grads["b"],
            "W2": self.layers["Affine2"].grads["W"],
            "b2": self.layers["Affine2"].grads["b"]
        }

        return grads, loss

