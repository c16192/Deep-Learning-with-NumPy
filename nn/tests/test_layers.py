import unittest
from ..layers import *
from ..functions import grad_numerical
import numpy as np


def test_layers(layer, in_shape, out_shape, t=None, n_trials=1, type="default"):
    for i in range(n_trials):
        x = np.random.random(in_shape)
        rand = np.random.random(out_shape)

        if t is None:
            layer.forward(x)
        else:
            layer.forward(x, t)
        dout = rand.copy()
        dLdx = layer.backward(dout)

        f = lambda w: np.sum(rand * (layer.forward(x) if t is None else layer.forward(x, t)))
        if type == "dropout":
            f = lambda w: np.sum(rand * (layer.forward(x, renew_mask=False)))
        dLdx_num = grad_numerical(f, x)

        np.testing.assert_array_almost_equal(
            dLdx_num, dLdx
        )

        for key in layer.grads:
            dLdParam_num = grad_numerical(f, layer.params[key])
            dLdParam = layer.grads[key]

            np.testing.assert_array_almost_equal(
                dLdParam_num, dLdParam
            )


class TestLayers(unittest.TestCase):
    
    def test_affine(self):
        batch_size = 16
        dim1 = 10
        dim2 = 8
        in_shape = (batch_size, dim1)
        out_shape = (batch_size, dim2)
        W = 1 / np.sqrt(dim1) * np.random.randn(dim1, dim2)
        b = np.zeros(dim2)
        layer = Affine(W, b)

        test_layers(layer, in_shape, out_shape)

    def test_sigmoid(self):
        layer = Sigmoid()
        shape = (10, 3)
        test_layers(layer, shape, shape)

    def test_relu(self):
        layer = ReLU()
        shape = (10, 3)
        test_layers(layer, shape, shape)

    def test_categorical(self):
        layer = BinaryCategory(onehot=True)
        in_shape = (10, 3)
        out_shape = (1, )
        t = np.zeros(in_shape)
        t_ = np.random.random(in_shape)
        t[np.arange(t.shape[0]), t_.argmax(axis=1)] = 1.0
        test_layers(layer, in_shape, out_shape, t)

    def test_binary(self):
        layer = BinaryCategory(onehot=False)
        in_shape = (10, 3)
        out_shape = (1, )
        t = np.where(np.random.random(in_shape) > 0.5, 1.0, 0.0)
        test_layers(layer, in_shape, out_shape, t)

    def test_batch_norm(self):
        N = 10
        D = 5
        gamma = np.ones((D,))
        beta = np.zeros((D,))
        layer = BatchNorm(gamma, beta)
        shape = (N, D)
        test_layers(layer, shape, shape)

    def test_dropout(self):
        layer = Dropout()
        shape = (10, 5)
        test_layers(layer, shape, shape, type="dropout")

    def test_conv2d(self):
        N, C, H, W = (2, 3, 8, 8)
        FN, C, FH, FW = (4, 3, 5, 5)
        in_shape = (N, C, H, W)
        out_shape = (2, 4, 4, 4)
        W = np.random.randn(FN, C, FH, FW)
        b = np.zeros(FN)
        layer = Conv2D(W, b)

        test_layers(layer, in_shape, out_shape)

    def test_pooling(self):
        in_shape = (2, 3, 8, 8)
        out_shape = (2, 3, 4, 4)
        layer = Pooling(2, 2, stride=2)

        test_layers(layer, in_shape, out_shape)

