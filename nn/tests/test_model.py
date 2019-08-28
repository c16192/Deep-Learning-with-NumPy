import unittest
from ..functions import *
from ..model import Vec2Class, ConvNet
import numpy as np


class TestModel(unittest.TestCase):
    
    def test_vec_2_class(self):
        x = np.random.random((10, 2))
        t = np.where(np.random.random((10, 1)) > 0.5, 1, 0)
        model = Vec2Class(dims=[2, 10, 1])
        grads_num = model.grad_numerical(x, t)[0]
        grads_bp = model.grads(x, t)[0]

        for key in grads_num:
            np.testing.assert_array_almost_equal(
                grads_num[key], grads_bp[key], decimal=5
            )

    def test_vec_2_class_with_batch(self):
        x = np.random.random((10, 2))
        t = np.where(np.random.random((10, 1)) > 0.5, 1, 0)
        model = Vec2Class(dims=[2, 10, 1], batch_norm=True)
        grads_num = model.grad_numerical(x, t)[0]
        grads_bp = model.grads(x, t)[0]

        for key in grads_num:
            np.testing.assert_array_almost_equal(
                grads_num[key], grads_bp[key], decimal=5
            )

    def test_convnet(self):
        input_shape = (1, 6, 6)
        x = np.random.random((2,) + input_shape)
        t = np.zeros((2, 3))
        t_ = np.random.random((2, 3)).argmax(axis=1)
        t[np.arange(2), t_] = 1.0
        model = ConvNet(input_shape, hidden_size=16, output_size=3)
        grads_num = model.grad_numerical(x, t)[0]
        grads_bp = model.grads(x, t)[0]

        for key in grads_num:
            np.testing.assert_array_almost_equal(
                grads_num[key], grads_bp[key], decimal=5
            )
