import unittest
from ..functions import *
import numpy as np


class TestFunctions(unittest.TestCase):
    
    def test_sigmoid(self):
        np.testing.assert_array_almost_equal(
            sigmoid(np.array([[-np.inf, 0], [np.inf, 1]])), 
            np.array([[0.0, 0.5], [1.0, 0.73105857863]])
        )
    
    def test_softmax(self):
        y = softmax(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        assert y.sum() == 1.0
    
        x = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [-np.inf, 0.0, 3.0, 3000, 400]])
        y = softmax(x)
        np.testing.assert_array_equal(y.sum(axis=1), [1.0, 1.0])
    
    def test_grad_numerical(self):
        f = lambda x: (x ** 2).sum()
        t = np.array([-3.0, 4.0])
        grad = grad_numerical(f, t)
        np.testing.assert_array_almost_equal(grad, [-6.0, 8.0])
    
    def test_grads_numerical(self):
        params = {
            "x": np.array(-3.0),
            "y": np.array(4.0)
        }
        f = lambda x: params["x"] ** 2 + params["y"] ** 2
        grads = grads_numerical(f, params)
        np.testing.assert_almost_equal(grads["x"], -6.0)
        np.testing.assert_almost_equal(grads["y"], 8.0)

    def test_im2col(self):
        x1 = np.random.random((1, 3, 7, 7))
        col1 = im2col(x1, 5, 5, stride=1, pad=0)
        assert col1.shape == (9, 75)

        x2 = np.random.random((10, 3, 7, 7))
        col2 = im2col(x2, 5, 5, stride=1, pad=0)
        assert col2.shape == (90, 75)