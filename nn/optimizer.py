import numpy as np


class Optimizer(object):
    def __init__(self):
        pass

    def reset(self):
        pass

    def update(self, params, grads):
        pass


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super(SGD, self).__init__()
        self.lr = lr

    def update(self, params, grads):
        for key in grads:
            params[key] -= self.lr * grads[key]


class Momentum(Optimizer):
    def __init__(self, lr=0.01, eta=0.9):
        super(Momentum, self).__init__()
        self.lr = lr
        self.eta = eta
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in grads:
            self.v[key] = self.eta * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in grads:
            self.h[key] += grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop(Optimizer):
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in grads:
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam(Optimizer):
    def __init__(self, alpha=0.01, beta1=0.9, beta2=0.99):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.moment1 = None
        self.moment2 = None

    def update(self, params, grads):
        if self.moment1 is None or self.moment2 is None:
            self.moment1, self.moment2 = {}, {}
            for key, val in params.items():
                self.moment1[key] = np.zeros_like(val)
                self.moment2[key] = np.zeros_like(val)

        self.t += 1
        lr_t = self.alpha * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)

        for key in grads:
            self.moment1[key] = self.beta1 * self.moment1[key] + (1.0 - self.beta1) * grads[key]
            self.moment2[key] = self.beta2 * self.moment2[key] + (1.0 - self.beta2) * grads[key] ** 2
            params[key] -= lr_t * self.moment1[key] / (np.sqrt(self.moment2[key]) + 1e-7)
