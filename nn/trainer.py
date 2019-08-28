import numpy as np
from ..data.exp.load import load_data
from ..data.exp.plot import plot_data
from .model import Vec2Class
from .optimizer import SGD
import matplotlib.pyplot as plt
import pickle


class Trainer(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.history = {}
        self.reset()

    def reset(self):
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_acc": []
        }

    def step(self, x, t):
        grads, loss = self.model.grads(x, t)
        self.optimizer.update(self.model.params, grads)
        return loss

    def fit(self, x_train, t_train, x_test, t_test, batch_size=128, n_iters=100000):
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size / batch_size, 1)

        for i in range(n_iters):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            loss = self.step(x_batch, t_batch)
            self.history["train_loss"].append(loss)

            if i % iter_per_epoch < 1.0:
                train_acc = self.model.accuracy(x_train, t_train)
                test_acc = self.model.accuracy(x_test, t_test)
                self.history["train_acc"].append(train_acc)
                self.history["test_acc"].append(test_acc)
                print("Epoch %d: train acc: %f, test acc: %f" % (i // iter_per_epoch, train_acc, test_acc))


if __name__ == "__main__":
    x_train, t_train, x_test, t_test = load_data()
    # plot_data(x_train, t_train)

    model = Vec2Class(dims=[2, 10, 1])
    optimizer = SGD()

    trainer = Trainer(model, optimizer)
    trainer.fit(x_train, t_train, x_test, t_test)
