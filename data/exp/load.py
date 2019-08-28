import numpy as np
import os, sys, csv

from .config import train_path, test_path
from .plot import plot_data


def load_data():
    train_data = np.loadtxt(train_path, delimiter=',', dtype='float32')
    test_data = np.loadtxt(test_path, delimiter=',', dtype='float32')
    return train_data[:, 0:2], train_data[:, 2:], test_data[:, 0:2], test_data[:, 2:]


if __name__ == "__main__":
    x_train, t_train, x_test, t_test = load_data()
    print(t_train.mean(), t_test.mean())
    plot_data(x_train, t_train)