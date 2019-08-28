import csv, sys
import numpy as np

from .plot import plot_data
from .config import train_path, test_path

def func(X):
    return np.cos(X[:,0]**2 + X[:,1]**2) > 0

def generate_data(n, max=3, min=-3):
    X = np.random.random((n, 2)) * (max-min) + min
    y = func(X)
    y = y.reshape(-1,1)
    data = np.concatenate((X, y), axis=1)
    return data

train_data = generate_data(8000)
test_data = generate_data(2000)

np.savetxt(train_path, train_data, delimiter=',')
np.savetxt(test_path, test_data, delimiter=',')

if __name__ == "__main__":
    plot_data(train_data[:, 0:2], train_data[:, 2])