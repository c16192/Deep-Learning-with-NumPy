import numpy as np
import os, sys, csv

from config import train_path, test_path
from plot import plot_data

train_data = np.loadtxt(train_path, delimiter=',', dtype='float32')
test_data = np.loadtxt(test_path, delimiter=',', dtype='float32')

if __name__ == "__main__":
    plot_data(train_data)