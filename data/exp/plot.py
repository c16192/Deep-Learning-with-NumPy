import matplotlib.pyplot as plt


def plot_data(x, t):
    plt.scatter(x[:, 0], x[:, 1], s=5, c=t.reshape(-1), cmap='Blues')
    plt.show()