import matplotlib.pyplot as plt

def plot_data(data):
    x, y, c = data.T
    plt.scatter(x, y, s=5, c=c, cmap='Blues')
    plt.show()