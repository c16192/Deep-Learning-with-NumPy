import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from .load import load_mnist
from .plot import plot_img

filename = os.path.join(os.path.dirname(__file__), "model_Adam.pkl")

with open(filename, "rb") as f:
    trainer = pickle.load(f)

losses = trainer.history["train_loss"]
train_acc = trainer.history["train_acc"]
test_acc = trainer.history["test_acc"]
plt.figure()
plt.plot(np.arange(len(losses)), losses)
plt.figure()
plt.plot(np.arange(len(train_acc)), train_acc, label="Training")
plt.hold(True)
plt.plot(np.arange(len(test_acc)), test_acc, label="Testing")
plt.legend()
plt.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, one_hot_label=True)

pred_labels = trainer.model.predict(x_test[200:210]).argmax(axis=1)
true_labels = t_test[200:210].argmax(axis=1)
print(pred_labels, true_labels)
plot_img(x_test[200, 0])
