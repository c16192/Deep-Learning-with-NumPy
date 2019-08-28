import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from .generate import generate_data
from .plot import plot_data

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

new_data = generate_data(2000)

pred_labels = trainer.model.predict(new_data[:, 0:2])
plt.figure()
plot_data(new_data[:, 0:2], pred_labels)

plt.figure()
plot_data(new_data[:, 0:2], new_data[:, 2:])