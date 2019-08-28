from .load import load_mnist
from ...nn.model import ConvNet
from ...nn.optimizer import *
from ...nn.trainer import Trainer
import pickle, os

optimizers = {
    # "SGD": SGD(),
    # "Momentum": Momentum(),
    # "AdaGrad": AdaGrad(),
    "Adam": Adam()
}

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, one_hot_label=True)

x_train, t_train = x_train[:1000], t_train[:1000]
x_test, t_test = x_test[:500], t_test[:500]

print(x_train.shape)

for name in optimizers:
    model = ConvNet((1, 28, 28))
    optimizer = optimizers[name]
    trainer = Trainer(model, optimizer)
    print("Start training...")
    trainer.fit(x_train, t_train, x_test, t_test, n_iters=160)

    fname = os.path.join(os.path.dirname(__file__), "model_%s.pkl" % name)

    with open(fname, "wb") as f:
        pickle.dump(trainer, f)
