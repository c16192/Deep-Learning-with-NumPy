from .load import load_data
from ...nn.model import Vec2Class
from ...nn.optimizer import *
from ...nn.trainer import Trainer
import pickle, os

x_train, t_train, x_test, t_test = load_data()
# plot_data(x_train, t_train)

optimizers = {
    # "SGD": SGD(),
    # "Momentum": Momentum(),
    # "AdaGrad": AdaGrad(),
    "Adam": Adam()
}

for name in optimizers:
    model = Vec2Class(dims=[2, 32, 32, 32, 32, 32, 1], batch_norm=True)
    optimizer = optimizers[name]
    trainer = Trainer(model, optimizer)
    trainer.fit(x_train, t_train, x_test, t_test, n_iters=10000)

    fname = os.path.join(os.path.dirname(__file__), "model_%s.pkl" % name)

    with open(fname, "wb") as f:
        pickle.dump(trainer, f)
