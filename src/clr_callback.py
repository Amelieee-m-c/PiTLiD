"""
Cyclical Learning Rate callback for Keras (triangular / triangular2 / exp_range).
Standard implementation, after Bradley Kenstler's public-domain `clr_callback.py`
(https://github.com/bckenstler/CLR), used by the original PiTLiD repo
(`training approach/tlft.py`: `from clr_callback import *`).

Reference: Smith, L.N., "Cyclical Learning Rates for Training Neural Networks", 2017.
"""
import numpy as np
from tensorflow import keras


class CyclicLR(keras.callbacks.Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000.0,
                 mode="triangular2", gamma=1.0, scale_fn=None, scale_mode="cycle"):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == "triangular":
                self.scale_fn = lambda x: 1.0
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = lambda x: 1.0 / (2.0 ** (x - 1))
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.clr_iterations = 0.0
        self.trn_iterations = 0.0
        self.history = {}

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == "cycle":
            return self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        if self.clr_iterations == 0:
            self.model.optimizer.learning_rate.assign(self.base_lr)
        else:
            self.model.optimizer.learning_rate.assign(self.clr())

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault("lr", []).append(float(self.model.optimizer.learning_rate.numpy()))
        self.history.setdefault("iterations", []).append(self.trn_iterations)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.model.optimizer.learning_rate.assign(self.clr())
