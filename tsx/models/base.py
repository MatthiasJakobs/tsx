import numpy as np
import torch
import torch.nn as nn
import skorch

from seedpy import fixedseed
from typing import Union
from tsx.utils import get_device

class NeuralNetRegressor(skorch.NeuralNetRegressor):
    """ Regression wrapper for scikit-learn-like PyTorch training

    Args:
        module: A PyTorch model of type ``torch.nn.Module``
        random_state (optional): Seed training with either a fixed seed or None. Defaults to None.
        max_epochs (optional): How long to train for. Defaults to 10.
        device (optional): Indicate where the model should be trained. If None, it chooses the fastest available option automatically. Defaults to None.
        lr (optional): Set learning rate. Defaults to 2e-3
        batch_size (optional): Training batch size. Defaults to 32.
        verbose (optional): Print status updates to the console. Defaults to false
        callbacks (optional): Skorch callback list used for training. Defaults to None.
        **kwargs: Optional keyword arguments for skorch.NeuralNetRegressor

    """
    def __init__(self, module: nn.Module, random_state: Union[None, int] = None, max_epochs: int = 10, device: str = None, lr: float = 2e-3, batch_size: int = 32, verbose: bool = False, callbacks: skorch.callbacks.Callback = None, **kwargs):
        self.random_state = random_state
        self.verbose = verbose

        self.device = device
        if self.device is None:
            self.device = get_device()

        super().__init__(
            module, 
            max_epochs=max_epochs, 
            device=self.device,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            **kwargs
        )

    def fit(self, X, y=None):
        if self.verbose:
            print('Run training on', self.device)
        with fixedseed([torch, np], seed=self.random_state):
            super().fit(X, y)

    def __call__(self, X):
        if not self.initialized_:
            self.initialize()
        return self.forward(X)

class NeuralNetClassifier(skorch.NeuralNetClassifier):
    """ Classification wrapper for scikit-learn-like PyTorch training

    Args:
        module: A PyTorch model of type ``torch.nn.Module``
        random_state (optional): Seed training with either a fixed seed or None. Defaults to None.
        max_epochs (optional): How long to train for. Defaults to 10.
        device (optional): Indicate where the model should be trained. If None, it chooses the fastest available option automatically. Defaults to None.
        lr (optional): Set learning rate. Defaults to 2e-3
        batch_size (optional): Training batch size. Defaults to 32.
        verbose (optional): Print status updates to the console. Defaults to false
        callbacks (optional): Skorch callback list used for training. Defaults to None.
        **kwargs: Optional keyword arguments for skorch.NeuralNetClassifier

    """

    def __init__(self, module: nn.Module, random_state: Union[None, int] = None, max_epochs: int = 10, device: str = None, lr: float = 2e-3, batch_size: int = 32, verbose: bool = False, callbacks: skorch.callbacks.Callback = None, **kwargs):
        self.random_state = random_state
        self.verbose = verbose

        self.device = device
        if self.device is None:
            self.device = get_device()

        super().__init__(
            module, 
            max_epochs=max_epochs, 
            device=self.device,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            **kwargs
        )

    def fit(self, X, y=None):
        if self.verbose:
            print('Run training on', self.device)
        with fixedseed([torch, np], seed=self.random_state):
            super().fit(X, y=y)

class TSValidSplit:

    def __init__(self, valid_percent=0.3):
        self.valid_percent = valid_percent

    def __call__(self, dataset, y, **fit_params):
        n_batches = dataset.X.shape[0]
        n_valid = int(n_batches * self.valid_percent)
        n_train = n_batches - n_valid

        train_ds = skorch.dataset.Dataset(dataset.X[:n_train], dataset.y[:n_train])
        val_ds = skorch.dataset.Dataset(dataset.X[n_train:], dataset.y[n_train:])
        return train_ds, val_ds

