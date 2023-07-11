import numpy as np
import torch
import skorch

from seedpy import fixedseed
from tsx.utils import to_random_state

class NeuralNetRegressor(skorch.NeuralNetRegressor):
    """ Regression wrapper for scikit-learn-like PyTorch training

    Args:
        module: A PyTorch model of type ``torch.nn.Module``
        random_state (optional): Seed training with either a fixed seed or a ``np.random.RandomState`` object. Defaults to None.
        max_epochs (optional): How long to train for. Defaults to 10.
        device (optional): Indicate where the model should be trained. If None, it chooses the fastest available option automatically. Defaults to None.
        lr (optional): Set learning rate. Defaults to 2e-3
        batch_size (optional): Training batch size. Defaults to 32.
        verbose (optional): Print status updates to the console. Defaults to false
        callbacks (optional): Skorch callback list used for training. Defaults to None.
        **kwargs: Optional keyword arguments for skorch.NeuralNetRegressor

    """
    def __init__(self, module, random_state=None, max_epochs=10, device=None, lr=2e-3, batch_size=32, verbose=False, callbacks=None, **kwargs):
        self.random_state = random_state
        self.verbose = verbose

        self.device = device
        if self.device is None:
            self.device = 'cpu'
            self.device = 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else self.device
            self.device = 'cuda' if torch.cuda.is_available() else self.device

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
        super().fit(X, y=y)

class NeuralNetClassifier(skorch.NeuralNetClassifier):
    """ Classification wrapper for scikit-learn-like PyTorch training

    Args:
        module: A PyTorch model of type ``torch.nn.Module``
        random_state (optional): Seed training with either a fixed seed or a ``np.random.RandomState`` object. Defaults to None.
        max_epochs (optional): How long to train for. Defaults to 10.
        device (optional): Indicate where the model should be trained. If None, it chooses the fastest available option automatically. Defaults to None.
        lr (optional): Set learning rate. Defaults to 2e-3
        batch_size (optional): Training batch size. Defaults to 32.
        verbose (optional): Print status updates to the console. Defaults to false
        callbacks (optional): Skorch callback list used for training. Defaults to None.
        **kwargs: Optional keyword arguments for skorch.NeuralNetClassifier

    """

    def __init__(self, module, random_state=None, max_epochs=10, device=None, lr=2e-3, batch_size=32, verbose=False, callbacks=None, **kwargs):
        self.random_state = random_state
        self.verbose = verbose

        self.device = device
        if self.device is None:
            self.device = 'cpu'
            self.device = 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else self.device
            self.device = 'cuda' if torch.cuda.is_available() else self.device

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
