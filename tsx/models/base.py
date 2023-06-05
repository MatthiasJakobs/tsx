import numpy as np
import torch
import skorch

from seedpy import fixedseed

class NeuralNetRegressor(skorch.NeuralNetRegressor):

    def __init__(self, model, random_state=None, max_epochs=10, device=None, lr=2e-3, batch_size=32, verbose=False, callbacks=None):
        self.random_state = random_state
        self.verbose = verbose

        self.device = device
        if self.device is None:
            self.device = 'cpu'
            self.device = 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else self.device
            self.device = 'cuda' if torch.cuda.is_available() else self.device

        super().__init__(
            model, 
            max_epochs=max_epochs, 
            device=self.device,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )

    def fit(self, X, y):
        if self.verbose:
            print('Run training on', self.device)
        with fixedseed([torch, np], seed=self.random_state):
            super().fit(X, y)

class NeuralNetClassifier(skorch.NeuralNetClassifier):

    def __init__(self, model, random_state=None, max_epochs=10, device=None, lr=2e-3, batch_size=32, verbose=False, callbacks=None, **kwargs):
        self.random_state = random_state
        self.verbose = verbose

        self.device = device
        if self.device is None:
            self.device = 'cpu'
            self.device = 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else self.device
            self.device = 'cuda' if torch.cuda.is_available() else self.device

        super().__init__(
            model, 
            max_epochs=max_epochs, 
            device=self.device,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            **kwargs
        )

    def fit(self, X, y):
        if self.verbose:
            print('Run training on', self.device)
        with fixedseed([torch, np], seed=self.random_state):
            super().fit(X, y)
