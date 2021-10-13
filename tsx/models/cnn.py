"""
Base class for all convolutional neural network based models for both forecasting and classification
"""

import numpy as np
import torch
import torch.nn as nn

from abc import abstractmethod

from tsx.utils import prepare_for_pytorch, to_numpy

class BaseCNN(nn.Module):

    def __init__(self):
        super(BaseCNN, self).__init__()

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def forward(self, X: torch.tensor, return_features: bool = False):
        ...

    def gradcam(self, X: np.ndarray, output_indices) -> np.ndarray:
        def _single_gradcam(logits, features):
            grads = to_numpy(torch.autograd.grad(logits, features)[0].squeeze())
            if len(grads.shape) == 1:
                grads = np.expand_dims(grads, 0)

            features = to_numpy(features.squeeze())

            w = np.mean(grads, axis=1)

            cam = np.zeros_like(features[0])
            for k in range(features.shape[0]):
                cam += w[k] * features[k]

            return np.maximum(cam, 0).squeeze()

        if isinstance(output_indices, int):
            output_indices = (np.ones((len(X))) * output_indices).astype(np.int8)

        cams = np.zeros_like(X)
        X = prepare_for_pytorch(X)

        for k in range(len(X)):
            x = X[k].unsqueeze(0)
            output, feats = self.forward(x, return_features=True)
            c = _single_gradcam(output.squeeze()[output_indices[k]], feats)
            cams[k] = c

        return cams

    