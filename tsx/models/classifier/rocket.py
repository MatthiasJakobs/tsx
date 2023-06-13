import torch
import pickle
import torch.nn as nn
import numpy as np

from os.path import join

from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score
from tsx.models import NeuralNetClassifier

#self.classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))

class ROCKETTransform:

    def __init__(self, L, k=1_000, n_channels=1, ppv_only=False, use_sigmoid=False):
        self.L = L
        self.k = k
        self.n_channels = n_channels
        self.ppv_only = ppv_only
        self.use_sigmoid = use_sigmoid

    def build_kernels(self):
        self.kernels = []
        grouped_kernels = np.zeros((self.k, 3), dtype=int)
        kernel_weights = []
        kernel_biases = []
        for i in range(self.k):

            # Weights
            kernel_length = [7, 9, 11][np.random.randint(0, 3)]
            weights = torch.normal(torch.zeros(1,self.n_channels,kernel_length), 1)
            weights = weights - torch.mean(weights)
            bias = torch.rand(1)
            bias = (-1 - 1) * bias + 1
            grouped_kernels[i][0] = kernel_length

            # Dilation
            A = np.log2((self.L-1) / (float(kernel_length)-1))
            dilation = torch.floor(2**(torch.rand(1)*A)).long().item()
            grouped_kernels[i][2] = dilation

            # Padding
            padding = 0 if torch.rand(1)>0.5 else 1
            grouped_kernels[i][1] = padding

            kernel_weights.append(weights)
            kernel_biases.append(bias)

        unique_configs = np.unique(grouped_kernels, axis=0)
        for u in unique_configs:
            indices = np.prod(np.equal(grouped_kernels, u), axis=1)
            kernel = nn.Conv1d(self.n_channels, np.sum(indices), kernel_size=u[0], stride=1, padding=u[1], dilation=u[2], bias=True)
            indices = indices.nonzero()[0]
            kernel.weight = nn.Parameter(torch.cat([kernel_weights[i] for i in indices], axis=0), requires_grad=False)
            kernel.bias = nn.Parameter(torch.cat([kernel_biases[i] for i in indices], axis=0), requires_grad=False)
            kernel.require_grad = False
            self.kernels.append(kernel)

    def _ppv(self, x):
        # use sigmoid as a soft approximation for ">" activation
        if self.use_sigmoid:
            return torch.mean(torch.sigmoid(x), dim=-1)
        return torch.mean((x > 0).float(), dim=-1)

    def transform(self, X):
        if isinstance(X, type(np.zeros(1))):
            X = torch.from_numpy(X).float()

        return self._apply_kernels(X)

    def _apply_kernels(self, X):
        features_ppv = []
        features_max = []
        with torch.no_grad():
            for k in self.kernels:
                if len(X.shape) == 2:
                    X = X.unsqueeze(1) # missing channel information

                transformed_data = k(X)

                features_ppv.append(self._ppv(transformed_data))
                if not self.ppv_only:
                    features_max.append(torch.max(transformed_data, dim=-1)[0])

            features_ppv = torch.cat(features_ppv, -1)
            if self.ppv_only:
                return features_ppv
            else:
                features_max = torch.cat(features_max, -1)
                return torch.cat((features_ppv, features_max), -1)
    
