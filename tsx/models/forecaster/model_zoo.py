import numpy as np
import torch
import torch.nn as nn

def get_linear(L, H):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(L, H),
    )

def get_fcn(L, H, n_hidden=32, depth=3):
    assert depth >= 2
    model = nn.ModuleList()
    model.append(nn.Flatten())
    for i in range(depth):
        if i == 0:
            model.append(nn.Linear(L, n_hidden))
            model.append(nn.ReLU())
        elif i == depth-1:
            model.append(nn.Linear(n_hidden, H))
        else:
            model.append(nn.Linear(n_hidden, n_hidden))
            model.append(nn.ReLU())

    return nn.Sequential(*model)

def get_1d_cnn(L, H, n_channels=1, depth_feature=2, depth_classification=2, n_hidden_neurons=32, n_hidden_channels=16):
    class CNN(nn.Module):
        
        def __init__(self):
            super().__init__()
            self.feature_extractor = nn.ModuleList()
            self.forecaster = nn.ModuleList()

            for i in range(depth_feature):
                if i == 0:
                    self.feature_extractor.append(nn.Conv1d(n_channels, n_hidden_channels, kernel_size=3, padding='same'))
                    self.feature_extractor.append(nn.BatchNorm1d(n_hidden_channels))
                    self.feature_extractor.append(nn.ReLU())
                else:
                    self.feature_extractor.append(nn.Conv1d(n_hidden_channels, n_hidden_channels, kernel_size=3, padding='same'))
                    self.feature_extractor.append(nn.BatchNorm1d(n_hidden_channels))
                    self.feature_extractor.append(nn.ReLU())

            self.forecaster.append(nn.Flatten())
            if depth_classification == 1:
                self.forecaster.append(nn.Linear(L * n_hidden_channels, H))
            else:
                for i in range(depth_classification):
                    if i == 0:
                        self.forecaster.append(nn.Linear(L * n_hidden_channels, n_hidden_neurons))
                        self.forecaster.append(nn.ReLU())
                    elif i == depth_classification-1:
                        self.forecaster.append(nn.Linear(n_hidden_neurons, H))
                    else:
                        self.forecaster.append(nn.Linear(n_hidden_neurons, n_hidden_neurons))
                        self.forecaster.append(nn.ReLU())

            self.feature_extractor = nn.Sequential(*self.feature_extractor)
            self.forecaster = nn.Sequential(*self.forecaster)

        def forward(self, X):
            feats = self.feature_extractor(X)
            out = self.forecaster(feats)
            return out
        
        def predict(self, X):
            is_numpy = isinstance(X, np.ndarray)
            if is_numpy:
                X = torch.from_numpy(X).float()

            self.eval()
            with torch.no_grad():
                preds = self.forward(X)

            if is_numpy:
                preds = preds.numpy()

            return preds
    
    return CNN()
