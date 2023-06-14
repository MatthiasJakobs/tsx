import numpy as np
import torch

# ----- transforms for entire dataset -----

# mean centered, std=1
def normalize(X):
    if isinstance(X[0], type(np.zeros(1))):
        return ((X.T - np.mean(X, axis=-1)) / np.std(X, axis=-1)).T
    if isinstance(X[0], type(torch.zeros(1))):
        return ((X.T - torch.mean(X, axis=-1)) / torch.std(X, axis=-1)).T

def split_horizon(x, H, L=None):
    assert len(x.shape) == 1
    assert len(x) > H

    if L is None:
        L = 0

    return x[:-(L+H)], x[-(L+H):]

def windowing(x, L, z=1, H=1, use_torch=False):
    univariate = len(x.shape) == 1

    if univariate:
        x = x.reshape(-1, 1)

    assert len(x.shape) == 2
    n_features = x.shape[-1]

    X = []
    y = []

    if isinstance(x, torch.Tensor):
        x = x.numpy()

    if L + H - z >= len(x):
        raise RuntimeError(f'cannot window sequence of length {len(x)} with L={L}, H={H}, z={z}')

    for i in range(0, len(x)-H-L+1, z):
        X.append(x[i:(i+L)].reshape(1, -1, n_features))
        y.append(x[(i+L):(i+L+H)])

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    if X.shape[-1] == 1 and y.shape[-1] == 1:
        X = X.squeeze()
        y = y.squeeze()

    if use_torch:
        return torch.from_numpy(X).float(), torch.from_numpy(y)

    return X, y
