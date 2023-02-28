import numpy as np
import torch

# ----- transforms for entire dataset -----

# mean centered, std=1
def normalize(X):
    if isinstance(X[0], type(np.zeros(1))):
        return ((X.T - np.mean(X, axis=-1)) / np.std(X, axis=-1)).T
    if isinstance(X[0], type(torch.zeros(1))):
        return ((X.T - torch.mean(X, axis=-1)) / torch.std(X, axis=-1)).T

def windowing(x, lag, z=1, H=1, use_torch=False):
    assert len(x.shape) == 1

    X = []
    y = []

    if lag + H - z >= len(x):
        raise RuntimeError(f'cannot window sequence of length {len(x)} with L={lag}, H={H}, z={z}')

    for i in range(0, len(x)-H-lag+1, z):
        X.append(x[i:(i+lag)].reshape(1, -1))
        y.append(x[(i+lag):(i+lag+H)])

    X = np.concatenate(X, axis=0)
    y = np.array(y)

    if use_torch:
        return torch.from_numpy(X).float(), torch.from_numpy(y)

    return X, y
