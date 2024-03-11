import torch
import numpy as np

def smape(a, b, axis=None):
    if isinstance(a, type(torch.zeros(0))) and isinstance(b, type(torch.zeros(0))):
        fn_abs = torch.abs
        fn_mean = torch.mean
    elif isinstance(a, type(np.zeros(0))) and isinstance(b, type(np.zeros(0))):
        fn_abs = np.abs
        fn_mean = np.mean
    else:
        raise NotImplementedError("Only supports both inputs to be torch tensors or numpy arrays")

    nom = fn_abs(a - b)
    denom = fn_abs(a) + fn_abs(b)
    if axis is not None:
        if len(a.shape) == 1:
            return nom / denom
        return fn_mean(nom / denom, axis=axis)
    return fn_mean(nom / denom)

# TODO: Make pytorch-independent
def mae(a, b):
    return torch.mean(torch.abs(a - b))

def mse(a, b):
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    return torch.mean((a-b)**2, axis=0)

def mase(y_pred, y_true, X):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    nom = np.mean(np.abs(y_pred-y_true))
    denom = np.sum(np.abs(X[1:]-X[:-1])) / (len(X)-1)
    if denom == 0:
        denom = 1e-5

    return nom / denom
