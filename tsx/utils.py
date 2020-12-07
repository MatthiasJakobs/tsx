import torch
import pandas as pd
import numpy as np

def to_numpy(x):
    if isinstance(x, type(torch.zeros(1))):
        if x.requires_grad:
            return x.detach().numpy()
        else:
            return x.numpy()
    if isinstance(x, type(pd.Series(data=[1,2]))):
        return x.to_numpy()
    if isinstance(x, type(np.zeros(1))):
        return x

    raise ValueError("Input of type {} cannot be converted to numpy array".format(type(x)))