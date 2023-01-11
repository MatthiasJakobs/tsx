import torch
import numpy as np

# l1 (Manhattan) norm
def manhattan(s, t):
    return np.linalg.norm(s-t, 1)

# l2 (euclidean) norm
def euclidean(s, t):
    assert len(s) == len(t), "Inputs to euclidean need to be of same length"
    if isinstance(s, torch.Tensor):
        s = s.numpy()
    if isinstance(t, torch.Tensor):
        t = t.numpy()

    return np.sum((s-t)**2)
# linf norm
def linf(s, t):
    return np.linalg.norm(s-t, np.inf)