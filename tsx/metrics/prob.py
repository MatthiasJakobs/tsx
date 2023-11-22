import numpy as np
from scipy.stats import entropy as _entropy

def entropy(P, scale=True):
    _entr = _entropy(P)
    if scale:
        _entr /= np.log(len(P))

    return _entr

