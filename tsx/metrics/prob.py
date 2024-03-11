import numpy as np
from scipy.stats import entropy as _entropy

def entropy(P, scale=True):
    ''' Compute (scaled) entropy

    Args:
        P: Input to entropy
        scale: If True, scale entropy to be in `[0,1]` (default: True)

    Returns:
        (Scaled) entropy

    '''
    _entr = _entropy(P)
    if scale:
        _entr /= np.log(len(P))

    return _entr

