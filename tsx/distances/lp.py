import torch
import numpy as np

from typing import Union

# l1 (Manhattan) norm
def manhattan(s: Union[np.ndarray, torch.Tensor], t: Union[np.ndarray, torch.Tensor]):
    ''' 
    
    Args:
        s: First input
        t: Second input

    Returns:
        Calculated distance (float)

    '''
    if isinstance(s, torch.Tensor):
        s = s.numpy()
    if isinstance(t, torch.Tensor):
        t = t.numpy()
    return np.linalg.norm(s-t, 1)

# l2 (euclidean) norm
def euclidean(s: Union[np.ndarray, torch.Tensor], t: Union[np.ndarray, torch.Tensor]):
    ''' 
    
    Args:
        s: First input
        t: Second input

    Returns:
        Calculated distance (float)

    '''
    assert len(s) == len(t), "Inputs to euclidean need to be of same length"
    if isinstance(s, torch.Tensor):
        s = s.numpy()
    if isinstance(t, torch.Tensor):
        t = t.numpy()

    return np.sum((s-t)**2)
# linf norm
def linf(s: Union[np.ndarray, torch.Tensor], t: Union[np.ndarray, torch.Tensor]):
    ''' 
    
    Args:
        s: First input
        t: Second input

    Returns:
        Calculated distance (float)

    '''
    if isinstance(s, torch.Tensor):
        s = s.numpy()
    if isinstance(t, torch.Tensor):
        t = t.numpy()
    return np.linalg.norm(s-t, np.inf)