import torch
import numpy as np
from fastdtw import fastdtw

from typing import Union

def dtw(s: Union[np.ndarray, torch.Tensor], t: Union[np.ndarray, torch.Tensor]):
    ''' Dynamic Time Warping from `fastdtw` package
    
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
    return fastdtw(s, t)[0]
