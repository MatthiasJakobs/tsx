import numpy as np
import torch

# independent method for mean
def mean(X, axis=None, dtype=None, out=None, keepdims=False, where=[True]): #where is broadcasted to match dimensions of X
    if isinstance(X, torch.Tensor):
        return torch.mean(X, dim=axis, keepdim=keepdims, dtype=dtype, out=out)
    return np.mean(X, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where) #X needs to be arraylike, everything is accepted

#independent method for sum
def sum(X, axis=None, dtype=None, out=None, keepdims=False, initial=[0], where=[True]): #where and initial are broadcasted to match dimensions of X
    if isinstance(X, torch.Tensor):
        return torch.sum(X, dim=axis, keepdim=keepdims, dtype=dtype)
    return np.sum(X, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where) #X needs to be arraylike, everything is accepted

#independent method for where
def where(condition, X=None, Y=None):
    if isinstance(condition, torch.BoolTensor):
        if isinstance(X, type(None)) and isinstance(Y, type(None)): # where(condition) is allowed for torch, using nonzero() instead than is recommended
            return torch.nonzero(condition, as_tuple=True)
        if (isinstance(X, torch.Tensor) or np.isscalar(X)) and (isinstance(X, torch.Tensor) or np.isscalar(Y)): # X and Y can also be scalar for torch method
            return torch.where(condition, X, Y)
        raise Exception("X and Y have to be both None or a combination of tensor and scalar")
        
    if isinstance(X, type(None)) and isinstance(Y, type(None)): # where(condition) is allowed for np, using nonzero() instead than is recommended
        return np.asarray(condition).nonzero()
    return np.where(condition, X, Y) #Condition, X and Y need to be arraylike, everything is accepted

#independent method for argsort
def argsort(X, axis=-1, kind=None, order=None, descending=False):
    if isinstance(X, torch.Tensor):
        return torch.argsort(X, dim=axis, descending=descending)
    return np.argsort(X, axis=axis, kind=kind, order=order) #X needs to be arraylike, everything is accepted


#TODO unsqueeze, zeros, zeros_like, one, one_like, sqrt, log, ...