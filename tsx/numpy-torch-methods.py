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

#independent method for unsqueeze
def unsqueeze(X, axis):
    if isinstance(X, torch.Tensor):
        return torch.unsqueeze(X, axis)
    return np.expand_dims(X, axis)

#independent method for sqrt
def sqrt(X, out=None, where=True, **kwargs):
    if isinstance(X, torch.Tensor):
        return torch.sqrt(X, out=out)
    return np.sqrt(X, out=out, where=where, **kwargs)

#independent method for log
def log(X, out=None, where=True, **kwargs):
    if isinstance(X, torch.Tensor):
        return torch.log(X, out=out)
    return np.log(X, out=out, where=where, **kwargs)

#independent method for zeros
def zeros(shape, framework='numpy', dtype=None, order='C', like=np.zeros(0), out=None, layout=torch.strided, device=None, requires_grad=False):
    if framework == 'torch':
        return torch.zeros(shape, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    if isinstance(dtype, type(None)): # np.zeros takes default dtype float not None
        dtype = float
    return np.zeros(shape, dtype=dtype, order=order, like=like)

#independent method for zeros_like
def zeros_like(data,*args, framework='numpy', dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format, order='K', subok=True, newshape=None):
    if framework == 'torch':
        return torch.zeros_like(data, *args, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)
    return np.zeros_like(data, dtype=dtype, order=order, subok=subok, shape=newshape)

#independent method for ones
def ones(shape, framework='numpy', dtype=None, order='C', like=np.zeros(0), out=None, layout=torch.strided, device=None, requires_grad=False):
    if framework == 'torch':
        return torch.ones(shape, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    return np.ones(shape, dtype=dtype, order=order, like=like) # np.ones takes default dtype None

#independent method for ones_like
def ones_like(data, framework='numpy', dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format, order='K', subok=True, newshape=None):
    if framework == 'torch':
        return torch.ones_like(data, dtype=dtype, device=device, layout=layout, requires_grad=requires_grad, memory_format=memory_format)
    return np.ones_like(data, dtype=dtype, order=order, subok=subok, shape=newshape)