import numpy as np
import torch
import tempfile
import zipfile

from urllib.request import urlretrieve
from os.path import join, basename, dirname
from shutil import rmtree as remove_dir
from typing import Union

def download_and_unzip(url: str, name: str) -> str:
    """
    downloads a given dataset url and unzips it
    :param name: name of the dataset
    :return: path to the folder in which the dataset was saved
    """
    path = join(dirname(__file__), "data", name)
    dl_dir = tempfile.mkdtemp()
    zip_file_name = join(dl_dir, basename(url))
    urlretrieve(url, zip_file_name)

    zipfile.ZipFile(zip_file_name, "r").extractall(path)
    remove_dir(dl_dir)
    return path

# ----- transforms for entire dataset -----

# mean centered, std=1
def normalize(X):
    if isinstance(X[0], type(np.zeros(1))):
        return ((X.T - np.mean(X, axis=-1)) / np.std(X, axis=-1)).T
    if isinstance(X[0], type(torch.zeros(1))):
        return ((X.T - torch.mean(X, axis=-1)) / torch.std(X, axis=-1)).T

def split_horizon(x: Union[np.ndarray, torch.Tensor], H: int, L: Union[None, int] = None):
    ''' Split a time series into two parts, given a forecasting horizon

    Args:
        x: Input time series
        H: Forecast horizon
        L (optional): Amount of lag to use

    Returns:
        Two arrays (type depends on the type of `x`), the first one corresponding to everything before `H`

    '''
    assert len(x.shape) == 1
    assert len(x) > H

    if L is None:
        L = 0

    return x[:-(L+H)], x[-(L+H):]

def windowing(x: Union[np.ndarray, torch.Tensor], L: int, z: int = 1, H: int = 1, use_torch: bool = False):
    ''' Create sliding windows from input `x`

    Args:
        x: Input time series
        L: Amount of lag to use
        H: Forecast horizon
        z: Step length
        use_torch: Whether to return `np.ndarray` or `torch.Tensor`

    Returns:
        Windowed `X` and `y`, either as a Numpy array or PyTorch tensor

    '''
    univariate = len(x.shape) == 1

    if univariate:
        x = x.reshape(-1, 1)

    assert len(x.shape) == 2
    n_features = x.shape[-1]

    X = []
    y = []

    if isinstance(x, torch.Tensor):
        x = x.numpy()

    if L + H - z >= len(x):
        raise RuntimeError(f'cannot window sequence of length {len(x)} with L={L}, H={H}, z={z}')

    for i in range(0, len(x)-H-L+1, z):
        X.append(x[i:(i+L)].reshape(1, -1, n_features))
        y.append(x[(i+L):(i+L+H)])

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    if X.shape[-1] == 1 and y.shape[-1] == 1:
        X = X.squeeze()
        y = y.squeeze()

    if use_torch:
        return torch.from_numpy(X).float(), torch.from_numpy(y)

    return X, y

def split_proportion(X, proportions):
    assert len(proportions) >= 2
    assert sum(proportions) == 1

    n = len(X)

    l = int(proportions[0] * n)
    splits = [X[0:l]]
    for prop in proportions[1:]:
        l = int(prop * n)
        start = sum([len(x) for x in splits])
        splits.append(X[start:(start+l)])

    return splits

