import torch
import numpy as np

from typing import List
from tsx.model_selection import ROC_Member

def find_closest_rocs(x: torch.Tensor, rocs: List[List[torch.Tensor]], dist_fn: callable):
    ''' Given an input `x` and RoCs `rocs` return the closest RoC member for each model w.r.t. `dist_fn`

    Args:
        x: Input time series window
        rocs: List of Regions of Competences
        dist_fn: Distance function applicable for two time series windows

    Returns:
        A list of model indices and a list of correpsonding closest `ROC_Member` objects

    '''
    closest_rocs = []
    closest_models = []

    for model in range(len(rocs)):
        rs = rocs[model]
        distances = [dist_fn(x.squeeze(), r.r.squeeze()) for r in rs]
        if len(distances) != 0:
            closest_rocs.append(rs[np.argsort(distances)[0]])
            closest_models.append(model)
    return closest_models, closest_rocs

def find_best_forecaster(x: torch.Tensor, rocs: List[List[ROC_Member]], pool: List[torch.nn.Module], dist_fn: callable, topm: int = 1):
    ''' Given an input `x`, a pool of pretrained models `pool` and corresponding RoCs `rocs` return the `topm` best forecasters according to distance measure `dist_fn`

    Args:
        x: Input time series window
        pool: List of pretrained models
        rocs: List of Regions of Competences
        dist_fn: Distance function applicable for two time series windows
        topm: How many models to return

    Returns:
        A `np.ndarray` of the `topm` best model indices from `pool`

    '''
    model_distances = np.ones(len(pool)) * 1e10
    closest_rocs_agg = [None]*len(pool)

    for i in range(len(pool)):
        x = x.squeeze()
        for rm in rocs[i]:
            r = rm.r
            distance = dist_fn(r, x)
            if distance < model_distances[i]:
                model_distances[i] = distance
                closest_rocs_agg[i] = r

    top_models = np.argsort(model_distances)[:topm]
    return top_models[:topm]


def roc_matrix(rocs, z=1):
    lag = rocs.shape[-1]
    m = np.ones((len(rocs), lag + len(rocs) * z - z)) * np.nan

    offset = 0
    for i, roc in enumerate(rocs):
        m[i, offset:(offset+lag)] = roc
        offset += z

    return m

def roc_mean(roc_matrix):
    summation_matrix = roc_matrix.copy()
    summation_matrix[np.where(np.isnan(roc_matrix))] = 0
    sums = np.sum(summation_matrix, axis=0)
    nonzeros = np.sum(np.logical_not(np.isnan(roc_matrix)), axis=0)
    return sums / nonzeros