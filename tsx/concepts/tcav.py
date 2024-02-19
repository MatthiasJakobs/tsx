import torch
import numpy as np
import torch.nn as nn

from tsx.utils import to_random_state
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from tsx.concepts import generate_in_out_sets, sample_balanced

def get_cavs(model, concepts, A, size_per_concept=20, return_lms=False, verbose=False, random_state=None):
    ''' Get Class Activation Vectors of `concepts` for `model` 

    Args:
        model: Pytorch model with method `get_activation`, returning a latent representation
        concepts: 2d numpy array of concept samples
        A: Alphabet size
        size_per_concept: How many samples per concept to generate for training linear models 
        return_lms: If True, return list of `sklearn.linear_model.LogisticRegression` instead of just CAVs (default: False)
        verbose: If True, print linear model F1 score per concept (default: False)
        random_state: Input to `to_random_state`

    Returns:
        Either a list of Class Activation Vectors or a list of `sklearn.linear_model.LogisticRegression` models

    '''
    if not hasattr(model, 'get_activation'):
        raise AttributeError(f'Model {model} has no method get_activation')

    rng = to_random_state(random_state)
    model.eval()

    X, y = generate_in_out_sets(concepts, A, size_per_concept=size_per_concept, random_state=rng)
    X = X.astype(np.float32)

    cavs = []

    for in_concept in range(len(concepts)):
        _X = X.copy()
        _y = y.copy()
        
        _y[y != in_concept] = 0
        _y[y == in_concept] = 1

        # Subsample
        _X, _y = sample_balanced(_X, _y, random_state=rng)
        
        X_t = model.get_activation(torch.from_numpy(_X).unsqueeze(1))
        
        lm = LogisticRegression()
        lm.fit(X_t, _y.squeeze())
        if verbose:
            print(f'f1_score concept classifier {in_concept}', f1_score(lm.predict(X_t), _y.squeeze()))

        # Get unit vector orthogonal to lm.coefs_
        # TODO: Double check with reference implementation https://github.com/tensorflow/tcav/blob/master/tcav/tcav.py
        # Might not need to be inverted
        if return_lms:
            cavs.append(lm)
        else:
            cav = -lm.coef_.squeeze()
            cavs.append(cav)

    if not return_lms:
        cavs = np.vstack(cavs)
    return cavs

def get_tcav(model, cavs, X, y=None, aggregate='tcav_original'):
    ''' Get TCAV values of `cavs` for `model` 

    Args:
        model: Pytorch model with method `get_activation`, returning a latent representation and `model.predictor` returning a point forecast
        cavs: List of Class Activation Vectors from `get_cavs`
        X: 2D numpy array of input data
        y: If not None, return TCAV values for squared error (default: None)
        aggregate: How to aggregate TCAV values. Possible values: `[tcav_original, none]` (default: `tcav_original`, which counts number of positive TCAV values)

    Returns:
        (Aggregated) TCAV scores

    '''
    if not (hasattr(model, 'feature_extractor') and hasattr(model, 'predictor')):
        raise AttributeError(f'model needs to have an attribute to extract latent features and a separate predictor. Make sure the attributes `feature_extractor` and `predictor` are present')
    
    tcav_values = np.zeros((len(X), len(cavs)))

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    
    J = torch.autograd.functional.jacobian(model.predictor, model.feature_extractor(X))
    # Due to concatenation of batches, we need to sum out irrelevant dimensions
    J = torch.einsum('bibj->bj', J).numpy()

    # Calculate TCAV_Q from product of jacobian and unit vector. Count how often biger than zero on train(?) set
    for c_idx, cav in enumerate(cavs):
        s_C = np.dot(J, cav)
        if y is not None:
            with torch.no_grad():
                preds = model(X)
            factor = -2 * (preds - y).numpy()
        else:
            factor = 1

        J *= factor
        #s_C = (s_C > 0).astype(np.int8)
        tcav_values[:, c_idx] = s_C

    if aggregate == 'none':
        return tcav_values
    elif aggregate == 'tcav_original':
        return np.mean((tcav_values > 0).astype(np.int8), axis=0)
    else:
        raise NotImplementedError(f'Unknown aggregation method {aggregate}')

