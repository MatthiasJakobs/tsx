import numpy as np
import tqdm

from itertools import chain, combinations, product
from scipy.special import binom
from functools import lru_cache
from sklearn.metrics import mean_squared_error as mse

from tsx.utils import to_random_state
from tsx.quantizers import SAX, EmpiricalQuantized, z_norm

###
# General
###

class ShapleyValues:

    def __init__(self, coalition_sampler=None, value_function_estimator=None, weight_function=None):
        self.cs = coalition_sampler
        self.vfe = value_function_estimator

        if weight_function is None:
            self.wf = ShapleyWeightFunction()
        else:
            self.wf = weight_function

    # TODO: Use Y correctly
    def shap_values(self, f, X, Y, verbose=True):
        need_squeeze = False
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            need_squeeze = True

        N = X.shape[-1]
        shaps = np.zeros_like(X)

        coalitions = self.cs.get(N)
        total = 2**N - 1

        for S in tqdm.tqdm(coalitions, disable=not verbose, total=total):
            # Determine which features can be used with S
            feature_indices = np.array([idx for idx in range(N) if idx not in S])
            w = self.wf.w(len(S), N)
            vS = self.vfe.get_value(f, S, X, Y)
            for f_idx in feature_indices:
                vSi = self.vfe.get_value(f, (*S, f_idx), X, Y)
                shaps[:, f_idx] += w * (vSi - vS)

        if need_squeeze:
            shaps = shaps.squeeze()
        
        return shaps

class ShapleyWeightFunction:

    @lru_cache
    def w(self, S, N):
        return (1/N)*(1/binom(N-1, S))


class IndependentSampler:

    def __init__(self, n_samples=100, random_state=None):
        self.rng = to_random_state(random_state)
        self.n_samples = n_samples

    def sample(self, background):
        if self.n_samples is None:
            return background.copy()
        else:
            indices = rng.choice(len(background), size=self.n_samples, replace=True)
        return background[indices]

class ExpectedFunctionValue:

    def __init__(self, background, density_sampler=None):
        self.background = background
        self.density_sampler = density_sampler

    # Calculate E[f(x) | X_S = x_S]
    def get_value(self, f, S, X, Y):
        S = np.array(S)
        values = np.zeros((len(X)))
        samples = self.density_sampler.sample(self.background)

        n_samples = len(samples)
        n_data = X.shape[0]

        samples = np.tile(samples, (n_data, 1))

        for i in range(n_data):
            if len(S) != 0:
                samples[i*n_data:(i+1)*n_data, S] = X[i, S]
        values = f(samples).reshape(n_data, n_samples).mean(axis=1)
        
        return values

class AllCoalitions:

    def get(self, n_features):
        return chain(*(combinations(np.arange(n_features), r=coal_size) for coal_size in range(n_features)))

class SampleCoalitions:

    def __init__(self, n_samples, random_state=None):
        self.n_samples = n_samples
        self.rng = to_random_state(random_state)

    def get(self, n_features):
        if 2**n_features - 1 < self.n_samples:
            return AllCoalitions().get(n_features)

        coal_sizes = self.rng.choice(n_features-1, replace=True, size=self.n_samples) + 1
        coal_sizes = np.concatenate([[0], coal_sizes, [n_features]]).astype(np.int8)
        return [tuple(self.rng.choice(n_features, replace=False, size=s)) for s in coal_sizes]

class KernelShap(ShapleyValues):

    def __init__(self, background, max_coalition_samples=50, random_state=None):
        self.rng = to_random_state(random_state)

        density_sampler = IndependentSampler(n_samples=None, random_state=self.rng)
        value_function = ExpectedFunctionValue(background, density_sampler)
        coalition_sampler = SampleCoalitions(max_coalition_samples, random_state=self.rng)

        super().__init__(coalition_sampler=coalition_sampler, value_function_estimator=value_function)

###
# Discrete
###

class SAXDecodingValueFunction:
    def __init__(self, sax=None, nr_samples=50, random_state=None):
        self.rng = to_random_state(random_state)
        self.sax = sax
        self.nr_samples = nr_samples

    # Calculate E[f(x) | X_S = x_S]
    # TODO: Only one call to f for speed
    def get_value(self, f, S, X, Y):
        S = np.array(S)
        values = np.zeros((len(X)))
        samples = self.sax.generate_perturbations(X, self.nr_samples, random_state=self.rng)

        n_samples = len(samples)
        n_data = X.shape[0]

        for i in range(n_data):
            if len(S) != 0:
                samples[i*n_data:(i+1)*n_data, S] = X[i, S]
        values = f(samples).reshape(n_data, self.nr_samples).mean(axis=1)

        return values


class SAXDependentSampler:

    def __init__(self, background, mean_y=0, sax=None, normalize=False, random_state=None):
        self.rng = to_random_state(random_state)
        if normalize:
            self.dist = EmpiricalQuantized(sax.encode(z_norm(background)))
        else:
            self.dist = EmpiricalQuantized(sax.encode(background))
        self.mean_y = mean_y

    def get_samples(self, _x, S):
        samples = self.dist.get_samples(_x, S, replace=True, random_state=self.rng)
        if len(samples) == 0:
            raise NotImplementedError()
            # For now, return mean prediction over background if no (partial) dependence found
            #return np.array(self.mean_y).reshape(1, -1)
        return samples

class SAXIndependentSampler:

    def __init__(self, tokens, random_state=None):
        self.rng = to_random_state(random_state)
        self.tokens = tokens

    def get_samples(self, _x, S):
        # SAX with blind, independent perturbations
        samples = []
        n_features = _x.shape[-1]
        not_S = np.array([i for i in range(n_features) if i not in S])

        # TODO:
        max_nr_samples = 100
        total_possible_samples = len(self.tokens)**len(not_S)

        if len(S) == n_features:
            return _x.reshape(1, -1)

        if total_possible_samples < max_nr_samples:
            replacement_values = np.array(list(product(self.tokens, repeat=len(not_S))))
            samples = np.zeros((len(replacement_values), len(_x)))
            samples[:, S] = _x[S]
            samples[:, not_S] = replacement_values
        else:
            samples = []
            for _ in range(max_nr_samples):
                # Sample how many notS indices to sample
                nr_not_S = self.rng.randint(1, len(not_S)+1)
                _not_S = self.rng.choice(not_S, size=nr_not_S, replace=False)
                sample = _x.copy()
                not_S_values = self.rng.choice(self.tokens, nr_not_S, replace=False)
                sample[_not_S] = not_S_values
                samples.append(sample)

            samples = np.vstack(samples)

        samples = samples.astype(np.int8)
        return samples

class SAXLossValueFunction:

    def __init__(self, sampler=None, normalize=False, sax=None, explain_loss=False, random_state=None):
        self.rng = to_random_state(random_state)
        self.sax = sax
        self.sampler = sampler
        self.normalize = normalize
        self.explain_loss = explain_loss

    def get_value(self, f, S, X, Y=None):
        S = np.array(S)
        values = np.zeros((len(X)))

        if self.normalize:
            X, mu, std = z_norm(X, return_mean_std=True)

        _X = self.sax.encode(X)
        
        all_samples = []
        for idx, _x in enumerate(_X):
            all_samples.append(self.sampler.get_samples(_x, S))

        sample_indices = [idx for idx in range(len(_X)) if not all_samples[idx].shape[-1] == 1]
        _X_decoded = self.sax.decode(np.vstack([all_samples[idx] for idx in sample_indices]), n_samples=20, random_state=self.rng)
        
        # TODO: This needs work and cannot be done this way
        if self.normalize:
            _X_decoded = std * _X_decoded + mu

        preds = f(_X_decoded)

        _bs = np.zeros((len(_X)))
        prev_endpoint = 0
        for idx in range(len(_X)):
            segment = all_samples[idx]
            length_segment = len(segment)
            if length_segment == 1 and segment.shape[-1] == 1:
                _bs[idx] = segment[0][0]
            else:
                segment_preds = preds[prev_endpoint:(prev_endpoint + length_segment)]
                _bs[idx] = segment_preds.mean()
                prev_endpoint += length_segment

        if self.explain_loss:
            # TODO:
            _as = np.array(Y).squeeze()
            return (_as - _bs)**2
        else:
            return _bs



class SAXIndependent(ShapleyValues):

    def __init__(self, n_alphabet, max_coalition_samples, explain_loss=True, normalize=False, random_state=None):
        rng = to_random_state(random_state)
        
        # Value function
        tokens = np.arange(n_alphabet)
        lvf = SAXLossValueFunction(
            random_state=rng,
            sax=SAX(tokens),
            sampler=SAXIndependentSampler(tokens, random_state=rng),
            explain_loss=explain_loss,
            normalize=normalize,
        )

        # CoalitionSampler
        cs = SampleCoalitions(max_coalition_samples, random_state=rng)

        super().__init__(value_function_estimator=lvf, coalition_sampler=cs)

class SAXEmpiricalDependent(ShapleyValues):
    
    def __init__(self, background, n_alphabet, max_coalition_samples, explain_loss=True, normalize=False, random_state=None):
        rng = to_random_state(random_state)
        
        # Value function
        tokens = np.arange(n_alphabet)
        sax = SAX(tokens)
        lvf = SAXLossValueFunction(
            sax=sax,
            random_state=rng,
            sampler=SAXDependentSampler(background, sax=sax, normalize=normalize, random_state=rng),
            explain_loss=explain_loss,
            normalize=normalize,
        )

        # CoalitionSampler
        cs = SampleCoalitions(max_coalition_samples, random_state=rng)

        super().__init__(value_function_estimator=lvf, coalition_sampler=cs)

class SAXDecodingPerturbations(ShapleyValues):
    
    def __init__(self, n_alphabet, max_coalition_samples, nr_perturbations, random_state=None):
        rng = to_random_state(random_state)
        
        # Value function
        tokens = np.arange(n_alphabet)
        sax = SAX(tokens)
        lvf = SAXDecodingValueFunction(
            sax,
            nr_samples=nr_perturbations,
            random_state=rng,
        )

        # CoalitionSampler
        cs = SampleCoalitions(max_coalition_samples, random_state=rng)

        super().__init__(value_function_estimator=lvf, coalition_sampler=cs)
