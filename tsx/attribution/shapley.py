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
    def shap_values(self, X, Y, verbose=True):
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
            vS = self.vfe.get_value(S, X, Y)
            for f_idx in feature_indices:
                vSi = self.vfe.get_value((*S, f_idx), X, Y)
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

    def __init__(self, f, background, density_sampler=None):
        self.f = f
        self.background = background
        self.density_sampler = density_sampler

    # Calculate E[f(x) | X_S = x_S]
    def get_value(self, S, X):
        S = np.array(S)
        values = np.zeros((len(X)))
        for idx, x in enumerate(X):
            samples = self.density_sampler.sample(self.background)
            if len(S) != 0:
                samples[:, S] = x[S]
            values[idx] = self.f(samples).mean()
        
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

    def __init__(self, f, background, random_state=None):
        self.rng = to_random_state(random_state)

        density_sampler = IndependentSampler(n_samples=None, random_state=self.rng)
        value_function = ExpectedFunctionValue(f, background, density_sampler)
        coalition_sampler = AllCoalitions()

        super().__init__(coalition_sampler=coalition_sampler, value_function_estimator=value_function)

###
# Discrete
###

class SAXDependentSampler:

    def __init__(self, background, mean_y=0, sax=None, random_state=None):
        self.rng = to_random_state(random_state)
        self.dist = EmpiricalQuantized(sax.encode(z_norm(background)))
        self.mean_y = mean_y

    def get_samples(self, _x, S):
        samples = self.dist.get_samples(_x, S, replace=True, random_state=self.rng)
        if len(samples) == 0:
            # For now, return mean prediction over background if no dependence found
            # TODO: Maybe make the S smaller and see if samples get returned?
            #       [0, 1, 3] -> [0, 1] for example, should be more hits in general
            return self.mean_y
        return samples

class SAXIndependentSampler:

    def __init__(self, tokens, random_state=None):
        self.rng = to_random_state(random_state)
        self.tokens = tokens

    def get_samples(self, _x, S):
        from itertools import product
        # SAX with blind, independent perturbations
        samples = []
        n_features = _x.shape[-1]
        not_S = np.array([i for i in range(n_features) if i not in S])

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
                not_S = self.rng.choice(not_S, size=nr_not_S, replace=False)
                sample = _x.copy()
                not_S_values = self.rng.choice(self.tokens, nr_not_S, replace=False)
                sample[not_S] = not_S_values
                samples.append(sample)

            samples = np.vstack(samples)

        samples = samples.astype(np.int8)
        return samples

class SAXLossValueFunction:

    def __init__(self, f, sampler=None, normalize=False, sax=None, random_state=None):
        self.f = f
        self.rng = to_random_state(random_state)
        self.sax = sax
        self.sampler = sampler
        self.normalize = normalize

    def get_value(self, S, X, Y=None):
        S = np.array(S)
        values = np.zeros((len(X)))

        if self.normalize:
            X, mu, std = z_norm(X, return_mean_std=True)

        _X = self.sax.encode(X)

        for idx, _x in enumerate(_X):

            samples = self.sampler.get_samples(_x, S)
            if isinstance(samples, np.float32):
                _b = np.array(samples).reshape(1, -1)
            else:
                samples = self.sax.decode(samples, random_state=self.rng)

                if self.normalize:
                    samples = (std * samples) + mu

                _b = np.array(self.f(samples).mean()).reshape(1, -1)

            _a = np.array(Y[idx]).reshape(1, -1)
            values[idx] = mse(_a, _b)
        
        return values


class SAXIndependent(ShapleyValues):

    def __init__(self, f, n_alphabet, max_coalition_samples, normalize=False, random_state=None):
        rng = to_random_state(random_state)
        
        # Value function
        tokens = np.arange(n_alphabet)
        lvf = SAXLossValueFunction(
            f, 
            random_state=rng,
            sax=SAX(tokens),
            sampler=SAXIndependentSampler(tokens, random_state=rng),
            normalize=normalize,
        )

        # CoalitionSampler
        cs = SampleCoalitions(max_coalition_samples, random_state=rng)

        super().__init__(value_function_estimator=lvf, coalition_sampler=cs)

class SAXEmpiricalDependent(ShapleyValues):
    
    def __init__(self, f, background, n_alphabet, max_coalition_samples, normalize=False, random_state=None):
        rng = to_random_state(random_state)
        
        # Value function
        tokens = np.arange(n_alphabet)
        sax = SAX(tokens)
        lvf = SAXLossValueFunction(
            f,
            sax=sax,
            random_state=rng,
            sampler=SAXDependentSampler(background, mean_y=f(background).mean(), sax=sax, random_state=rng),
            normalize=normalize,
        )

        # CoalitionSampler
        cs = SampleCoalitions(max_coalition_samples, random_state=rng)

        super().__init__(value_function_estimator=lvf, coalition_sampler=cs)