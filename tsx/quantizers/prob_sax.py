import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
from scipy import integrate
from tqdm import tqdm
from tsx.utils import to_random_state

## lloy-max implementation inspired by https://github.com/JosephChataignon/Max-Lloyd-algorithm/blob/master/1%20dimension/max_lloyd_1D.py
def lloyd_max(x, density, n_alphabet, epochs=100, verbose=False, init_codewords='random', random_state=None):
    #assert init_codewords in ['random', 'kmeans++']
    assert init_codewords in ['random']
    rng = to_random_state(random_state)

    # Initialize codewords
    if init_codewords == 'random':
        c_min, c_max = x.min() - 1, x.max() + 1
        c = rng.uniform(c_min, c_max, size=n_alphabet)

    # Initialize boundaries
    b = np.zeros((n_alphabet + 1))
    b[0] = -np.inf
    b[-1] = np.inf

    # Run for e epochs
    for e in range(epochs):
        b_before = b[1:-1].copy()
        c_before = c.copy()

        for j in range(1, n_alphabet):
            b[j] = 0.5 * (c[j-1] + c[j])

        for i in range(len(c)):
            bi = b[i]
            biplus1 = b[i+1]
            if bi == biplus1:
                c[i] = 0
            else:
                nom = integrate.quad(lambda t: t * density(t), b[i], b[i+1])[0]
                denom = integrate.quad(density, b[i], b[i+1])[0]
                c[i] = nom / denom

        # Compute delta and see if it decreases
        b_delta = np.abs(b[1:-1] - b_before).mean()
        c_delta = np.abs(c - c_before).mean()

        if verbose:
            print(e, b_delta, c_delta)

    return b, c


class KernelSAX:

    def __init__(self, alphabet, kernel='gaussian', boundary_estimator='lloyd-max', bandwidth=0.2):
        assert boundary_estimator in ['lloyd-max']
        # TODO: epanechnikov is very slow and prints alot of warnings, but seems to work with lloyd-max
        assert kernel in ['gaussian']

        self.kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)

        self.tokens = alphabet
        self.n_alphabet = len(alphabet)

        self.is_fitted = False

    @staticmethod
    def bandwith_cross_validation(X, alphabet_size=7, k=5, kernel='gaussian', boundary_estimator='lloyd-max', random_state=None):
        kf = KFold(n_splits=k)

        params = {}

        for bandwidth in tqdm([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
            errors = []
            for (train_indices, test_indices) in kf.split(X):
                X_train = X[train_indices]
                X_test = X[test_indices]

                sax = KernelSAX(np.arange(alphabet_size), kernel=kernel, boundary_estimator=boundary_estimator, bandwidth=bandwidth)
                sax.fit(X_train)
                encoded = sax.encode(X_test)
                decoded = sax.decode(encoded)

                errors.append(mse(decoded, X_test))

            params[bandwidth] = np.mean(errors)

        return params

    def fit(self, X):
        # TODO: What about multiple samples? Estimate one KDE for each? 
        #assert len(X.shape) == 1
        # if len(X.shape) == 1:
        #     X = X.reshape(-1, 1)
        X = X.reshape(-1, 1)

        # Fit kernel density estimator 
        self.kde.fit(X)

        # Find cutpoints
        density = lambda t: np.exp(self.kde.score_samples(np.array(t).reshape(-1, 1)))
        self.boundaries, self.representatives = lloyd_max(X.squeeze(), density, self.n_alphabet, epochs=200, random_state=0)

        self.is_fitted = True

    def encode(self, X):
        if not self.is_fitted:
            raise RuntimeError('Not fitted')

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        enced = np.digitize(X, self.boundaries[1:], right=True)
        _enced = enced.copy()
        for idx, token in enumerate(self.tokens):
            _enced[np.where(enced == idx)] = token

        return _enced

    def decode(self, tokens):
        if not self.is_fitted:
            raise RuntimeError('Not fitted')

        if len(tokens.shape) == 1:
            tokens = tokens.reshape(1, -1)

        if isinstance(tokens, str):
            tokens = list(tokens)

        tokens = np.array(tokens)
        unique_tokens = np.unique(tokens)

        decoded = np.zeros_like(tokens, dtype=np.float64)
        for token in unique_tokens:
            token_index = np.argmax(self.tokens == token)
            decoded[np.where(tokens == token)] = self.representatives[token_index]

        return decoded

