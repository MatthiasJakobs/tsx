import numpy as np

from tsx.utils import to_random_state

class EmpiricalQuantized:

    # X: np.ndarray shape (batch, features)
    def __init__(self, X):
        self.X = X
        self.n_background = len(X)

    def get_samples(self, X, S, n_samples=None, build_up=True, replace=False, random_state=None):
        rng = to_random_state(random_state)
        S = np.array(S)
        X = X.squeeze()
        assert len(X.shape) == 1

        if len(S) == 0:
            all_sample_indices = np.arange(self.n_background)
        else:
            # Try subsets of S to ensure that at least some matches occur
            if build_up:
                matches = None
                # TODO: Random samples of length idx+1 better / less biased ? 
                for idx in range(1, len(S)+1):
                    subset = S[:idx]
                    if matches is None:
                        sub_X = self.X[:, subset]
                        matches = np.all(sub_X == X[subset], axis=1)
                    else:
                        sub_X = self.X[:, subset]
                        _matches = np.all(sub_X == X[subset], axis=1)
                        if _matches.sum() != 0:
                            matches = _matches
                        else:
                            break
            else:
                sub_X = self.X[:, S]
                matches = np.all(sub_X == X[S], axis=1)

            all_sample_indices = np.where(matches)[0]

        if replace == False:
            n_samples = min(n_samples, self.n_background)

        if n_samples is None:
            return self.X[all_sample_indices]

        return self.X[rng.choice(all_sample_indices, size=n_samples, replace=replace)]
