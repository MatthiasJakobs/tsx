import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SummaryStatistics(BaseEstimator, TransformerMixin):

    def _check_shape(self, X):
        n_dims = len(X.shape)
        if n_dims < 2 or n_dims > 3:
            raise NotImplementedError('Unsure which dimensions are time and n_channels for array', X.shape)

        # Assume that the two present timensions are (n_samples, n_timesteps)
        if n_dims == 2:
            return np.expand_dims(X, 1)
        # Otherwise, assume (n_samples, n_channels, n_timesteps)
        return X

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self._check_shape(X)
        
        t = np.concatenate([
            np.min(X, axis=-1, keepdims=True),
            np.max(X, axis=-1, keepdims=True),
            np.mean(X, axis=-1, keepdims=True),
            np.var(X, axis=-1, keepdims=True),
        ], axis=-1)

        if len(t.shape) == 3:
            batch_size, _, _ = t.shape
            t = t.reshape(batch_size, -1)

        return t