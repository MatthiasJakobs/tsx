import numpy as np

from tsx.utils import to_random_state
from tsx.quantizers import SAX, z_norm

# Simple quantization baseline that forecasts based on sampling from training data
# TODO: Very simple, this surely is a thing already?
class PropQuant:

    def __init__(self, sax_alphabet_size, n_decode_samples=10, majority_vote=False, random_state=None):
        self.rng = to_random_state(random_state)
        self.sax = SAX(np.arange(sax_alphabet_size))
        self.n_decode_samples = n_decode_samples
        self.majority_vote = majority_vote

    # X.shape=(n_datapoints, L(ag))
    # y.shape=(n_datapoints, H(orizon))
    def fit(self, X, y):
        # Save all seen pairs with their train output
        dist_dict = {}

        if len(y.shape) > 1:
            H = y.shape[-1]
        else:
            H = 1

        self.H = H

        # Normalize and encode
        Z = np.hstack([X, y])
        Z = z_norm(Z)
        Z_e = self.sax.encode(Z)

        X_e, y_e = Z_e[:, :-H], Z_e[:, -H:]
        for _x, _y in zip(X_e, y_e):
            x_string = ','.join([str(a) for a in _x.tolist()])
            y_string = ','.join([str(a) for a in _y.tolist()])
            try:
                dist_dict[x_string].append(y_string)
            except KeyError:
                dist_dict[x_string] = [y_string]

        self.dist_dict = dist_dict

    def predict_step(self, X):
        # Encode and normalize
        X, mu, std = z_norm(X, return_mean_std=True)
        _x = self.sax.encode(X).squeeze()

        # String format for dictionary lookup
        x_string = ','.join([str(a) for a in _x.tolist()])

        try:
            # Try getting all samples from empirical distribution
            samples = self.dist_dict[x_string]
            y_e = np.vstack([np.array(s.split(','), dtype=np.int8) for s in samples])

            if self.majority_vote:
                from scipy.stats import mode
                y_e = mode(y_e, axis=0, keepdims=False)[0]
                y_e = np.expand_dims(y_e, 0)
        except KeyError:
            # Return middle token of distribution
            y_e = np.ones((1, self.H), dtype=np.int8) * self.sax.tokens[self.sax.n_alphabet // 2]

        # Decode and denormalize
        y_hat = self.sax.decode(y_e, n_samples=self.n_decode_samples, random_state=self.rng)
        y_hat = std * y_hat + mu

        return y_hat
