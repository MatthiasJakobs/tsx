import numpy as np

from tsx.utils import to_random_state
from tsx.quantizers import SAX, z_norm

# Simple quantization baseline that forecasts based on sampling from training data
# TODO: Very simple, this surely is a thing already?
# TODO: Tested only with H=1
class PropQuant:

    def __init__(self, sax_alphabet_size, n_decode_samples=10, random_state=None):
        self.rng = to_random_state(random_state)
        self.sax = SAX(np.arange(sax_alphabet_size))
        self.n_decode_samples = n_decode_samples

    # X.shape=(n_datapoints, L(ag))
    # y.shape=(n_datapoints, H(orizon))
    def fit(self, X, y):
        # Save all seen pairs with their train output
        dist_dict = {}

        if len(y.shape) > 1:
            H = y.shape[-1]
        else:
            H = 1

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
        X_start = X[0]
        X, mu, std = z_norm(X, return_mean_std=True)
        Z = self.sax.encode(X)

        predictions = []

        for idx, _x in enumerate(Z):
            x_string = ','.join([str(a) for a in _x.tolist()])
            try:
                # Sample from empirical distribution of training data
                sample = self.rng.choice(self.dist_dict[x_string], size=1)[0]
                y_e = np.array(sample.split(','), dtype=np.int8)
            except KeyError:
                # Return middle token of distribution
                y_e = self.sax.tokens[self.sax.n_alphabet // 2]

            # Decode and denormalize
            if len(y_e.shape) <= 1:
                y_e = y_e.reshape(1, 1)
            y_hat = self.sax.decode(y_e, n_samples=self.n_decode_samples, random_state=self.rng).squeeze()
            y_hat = std[idx] * y_hat + mu[idx]

            predictions.append(y_hat.squeeze())

        return np.concatenate([X_start, np.array(predictions)])






