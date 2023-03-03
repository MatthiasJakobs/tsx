import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from tsx.utils import to_random_state

def plot_sax_encoding(x, sax, outpath='saxplot.png'):
    bin_eps = 0.5
    label_eps = 0.5

    x_encoded = sax.encode(x).squeeze()

    boundaries = np.array(sax.boundaries[1:][:-1])
    r = np.maximum(abs(np.max(x)), abs(np.min(x))) + bin_eps
    y_min = -r
    y_max = r

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(x, color='C0')
    ax.scatter(np.arange(len(x)), x, marker='x', color='C0')
    for b in boundaries:
        plt.axhline(b, color='C1', linestyle='--', alpha=0.7)
    
    # Set encoding text
    _boundaries = np.array([y_min, *boundaries, y_max])
    y_pos = (_boundaries[1:] + _boundaries[:-1]) / 2
    x_max = len(x) - 1 + label_eps
    
    alphabet = [chr(65 + i) for i in range(len(sax.tokens))]
    tokenized = [chr(65 + i) for i in range(len(sax.tokens))]
    for character, _y in zip(alphabet, y_pos):
        plt.text(x=x_max, y=_y, s=character, fontweight='bold')

    x_encoded = ''.join([alphabet[i-1] for i in x_encoded])

    plt.title(r'$x_{enc}=$' + x_encoded)
    plt.ylim(y_min, y_max)
    plt.xticks(ticks=np.arange(len(x)), labels=np.arange(len(x)))
    plt.xlabel(r'$t$')
    #plt.yticks([])

    padding = 0.1
    gauss_width = 0.1 + padding
    plt.subplots_adjust(left=gauss_width)
    old_position = ax.get_position().bounds
    # [left, bottom, width, height]
    new_position = (padding, old_position[1], gauss_width-padding, old_position[3])
    cax = plt.axes(new_position, sharey=ax)

    support = np.linspace(-3, 3, 1000)
    height = -0.3
    sd = 1
    gaussian = np.exp((-support ** 2.0) / (2 * sd ** 2.0))
    gaussian /= gaussian.max()
    gaussian *= height

    #cax.plot([1, 2, 3])
    cax.plot(gaussian, support)
    cax.set_xticks([])

    plt.savefig(outpath)

def z_norm(X, return_mean_std=False):
    if len(X.shape) == 1:
        X = X.reshape(1, -1)

    mu = X.mean(axis=1)
    std = X.std(axis=1)

    # Dirty hack for when all values are identical
    std[np.where(std == 0)] = 1

    normalized = ((X.T - mu) / std).T

    if return_mean_std:
        return normalized, mu, std
    else:
        return normalized


def paa(X, M=None, return_indices=False):
    n = len(X)
    if M is None or n == M:
        return X 

    transformed = np.zeros((M))

    for i in range(n * M):
        idx = i // n 
        pos = i // M
        transformed[idx] = transformed[idx] + X[pos]

    if return_indices:
        boundaries = np.linspace(0, n, M+1).astype(np.int16)
        indices = [(a + b) / 2 for a, b in zip(boundaries[:-1], boundaries[1:])]

        return indices, transformed / n

    return transformed / n


class SAX:

    def __init__(self, alphabet):
        self.tokens = alphabet
        self.n_alphabet = len(alphabet)

        # Get alpha percentiles
        percentile_numbers = np.linspace(0, 1, self.n_alphabet+1)[1:][:-1]
        self.boundaries = [np.float32('-inf')] + [norm.ppf(p) for p in percentile_numbers] + [np.float32('inf')]
        self.boundary_samples = { key: None for key in zip(self.boundaries[:-1], self.boundaries[1:]) }

    # TODO: May be numerically instable for small |b1-b2|
    def sample_from_range(self,size, b1, b2, random_state=None):
        t1, t2 = norm.cdf(b1), norm.cdf(b2)

        rv = uniform(t1, t2-t1)
        ys = rv.rvs(size=size, random_state=random_state)
        return norm.ppf(ys)

    def fast_sample_from_range(self, size, b1, b2, random_state=None):
        # If the buffer is not build up already, create it
        if self.boundary_samples[(b1, b2)] is None:
            self.boundary_samples[(b1, b2)] = self.sample_from_range(10_000, b1, b2, random_state=random_state)

        rng = to_random_state(random_state)
        return rng.choice(self.boundary_samples[(b1, b2)], size=size)

    def encode(self, X):
        enced = np.digitize(X, self.boundaries[1:], right=True)
        _enced = enced.copy()
        for idx, token in enumerate(self.tokens):
            _enced[np.where(enced == idx)] = token

        return _enced


    def decode(self, tokens, n_samples=1, random_state=None):
        random_state = to_random_state(random_state)

        if isinstance(tokens, str):
            tokens = list(tokens)

        tokens = np.array(tokens)

        unique_tokens, counts = np.unique(tokens, return_counts=True)

        decoded = np.zeros_like(tokens, dtype=np.float64)
        for token, count in zip(unique_tokens, counts):
            token_index = np.argmax(self.tokens == token)
            lb = self.boundaries[token_index]
            ub = self.boundaries[token_index+1]
            decoded[np.where(tokens == token)] = self.fast_sample_from_range((count, n_samples), lb, ub, random_state=random_state).mean(axis=1)

        return decoded


    def generate_perturbations(self, X, size, random_state=None):
        random_state = to_random_state(random_state)
        encoding = self.encode(X)
        unique_tokens, counts = np.unique(encoding, return_counts=True)

        # Sample for each literal in encoding and save to _d
        _d = {}
        for token, count in zip(unique_tokens, counts):
            token_index = np.argmax(self.tokens == token)
            lb = self.boundaries[token_index]
            ub = self.boundaries[token_index+1]
            samples = self.fast_sample_from_range(count*size, lb, ub, random_state=random_state)
            _d[token] = samples

        # Populate the samples array per encoding literal
        samples = np.tile(encoding, (size, 1))
        samples = samples.astype(np.float32)

        for literal in unique_tokens:
            samples[np.where(samples == literal)] = _d[literal]

        return samples
