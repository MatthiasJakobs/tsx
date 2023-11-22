import numpy as np
from tsx.utils import to_random_state

from itertools import product

def n_uniques(A, L):
    return A**L - (A-1)**L

# TODO: This surely can be constructed cleverly
def generate_unique_concepts(L, A):
    # Create all combinations
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:A]
    combs_string = np.array(list(product(alphabet, repeat=L)))
    combs = np.array(list(product(np.arange(A)+1, repeat=L)))

    # Find uniques
    uniques = []
    unique_indices = []
    for idx, c in enumerate(combs):
        # Normalize by last entry
        c_norm = c[-1]
        _c = c - c_norm

        if np.any([np.all(u == _c) for u in uniques]):
            continue

        unique_indices.append(idx)
        uniques.append(_c)

    #uniques = np.vstack(uniques)
    uniques = combs_string[np.array(unique_indices)]
    return [ ''.join(x) for x in uniques ] 

def generate_all_concepts(L, A):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:A]
    return [ ''.join(x) for x in product(alphabet, repeat=L) ] 

# Convenience method to map string representations to indices of bounds
def _map_key_to_indices(key):
    chars = list(key)
    mapping = {c: ord(c)-65 for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
    indices = [mapping[c] for c in chars]
    return np.array(indices)

# Given the size of an alphabet, return cut points
def _get_bounds(n_alphabet):
    low = -n_alphabet
    last = low
    bounds = []
    for _ in range(n_alphabet):
        bounds.append([last, last+2])
        last += 2
    return np.array(bounds)

# Generate datapoints for a given concept
def generate_samples(concept_key, size, n_alphabet, random_state=None):
    rng = to_random_state(random_state)
    concept_indices = _map_key_to_indices(concept_key)
    bounds = _get_bounds(n_alphabet)
    samples = np.zeros((size, len(concept_indices)))
    for _i, idx in enumerate(concept_indices):
        low, high = bounds[idx]
        samples[:, _i] = rng.uniform(low, high, size=size)

    return samples

# Generate datasets from given concepts
def generate_in_out_sets(concepts, n_alphabet, size_per_concept, random_state=None):
    rng = to_random_state(random_state)
    n_steps = len(concepts[0])
    X = np.zeros((size_per_concept*len(concepts), n_steps))
    y = np.zeros((size_per_concept*len(concepts), 1))
    for n_idx, concept in enumerate(concepts):
        X[n_idx*size_per_concept:n_idx*size_per_concept+size_per_concept] = generate_samples(concept, size_per_concept, n_alphabet, random_state=rng)
        y[n_idx*size_per_concept:n_idx*size_per_concept+size_per_concept] = n_idx

    return X, y.astype(np.int32)

# Return balanced binary dataset with size 2*|`class_idx`|
def sample_balanced(X, y, class_idx=1, random_state=None):
    rng = to_random_state(random_state)
    one_indices = np.where(y == class_idx)[0]
    zero_indices = np.where(y != class_idx)[0]
    zero_indices = rng.choice(zero_indices, size=len(one_indices), replace=False)
    all_indices = np.concatenate([one_indices, zero_indices])
    return X[all_indices], y[all_indices]

def find_closest_concepts(X, concepts):
    # Normalize X and concepts (for good measure)
    mu, std = np.mean(X, axis=-1), np.std(X, axis=-1)
    std[std == 0] = 1
    _X = (X - mu[:, None]) / std[:, None]

    mu, std = np.mean(concepts, axis=-1), np.std(concepts, axis=-1)
    std[std == 0] = 1
    _concepts = (concepts - mu[:, None]) / std[:, None]

    closest_concepts = []
    for x in _X:
        distances = np.sum((_concepts - x[None, :])**2, axis=1)
        closest_concepts.append(np.argmin(distances))

    return closest_concepts

# X.shape = (batch_size, L)
def get_concept_distributions(X, concepts, normalize=True):
    concept_dist = np.zeros((len(concepts)))
    probable_concepts = find_closest_concepts(X, concepts)
    for c in probable_concepts:
        concept_dist[c] += 1

    if normalize:
        concept_dist = concept_dist / concept_dist.sum()

    return concept_dist

def get_normalized_concept_prototypes(L, A):
    unique_concepts = generate_unique_concepts(L, A)
    normalized_concepts = np.zeros((len(unique_concepts), L))
    for idx, concept_key in enumerate(unique_concepts):
        samples = generate_samples(concept_key, 30, A, random_state=12345)
        normalized_concepts[idx] = samples.mean(axis=0)

    # Normalize samples
    mu, std = normalized_concepts.mean(axis=-1), normalized_concepts.std(axis=-1)
    std[std==0] = 1
    normalized_concepts = (normalized_concepts - mu[:, None]) / std[:, None]

    return normalized_concepts
