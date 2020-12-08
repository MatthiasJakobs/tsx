import torch
import pandas as pd
import numpy as np

from tqdm import trange
from itertools import combinations


class NSGA2:
    # based on implementation from https://github.com/haris989/NSGA-II/blob/master/NSGA%20II.py
    # NSGA2 is configured to always minimize (with 0 being optimal), so make sure to configure your criteria functions accordingly

    def __init__(self, criterias, parent_size=10, offspring_size=10, dimensions=3, generations=10):
        self.criterias = criterias
        self.parent_size = parent_size
        self.offspring_size = offspring_size
        self.dimensions = dimensions
        self.generations = generations

    def _random_individuals(self, n):
        return np.random.binomial(1, p=0.5, size=3*n).reshape(n, 3)

    def _apply_functions(self, X):
        result = np.zeros((len(self.criterias), len(X)))
        for i, f in enumerate(self.criterias):
            result[i] = f(X)

        return result.T

    # input: (batch_size, len(self.criterias))
    def fast_non_dominated_sort(self, individual_performs):
        fronts = []
        indices = np.arange(len(individual_performs))
        while len(indices) != 0:
            permutations = list(combinations(np.arange(len(indices)), 2))
            domination_count = np.zeros(len(indices))
            dominates = [[] for _ in range(len(indices))]
            for (ix1, ix2) in permutations:
                if self._dominates(individual_performs[ix1], individual_performs[ix2]):
                    domination_count[ix2] += 1
                    dominates[ix1].append(ix2)
                if self._dominates(individual_performs[ix2], individual_performs[ix1]):
                    domination_count[ix1] += 1
                    dominates[ix2].append(ix1)

            domination_mask = domination_count == 0
            inv_domination_mask = np.logical_not(domination_mask)
            non_dominated = np.where(domination_mask)[0]
            for ix in non_dominated:
                for i in dominates[ix]:
                    domination_count[i] -= 1
            if np.all(inv_domination_mask):
                fronts.append(indices[inv_domination_mask])
                return fronts
            fronts.append(indices[non_dominated])
            indices = indices[inv_domination_mask]
        return fronts

    def recombination(self, x):
        return self._random_individuals(len(x))

    def mutation(self, x):
        return self._random_individuals(len(x))

    def run(self):
        parents = self._random_individuals(self.parent_size)
        offspring = self._random_individuals(self.offspring_size)

        for g in trange(self.generations):
            offspring = self.recombination(self.mutation(parents))
            population = np.concatenate((parents, offspring))

            evaluation = self._apply_functions(population)
            fronts = self.fast_non_dominated_sort(evaluation)

            parent_indices = []
            last_complete_front = -1
            for f in fronts:
                if len(f) + len(parent_indices) <= self.parent_size:
                    parent_indices.extend(list(f))
                    last_complete_front += 1

            # need to do crowding_distanc:
            individuals_left = self.parent_size - len(parent_indices)
            if individuals_left != 0 and last_complete_front != len(fronts)-1:
                if last_complete_front == -1:
                    individuals = evaluation[np.concatenate(fronts)]
                    parent_indices = self.crowding_distance(individuals, n=individuals_left)
                else:
                    individuals = evaluation[np.concatenate(fronts[last_complete_front + 1 :])]
                    parent_indices.extend(self.crowding_distance(individuals, n=individuals_left))

            parents = population[parent_indices]
            evaluation = evaluation[parent_indices]

        return evaluation, parents

                    
    # n is the number of individuals to extract from remaining fronts
    def crowding_distance(self, individuals, n):
        distances = np.zeros(len(individuals))
        distances[0] = 1e20
        distances[-1] = 1e20

        for i in range(len(self.criterias)):
            sorted_indices = np.argsort(individuals[:, i])

            for j in range(1, len(distances)-1):
                distances[j] = distances[j] + (individuals[sorted_indices[j+1]][i] - individuals[sorted_indices[j-1]][i]) / (np.max(individuals[:, i]) - np.min(individuals[:, i]))

        return np.argsort(distances)[-n:]
        

    # a, b shape = (f_1, f_2, ...)
    def _dominates(self, a, b):
        return np.all(a <= b)

def to_numpy(x):
    if isinstance(x, type(torch.zeros(1))):
        if x.requires_grad:
            return x.detach().numpy()
        else:
            return x.numpy()
    if isinstance(x, type(pd.Series(data=[1,2]))):
        return x.to_numpy()
    if isinstance(x, type(np.zeros(1))):
        return x

    raise ValueError("Input of type {} cannot be converted to numpy array".format(type(x)))
