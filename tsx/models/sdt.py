###
# Soft decision trees
###

import numpy as np
import torch
import torch.nn as nn

# Soft decision node
class Node(nn.Module):

    def __init__(self, n_features, left_child, right_child, classifier=True):
        super().__init__()
        self.n_features = n_features
        self.is_classifier = classifier
        self.is_leaf = left_child is None and right_child is None

        if self.is_leaf and not classifier:
            self.z = nn.Parameter(torch.normal(torch.zeros(1), torch.ones(1)))
        
        self.weight = nn.Linear(n_features, 1)

        self.left_child = left_child
        self.right_child = right_child

    def forward(self, X):
        out = self.weight(X)
        right = torch.sigmoid(out)
        left = 1-torch.sigmoid(out)
        if self.is_leaf:
            if self.is_classifier:
                return right
            else:
                return self.z
        return left * self.left_child(X) + right * self.right_child(X)

    def num_parameters(self):
        N = self.weight.weight.numel() + self.weight.bias.numel()
        if self.left_child is not None and self.right_child is not None:
            return N + self.left_child.num_parameters() + self.right_child.num_parameters()
        else:
            return N

def _create_sdt(n_features, depth, classifier=True):
    if depth == 1:
        return Node(n_features, None, None, classifier=classifier)
    else:
        return Node(n_features, _create_sdt(n_features, depth-1, classifier=classifier), _create_sdt(n_features, depth-1, classifier=classifier), classifier=classifier)

class SDTBase(nn.Module):

    def __init__(self, n_features, depth, classifier):
        super().__init__()
        self.model = _create_sdt(n_features, depth, classifier)

    def forward(self, X):
        return self.model(X)

    def num_parameters(self):
        return self.model.num_parameters()

class SoftDecisionTreeRegressor(SDTBase):

    def __init__(self, n_features, depth):
        super().__init__(n_features, depth, classifier=False)

    @torch.no_grad()
    def predict(self, X):
        return self.model(X).cpu().numpy().squeeze()

class SoftDecisionTreeClassifier(SDTBase):

    def __init__(self, n_features, depth):
        super().__init__(n_features, depth, classifier=True)

    @torch.no_grad()
    def predict(self, X):
        return (self.model(X) >= 0.5).cpu().numpy().squeeze()

class SoftEnsembleClassifier(nn.Module):

    def __init__(self, n_trees, n_features, depth):
        super().__init__()
        self.trees = nn.ModuleList([SoftDecisionTreeClassifier(n_features, depth) for _ in range(n_trees)])

    def forward(self, X):
        single_preds = torch.cat([tree(X) for tree in self.trees], axis=-1)
        return torch.mean(single_preds, axis=-1)

    def predict(self, X):
        with torch.no_grad():
            return (self.forward(X).cpu().numpy() >= 0.5).astype(np.int8)

class SoftEnsembleRegressor(nn.Module):

    def __init__(self, n_trees, n_features, depth):
        super().__init__()
        self.trees = nn.ModuleList([SoftDecisionTreeRegressor(n_features, depth) for _ in range(n_trees)])

    def forward(self, X):
        single_preds = torch.cat([tree(X) for tree in self.trees], axis=-1)
        return torch.mean(single_preds, axis=-1)

    def predict(self, X):
        with torch.no_grad():
            return self.forward(X).cpu().numpy()


