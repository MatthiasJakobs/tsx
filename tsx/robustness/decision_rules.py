import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def test_rules(model, thresholds, n_features, x_min=-1e3, x_max=1e3, n_samples=10000):

    rng = np.random.RandomState(0)
    samples = np.zeros((n_samples, n_features))

    # In case there are +- inf boundaries
    thresholds = np.maximum(thresholds, x_min)
    thresholds = np.minimum(thresholds, x_max)

    for i in range(n_features):
        f = rng.uniform(low=thresholds[i, 0], high=thresholds[i, 1], size=n_samples)
        samples[:, i] = f

    preds = model.predict(samples)
    return np.all(preds[0] == preds[1:])

def extract_rules(x, model):
    x = x.reshape(1, -1)
    sample_id = 0

    feature_thresholds = np.zeros((x.shape[-1], 2))
    feature_thresholds[:, 0] = -np.inf
    feature_thresholds[:, 1] = np.inf

    if isinstance(model, DecisionTreeRegressor):
        trees = [model]
    elif isinstance(model, RandomForestRegressor):
        trees = model.estimators_
    elif isinstance(model, GradientBoostingRegressor):
        trees = model.estimators_.squeeze()
    else:
        raise NotImplementedError('Cannot extract subtrees from type', type(model))

    for tree in trees:

        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        node_indicator = tree.decision_path(x)
        leave_id = tree.apply(x)
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]

        for node_id in node_index:
            if leave_id[sample_id] == node_id:
                continue

            if (x[sample_id, feature[node_id]] <= threshold[node_id]):
                old = feature_thresholds[feature[node_id], 1]
                new = min(old, threshold[node_id])
                feature_thresholds[feature[node_id], 1] = new
            else:
                old = feature_thresholds[feature[node_id], 0]
                new = max(old, threshold[node_id])
                feature_thresholds[feature[node_id], 0] = new

    return feature_thresholds
