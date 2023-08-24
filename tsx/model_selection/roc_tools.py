import numpy as np

def find_closest_rocs(x, rocs, dist_fn):
    closest_rocs = []
    closest_models = []

    for model in range(len(rocs)):
        rs = rocs[model]
        distances = [dist_fn(x.squeeze(), r.r.squeeze()) for r in rs]
        if len(distances) != 0:
            closest_rocs.append(rs[np.argsort(distances)[0]])
            closest_models.append(model)
    return closest_models, closest_rocs

def find_best_forecaster(x, rocs, pool, dist_fn, topm=1):
    model_distances = np.ones(len(pool)) * 1e10
    closest_rocs_agg = [None]*len(pool)

    for i in range(len(pool)):
        x = x.squeeze()
        for rm in rocs[i]:
            r = rm.r
            distance = dist_fn(r, x)
            if distance < model_distances[i]:
                model_distances[i] = distance
                closest_rocs_agg[i] = r

    top_models = np.argsort(model_distances)[:topm]
    closest_rocs = []
    for i in top_models:
        if closest_rocs_agg[i] is not None:
            closest_rocs.append(closest_rocs_agg[i])

    return top_models[:topm]


def roc_matrix(rocs, z=1):
    lag = rocs.shape[-1]
    m = np.ones((len(rocs), lag + len(rocs) * z - z)) * np.nan

    offset = 0
    for i, roc in enumerate(rocs):
        m[i, offset:(offset+lag)] = roc
        offset += z

    return m

def roc_mean(roc_matrix):
    summation_matrix = roc_matrix.copy()
    summation_matrix[np.where(np.isnan(roc_matrix))] = 0
    sums = np.sum(summation_matrix, axis=0)
    nonzeros = np.sum(np.logical_not(np.isnan(roc_matrix)), axis=0)
    return sums / nonzeros