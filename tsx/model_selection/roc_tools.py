import numpy as np

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
