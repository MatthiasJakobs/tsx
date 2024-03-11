import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error
from torch import where

from tsx.utils import to_random_state

class OMS_ROC:

    ''' RoC-based model-agnostic selection method utilizing K-Means clustering of validation data to build Regions of Competence

    Args:
        pool: Pool of pretrained models to do forecasting
        random_state: Valid input to `to_random_state`
    '''

    def __init__(self, pool, random_state=None):
        self.rng = to_random_state(random_state)
        self.pool = pool

    # Simple version to determine K
    def _find_nr_clusters(self, x, nc_max=10):
        ks = (np.arange(nc_max-2)+2).astype(np.int8)
        sscores = []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=self.rng.integers(0, 10_000, 1)[0])
            _x = km.fit_predict(x)
            sscores.append(silhouette_score(x, _x))

        return ks[np.argmax(sscores)]


    def run(self, x_val, y_val, x_test):
        ''' Compute model selection and prediction

        Args:
            x_val: Input for training KNN
            y_val: Label for training KNN
            x_test: Input to forecast

        Returns:
           Tuple of `predictions` and `selection`

        '''
        K = self._find_nr_clusters(x_val)

        km = KMeans(n_clusters=K, random_state=self.rng.integers(0, 10_000, 1)[0])
        C = km.fit_predict(x_val)

        cluster_experts = {}

        for c in range(K):
            indices = np.where(C == c)[0]
            _x = x_val[indices]
            _y = y_val[indices]

            best_model = np.argmin([mean_squared_error(m.predict(_x), _y) for m in self.pool])
            cluster_experts[c] = best_model

        # Inference
        selection = np.zeros((len(x_test)))
        preds = np.zeros((len(x_test)))
        for idx, x in enumerate(x_test):
            c = int(np.argmin(np.mean((km.cluster_centers_ - x[None, :])**2, axis=1)))
            selection[idx] = cluster_experts[c]
            preds[idx] = self.pool[cluster_experts[c]].predict(x.reshape(1, -1)).squeeze()

        return preds, selection.astype(np.int8)



            

