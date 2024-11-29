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

    def __init__(self, nc_max=15, random_state=None):
        self.rng = to_random_state(random_state)
        self.nc_max = nc_max

    # Simple version to determine K
    def _find_nr_clusters(self, x):
        ks = (np.arange(self.nc_max-2)+2).astype(np.int8)
        sscores = []
        for k in ks:
            km = KMeans(n_init='auto', n_clusters=k, random_state=self.rng)
            _x = km.fit_predict(x)
            sscores.append(silhouette_score(x, _x))

        return ks[np.argmax(sscores)]


    def run(self, X_train, y_train, train_preds, X_test, y_test, test_preds):
        ''' Compute model selection and prediction

        Args:
            X_train: Input for training meta learners
            y_train: Label for training meta learners
            train_preds: shape (n_learner, T_train) predictions on training data for each model
            X_test: Test inputs
            y_test: Test labels
            test_preds: shape (n_learner, T_test) predictions on test data for each model

        Returns:
           Tuple of `predictions` and `selection`

        '''
        n_learner = len(train_preds)
        K = self._find_nr_clusters(X_train)

        km = KMeans(n_clusters=K, n_init='auto', random_state=self.rng)
        C = km.fit_predict(X_train)

        cluster_experts = {}

        for c in range(K):
            indices = np.where(C == c)[0]
            _x = X_train[indices]
            _y = y_train[indices]

            #best_model = np.argmin([mean_squared_error(m.predict(_x).reshape(_x.shape[0]), _y) for m in self.pool])
            best_model = np.argmin([np.mean((_y - train_preds[m_idx][indices])**2) for m_idx in range(n_learner)])
            cluster_experts[c] = best_model

        # Inference
        selection = np.zeros((len(X_test)))
        preds = np.zeros((len(X_test)))
        for idx, x in enumerate(X_test):
            c = int(np.argmin(np.mean((km.cluster_centers_ - x[None, :])**2, axis=1)))
            selection[idx] = cluster_experts[c]
            preds[idx] = test_preds[cluster_experts[c]][idx]

        return preds, selection.astype(np.int8)



            

