import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from tsx.utils import to_random_state

class KNNRoC:

    ''' Train KNN classifier based on Regions of Competence

    Args:
        pool: Pool of pretrained models to do forecasting
        random_state: Valid input to `to_random_state`
    '''

    def __init__(self, pool, random_state=None):
        self.rng = to_random_state(random_state)
        self.pool = pool

    def build_rocs(self, x_val, y_val):
        val_losses = np.vstack([(m.predict(x_val).squeeze() - y_val.squeeze())**2 for m in self.pool])
        self.rocs = [ [] for _ in range(len(self.pool)) ]
        best_models = np.argmin(val_losses, axis=0)
        for m_idx in range(len(self.pool)):
            self.rocs[m_idx] = x_val[np.where(best_models == m_idx)]

    # TODO: Support DTW
    def train_knn(self):
        self.knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        # Prepare train data 
        y = np.concatenate([np.ones(len(_x))*m_idx for m_idx, _x in enumerate(self.rocs)]).astype(np.int8)
        x = np.concatenate(self.rocs, axis=0)
        
        self.knn.fit(x, y)

    def run(self, x_val, y_val, x_test):
        ''' Compute model selection and prediction

        Args:
            x_val: Input for training KNN
            y_val: Label for training KNN
            x_test: Input to forecast

        Returns:
           Tuple of `predictions` and `selection`

        '''
        self.build_rocs(x_val, y_val)
        self.train_knn()

        selection = self.knn.predict(x_test).astype(np.int8)

        preds = np.zeros((len(x_test)))
        for m_idx, m in enumerate(self.pool):
            to_predict = np.where(selection == m_idx)[0]
            preds[to_predict] = m.predict(x_test[to_predict]).squeeze()

        return preds, selection
