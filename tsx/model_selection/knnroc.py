import numpy as np

from sklearn.neighbors import KNeighborsClassifier

class KNNRoC:

    ''' Train KNN classifier based on Regions of Competence

    Args:
        pool: Pool of pretrained models to do forecasting
    '''

    def build_rocs(self, x_val, y_val, val_preds):
        n_learner = val_preds.shape[0]
        y_val = y_val.squeeze()
        val_losses = (val_preds - y_val[None, :])**2
        self.rocs = [ [] for _ in range(n_learner) ]
        best_models = np.argmin(val_losses, axis=0)
        for m_idx in range(n_learner):
            self.rocs[m_idx] = x_val[np.where(best_models == m_idx)]

    # TODO: Support DTW
    def train_knn(self):
        self.knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        # Prepare train data 
        y = np.concatenate([np.ones(len(_x))*m_idx for m_idx, _x in enumerate(self.rocs)]).astype(np.int8)
        x = np.concatenate(self.rocs, axis=0)
        
        self.knn.fit(x, y)

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
        self.build_rocs(X_train, y_train, train_preds)
        self.train_knn()

        n_learner = len(self.rocs)

        selection = self.knn.predict(X_test).astype(np.int8)

        preds = np.zeros((len(X_test)))
        for m_idx in range(n_learner):
            to_predict = np.where(selection == m_idx)[0]
            if len(to_predict) > 0:
                preds[to_predict] = test_preds[m_idx, to_predict].squeeze()

        return preds, selection
