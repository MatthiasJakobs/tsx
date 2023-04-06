import numpy as np
from tsx.datasets import windowing

class BestCaseSelector:

    def __init__(self, models, lag=5):
        self.models = models
        self.lag = lag

    def run(self, X_train, X_val, X_test):
        X, y = windowing(X_test, lag=self.lag)
        self.test_predictors = []

        preds = []
        for _x, _y in zip(X, y):
            _x = _x.reshape(1, -1)
            model_losses = [(m.predict(_x).squeeze() - _y)**2 for m in self.models]
            best_forecaster = np.argmin(model_losses)
            self.test_predictors.append(best_forecaster)
            preds.append(self.models[best_forecaster].predict(_x).squeeze())

        return np.concatenate([X_test[:self.lag], np.array(preds)])

class WorstCaseSelector:

    def __init__(self, models, lag=5):
        self.models = models
        self.lag = lag

    def run(self, X_train, X_val, X_test):
        X, y = windowing(X_test, lag=self.lag)
        self.test_predictors = []

        preds = []
        for _x, _y in zip(X, y):
            _x = _x.reshape(1, -1)
            model_losses = [(m.predict(_x).squeeze() - _y)**2 for m in self.models]
            worst_forecaster = np.argmax(model_losses)
            self.test_predictors.append(worst_forecaster)
            preds.append(self.models[worst_forecaster].predict(_x).squeeze())

        return np.concatenate([X_test[:self.lag], np.array(preds)])
