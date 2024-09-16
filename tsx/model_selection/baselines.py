import numpy as np
from tsx.datasets import windowing

class BaseSelector:

    def __init__(self, models, L=5):
        self.models = models
        self.L = L

    def run(self, X_train, X_val, X_test, return_predictors=False):
        X, y = windowing(X_test, L=self.L, H=1)
        self.test_predictors = []

        preds = []

        model_preds = np.array([m.predict(X).squeeze() for m in self.models]).T

        losses = (model_preds - y[:, None])**2
        test_predictors = self.choose_based_on_loss(losses)

        for y_idx in range(len(y)):
            m_idx = test_predictors[y_idx]
            preds.append(model_preds[y_idx, m_idx])

        preds = np.array(preds)

        if return_predictors:
            return preds, test_predictors

        return preds

# TODO: Only support H=1 right now
class BestCaseSelector(BaseSelector):

    def choose_based_on_loss(self, losses):
        best_forecaster = np.argmin(losses, axis=1)
        best_forecaster = best_forecaster.squeeze()
        return best_forecaster

# TODO: Only support H=1 right now
class WorstCaseSelector(BaseSelector):

    def choose_based_on_loss(self, losses):
        worst_forecaster = np.argmax(losses, axis=1)
        worst_forecaster = worst_forecaster.squeeze()
        return worst_forecaster
