import numpy as np

from sklearn.ensemble import RandomForestRegressor
from tsx.utils import to_random_state
from scipy.stats import pearsonr

# Paper
#   - https://link.springer.com/article/10.1007/s10994-018-05774-y
#   - https://link.springer.com/chapter/10.1007/978-3-319-71246-8_29
class ADE:
    ''' Reimplementation of ADE from from https://link.springer.com/article/10.1007/s10994-018-05774-y

    Args:
        random_state : Input to `to_random_state`
    
    '''

    def __init__(self, random_state=None):
        self.rng = to_random_state(random_state)

    def run(self, X_train, y_train, train_preds, X_test, y_test, test_preds, _omega=0.5, _lambda=50, only_best=False):
        ''' Compute model selection and prediction

        Args:
            X_train: Input for training meta learners
            y_train: Label for training meta learners
            train_preds: shape (n_learner, T_train) predictions on training data for each model            X_test: Test input data
            X_test: Test inputs
            y_test: Test labels
            test_preds: shape (n_learner, T_test) predictions on test data for each model
            _omega: Committee ratio
            _lambda: Window size (how much old data timesteps to include for penalty)
            only_best: If True, return only best model. Otherwise, return ensemble weights (default: False)

        Returns:
           Tuple of `predictions` and `weights`. `weights` is a list of indices if `only_best==True` 
        '''

        n_learner = len(train_preds)

        # Initialize meta learner
        self.meta_learner = [RandomForestRegressor(random_state=self.rng.integers(0, 10_000, 1)[0]) for _ in range(n_learner)]

        # Train meta learner on absolute error of training data
        for idx in range(n_learner):
            AE = np.abs(train_preds[idx].squeeze() - y_train)
            self.meta_learner[idx].fit(X_train, AE)

        prediction_history = train_preds.copy()
        label_history = y_train.copy()

        predictions = np.zeros((len(y_test)))
        weights = np.zeros((len(y_test), n_learner))

        for t in range(len(y_test)):
            _X = X_test[t].reshape(1, -1)

            # Form committees
            avg_errors = np.abs(prediction_history-label_history[None, :])[:, -_lambda:].mean(axis=1)
            to_pick = int(n_learner * _omega)
            committee_indices = np.argsort(avg_errors)[:to_pick]
            ML = [self.meta_learner[idx] for idx in committee_indices]

            # Get loss predictions
            loss_predictions = np.array([_ml.predict(_X).squeeze() for _ml in ML])

            # TODO: Weighting correct?
            # _min, _max = loss_predictions.min(), loss_predictions.max()
            # loss_predictions = (loss_predictions - _min) / (_max - _min)
            # local_weights = -loss_predictions / (-loss_predictions).sum()
            # weights[t, committee_indices] = local_weights
            # TODO: Use exp instead of min-max
            local_weights = np.exp(-loss_predictions) / (np.exp(-loss_predictions)).sum()
            weights[t, committee_indices] = local_weights

            # Sequential reweighting
            weight_sorting = committee_indices[np.argsort(-local_weights)]

            for idx, i in enumerate(weight_sorting):
                w_i = weights[t, i]
                for j in weight_sorting[idx:]:
                    if i == j:
                        continue
                    w_j = weights[t, j]

                    penalty = pearsonr(prediction_history[i][-_lambda:], prediction_history[j][-_lambda:]).statistic * w_j * w_i
                    w_j += penalty
                    w_i -= penalty
                    weights[t, i] = w_i
                    weights[t, j] = w_j

            # Prediction 
            if only_best:
                highest_weight_index = np.argmax(weights[t])
                current_prediction = weights[t, highest_weight_index] * test_preds[highest_weight_index, t]
            else:
                current_prediction = (weights[t] * test_preds[:, t]).sum()
            predictions[t] = current_prediction

            # Add to prediction history and label history
            prediction_history = np.concatenate([prediction_history, test_preds[:, t].reshape(-1, 1)], axis=1)
            label_history = np.concatenate([label_history, y_test[t].reshape(-1)])

        if only_best:
            return predictions, np.argmax(weights, axis=1)
        return predictions, weights
