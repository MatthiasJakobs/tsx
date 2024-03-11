import numpy as np
from tsx.utils import to_random_state
from tsx.metrics import mase
from scipy.special import erfc

def EMASE(_mase):
    return erfc(2*_mase)

# Paper
#   - https://ieeexplore.ieee.org/abstract/document/8259783?casa_token=yA69YjHH3OEAAAAA:KSJg6CPyOOOC2KkbypuUA0BEPjuUNsqcgHVHDCM3sxHH4p0jMfnq8Ev1-JYGEHy56x7CI1gCZQ
class DETS:

    # Lambda: Committee size (top-lambda percent)
    # P: Window size (how much old data to include for penalty)
    # train_preds: shape (n_learner, T_train) predictions on training data for each model
    # test_preds: shape (n_learner, T_test) predictions on test data for each model
    def run(self, X_train, y_train, train_preds, X_test, y_test, test_preds, P=50, _lambda=0.5, only_best=False):
        n_learner = len(train_preds)

        prediction_history = train_preds.copy()
        label_history = y_train.copy()
        input_history = X_train.copy()

        predictions = np.zeros((len(y_test)))
        weights = np.zeros((len(y_test), n_learner))

        # Compute initial EMASE scores
        last_predictions = prediction_history[:, -P:] 
        last_labels = label_history[-P:]
        last_inputs = input_history[-P:, -1].reshape(-1)
        mase_errors = np.array([mase(last_predictions[i], last_labels, last_inputs) for i in range(n_learner)])
        # min-max norm errors
        _min, _max = min(mase_errors), max(mase_errors)
        mase_errors = (mase_errors - _min) / (_max - _min)
        assert np.all(mase_errors >= 0) and np.all(mase_errors <= 1)
        emase_errors = EMASE(mase_errors)
        # Pick top _lambda percent
        to_pick = int(n_learner * _lambda)
        committee_indices = np.argsort(-emase_errors)[:to_pick]


        for t in range(len(y_test)):
            # Use current committee to predict
            if only_best:
                highest_weight_index = committee_indices[0]
                preds = test_preds[highest_weight_index, t].squeeze()
                weights[t, highest_weight_index] = 1.0
            else:
                preds = test_preds[committee_indices, t]
                local_weights = emase_errors[committee_indices] / emase_errors[committee_indices].sum()
                preds = (local_weights * preds).sum()
                weights[t, committee_indices] = local_weights

            # Log predictions and weights
            predictions[t] = preds

            # Update history
            prediction_history = np.concatenate([prediction_history, test_preds[:, t].reshape(-1, 1)], axis=1)
            label_history = np.concatenate([label_history, y_test[t].reshape(-1)])
            input_history = np.concatenate([input_history, X_test[t].reshape(1, -1)], axis=0)

            # ------ Update selection

            # Compute initial EMASE scores
            last_predictions = prediction_history[:, -P:] 
            last_labels = label_history[-P:]
            last_inputs = input_history[-P:, -1].reshape(-1)
            mase_errors = np.array([mase(last_predictions[i], last_labels, last_inputs) for i in range(n_learner)])
            # min-max norm errors
            _min, _max = min(mase_errors), max(mase_errors)
            if _min == _max:
                # TODO: How to handle the case of equal values correctly?
                mase_errors = np.array([0.5, 0.5])
            else:
                mase_errors = (mase_errors - _min) / (_max - _min)
            assert np.all(mase_errors >= 0) and np.all(mase_errors <= 1)
            emase_errors = EMASE(mase_errors)
            # Pick top _lambda percent
            to_pick = int(n_learner * _lambda)
            committee_indices = np.argsort(-emase_errors)[:to_pick]

        if only_best:
            return predictions, np.argmax(weights, axis=1)
        return predictions, weights
