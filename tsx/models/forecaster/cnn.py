import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from .base import BasePyTorchForecaster

class Shallow_FCN(BasePyTorchForecaster):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.forecaster = nn.Sequential(
            nn.Linear(736, 1)
        )

    def predict(self, x):
        with torch.no_grad():
            prediction = self.forward(x)
            return prediction.squeeze().numpy()

    def forward(self, x, return_intermediate=False):

        feats = self.feature_extractor(x)
        flatted = self.flatten(feats)
        prediction = self.forecaster(flatted)

        if return_intermediate:
            to_return = {}
            to_return['feats'] = feats
            to_return['logits'] = prediction
            to_return['output'] = prediction
            return to_return
        else:
            return prediction

    def reset_gradients(self):
        self.feature_extractor.zero_grad()
        self.forecaster.zero_grad()

    def preprocessing(self, X_train, y_train, X_test=None, y_test=None):
        return X_train, y_train, X_test, y_test


class Shallow_CNN_RNN(BasePyTorchForecaster):

    def __init__(self, cnn_filters=64, transformed_ts_length=11, output_size=1, hidden_states=100, **kwargs):
        super().__init__(**kwargs)

        self.cnn_filters = cnn_filters
        self.transformed_ts_length = transformed_ts_length
        self.output_size = output_size
        self.hidden_states = hidden_states

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, self.cnn_filters, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(self.cnn_filters, self.hidden_states)
        self.reset_hidden_states()

        self.dense = nn.Linear(transformed_ts_length, output_size)
        
    def reset_hidden_states(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        self.lstm_hidden_cell = (
            torch.zeros(1,batch_size, self.hidden_states), 
            torch.zeros(1,batch_size, self.hidden_states)
        )

    def forward(self, x, return_intermediate=False):
        feats = self.feature_extractor(x)
        batch_size, nr_filters, seq_length = feats.shape
        lstm_out, self.lstm_hidden_cell = self.lstm(feats.view(seq_length, batch_size, nr_filters), self.lstm_hidden_cell)
        prediction = self.dense(lstm_out[..., -1].view(batch_size, -1))

        if return_intermediate:
            output = {}
            output['feats'] = feats
            output['prediction'] = prediction
            output['logits'] = prediction
            return output

        return prediction

    def preprocessing(self, X_train, y_train, X_test=None, y_test=None):
        return X_train, y_train, X_test, y_test