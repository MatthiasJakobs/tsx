import torch
import torch.nn as nn

from tsx.models.classifier import BasePyTorchClassifier

class TimeSeries1DNet(BasePyTorchClassifier):

    def __init__(self, input_size=1, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = self._conv1dblock(input_size,   32, kernel_size, padding=kernel_size // 2)
        self.conv2 = self._conv1dblock(32,  64, kernel_size, padding=kernel_size // 2)
        self.conv3 = self._conv1dblock(64, 128, kernel_size, padding=kernel_size // 2)

        self.avg_pool = nn.AvgPool1d(80)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(128, self.n_classes)

    def _conv1dblock(self, in_features, out_features, kernel_size=3, padding=0):
        return nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )

    def preprocessing(self, x_train, y_train, X_test=None, y_test=None):
        return x_train, y_train, X_test, y_test

    def forward(self, x):
        x = self.get_features(x)

        x = self.avg_pool(x)
        x = self.flatten(x)

        x = self.dense(x)
        return x

    def predict(self, X):
        prediction = nn.functional.softmax(self.forward(X), dim=-1)
        return torch.argmax(prediction, axis=-1).squeeze()

    def get_features(self, x, numpy=False):
        features = self.conv3(self.conv2(self.conv1(x))) 
        if numpy:
            return features.detach().numpy()
        return features

    def get_class_weights(self, numpy=False):
        w = self.dense.weight.clone()
        w = w.detach()
        if numpy:
            return w.numpy()
        return w