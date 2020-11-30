from tsx.models.classifier import ROCKET
from tsx.datasets.ucr import load_ecg200
from tsx.datasets.utils import normalize
import torch

ds = load_ecg200(transforms=[normalize])
X_train, y_train = ds.torch(train=True)
X_test, y_test = ds.torch(train=False)
y_train = ((y_train + 1) / 2).long()
y_test = ((y_test + 1) / 2).long()

# Train model with Ridge regression
config_ridge = {
    "n_classes": 2,
    "ridge": True,
    "input_length": len(X_train[0])
}

model = ROCKET(**config_ridge)
model.fit(X_train.unsqueeze(1), y_train, X_test=X_test.unsqueeze(1), y_test=y_test)

# Train model with logistic regression
config_logistic = {
    "n_classes": 2,
    "ridge": False,
    "learning_rate": 1e-5,
    "epochs": 100,
    "input_length": len(X_train[0])
}

model = ROCKET(**config_logistic)
model.fit(X_train.unsqueeze(1), y_train, X_test=X_test.unsqueeze(1), y_test=y_test)
