import torch
import torch.nn as nn
import pandas as pd

from tsx.metrics import smape

class BaseForecaster:

    def fit(self, X, y):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def forecast_single(self, X):
        raise NotImplementedError()

    def forecast_multi(self, X):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def inform(self, string):
        if self.verbose:
            print(string)

    def preprocessing(self, X_train, y_train, X_test=None, y_test=None):
        raise NotImplementedError()

class BasePyTorchForecaster(nn.Module, BaseForecaster):

    def __init__(self, loss=torch.nn.MSELoss, optimizer=torch.optim.Adam, batch_size=5, learning_rate=1e-3, verbose=False, epochs=5):
        super(BasePyTorchForecaster, self).__init__()
        self.classifier = False
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.epochs = epochs
        self.fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Expects X, y to be Pytorch tensors 
        X_train, y_train, X_val, y_val = self.preprocessing(X_train, y_train, X_test=X_val, y_test=y_val)

        ds = torch.utils.data.TensorDataset(X_train, y_train)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        loss_fn = self.loss()
        optim = self.optimizer(self.parameters(), lr=self.learning_rate)

        if X_val is not None and y_val is not None:
            logs = pd.DataFrame(columns=["train_loss", "val_loss", "val_smape"])
        else:
            logs = pd.DataFrame(columns=["train_loss"])

        for epoch in range(self.epochs):
            print_epoch = epoch + 1
            epoch_loss = 0.0
            for i, (X, y) in enumerate(dl):
                # reset repackage hidden states of RNN before next batch
                if hasattr(self, 'reset_hidden_states'):
                    self.reset_hidden_states(batch_size=len(X))
                optim.zero_grad()
                prediction = self.forward(X)
                loss = loss_fn(prediction, y)
                loss.backward()
                epoch_loss += loss.item()
                optim.step()

            if X_val is not None and y_val is not None:
                with torch.no_grad():
                    if hasattr(self, 'reset_hidden_states'):
                        self.reset_hidden_states(batch_size=len(X_val))
                    val_precition = self.forward(X_val)
                    val_loss = loss_fn(val_precition, y_val).item()
                    val_smape = smape(val_precition, y_val).item()

                logs.loc[epoch] = [epoch_loss, val_loss, val_smape]

                print("Epoch {} train_loss {} val_loss {}".format(print_epoch, epoch_loss, val_loss))
            else:
                print("Epoch {} train_loss {} ".format(print_epoch, epoch_loss))
                logs.loc[epoch] = [epoch_loss]

        self.fitted = True
        return logs
