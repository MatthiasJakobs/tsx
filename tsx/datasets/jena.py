import torch
import numpy as np
import pandas as pd
from tsx.datasets.utils import download_and_unzip
from os.path import join

URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'

def load_jena(full_features: bool = False, resample: str ='60T', return_numpy: bool = False, return_pytorch: bool = False):
    """ Returns the Jena Climate 2009 - 2016 dataset

    Args:
        full_feature: return all features (true) or selection of informative features
        resample: string in pandas resample notation
        return_numpy: returns dataset as a numpy array
        return_pytorch: returns dataset as a pytorch tensor
    """
    path = download_and_unzip(URL, 'jena')
    path = join(path, 'jena_climate_2009_2016.csv')

    X = pd.read_csv(path)
    X['Date Time'] = pd.to_datetime(X['Date Time'], format='%d.%m.%Y %H:%M:%S')
    X = X.set_index('Date Time')

    if not full_features:
        selected_features = [X.columns[i] for i in [0, 1, 5, 7, 8, 10, 11]]
        X = X[selected_features]

    # Resample to desired length. Default is 60 minutes (as suggested in https://github.com/keras-team/keras-io/blob/master/examples/timeseries/timeseries_weather_forecasting.py) 
    X = X.resample(resample).mean()

    if return_numpy:
        return X.to_numpy().astype(np.float32)
    if return_pytorch:
        return torch.from_numpy(X.to_numpy()).float()

    return X
