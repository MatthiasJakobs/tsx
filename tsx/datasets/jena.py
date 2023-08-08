import torch
import numpy as np
import pandas as pd
from tsx.datasets.utils import download_and_unzip
from os.path import join

URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'

def load_jena(resample: str ='60T', return_numpy: bool = False, return_pytorch: bool = False):
    """
    returns the Jena Climate 2009 - 2016 dataset
    :param resample: String in pandas resample notation
    :param return_numpy: returns dataset as a numpy array
    :param return_pytorch: returns dataset as a pytorch tensor
    :return: the dataset 
    """
    path = download_and_unzip(URL, 'jena')
    path = join(path, 'jena_climate_2009_2016.csv')

    X = pd.read_csv(path)
    X['Date Time'] = pd.to_datetime(X['Date Time'], format='%d.%m.%Y %H:%M:%S')
    X = X.set_index('Date Time')

    # Resample to desired length. Default is 60 minutes (as suggested in https://github.com/keras-team/keras-io/blob/master/examples/timeseries/timeseries_weather_forecasting.py) 
    X = X.resample(resample).mean()

    if return_numpy:
        return X.to_numpy().astype(np.float32)
    if return_pytorch:
        return torch.from_numpy(X.to_numpy()).float()

    return X
