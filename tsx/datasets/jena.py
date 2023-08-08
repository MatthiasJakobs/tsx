import pandas as pd
from tsx.datasets.utils import download_and_unzip
from os.path import join

URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'

def load_jena(return_numpy: bool = False):
    path = download_and_unzip(URL, 'jena')
    path = join(path, 'jena_climate_2009_2016.csv')

    X = pd.read_csv(path)

    # Drop date column if numpy
    if return_numpy:
        X = X.drop(columns=['Date Time'])
        return X.to_numpy()

    return X
