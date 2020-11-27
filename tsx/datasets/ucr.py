import pandas as pd
import numpy as np
import zipfile
import tempfile
import torch

from os.path import join, basename, dirname, exists
from urllib.request import urlretrieve
from shutil import rmtree as remove_dir

class UCR_UEA_Dataset:

    def __init__(self, name, path=None, download=True):
        self.name = name
        self.download = download
        self.path = path

        if self.path is None and self.download == False:
            raise ValueError("If you do not want to download the dataset, you need to provide a path!")

        self.download_or_load()

        self.x_train, self.y_train = self.parseData(self.train_path)
        self.x_test, self.y_test = self.parseData(self.test_path)

    def download_or_load(self):
        # based on code from https://github.com/alan-turing-institute/sktime/blob/master/sktime/datasets/base.py
        if self.path is None:
            self.path = join(dirname(__file__), "data", self.name)

        if self.download:
            if not exists(self.path):
                url = "http://timeseriesclassification.com/Downloads/{}.zip".format(self.name)
                dl_dir = tempfile.mkdtemp()
                zip_file_name = join(dl_dir, basename(url))
                urlretrieve(url, zip_file_name)

                zipfile.ZipFile(zip_file_name, "r").extractall(self.path)
                remove_dir(dl_dir)

        self.train_path = join(self.path, self.name + "_TRAIN.txt")
        self.test_path = join(self.path, self.name + "_TEST.txt")

    def parseData(self, path):
        features = []
        labels = []
        with open(path, "r") as fp:
            for line in fp:
                tokens = [s for s in line.split(" ") if s != ""]
                label = tokens[0]
                feature = np.array(tokens[1:]).astype(np.float32)

                features.append(pd.Series(feature))
                labels.append(label)

        self.same_length = len(np.unique([len(x) for x in features])) == 1
        if self.same_length:
            features = np.array(features)
        else:
            features = np.array(features, dtype=object)

        labels = np.array(labels).astype(float).astype(int)
        return np.array(features), np.array(labels).astype(float).astype(int)

    def torch(self, train=True):
        if self.same_length:
            if train:
                return torch.tensor(self.x_train).float(), torch.tensor(self.y_train).long() 
            else:
                return torch.tensor(self.x_test).float(), torch.tensor(self.y_test).long() 
        else:
            raise ValueError("Dataset {} contains time-series data with different length. Conversion to pytorch failed".format(self.name))
            if train:
                return self.x_train, self.y_train
            else:
                return self.x_test, self.y_test 


def load_ecg200(**kwargs):
    name = "ECG200"
    return UCR_UEA_Dataset(name, **kwargs)

def load_ecg5000(**kwargs):
    name = "ECG5000"
    return UCR_UEA_Dataset(name, **kwargs)

def load_dodgerloop_day(**kwargs):
    name = "DodgerLoopDay"
    return UCR_UEA_Dataset(name, **kwargs)
