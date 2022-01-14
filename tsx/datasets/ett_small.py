"""
ETT-small Dataset presented in https://arxiv.org/pdf/2012.07436.pdf
"""

import torch
import pandas as pd
from urllib.request import urlretrieve
from os.path import join, dirname, exists
from dataclasses import dataclass

urls = {
    "h1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/450ea779973f463406e3c75e4ec154ebc1add6f5/ETT-small/ETTh1.csv",
    "h2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/450ea779973f463406e3c75e4ec154ebc1add6f5/ETT-small/ETTh2.csv",
    "m1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/450ea779973f463406e3c75e4ec154ebc1add6f5/ETT-small/ETTm1.csv",
    "m2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/450ea779973f463406e3c75e4ec154ebc1add6f5/ETT-small/ETTm2.csv",
}

@dataclass
class ETTSmall:

    def __init__(self, name):
        self.name = name
        self.download_or_load()

    def download_or_load(self):
        dataset_file_path = join(dirname(__file__), "data", f"ett{self.name}.csv")
        if not exists(dataset_file_path):
            urlretrieve(urls[self.name], dataset_file_path)

        df = pd.read_csv(dataset_file_path)
        df = df.drop(columns=["date"])
        self.columns = list(df.columns)
        self.X = df.to_numpy()

    def torch(self):
        return torch.from_numpy(self.X).float()

    def numpy(self):
        return self.X