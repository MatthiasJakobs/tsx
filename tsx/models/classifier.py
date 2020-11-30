import torch
import torch.nn as nn
import numpy as np

from sklearn.linear_model import RidgeClassifier

class ROCKET(nn.Module):

    # TODO: Only for equal-length datasets?
    def __init__(self, nr_classes, input_length=10, k=10_000, ridge=True):
        super(ROCKET, self).__init__()
        self.k = k
        self.nr_classes = nr_classes
        self.ridge = ridge
        self.input_length = input_length

        self.kernels = []
        self.build_kernels()

        if self.ridge:
            raise NotImplementedException("TODO: Ridge regression")
        else:
            self.classifier = nn.Sequential(nn.Linear(2*self.k, self.nr_classes), nn.Softmax(dim=-1))

    def build_kernels(self):
        for i in range(self.k):
            kernel_length = [7, 9, 11][np.random.randint(0, 3)]

            weights = torch.normal(torch.zeros(1,1,kernel_length), 1)
            weights = weights - torch.mean(weights)

            bias = torch.rand(1)
            bias = (-1 - 1) * bias + 1

            # Parameter for dilation
            A = np.log2((self.input_length-1) / (float(kernel_length)-1))
            dilation = torch.floor(2**(torch.rand(1)*A)).long().item()

            padding = 0 if torch.rand(1)>0.5 else 1

            kernel = nn.Conv1d(1, 1, kernel_size=kernel_length, stride=1, padding=padding, dilation=dilation, bias=True)
            kernel.weight = nn.Parameter(weights, requires_grad=False)
            kernel.bias = nn.Parameter(bias, requires_grad=False)
            kernel.require_grad = False

            self.kernels.append(kernel)

    def ppv(self, x):
        return torch.mean((x > 0).float())

    def forward(self, x):
        transformed = []
        for i in range(self.k):
            transformed.append(self.kernels[i](x))

        features_ppv = [self.ppv(x) for x in transformed]
        features_max = [torch.max(x) for x in transformed]
        features = torch.tensor(features_ppv + features_max)
        return self.classifier(features)
