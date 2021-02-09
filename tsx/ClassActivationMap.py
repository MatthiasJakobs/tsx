import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import TensorDataset


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv1DBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=0, activation=F.relu):
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class TimeSeries1DNet(nn.Module):

    def __init__(self, n_classes=2, kernel_size=7):
        super().__init__()
        self.conv1 = Conv1DBlock(1,   32, kernel_size, padding=kernel_size // 2)
        self.conv2 = Conv1DBlock(32,  64, kernel_size, padding=kernel_size // 2)
        self.conv3 = Conv1DBlock(64, 128, kernel_size, padding=kernel_size // 2)

        self.avg_pool = nn.AvgPool1d(40)
        # self.avg_pool2 = nn.AvgPool1d(22)

        self.flatten = Flatten()

        self.dense = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.avg_pool(x)
        x = self.flatten(x)

        x = self.dense(x)
        return x

    def cam(self, x, class_id):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        clz_weights = self.dense.weight[class_id]  # .view(-1, 1)

        batch_cam = torch.zeros(size=(x.shape[0], x.shape[2]))

        # Can we do this in parallel for each element? e.g. by using batched matrix multiplication?
        for i in range(x.size()[0]):
            sample_i = x[i]
            # cam = torch.zeros(sample_i.shape[1])
            clz_weights_i = clz_weights[i]
            # Iterate filters
            for m in range(sample_i.shape[0]):  # Zip loop?
                batch_cam[i, :] += clz_weights_i[m].item() * sample_i[m]

            ...
            # out = cam

            # Is close is really close...

            # out = torch.matmul(sample_i.T, clz_weights)
            # out = out.reshape(1, -1)

            # You might want to apply smoothing.

        # Apply min max normalization
        batch_cam = (batch_cam - torch.min(batch_cam, axis=1, keepdim=True)[0]) / (torch.max(batch_cam, axis=1, keepdim=True)[0] - torch.min(batch_cam, axis=1, keepdim=True)[0])
        return batch_cam


N_TS = 1000
N_SZ = 40
N_DIM = 1

class1_peak = 15
class2_peak = 30


def generate_class(n_ts, n_sz, peak_begin, peak_end, peak_height):
    wn = np.random.randn(n_ts, n_sz)
    # Mix two signals
    x = np.zeros(shape=(n_ts, n_sz))
    x[:, peak_begin:peak_end] = peak_height
    return wn + x


clz1_x = generate_class(N_TS, N_SZ, 10, 20, 5)
clz1_y = np.zeros(N_TS, dtype=np.int64)

clz2_x = generate_class(N_TS, N_SZ, 25, 30, -2)
clz2_y = np.ones(N_TS, dtype=np.int64)

data_x = np.concatenate([clz1_x, clz2_x], axis=0)
data_y = np.concatenate([clz1_y, clz2_y])

idx = np.arange(2 * N_TS)
np.random.shuffle(idx)

data_x = data_x.reshape(2 * N_TS, 1, N_SZ)
data_x = data_x[idx]
data_y = data_y[idx]

dataset = TensorDataset(torch.Tensor(data_x), torch.Tensor(data_y))
loader = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    batch_size=32
)


def train(data_loader, model, epochs):
    opt = SGD(model.parameters(), lr=0.001)
    loss_fn = CrossEntropyLoss()
    for i in range(epochs):

        for x, y in data_loader:
            opt.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y.type(torch.LongTensor))
            loss.backward()
            opt.step()
            print('ACC: {}. LOSS {}'.format(accuracy_score(F.softmax(y_hat).argmax(axis=1), y), loss.item()))


data = torch.randn(N_TS, N_DIM, N_SZ)

model = TimeSeries1DNet()

train(loader, model, 2)

for x, y in loader:
    cam = model.cam(x, y.type(torch.LongTensor))
    print(y.type(torch.LongTensor), cam)

print(model)

model(data)
