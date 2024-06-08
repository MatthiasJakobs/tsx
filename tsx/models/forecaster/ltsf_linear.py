# From https://github.com/cure-lab/LTSF-Linear

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, L, H, n_features, kernel_size=25, individual=False, normalize=False):
        super(DLinear, self).__init__()
        self.L = L
        self.H = H

        self.decompsition = SeriesDecomposition(kernel_size)
        self.channels = n_features
        self.individual = individual
        self.normalize = normalize

        if self.individual:
            self.seasonal = nn.ModuleList()
            self.trend = nn.ModuleList()
            
            for _ in range(self.channels):
                self.seasonal.append(nn.Linear(self.L, self.H))
                self.trend.append(nn.Linear(self.L, self.H))
        else:
            self.seasonal = nn.Linear(self.L, self.H)
            self.trend = nn.Linear(self.L, self.H)


    # x: [Batch, channel, Input length]
    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.normalize:
            seq_last = x[:,-1:,:].detach()
            x = x - seq_last            

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.H],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.H],dtype=trend_init.dtype).to(trend_init.device)

            for i in range(self.channels):
                seasonal_output[:,i,:] = self.seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.seasonal(seasonal_init)
            trend_output = self.trend(trend_init)

        x = seasonal_output + trend_output
        
        if self.normalize:
            x = x + seq_last   
        
        return x

class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, L, H, n_features, individual=False):
        super(NLinear, self).__init__()
        self.seq_len = L
        self.pred_len = H
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = n_features
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    # x: [Batch, Channel, input length]
    def forward(self, x):
        x = x.permute(0, 2, 1)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        x = x.permute(0, 2, 1)
        return x # [Batch, Channel, Output length]

