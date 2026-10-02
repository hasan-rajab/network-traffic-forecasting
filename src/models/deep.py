import torch
from torch import nn
class LSTMForecaster(nn.Module):
    def __init__(self,n_features=1,hidden_size=32):
        super().__init__(); self.lstm=nn.LSTM(n_features,hidden_size,batch_first=True); self.head=nn.Linear(hidden_size,1)
    def forward(self,x):
        y,_=self.lstm(x); return self.head(y[:,-1]).squeeze(-1)
