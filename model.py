import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import sin, cos, pow, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


def convolve(in_channel: int, out_channel: int, kernel: int = 3, dropout: int = 0.2):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel,
                  kernel_size=kernel, padding=kernel//2),
        nn.Dropout1d(p=dropout),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
        nn.Conv1d(out_channel, out_channel,
                  kernel_size=kernel, padding=kernel//2),
        nn.Dropout1d(p=dropout),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    )


class CNN(nn.Module):
    def __init__(self, config: dict) -> None:
        super(CNN, self).__init__()

        kernel = config['kernel size']
        dropout = config['Dropout']
        data_length = config['Data Length']
        self.device = config['device']
        self.pool = nn.MaxPool1d(2)

        self.layers = nn.ModuleList([])
        last = 4 + config['n order differences']
        for feature in config['CNN features']:
            data_length = data_length // 2
            layer = convolve(last, feature, kernel=kernel, dropout=dropout)
            self.layers.append(layer)
            last = feature

        self.output = nn.ModuleList([])
        last = data_length*last
        for feature in config['Output features']:
            layer = nn.Linear(last, feature)
            self.output.append(layer)
            self.output.append(nn.ReLU())
            last = feature

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        for layer in self.output:
            x = layer(x)
        return x


class RNN(nn.Module):
    def __init__(self, config: dict):
        super(RNN, self).__init__()

        bidirectional = config['bidirectional']
        hidden_size = config['hidden size']
        input_size = 4 + config['n order differences']
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(config['Dropout'])
        hidden_size *= 1 + bidirectional
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x_length):
        x, lengths = x_length
        x = pack_padded_sequence(
            x, lengths, enforce_sorted=False
        )
        x, hidden = self.rnn(x)
        x, _ = pad_packed_sequence(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x
