import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import sin, cos, pow, Tensor


def convolve(in_channel: int, out_channel: int, kernel=3):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size=kernel),
        nn.Dropout1d(p=0.25),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
        nn.Conv1d(out_channel, out_channel, kernel_size=kernel),
        nn.Dropout1d(p=0.25),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    )


class CNN(nn.Module):
    def __init__(self, config: dict) -> None:
        super(CNN, self).__init__()

        kernel = config['kernel size']
        self.device = config['device']
        self.pool = nn.MaxPool1d(2)
        self.layers = nn.ModuleList([
            convolve(4, 200, kernel=kernel),
            convolve(200, 400, kernel=kernel),
            convolve(400, 400, kernel=kernel),
        ])
        self.output = nn.Linear(400 * 9, 1)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.output(x)
        return x
