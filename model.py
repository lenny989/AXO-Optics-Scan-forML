"""Fully connected network: flattened image -> (dX, dZ). Everything is configurable."""
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class FCNet(nn.Module):
    def __init__(self, in_dim=128 * 128, hidden=(1024, 512, 128), p_drop=0.2,
                 activation="relu", batchnorm=True):
        super().__init__()
        layers, d = [nn.Flatten()], in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(ACTIVATIONS[activation]())
            if p_drop > 0:
                layers.append(nn.Dropout(p_drop))
            d = h
        layers.append(nn.Linear(d, 2))   # regression output: no activation
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    pass