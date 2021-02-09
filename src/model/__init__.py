import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


# ---------- base model ---------- #
class Model(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def get_paramaters(self):
        pass


# ---------- layers ---------- #
class Conv2dBNLeaky(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding=0):
        super(Conv2dBNLeaky, self).__init__()

        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.activation(x)
        return out


class Route(nn.Module):
    def __init__(self, layers):
        super(Route, self).__init__()
        self.layers = layers

    def forward(self, xs):
        return torch.cat(xs, axis=1)


class Reorg(nn.Module):
    def __init__(self, stride):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.data.size()
        ws = self.stride
        hs = self.stride
        x = x.view(b, c, int(h / hs), hs, int(w / ws), ws).transpose(3, 4).contiguous()
        x = x.view(b, c, int(h / hs * w / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(b, c, hs * ws, int(h / hs), int(w / ws)).transpose(1, 2).contiguous()
        x = x.view(b, hs * ws * c, int(h / hs), int(w / ws))
        return x
