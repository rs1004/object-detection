import torch.nn as nn
from abc import ABCMeta, abstractmethod


class Model(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def get_paramaters(self):
        pass
