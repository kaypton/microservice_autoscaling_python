from abc import ABC

import numpy
import abc

import torch


class Serializer(object):
    @abc.abstractmethod
    def to_ndarray(self) -> numpy.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_torch_tensor(self) -> torch.Tensor:
        raise NotImplementedError()


class DDPGState(Serializer, ABC):
    def __init__(self):
        pass


class DDPGAction(Serializer, ABC):
    def __init__(self):
        pass


