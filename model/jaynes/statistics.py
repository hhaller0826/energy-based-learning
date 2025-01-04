import torch
from abc import ABC, abstractmethod

from model.variable.layer import Layer
from model.function.network import Network
from training.statistics import Statistic

"""
Helper functions to get desired statistics and build an 
output file that records/displays these statistics.
"""

def layer_to_iota(layer:Layer):
    """
    Given a tensor of neuron values [y_i], returns 
    """
    return torch.mul(layer.state, layer.activate())

class CRISStatistic(Statistic):
    @abstractmethod
    def display_string(self) -> str:
        """Returns a display string that should be added to an output file"""
        return str(type(self))
    
class IotaStatistic(CRISStatistic):
    def __init__(self, layer:Layer, filename:str):
        self._layer = layer
        self._filename = filename
        self._name = 'Iota for ' + layer.name()

    def do_measurement(self):
        return layer_to_iota(self._layer)
    
    def display_string(self) -> str:
        return self._name + ': ' + str(self.do_measurement())
    