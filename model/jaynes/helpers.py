import math
import torch
from torch import Tensor
from enum import Enum

from model.variable.layer import Layer
from model.variable.parameter import Parameter

class VariableType(Enum):
    NEURON = Layer
    CONNECTION = Parameter

def ln_continuous_factorial(x) -> float:
    '''
    Returns ln(x!)
    '''
    if type(x) is torch.Tensor:
        return torch.lgamma(x+1)
    return math.lgamma(x+1)

def scale_magnitude(x: Tensor) -> Tensor:
    '''
    Scales the magnitude of all weights in the tensor into
    a range of 0 to 1
    '''
    # if there are negative values, shift all values up
    min = torch.min(x).item()
    if min < 0:
        x = torch.sub(x,min)

    # if there are values greater than 1, scale all values down
    max = torch.max(x).item()
    if max > 1:
        return torch.div(x,max)

    return x