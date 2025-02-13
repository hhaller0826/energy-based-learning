import torch
from torch import Tensor
from helpers import *
from abc import ABC, abstractmethod
from model.variable.variable import Variable

def network_entropy(values: list[Variable], entropy_type: EntropyType, num_bins=None):
    def s_n(layer: Layer):
        if entropy_type is EntropyType.BINS:
            if type(num_bins) is int: entropy = BinEntropy(layer.state, num_bins)
            else: entropy = BinEntropy(layer.state)
        else: # Default to soft argmax
            entropy = SoftArgmaxEntropy(layer.state)
        return entropy.eval()

    entropies = [s_n(layer) for layer in values]

    return sum(entropies) 
class Entropy(ABC):
    """
    Abstract class to calculate the entropy of a Tensor of values. These values will typically be the 
    state vectors of a Parameter or Layer object, which correspond to a network's weights and neurons respecitvely.

    Attributes
    ----------
    _distribution (Tensor): distribution of the provided values that is appropriate for the given entropy technique

    Methods
    -------
    eval()
        Evaluates the entropy of the distribution
    """
    def __init__(self, values: Tensor):
        self._distribution = self.init_distribution(values)

    @abstractmethod
    def init_distribution(self, values):
        """Initializes the variable"""
        pass
    
    @abstractmethod
    def eval(self):
        """Evaluate the entropy"""
        pass 

class BinEntropy(Entropy):
    def __init__(self, values: Tensor, num_bins: int=10):
        self.num_bins = num_bins
        Entropy.__init__(self, values)

    def init_distribution(self, values):
        """Scale the values 0-1 and bin them appropriately"""
        return torch.histc(input=scale_magnitude(values), bins=self.num_bins, min=0, max=1)
    
    def eval(self):
        """Calculates the entropy function as given in "Arbitrage equilibrium and the emergence
            of universal microstructure" by V. Venkatasubramanian and those other guys:
            weights: 1/M^l * ln[ M^l! / prod[M^l*x_k] ] 
            neurons: 1/N^l * ln[ N^l! / prod[N^l_q] ]
        """
        total_count = sum(self._distribution)
        ln_prod_factorial = torch.sum(ln_continuous_factorial(self._distribution))
        ln_count_factorial = ln_continuous_factorial(total_count)
        entropy = (ln_count_factorial - ln_prod_factorial) / total_count
        return entropy.item()
    
class SoftArgmaxEntropy(Entropy):
    def init_distribution(self, values):
        """Returns the softmax function applied to the layer's state"""
        return torch.nn.functional.softmax(values, dim=-1)
    
    def eval(self):
        """Returns -sum[xlogx] summing over the x's in the distribution"""
        entropy_list = torch.mul(self._distribution, torch.log(self._distribution))
        return -1 * torch.sum(entropy_list).item()
