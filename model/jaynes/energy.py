import torch
from torch import Tensor
from model.jaynes.helpers import *
from model.variable.variable import Variable
from model.function.network import Network
from model.function.interaction import Function
from model.variable.parameter import Bias
from model.jaynes.entropy import *

"""
Extremely generalized functions to calculate Jaynes energy so that they can be 
applied to most objects.
"""

def jaynes_network_neuron_potential(network, num_bins:int=10, utility_scale:float=1., disutility_scale:float=1.):
    """Get the Network-wide Jaynes neuron-potential for the given network: 
    sum [ eta * sum[x_k ln(Z_q)] - zeta * sum[x_k (ln(Z_q))^2] + entropy ]
    """
    phi_N = get_network_potentials(VariableType.NEURON, network, num_bins, utility_scale, disutility_scale)
    return sum(phi_N)

def jaynes_network_connection_potential(network, num_bins:int=10, utility_scale:float=1., disutility_scale:float=1.):
    """Get the Network-wide Jaynes connection-potential for the given network: 
    sum [ alpha * sum[x_k ln|w_ijk|] - beta * sum[x_k (ln|w_ijk|)^2] + entropy ]
    """
    phi_W = get_network_potentials(VariableType.CONNECTION, network, num_bins, utility_scale, disutility_scale)
    return sum(phi_W)

def get_network_potentials(variable_type:VariableType, network, num_bins:int=10, utility_scale:float=1., disutility_scale:float=1.):
    """Get a list Jaynes connection or neuron potentials for each layer the given network.

    Args: 
        potential_type (PotentialType):         NEURON or CONNECTION
        network (potential_type or Network or Function): the neural network we are getting the potential for
        num_bins (int, optional):               Number of bins over which we should distribute the network values
        utility_scale (float, optional):        alpha / eta
        disutility_scale (float, optional):     beta / zeta

    Raises:
        AssertionError: if the network argument is not an expected type
    """
    if type(network) is variable_type.value:
        # Can treat one parameter object as a single-layer or single-connection-layer network
        return jaynes_layer_potential(network, num_bins, utility_scale, disutility_scale)
    
    fn = (lambda: network, lambda: network._function)[type(network) is Network]
    assert type(fn) is Function, "Network type is not supported by this function"
    
    def phi_(p): return jaynes_layer_potential(p, num_bins, utility_scale, disutility_scale)
    
    match variable_type:
        case VariableType.NEURON:
            potentials = [phi_(p) for p in Function(fn).layers]
        case VariableType.CONNECTION:
            potentials = [phi_(p) for p in Function(fn).params if type(p) is not Bias]
    
    return potentials

def jaynes_layer_potential(variable:Variable, num_bins:int=10, utility_scale:float=1., disutility_scale:float=1., entropy:Entropy=None):
    '''
    Args: 
        utility_scale: alpha / eta
        disutility_scale: beta / zeta

    Returns the Jaynes potential for this layer as given in "Arbitrage equilibrium 
        and the emergence of universal microstructure" by V. Venkatasubramanian 
        and those other guys:
            weights: alpha * sum[x_k ln|w_ijk|] - beta * sum[x_k (ln|w_ijk|)^2] + entropy
            neurons: eta * sum[x_k ln(Z_q)] - zeta * sum[x_k (ln(Z_q))^2] + entropy
    '''
    if entropy is None: entropy = BinEntropy(variable._state, num_bins)
    return layer_potential(variable, num_bins, utility_scale, disutility_scale) + entropy.eval()

def layer_potential(variable:Variable, num_bins:int=10, utility_scale:float=1., disutility_scale:float=1.):
    """Get the layer's potential energy withOUT the entropy term
    """
    values = scale_magnitude(variable.state)
    bin_counts = torch.histc(input=values, bins=num_bins, min=0, max=1)
    weights = torch.linspace(0, 1, steps=num_bins+1)[:-1]

    x_l = bin_counts / len(values)
    ln = torch.cat((Tensor([0]), torch.log(weights[1:])), dim=0)

    u = utility_scale * torch.mul(x_l, ln)
    v = disutility_scale * torch.mul(x_l, torch.square(ln))

    return torch.sum(torch.sub(u,v)).item()
