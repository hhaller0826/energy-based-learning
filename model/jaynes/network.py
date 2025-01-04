import torch
from model.jaynes.layer import *
from model.function.network import Network
from model.function.interaction import Function
from model.variable.parameter import Bias
from model.jaynes.energy import *
from model.jaynes.entropy import *

class JaynesNetwork:
    def __init__(self, network: Network):
        # TODO: THIS IS BASICALLY PSEUDOCODE 
        self.network = network
        self.weights: list[Parameter] = [w for w in Function(network._function).params if type(w) is not Bias]
        self.neurons: list[Layer] = network.free_layers()
        self.num_bins = 100

    def weight_potential(self, alpha, beta):
        '''
        Return the Jaynes Potential Energy across all layers of weights in the network.
        '''
        return jaynes_network_connection_potential(self.network,self.num_bins,alpha,beta)
    
    def network_connection_entropy(self, entropy_type: EntropyType):
        '''
        Return the network-wide connection entropy
        '''
        return network_entropy(self.weights, entropy_type, self.num_bins)
    
    def neuronal_potential(self, eta, zeta):
        '''
        Return the Jaynes Potential Energy across all layers of neurons in the network.
        '''
        return jaynes_network_neuron_potential(self.network,self.num_bins,eta,zeta)
    
    def network_neuron_entropy(self, entropy_type: EntropyType):
        '''
        Return the network-wide neuronal entropy
        '''
        return network_entropy(self.neurons, entropy_type, self.num_bins) 
    
    
