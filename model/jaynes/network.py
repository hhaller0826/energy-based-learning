from layer import *

class JaynesNetwork:
    def __init__(self, data, num_bins):
        # TODO: THIS IS BASICALLY PSEUDOCODE 
        self.weights = []
        self.neurons = []
        self.num_layers = len(data)
        for layer in range(self.num_layers):
            self.weights.append(JaynesConnectionLayer(data[layer].weights, num_bins))
            self.neurons.append(JaynesNeuronalLayer(data[layer].neurons, num_bins))

    def weight_potential(self, alpha, beta):
        '''
        Return the Jaynes Potential Energy across all layers of weights in the network.
        '''
        phi_w = 0
        weight_layer : JaynesConnectionLayer
        for weight_layer in self.weights:
            phi_w += weight_layer.potential(alpha, beta)
        return phi_w
    
    def network_connection_entropy(self):
        '''
        Return the network-wide connection entropy
        '''
        entropy_w = 0
        weight_layer : JaynesConnectionLayer
        for weight_layer in self.weights:
            entropy_w += weight_layer.entropy()
        return entropy_w 
    
    def neuronal_potential(self, eta, zeta):
        '''
        Return the Jaynes Potential Energy across all layers of neurons in the network.
        '''
        phi_N = 0
        neuron_layer : JaynesNeuronalLayer
        for neuron_layer in self.neurons:
            phi_N += neuron_layer.potential(eta, zeta)
        return phi_N
    
    def network_neuron_entropy(self):
        '''
        Return the network-wide neuronal entropy
        '''
        entropy_n = 0
        neuron_layer : JaynesNeuronalLayer
        for neuron_layer in self.neurons:
            entropy_n += neuron_layer.entropy()
        return entropy_n 
