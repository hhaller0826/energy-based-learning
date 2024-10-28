import math
from helpers import *

class JaynesMachine:
    # TODO: implement Jaynes machine that can go through layer-by-layer
    def __init__(self, weights):
        self.W = weights

# TODO: Jaynes Neuronal Layer

class JaynesConnectionLayer:
    def __init__(self, weights, num_bins):
        self.W = weights
        self.M_l = len(weights)
        self.m = num_bins
        self.bins = bin_values(weights, num_bins)
        
    def potential(self, utility_scale, disutility_scale):
        '''
        Args: 
            utility_scale: alpha
            disutility_scale: beta

        Returns the Jaynes potential for this layer as given in "Arbitrage equilibrium 
            and the emergence of universal microstructure" by V. Venkatasubramanian 
            and those other guys:
            alpha * sum[x_k ln|w_ijk|] - beta * sum[x_k (ln|w_ijk|)^2] + entropy
        '''
        phi_l
        for k in range(0,self.m):
            phi_l += self.bin_connection(k, utility_scale, disutility_scale)

        return phi_l + self.entropy()

    
    def bin_connection(self, k, utility_scale, disutility_scale):
        '''
        Args:
            k: the index of the bin
            utility_scale: alpha
            disutility_scale: beta

        Returns the Jaynes potential without entropy for a specified bin
        '''
        x_l_k = self.bins[k] / self.M_l
        # TODO: genuinely kinda unsure abt |w_ijk|
        #       currently treating it as the value of the bin itself
        w_ijk = k / self.m 
        ln = math.log(w_ijk)

        u = utility_scale * x_l_k * ln
        v = disutility_scale * x_l_k * ln * ln
        return u - v


    def entropy(self):
        '''
        Returns the entropy function as given in "Arbitrage equilibrium and the emergence
            of universal microstructure" by V. Venkatasubramanian and those other guys:
            1/M_l * ln[M_l! / prod[M_l*x_k]]
        '''
        product = 1
        for M_l_k in self.bins:
            product = product * M_l_k
        
        log_term = math.factorial(self.M_l) / math.factorial(product)
        return 1/self.M_l * math.log(log_term)

