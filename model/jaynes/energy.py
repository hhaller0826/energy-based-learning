import math
import matplotlib.pyplot as plt
from helpers import *

class JaynesMachine:
    # TODO: implement Jaynes machine that can go through layer-by-layer
    def __init__(self, weights):
        self.W = weights

# TODO: Jaynes Neuronal Layer

class JaynesConnectionLayer:
    def __init__(self, weights, num_bins):
        self.M_l = len(weights)
        self.m = num_bins
        self.bin_counts = bin_values(weights, num_bins)
        self.bin_weights = [k / self.m for k in range(0,self.m)]
        
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
        phi_l = 0
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
        if self.bin_counts[k] == 0:
            return 0
        x_l_k = self.bin_counts[k] / self.M_l
        # TODO: genuinely kinda unsure abt |w_ijk|
        #       currently treating it as the value of the bin itself
        w_ijk = self.bin_weights[k]
        if w_ijk == 0: ln = 0
        else: ln = math.log(w_ijk)

        u = utility_scale * x_l_k * ln
        v = disutility_scale * x_l_k * ln * ln
        return u - v


    def entropy(self):
        '''
        Returns the entropy function as given in "Arbitrage equilibrium and the emergence
            of universal microstructure" by V. Venkatasubramanian and those other guys:
            1/M_l * ln[M_l! / prod[M_l*x_k]]
        '''
        d = 0
        for M_l_k in self.bin_counts:
            if M_l_k != 0: d += math.lgamma(M_l_k)
        
        n = math.lgamma(self.M_l)
        log_term = n - d
        return log_term / self.M_l
    
    def plot(self):
        '''
        Plot a histogram for this layer
        '''
        plt.bar(x=self.bin_weights, height=self.bin_counts, width=1/self.m, align='edge')
        plt.show()

    def test_print(self, alpha, beta):
        '''
        Print helpful testing info for the given alpha & beta vals
        '''
        print("    bin | count | bin connection")
        print("--------------------------------")
        for k in range(0, self.m):
            print_row = "bin " + str(self.bin_weights[k])
            print_row += " | " + str(self.bin_counts[k])
            print_row += "     | " + str(self.bin_connection(k,alpha, beta))
            print(print_row)
        print("Potential: ", self.potential(alpha, beta))
        print("Entropy:", self.entropy())