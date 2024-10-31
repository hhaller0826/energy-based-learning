import math
import matplotlib.pyplot as plt
from helpers import *

class JaynesLayer:
    """
    Necessary functions to get the energy/entropy of a Neural Network layer (could be a 
    layer of weights or neurons) using the Jaynes Machine energy functions as described in 
    "Arbitrage equilibrium and the emergence of universal microstructure" 
    by V. Venkatasubramanian and those other guys.
    """
    def __init__(self, values, num_bins):
        self.total_count = len(values) # M_l or N_l
        self.num_bins = num_bins # m or n
        self.bin_counts = bin_counts(values, num_bins)

    def potential(self, utility_scale, disutility_scale):
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
        phi = 0
        for bin in range(0,self.num_bins):
            phi += self.bin_connection(bin, utility_scale, disutility_scale)

        return phi + self.entropy()
    
    def bin_connection(self, bin, utility_scale, disutility_scale):
        '''
        Args:
            bin: the index of the bin, usually denoted k or q
            utility_scale: alpha / eta
            disutility_scale: beta / zeta

        Returns the Jaynes potential without entropy for the specified bin
                  = [marginal utility] - [marginal disutility]
        (weights) = [alpha * x_k * ln(w_ijk)] + [beta * x_k * (ln(w_ijk))^2]
        (neurons) = [eta * x_q * ln(Z_q)] + [beta * x_q * (ln(Z_q))^2]
        '''
        if self.bin_counts[bin] == 0:
            return 0
        x_l_bin = self.bin_counts[bin] / self.total_count
        val = self.bin_value(bin)
        if val == 0: ln = 0
        else: ln = math.log(val)

        u = utility_scale * x_l_bin * ln
        v = disutility_scale * x_l_bin * ln * ln
        return u - v
    
    def bin_value(self, bin):
        '''
        weights: w_ijk
        neurons: Z_q

        Returns the value (weight or neuronal iota) of the specified bin.
        '''
        return bin / self.num_bins
    
    def entropy(self):
        '''
        Returns the entropy function as given in "Arbitrage equilibrium and the emergence
            of universal microstructure" by V. Venkatasubramanian and those other guys:
            weights: 1/M^l * ln[M^l! / prod[M^l*x_k]] 
            neurons: 1/N^l * ln[N^l! / prod[N^l_q]]  
        '''
        ln_prod_factorial = 0
        for bin_count in self.bin_counts:
            if bin_count != 0: ln_prod_factorial += ln_continuous_factorial(bin_count)
        
        ln_count_factorial = ln_continuous_factorial(self.total_count)
        return (ln_count_factorial - ln_prod_factorial) / self.total_count

    
    def plot(self):
        '''
        Plot a histogram for this layer
        '''
        weights = [self.bin_value(k) for k in range(self.num_bins)]
        plt.bar(x=weights, height=self.bin_counts, width=1/self.num_bins, align='edge')
        plt.show()

    def test_print(self, alpha, beta):
        '''
        Print helpful testing info for the given alpha & beta vals
        '''
        print("    bin | count | bin connection")
        print("--------------------------------")
        for k in range(0, self.m):
            print_row = "bin " + str(self.bin_value(k))
            print_row += " | " + str(self.bin_counts[k])
            print_row += "     | " + str(self.bin_connection(k,alpha, beta))
            print(print_row)
        print("Potential: ", self.potential(alpha, beta))
        print("Entropy:", self.entropy())


class JaynesNeuronalLayer(JaynesLayer):
    def __init__(self, iota, num_bins):
        JaynesLayer.__init__(self, iota, num_bins)


class JaynesConnectionLayer(JaynesLayer):
    def __init__(self, weights, num_bins):
        JaynesLayer.__init__(self, weights, num_bins)

    