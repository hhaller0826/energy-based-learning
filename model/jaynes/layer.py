import matplotlib.pyplot as plt
import torch
from torch import Tensor
from helpers import *

class JaynesVariable:
    """
    Necessary functions to get the energy/entropy of a Neural Network layer (could be a 
    layer of weights or neurons) using the Jaynes Machine energy functions as described in 
    "Arbitrage equilibrium and the emergence of universal microstructure" 
    by V. Venkatasubramanian and those other guys.
    """
    # TODO: should we scale the values here or assume they're scaled?
    def __init__(self, values: torch.Tensor, num_bins: int):
        self.values = values
        self.total_count = len(values) # M_l or N_l
        self.num_bins = num_bins # m or n
        self.bin_counts = torch.histc(input=values, bins=self.num_bins, min=0, max=1)
        self.weights = torch.linspace(0, 1, steps=num_bins+1)[:-1]
        # note to self: weights calculated like that bc, for example if u want 2 bins,
        # torch.linspace(0, 1, 2) = [0, 1] which to us is only 1 bucket; 
        # torch.linspace(0, 1, 3) = [0, 0.5, 1] which is correct, but we don't need the final value, hence [:-1]

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
        x_l = self.bin_counts / self.total_count
        ln = torch.cat((torch.tensor([0]), torch.log(self.weights[1:])), dim=0)

        u = utility_scale * torch.mul(x_l, ln)
        v = disutility_scale * torch.mul(x_l, torch.square(ln))

        return torch.sum(torch.sub(u,v)).item() + self.entropy()
    
    def entropy(self) -> float:
        '''
        Returns the entropy function as given in "Arbitrage equilibrium and the emergence
            of universal microstructure" by V. Venkatasubramanian and those other guys:
            weights: 1/M^l * ln[ M^l! / prod[M^l*x_k] ] 
            neurons: 1/N^l * ln[ N^l! / prod[N^l_q] ]  
        '''
        ln_prod_factorial = torch.sum(ln_continuous_factorial(self.bin_counts))
        ln_count_factorial = ln_continuous_factorial(self.total_count)
        entropy = (ln_count_factorial - ln_prod_factorial) / self.total_count
        return entropy.item()

    
    def plot(self):
        '''
        Plot a histogram for this layer
        '''
        plt.bar(x=self.weights, height=self.bin_counts, width=1/self.num_bins, align='edge')
        plt.show()

    def test_print(self, alpha, beta):
        '''
        Print helpful testing info for the given alpha & beta vals
        '''
        # print("    bin | count ")
        # print("--------------------------------")
        # for k in range(0, self.num_bins):
        #     print_row = "bin " + str(round(self.weights[k].item(),2))
        #     print_row += " | " + str(self.bin_counts[k].item())
        #     print(print_row)
        print("Potential: ", round(self.potential(alpha, beta),4))
        print("Entropy:", round(self.entropy(),4))


class JaynesNeuronalLayer(JaynesVariable):
    def __init__(self, iota, num_bins):
        super().__init__(iota, num_bins)


class JaynesConnectionLayer(JaynesVariable):
    def __init__(self, weights, num_bins):
        super().__init__(weights, num_bins)
