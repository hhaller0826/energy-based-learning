import torch
import torch.nn as nn

"""
dist_from_center should be a function that is about 1 when |dists/sigma| <= 1
and about 0 otherwise. Approximation of smooth uniform.
"""


def generalized_gaussian(dists, sigma, beta = 10): 
    # Strange behavior when dists/sigma is exactly 1 - 
    # this would be very likely to happen if sigma is
    # a negative power of 10 (e.g. 10^-5). Add irrational stabilizer to prevent this
    

    exponent = -0.5 * (dists / sigma) ** beta

    
    return torch.exp(exponent)

def weighted_softargmax(dists, sigma, beta = -1000): #sigma is unused

    print(dists)

    soft_arg_max = nn.Softmax(dim=0)

    return soft_arg_max(torch.abs(dists) * beta)

    

def ln_continuous_factorial(x):
    return torch.lgamma(x+1)

def comb_entropy_no_diff(W, num_bins):

    M_l = torch.Tensor([len(W)])

    M_k_l = torch.histc(input = W, bins = num_bins, min = 0, max = 1)

    final_entr = 1/M_l * (ln_continuous_factorial(M_l) - torch.sum(ln_continuous_factorial(M_k_l)))
    return final_entr.item()

def comb_entropy_diff(W, num_bins, dist_from_center = weighted_softargmax):

    #M_l = torch.Tensor([len(W)])
    
    
    range_step = 1/num_bins

    bin_centers = torch.arange(start= range_step/2, end = 1, step=range_step)

    dists = W.reshape(1, -1) - bin_centers.reshape(-1, 1) #of shape (num_bins, num_weights)


    dists = dist_from_center(dists=dists, sigma = range_step/2)

    print(dists)

    M_k_l = torch.sum(dists, dim = -1)

    M_l = torch.sum(dists) # This seems to perform better than using number of weights directly


    #print("M_l", M_l)
    #print("M_k_l", M_k_l)

    #print(torch.sum(ln_continuous_factorial(M_k_l)))

    final_entr = 1/M_l * (ln_continuous_factorial(M_l) - torch.sum(ln_continuous_factorial(M_k_l)))
    return final_entr.item()

num_bins = 23

#generate indices and weights
W_size = num_bins

W = torch.rand(W_size)

print("Differentiable Combinatorial Entropy (Approximation) Using Softmax", comb_entropy_diff(W, num_bins=num_bins))
print("Combinatorial Entropy True Value", comb_entropy_no_diff(W, num_bins=num_bins))