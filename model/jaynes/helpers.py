import math

def bin_counts(values, m):
    '''
    Return an array of size m where each value of the array
    corresponds to the number of values in that bin.

    ex.: [0.05, 0.09, 0.15, 0.20, 0.33, 0.39, 0.80] with 10 bins 
        would return [2, 1, 1, 2, 0, 0, 0, 0, 1, 0]

    Assumes all values are scaled 0-1
    '''
    bins = [0] * m
    bin_size = 1 / m
    for val in values:
        bins[int(val / bin_size)] += 1
    return bins

def ln_continuous_factorial(x):
    '''
    Returns ln(x!)
    '''
    return math.lgamma(x+1)