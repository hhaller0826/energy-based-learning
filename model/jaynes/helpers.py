
def bin_values(values, m):
        '''
        Return an array of size m where each value of the array
        corresponds to the number of values in that bin.

        Assumes all values are scaled 0-1
        '''
        bins = [0] * m
        bin_size = 1 / m
        for val in values:
            bins[int(val / bin_size)] += 1
        return bins