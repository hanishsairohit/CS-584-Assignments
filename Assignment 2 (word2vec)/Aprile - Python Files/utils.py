#!/usr/bin/env python

import numpy as np

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    ### YOUR CODE HERE

    # Using l2-normalization because 11-normalization is less stable
    # Get sum of squares for each row
    sum_of_squares = np.sum(x**2, axis=1).reshape((len(x), 1))

    # Normalize x
    x = x / np.sqrt(sum_of_squares)

    ### END YOUR CODE
    return x
    

def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. 

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    ### YOUR CODE HERE

    # Matrix - requires specifying axis so row versus column sum
    if x.ndim > 1:
        x = np.exp(x) / np.sum(np.exp(x), axis=1)

    # Vector - does not have a second dimension, so cannot use axis=1
    else:
        x = np.exp(x) / np.sum(np.exp(x), axis=0)

    ### END YOUR CODE
    return x