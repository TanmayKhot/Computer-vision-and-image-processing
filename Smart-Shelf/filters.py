"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np
import scipy


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))


    # YOUR CODE HERE
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # YOUR CODE HERE
    for i in range(Hi - 1):
        for j in range(Wi - 1):
            for m in range(Hk):
                for n in range(Wk):
                    out[i, j] = out[i, j] + \
                        image[i - m + 1, j - n + 1] * kernel[m, n]

    # END YOUR CODE



    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """


    ### YOUR CODE HERE
    H, W = image.shape
    out = np.zeros((pad_height * 2 + H, pad_width * 2 + W))
    for i in range(H):
        for j in range(W):
            out[i + pad_height, j + pad_width] = image[i, j]
    # END YOUR CODE
    return out


def conv_fast(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # YOUR CODE HERE
    image = zero_pad(image, Hk, Wk)
    kernel = np.flip(np.flip(kernel, 1), 0)
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = sum(
                sum(kernel * image[i + 2:i + 2 + Hk, j + 2:j + 2 + Wk]))
    # END YOUR CODE

    return out
def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = np.flip(np.flip(g, 0), 1)
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    out = None
    ### YOUR CODE HERE
    g = g - np.mean(g)
    g = np.flip(np.flip(g, 0), 1)
    out = conv_fast(f, g)

    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
