import math
import numpy as np

##################################
#####   Some functions    ########
##################################

def relu(x):
    """
    relu function
    """
    fx = np.maximum(0, x)
    return fx

def leakyrelu(x, negative_slope=0.1):
    """
    leaky relu function
    """
    fx = np.maximum(negative_slope*x, x)
    return fx

def sigmoid(x):
    """
    sigmoid function
    """
    fx = 1./(1. + np.exp(-x))
    return fx

def softmax(x, axis=1):
    """
    softmax function
    """
    x = x - np.max(x, axis=axis)[:,None]
    exp = np.exp(x)
    fx = exp/exp.sum(axis=axis)[:,None]
    return fx

def tanh(x):
    """
    tanh function
    """
    exp = np.exp(x)
    exp_ = np.exp(-x)
    fx = (exp - exp_)/(exp + exp_)
    return fx

def one_hot(y, num_classes=2):
    """
    one-hot encoder
    """
    encod = np.zeros((len(y), num_classes))
    encod[range(len(y)), list(y)] = 1.
    return encod

def get_indices(input_shape, kernel_size=(3,3), stride=(1,1), padding=(0,0)):
    """
    function to get indices for unfold and unfold 
    """
    batch_size, in_chan, in_height, in_width = input_shape

    out_height = (in_height + 2 * padding[0] - kernel_size[0])// stride[0] + 1
    out_width = (in_width + 2 * padding[1] - kernel_size[1])// stride[1] + 1

    i0 = np.repeat(np.arange(kernel_size[0]), kernel_size[1])
    i0 = np.tile(i0, in_chan)
    i1 = stride[0] * np.repeat(np.arange(out_height), out_width)
    
    j0 = np.tile(np.arange(kernel_size[1]), kernel_size[0] * in_chan)
    j1 = stride[1] * np.tile(np.arange(out_width), out_height)
    
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(in_chan), kernel_size[0] * kernel_size[1]).reshape(-1, 1)
    
    k, i, j = k.astype(int), i.astype(int), j.astype(int)
    
    return (k, i, j)


def unfold(input, kernel_size=(3,3), stride=(1,1), padding=(0,0)):
    """
    function to unfold(im2col) based on fancy indexing
    """
    batch_size, in_chan, in_height, in_width = input.shape
    
    vals = ((0,0), (0,0), (padding[0], padding[0]), (padding[1], padding[1]))
    input_padded = np.pad(input, vals, mode='constant')

    k, i, j = get_indices(input.shape, kernel_size, stride, padding)
    
    input_f = input_padded[:, k, i, j]

    input_f = input_f.transpose(1, 2, 0).reshape(in_chan*kernel_size[0]*kernel_size[1], -1)
    return input_f


def fold(input_f, input_shape, kernel_size=(3,3), stride=(1,1), padding=(0,0)):
    """
    function to fold(col2im) based on fancy indexing
    """
    batch_size, in_chan, in_height, in_width = input_shape
    
    height_padded, width_padded = in_height + 2*padding[0], in_width + 2*padding[1]
    input_padded = np.zeros((batch_size, in_chan, height_padded, width_padded), dtype=input_f.dtype)
    
    k, i, j = get_indices(input_shape, kernel_size, stride, padding)
    
    input = input_f.reshape(in_chan*kernel_size[0]*kernel_size[1], -1, batch_size)
    input = input.transpose((2, 0, 1))
    np.add.at(input_padded, (slice(None), k, i, j), input)
    if padding == (0,0):
        return input_padded
    return input_padded[:, :, padding[0]: -padding[0], padding[1]: -padding[1]]