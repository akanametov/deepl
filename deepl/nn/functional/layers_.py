import math
import numpy as np
from . import __base__ as T

##################################
#######  Layer functions  ########
##################################

def relu_forward(input):
    """
    forward pass of ReLU
    """
    out = T.relu(input)
    return out

def relu_backward(grad, input):
    """
    backward pass of ReLU
    """
    grad_input = None
    grad_input = grad * T.relu(input)
    return grad_input

def leakyrelu_forward(input, negative_slope=0.1):
    """
    forward pass of LeakyReLU
    """
    out = T.leakyrelu(input, negative_slope)
    return out

def leakyrelu_backward(grad, input, negative_slope=0.1):
    """
    backward pass of LeakyReLU
    """
    grad_input = None
    grad_input = grad * T.leakyrelu(input, negative_slope)
    return grad_input

def sigmoid_forward(input):
    """
    forward pass of Sigmoid
    """
    out = T.sigmoid(input)
    return out

def sigmoid_backward(grad, input):
    """
    backward pass of Sigmoid
    """
    grad_input = None
    grad_input = grad * T.sigmoid(input) * (1. - T.sigmoid(input))
    return grad_input

def tanh_forward(input):
    """
    forward pass of Tanh
    """
    out = T.tanh(input)
    return out

def tanh_backward(grad, input):
    """
    backward pass of Tanh
    """
    grad_input = None
    grad_input = grad * (1. - T.tanh(input)**2)
    return grad_input

def softmax_forward(input, axis=1):
    """
    forward pass of Softmax
    """
    out = T.softmax(input, axis=axis)
    return out

def softmax_backward(grad, input, axis=1):
    """
    backward pass of Softmax
    """
    grad_input = None
    grad_input = grad * T.softmax(input, axis=axis) * (1. - T.softmax(input, axis=axis))
    return grad_input

def flatten_forward(input):
    """
    forward pass of Flatten
    """
    out = input.reshape(len(input), -1)
    return out

def flatten_backward(grad, input):
    """
    backward pass of Flatten
    """
    grad_input = None
    grad_input = grad.reshape(input.shape)
    return grad_input

def linear_forward(input, weight, bias=None):
    """
    forward pass of Linear
    """
    out = input @ weight.T
    if bias is not None:
        out += bias
    return out

def linear_backward(grad, input, weight, bias=None):
    """
    bacward pass of Linear
    """
    grad_input = grad_weight = grad_bias = None
    
    grad_input = grad @ weight
    
    grad_weight = (grad.T @ input)/len(input)
    
    if bias is not None:
        grad_bias = grad.sum(axis=0)/len(input)
        
    return (grad_input, grad_weight, grad_bias)

def conv2d_forward(input, weight, bias=None, stride=(1,1), padding=(0,0)):
    """
    forward pass of Conv2d
    """
    batch_size, in_chan, in_height, in_width = input.shape
    out_chan, _, filter_height, filter_width = weight.shape
    kernel_size = (filter_height, filter_width)

    #assert (in_height + 2 * padding[0] - kernel_size[0]) % stride[0] == 0, 'Image height size is incorrect'
    #assert (in_width + 2 * padding[1] - kernel_size[1]) % stride[1] == 0, 'Image width size is incorrect'

    out_height = (in_height + 2 * padding[0] - kernel_size[0])// stride[0] + 1
    out_width = (in_width + 2 * padding[1] - kernel_size[1])// stride[1] + 1

    input_f = T.unfold(input, kernel_size, stride, padding)
    weight_f = weight.reshape(out_chan, -1)
    
    out = weight_f @ input_f
    
    if bias is not None:
        out += bias.reshape(-1, 1)
    
    out = out.reshape(out_chan, out_height, out_width, batch_size)
    out = out.transpose((3, 0, 1, 2))

    return out

def conv2d_backward(grad, input, weight, bias=None, stride=(1,1), padding=(0,0)):
    """
    backward pass of Conv2d
    """
    grad_input = grad_weight = grad_bias = None
    
    batch_size, in_chan, in_height, in_width = input.shape
    out_chan, _, filter_height, filter_width = weight.shape
    kernel_size = (filter_height, filter_width)

    grad_f = grad.transpose((1, 2, 3, 0)).reshape(out_chan, -1)
    input_f = T.unfold(input, kernel_size, stride, padding)

    grad_input_f = weight.reshape(out_chan, -1).T @ grad_f
    grad_input = T.fold(grad_input_f, input.shape, kernel_size, stride, padding)
    
    grad_weight_f = grad_f @ input_f.T
    grad_weight = grad_weight_f.reshape(weight.shape)
    
    if bias is not None:
        grad_bias = np.sum(grad, axis=(0, 2, 3))[:, None, None, None]

    return (grad_input, grad_weight, grad_bias)

def maxpool2d_forward(input, kernel_size=(2,2), stride=(2,2)):
    """
    forward pass of MaxPool2d
    """
    batch_size, in_chan, in_height, in_width = input.shape

    #assert (in_height - kernel_size[0]) % stride[0] == 0, 'Invalid height'
    #assert (in_width - kernel_size[1]) % stride[1] == 0, 'Invalid width'

    out_height = (in_height - kernel_size[0])//stride[0] + 1
    out_width = (in_width - kernel_size[1])//stride[1] + 1

    input_r = input.reshape(batch_size * in_chan, 1, in_height, in_width)
    input_f = T.unfold(input_r, kernel_size, stride)
    
    idx = np.argmax(input_f, axis=0)
    
    out = input_f[idx, np.arange(input_f.shape[1])]
    out = out.reshape((out_height, out_width, batch_size, in_chan))
    out = out.transpose((2, 3, 0, 1))

    return out

def maxpool2d_backward(grad, input, kernel_size=(3,3), stride=(1,1)):
    """ 
    backward pass of MaxPool2d
    """
    grad_input = None
    
    batch_size, in_chan, in_height, in_width = input.shape

    grad_f = grad.transpose((2, 3, 0, 1)).flatten()
    input_r = input.reshape(batch_size * in_chan, 1, in_height, in_width)
    input_f = T.unfold(input_r, kernel_size, stride)
    grad_input_f = np.zeros_like(input_f)
    
    idx = np.argmax(input_f, axis=0)
    grad_input_f[idx, np.arange(grad_input_f.shape[1])] = grad_f
    
    grad_input_r = T.fold(grad_input_f, input_r.shape, kernel_size, stride)
    grad_input = grad_input_r.reshape(input.shape)

    return grad_input