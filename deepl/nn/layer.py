import math
from .__base__ import Module, Parameter
from . import functional as F

#########################
######  Layers   ########
#########################
    
class ReLU(Module):
    """
    ReLU (activation) layer
    """
    def __init__(self,):
        super().__init__()
        
    def forward(self, x):
        fx = F.relu_forward(x)
        return fx
    
    def backward(self, grad):
        grad_input = F.relu_backward(grad, self.input)
        return grad_input
    
class LeakyReLU(Module):
    """
    LeakyReLU (activation) layer
    """
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope=negative_slope
        
    def forward(self, x):
        fx = F.leakyrelu_forward(x, self.negative_slope)
        return fx
    
    def backward(self, grad):
        grad_input = F.leakyrelu_backward(grad, self.input, self.negative_slope)
        return grad_input
    
class Sigmoid(Module):
    """
    Sigmoid (activation) layer
    """
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        fx = F.sigmoid_forward(x)
        return fx
    
    def backward(self, grad):
        grad_input = F.sigmoid_backward(grad, self.input)
        return grad_input
    
class Softmax(Module):
    """
    Softmax (activation) layer
    """
    def __init__(self, axis=1):
        super().__init__()
        self.axis=axis

    def forward(self, x):
        fx = F.softmax_forward(x, self.axis)
        return fx
    
    def backward(self, grad):
        grad_input = F.softmax_backward(grad, self.input, self.axis)
        return grad_input
    
class Tanh(Module):
    """
    Tanh (activation) layer
    """
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        fx = F.tanh_forward(x)
        return fx
    
    def backward(self, grad):
        grad_input = F.tanh_backward(grad, self.input)
        return grad_input
    
class Flatten(Module):
    """
    Flatten layer
    """
    def __init__(self,):
        super().__init__()
    
    def forward(self, x):
        fx = F.flatten_forward(x)
        return fx
    
    def backward(self, grad):
        grad_input = F.flatten_backward(grad, self.input)
        return grad_input
    
class MaxPool2d(Module):
    """
    MaxPool2d (Maximum pooling) layer
    """
    def __init__(self,):
        super().__init__()
    
    def forward(self, x):
        fx = F.maxpool2d_forward(x)
        return fx
    
    def backward(self, grad):
        grad_input = F.maxpool2d_backward(grad, self.input)
        return grad_input

class Linear(Module):
    """
    Linear (Fully-connected) layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        
        self.weight = Parameter(F.init.kaiming_uniform((out_features, in_features)))
        self.bias = Parameter(F.init.uniform((out_features,), math.sqrt(1/in_features)))
        
    def forward(self, x):
        fx = F.linear_forward(x, self.weight(), self.bias())
        return fx
    
    def backward(self, grad):
        grad_input, grad_weight, grad_bias = F.linear_backward(grad, self.input,
                                                               self.weight(), self.bias())
        self.weight.grad = grad_weight
        self.bias.grad = grad_bias
        return grad_input
    
class Conv2d(Module):
    """
    Conv2d (Convolutional) layer
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=(3,3),
                 stride=(1,1),
                 padding=(0,0)):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        
        self.weight = Parameter(F.init.kaiming_uniform((out_channels, in_channels, *kernel_size)))
        self.bias = Parameter(F.init.uniform((out_channels, 1, 1, 1), math.sqrt(1/in_channels)))
        
    def forward(self, x):
        fx = F.conv2d_forward(x, self.weight(), self.bias(), self.stride, self.padding)
        return fx
    
    def backward(self, grad):
        grad_input, grad_weight, grad_bias = F.conv2d_backward(grad, self.input, self.weight(),
                                                               self.bias(), self.stride, self.padding)
        self.weight.grad = grad_weight
        self.bias.grad = grad_bias
        return grad_input