import math
import numpy as np

#########################
###### Parameter ########
#########################

class Parameter(object):
    """
    Parameter
    """
    def __init__(self, data):
        self.data=data
        self.grad=None
    
    def __call__(self,):
        return self.data
    
#########################
####   Module base  #####
#########################

class Module(object):
    """
    Module (basic class for layers)
    """
    def __init__(self,):
        pass
    
    def forward(self, x):
        pass
    
    def backward(self, df):
        pass
    
    def __call__(self, x):
        self.input=x
        return self.forward(x)
    
    def train(self,):
        self.train=True
        return None
    
    def eval(self,):
        self.train=False
        return None
    
    def params(self,):
        self.param_names=[attr for attr in self.__dict__.keys() \
                          if isinstance(getattr(self, attr), Parameter)]
        return (getattr(self, p) for p in self.param_names)
        
    def parameters(self,):
        return (p.data for p in self.params())
    
    def named_parameters(self,):
        return ((n, p.data) for (n,p) in zip(self.param_names, self.params()))
    
    def grads(self,):
        return (p.grad for p in self.params())
    
    def named_grads(self,):
        return (('.'.join([n,'.grad']), p.data) for (n,p) in zip(self.param_names, self.params()))
    
    def update(self, params):
        for i, p in enumerate(self.param_names):
            getattr(self, p).data = params[i]
        return None

    def update_named(self, params):
        for p in self.param_names:
            getattr(self, p).data = params[p]
        return None
    
#########################
###  Network base  ######
#########################

class Network(object):
    """
    Network (basic class for neural network)
    """
    def __init__(self,):
        pass
    
    def __call__(self, x):
        return self.forward(x)
    
    def layers(self,):
        self.layer_names=[attr for attr in self.__dict__.keys() \
                          if isinstance(getattr(self, attr), Module)]
        return (getattr(self, l) for l in self.layer_names if len(list(getattr(self, l).parameters()))!=0)
    
    def all_layers(self,):
        self.all_layer_names=[attr for attr in self.__dict__.keys() \
                          if isinstance(getattr(self, attr), Module)]
        return (getattr(self, l) for l in self.all_layer_names)
    
    def backward(self, grad, return_grad=False):
        for layer in reversed(list(self.all_layers())):
            grad = layer.backward(grad)
        if return_grad:
            return grad
        return None
    
    def parameters(self,):
        return (list(l.parameters()) for l in self.layers())
    
    def named_parameters(self,):
        return ((n, dict(l.named_parameters())) for (n, l) in zip(layer_names, self.layers()))
    
    def grads(self,):
        return (list(l.grads()) for l in self.layers())
    
    def named_grads(self,):
        return ((n, dict(l.named_grads())) for (n, l) in zip(layer_names, self.layers()))
    
    def update(self, params):
        for i, l in enumerate(self.layers()):
            l.update(params[i])
        return None
    
    def update_named(self, params):
        for layer in self.layer_names:
            getattr(self, l).update(params[l])
        return None