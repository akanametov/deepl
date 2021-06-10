import math
import numpy as np
from ..nn.functional import __base__ as T
    
def bce_loss(x, y, epsilon=1e-6):
    """
    binary cross-entropy loss
    """
    bce = np.mean(x - x*y + np.log(1. + np.exp(-x)+epsilon))
    return bce

def focal_loss(x, y, gamma=2, epsilon=1e-6):
    """
    focal loss
    """
    fx = T.sigmoid(x)
    
    focal = -np.mean((1-fx)**gamma *y *np.log(fx+epsilon)\
                     + (1-y)*np.log(1-fx+epsilon)) 
    
    return focal

def xentropy_loss(x, y, epsilon=1e-6):
    """
    cross-entropy loss
    """
    fx = T.softmax(x)
    y = T.one_hot(y, np.max(y)+1)
    xentropy = - np.sum(y * np.log(fx+epsilon))/len(y)
    return xentropy

def bce_forward(fx, y, epsilon=1e-6):
    """
    forward pass of binary cross-entropy loss
    """
    loss = - (y.T @ np.log(fx + epsilon) + (1.-y).T @ np.log(1 - fx + epsilon))/len(y)
    return loss

def bce_backward(fx, y, epsilon=1e-6):
    """
    backward pass of binary cross-entropy loss
    """
    dloss = - (y/(fx + epsilon) - (1.-y)/(1 - fx + epsilon))/len(y)
    return dloss

def bcelogits_forward(x, y, epsilon=1e-6):
    """
    forward pass of binary cross-entropy loss with logits
    """
    fx = T.sigmoid(x)
    loss = bce_forward(fx, y, epsilon)
    return loss

def bcelogits_backward(x, y, epsilon=1e-6):
    """
    backward pass of binary cross-entropy loss with logits
    """
    fx = T.sigmoid(x)
    dloss = bce_backward(fx, y, epsilon)
    dfx = dloss * fx * (1. - fx)
    return dfx

def xentropy_forward(x, y, epsilon=1e-6):
    """
    forward pass of cross-entropy loss
    """
    fx = T.softmax(x)
    y = T.one_hot(y, np.max(y)+1)
    loss = - np.sum(y * np.log(fx+epsilon))/len(y)
    return loss

def xentropy_backward(x, y, epsilon=1e-6):
    """
    backward pass of cross-entropy loss
    """
    fx = T.softmax(x)
    i = range(len(y))
    j = list(y)
    fx[i, j] -= 1
    dfx = fx/len(y)
    return dfx

class Criterion(object):
    """
    Citerion base Class
    """
    def __init__(self,):
        pass
    
    def forward(self, pred, target):
        pass
    
    def __call__(self, pred, target):
        self.pred = pred
        self.target = target
        return self.forward(pred, target)
    
    def backward(self,):
        pass
