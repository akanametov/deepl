import math
import numpy as np
from .__base__ import Optimizer
    
class SGD(Optimizer):
    """
    SGD (Stochastic Gradient Descent) optimizer
    """
    def __init__(self, lr=0.001):
        self.lr=lr
        self.grads=None
    
    def step(self, parameters, grads):
        self.grads=list(grads).copy()
        params=[[w - self.lr* dw,
                 b - self.lr* db] \
                 for ((w, b), (dw, db)) in zip(parameters, self.grads)]
        return params
    
class Momentum(Optimizer):
    """
    SGD with momentum (Stochastic Gradient Descent with momentum) optimizer
    """
    def __init__(self, lr=0.001, beta=0.9):
        self.lr=lr
        self.beta=beta
        self.grads=None
        self.moments=None
    
    def step(self, parameters, grads):
        self.grads=list(grads).copy()
        if self.moments is None:
            self.moments=[[self.lr* dw,
                           self.lr* db] \
                           for (dw, db) in self.grads]
        else:
            self.moments=[[self.beta*wm + self.lr* dw,
                           self.beta*bm + self.lr* db] \
                           for ((wm, bm), (dw, db)) in zip(self.moments, self.grads)]
        
        params=[[w - wm,
                 b - bm] \
                 for ((w, b), (wm, bm)) in zip(parameters, self.moments)]
        return params
    
class AdaGrad(Optimizer):
    """
    AdaGrad (Adaptive Gradient) optimizer
    """
    def __init__(self, lr=0.001, epsilon=1e-8):
        self.lr=lr
        self.epsilon=epsilon
        self.grads=None
        self.squares=None
    
    def step(self, parameters, grads):
        self.grads=list(grads).copy()
        if self.squares is None:
            self.squares=[[dw*dw,
                           db*db] \
                           for (dw, db) in self.grads]
        else:
            self.squares=[[ws + dw*dw,
                           bs + db*db] \
                           for ((ws, bs), (dw, db)) in zip(self.squares, self.grads)]
        
        params=[[w - self.lr*dw/(np.sqrt(ws)+self.epsilon),
                 b - self.lr*db/(np.sqrt(bs)+self.epsilon)] \
                 for ((w, b), (dw, db), (ws, bs)) in zip(parameters, self.grads, self.squares)]
        return params
    
class RMSprop(Optimizer):
    """
    RMSprop (Root Mean Squared propogation) optimizer
    """
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        self.lr=lr
        self.beta=beta
        self.epsilon=epsilon
        self.grads=None
        self.squares=None
    
    def step(self, parameters, grads):
        self.grads=list(grads).copy()
        if self.squares is None:
            self.squares=[[(1.-self.beta)*dw*dw,
                           (1.-self.beta)*db*db] \
                           for (dw, db) in self.grads]
        else:
            self.squares=[[self.beta*ws + (1.-self.beta)*dw*dw,
                           self.beta*bs + (1.-self.beta)*db*db] \
                           for ((ws, bs), (dw, db)) in zip(self.squares, self.grads)]

        params=[[w - self.lr*dw/(np.sqrt(ws)+self.epsilon),
                 b - self.lr*db/(np.sqrt(bs)+self.epsilon)] \
                 for ((w, b), (dw, db), (ws, bs)) in zip(parameters, self.grads, self.squares)]

        return params
    
class Adam(Optimizer):
    """
    Adam (Adaptive momentum) optimizer
    """
    def __init__(self, lr=0.001, betas=(0.9, 0.999), epsilon=1e-8):
        self.lr=lr
        self.betas=betas
        self.epsilon=epsilon
        self.grads=None
        self.moments=None
        self.squares=None
        self.t=0
        
        
    
    def step(self, parameters, grads):
        self.grads=list(grads).copy()
        self.t += 1
        t = self.t
        lr_t = self.lr*np.sqrt(1-pow(self.betas[1], t))/(1-pow(self.betas[0], t))
        
        if (self.moments is None) and (self.squares is None):
            self.moments=[[(1.-self.betas[0])*dw,
                           (1.-self.betas[0])*db] \
                           for (dw, db) in self.grads]
            self.squares=[[(1.-self.betas[1])*dw*dw,
                           (1.-self.betas[1])*db*db] \
                           for (dw, db) in self.grads]
        else:
            self.moments=[[self.betas[0]*wm + (1.-self.betas[0])*dw,
                           self.betas[0]*bm + (1.-self.betas[0])*db] \
                           for ((wm, bm), (dw, db)) in zip(self.moments, self.grads)]
            self.squares=[[self.betas[1]*ws + (1.-self.betas[1])*dw*dw,
                           self.betas[1]*bs + (1.-self.betas[1])*db*db] \
                           for ((ws, bs), (dw, db)) in zip(self.squares, self.grads)]
        
        params=[[w - lr_t*wm/(np.sqrt(ws)+self.epsilon),
                 b - lr_t*bm/(np.sqrt(bs)+self.epsilon)] \
                 for ((w, b), (wm, bm), (ws, bs)) in zip(parameters, self.moments, self.squares)]
        return params