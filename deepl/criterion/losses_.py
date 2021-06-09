import math
import numpy as np
from . import __base__ as F
from .__base__ import Criterion
    
class BCELoss(Criterion):
    """
    Binary Cross Entropy loss
    """
    def __init__(self,):
        super().__init__()
        self.epsilon=1e-6
        
    def forward(self, pred, target):
        loss = F.bce_forward(pred, target)
        return loss
    
    def backward(self,):
        pred = self.pred
        target = self.target
        
        grad = F.bce_backward(pred, target)
        return grad[:, None]
    
class BCEWithLogitsLoss(Criterion):
    """
    Binary Cross Entropy loss with Logits
    (no need in nn.Sigmoid layer)
    """
    def __init__(self,):
        super().__init__()
        self.epsilon=1e-6
        
    def forward(self, pred, target):
        loss = F.bcelogits_forward(pred, target)
        return loss
    
    def backward(self,):
        pred = self.pred
        target = self.target
        
        grad = F.bcelogits_backward(pred, target)
        return grad[:, None]
    
class CrossEntropyLoss(Criterion):
    """
    Cross Entropy loss
    (no need in nn.Softmax layer)
    """
    def __init__(self,):
        super().__init__()
        self.epsilon=1e-6
        
    def forward(self, pred, target):
        loss = F.xentropy_forward(pred, target)
        return loss
    
    def backward(self,):
        pred = self.pred
        target = self.target
        
        grad = F.xentropy_backward(pred, target)
        return grad