import numpy as np
from sklearn.metrics import confusion_matrix

def accuracy(y_pred, y_true):
    """ accuracy score """
    return np.mean(y_pred == y_true)

def precision(y_pred, y_true, mean=True, epsilon=1e-8):
    """ precision score """
    cm = confusion_matrix(y_true, y_pred)
    pre = np.diag(cm)/(np.sum(cm, axis=0)+epsilon)
    if mean:
        return np.mean(pre)
    return pre

def recall(y_pred, y_true, mean=True, epsilon=1e-8):
    """ recall score """
    cm = confusion_matrix(y_true, y_pred)
    rec = np.diag(cm)/(np.sum(cm, axis = 1)+epsilon)
    if mean:
        return np.mean(rec)
    return rec

def f1_score(y_pred, y_true, mean=True, epsilon=1e-8):
    """ F1 score """
    pre = precision(y_pred, y_true, mean, epsilon)
    rec = recall(y_pred, y_true, mean, epsilon)
    f1 = 2*(rec * pre)/(rec + pre + epsilon)
    return f1

def iou(fx, y):
    """ Intersection over Union"""
    inter = (fx * y).sum()
    union = fx.sum() + y.sum()

    return inter/(union - inter + 1)

def dice(fx, y):
    """ DICE """
    inter = (fx * y).sum()
    union = fx.sum() + y.sum()

    return 2*inter/(union + 1)

def tversky(fx, y, betas=(0.5, 0.5)):
    """ Tversky """
    inter = (fx * y).sum()
    fps = (fx * (1 - y)).sum()
    fns = ((1 - fx) * y).sum()
    denom = inter + betas[0]*fps + betas[1]*fns

    return inter/(denom + 1)