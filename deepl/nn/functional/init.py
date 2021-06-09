import math
import numpy as np

def kaiming_uniform(size: tuple, a: float=math.sqrt(5), mode: str='fan_in'):
    fan_out, fan_in = size[-2:]
    if mode=='fan_in':
        fan = fan_in
    else:
        fan = fan_out
    gain = math.sqrt(2.0 / (1 + a ** 2))
    bound = gain * math.sqrt(3/fan_in)
    return np.random.uniform(-bound, bound, size=size)

def uniform(size=tuple, bound: float=1):
    return np.random.uniform(-bound, bound, size=size)