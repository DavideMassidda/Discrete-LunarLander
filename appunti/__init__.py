import numpy as np
import time
from . import DQN
from . import PGM

def running_mean(x, window=50):
    x = np.array(x)
    kernel = np.ones(window)
    conv_len = x.shape[0]-window
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+window]
        y[i] /= window
    return y
