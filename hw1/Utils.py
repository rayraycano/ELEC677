# Raymond Cano
# Utility functions for our neural networks
import numpy as np

def random_init(indim, outdim):
    return np.random.randn(indim, outdim) / np.sqrt(indim)