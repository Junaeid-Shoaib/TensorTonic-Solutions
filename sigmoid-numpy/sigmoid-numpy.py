import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    arr = np.array(x)
    y = 1/(1+np.exp(-arr))

    return y