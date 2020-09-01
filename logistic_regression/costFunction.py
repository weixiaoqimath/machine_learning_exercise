import numpy as np
from sigmoid import sigmoid
def costFunction(X, y, theta):
    """
        theta is a parameter column vector. X is trainX with one column added.
    """
    m = y.shape[0]
    return 1/m*(-np.transpose(y)@np.log(sigmoid(X@theta))-(1-np.transpose(y))@np.log(1 - sigmoid(X@theta)))


# In[ ]:




