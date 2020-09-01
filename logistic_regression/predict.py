import numpy as np
from sigmoid import sigmoid
def predict(X, theta):
    return np.round(sigmoid(X @ theta))