import numpy as np
def computeCostMulti(X, y, theta):
    # theta is the column vector of parameters

    m = X.shape[0]
    return 1/(2*m)*np.transpose(X@theta - y)@(X@theta - y)




