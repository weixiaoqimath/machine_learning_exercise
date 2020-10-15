import numpy as np
def scaler(X, mean, std):
    """
        mean and std are two row vectors. For each row of X, it substract A and divide the resulting row by B elementwise. 
    """
    # In case std has zeros. Replace zeros by ones. 
    A = np.zeros((1, X.shape[1]))
    for i in np.arange(X.shape[1]):
        if std[0, i] == 0:
            A[0, i] = 1
        else:
            A[0, i] = std[0, i]
    return (X - np.ones((X.shape[0],1))@mean)/(np.ones((X.shape[0],1))@A)