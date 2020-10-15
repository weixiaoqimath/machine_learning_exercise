import math
import numpy as np
import matplotlib.pyplot as plt

def accuracy(trueLabels, predLabels):
        return float(sum(trueLabels == predLabels))/ float(len(trueLabels))

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

class SVM():
    """
        simple smo
    """
    def __init__(self, iterations=500, kernel="poly", C=1.0, tol = 0.001, gamma = "auto", degree = 3, coef0=0.0):
        self.kernels = {
            "linear" : self.linear,
            "poly" : self.poly,
            "rbf" : self.rbf,
            "sigmoid" : self.sigmoid
        }
        self.iterations = iterations
        self.kernel = self.kernels[kernel]
        self.C = C
        self.tol = tol
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def fit(self, X, y):
        """
            X: dataset without extended column of ones 
            y: target or label.
        """
        m = X.shape[0]
        if m != y.shape[0]:
            print("Error of simple_smo: Dimension of data and target don't match.")
            return None
        if self.gamma == "auto":
            self.gamma = 1/X.shape[1] # This is helpful for rbf and sigmoid.
        KerMat = self.kernel_matrix(X) # All product of xi and xj
        alphas = np.zeros((m,1))
        b = 0
        passes = 0
        while (passes < self.iterations):
            num_changed_alphas = 0
            for i in np.arange(m):
                Ei = KerMat[i,:]@(alphas*y) + b - y[i]
                if (y[i, 0]*Ei < - self.tol and alphas[i,0] < self.C) or ((y[i, 0]*Ei > self.tol) and (alphas[i,0] > 0)):
                    j = self.select_j(i, m)
                    Ej = KerMat[j,:]@(alphas*y) + b - y[j]
                    alphaIold = np.copy(alphas[i]); # prevent alphaIold being modified unexpectedly
                    alphaJold = np.copy(alphas[j]);
                    if (y[i] != y[j]):
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - self.C)
                        H = min(self.C, alphas[j] + alphas[i])
                    if L==H:
                        continue # continue to next i
                    eta = 2.0 * KerMat[i,j] - KerMat[i,i] - KerMat[j,j]
                    if eta >= 0:
                        continue
                    # Compute and clip new value for alphas[j]
                    alphas[j] = alphas[j] - y[j]*(Ei - Ej)/eta  
                    alphas[j] = self.clip(alphas[j], L, H)
                    if abs(alphaJold - alphas[j]) < 0.00001:
                        continue 
                    alphas[i] = alphas[i] + y[i]*y[j]*(alphaJold - alphas[j])
                    # compute b1 and b2
                    b1 = b - Ei- y[i]*(alphas[i]-alphaIold)*KerMat[i,i] - y[j]*(alphas[j]-alphaJold)*KerMat[i,j]
                    b2 = b - Ej- y[i]*(alphas[i]-alphaIold)*KerMat[i,j] - y[j]*(alphas[j]-alphaJold)*KerMat[j,j]
                    # compute b
                    if 0 < alphas[i] and alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] and alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2)/2.0  
                    num_changed_alphas += 1
                    # end if
            # end for
            if (num_changed_alphas == 0): 
                passes += 1
            else: 
                passes = 0
        self.b = b
        self.SV = X[np.where(alphas != 0)[0]] # support vectors
        self.SValphas = alphas[np.where(alphas != 0)[0]]
        self.SVlabels = y[np.where(alphas != 0)[0]]

    def clip(self, alpha, L, H):
        ''' 
            clip alpha to lie within range [L, H].
        '''
        if alpha < L:
            return L
        elif alpha > H:
            return H
        else:
            return alpha
    
    def select_j(self, i, m):
        """
            Select a number in np.arange(m) that doesn't equal to i.
        """
        j = np.random.choice(m-1, 1)
        if j == i:
            j = j + 1
        return j

    def predict(self, Xtest, prob = False):
        m = self.SV.shape[0]
        n = Xtest.shape[0]
        predLabels = np.zeros((n,1))
        for i in np.arange(n):
            inter = np.zeros((1, m))
            for j in np.arange(m):
                inter[0,j]=self.kernel(Xtest[i, :], self.SV[j, :])
            f = inter@(self.SValphas*self.SVlabels) + self.b
            # end calculation of f
            if prob == True:
                predLabels[i, 0] = f
            else:
                if f >= 0:
                    predLabels[i, 0] = 1
                else:
                    predLabels[i, 0] = -1
        if n == 1:
            return predLabels[0,0] # value instead of array
        return predLabels

    def kernel_matrix(self, X):
        """
            Return a matrix M such that M[i,j] = Kernel(X[i,:], X[j,:])
        """
        m = X.shape[0]
        KerMat = np.zeros(m**2).reshape((m,m))
        for i in np.arange(m):
            for j in np.arange(m):
                if j >= i:
                    KerMat[i, j] = self.kernel(X[i,:], X[j,:])
                else:
                    KerMat[i, j] = KerMat[j, i]
        return KerMat 

    def linear(self, u, v):
        return u@(v.T)

    def poly(self, u, v):
        """
            polynomial kernel (coef0 + \gamma u \cdot v)^d where d is degree. u and v are row vectors.
        """
        return (self.coef0 + self.gamma * u@(v.T))**self.degree

    def rbf(self, u, v):
        return math.exp(-self.gamma * np.dot((u-v),(u-v).T))

    def sigmoid(self, u, v):
        return math.tanh(self.gamma * u@v + self.coef0)

# In[25]:




