import math
import numpy as np
import matplotlib.pyplot as plt

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
        self.data = X
        self.labels = y
        m = X.shape[0]
        if m != y.shape[0]:
            print("Error of simple_smo: Dimension of data and target don't match.")
            return None
        if self.gamma == "auto":
            self.gamma = 1/X.shape[1]
        #KerMat = self.kernel_matrix(X) # All product of xi and xj
        alphas = np.zeros((m, 1))
        b = 0
        passes = 0
        while (passes < self.iterations):
            num_changed_alphas = 0
            for i in np.arange(m):
                Ei = self.f(X[i,:], alphas, b) - y[i]
                #print(Ei)
                if (y[i, 0]*Ei < - self.tol and alphas[i, 0] < self.C) or ((y[i, 0]*Ei > self.tol) and (alphas[i, 0] > 0)):
                    j = self.select_j(i, m)
                    Ej = self.f(X[j,:], alphas, b) - y[j]
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
                    eta = 2.0 * self.kernel(X[i,:], X[j,:]) - self.kernel(X[i,:], X[i,:]) - self.kernel(X[j,:], X[j,:])
                    if eta >= 0:
                        continue
                    # Compute and clip new value for alphas[j]
                    alphas[j] = alphas[j] - y[j]*(Ei - Ej)/eta  
                    alphas[j] = self.clip(alphas[j], L, H)
                    if abs(alphaJold - alphas[j]) < 0.00001:
                        continue 
                    alphas[i] = alphas[i] + y[i]*y[j]*(alphaJold - alphas[j])
                    # compute b1 and b2
                    b1 = b - Ei- y[i]*(alphas[i]-alphaIold)*self.kernel(X[i,:], X[i,:]) - y[j]*(alphas[j]-alphaJold)*self.kernel(X[i,:], X[j,:])
                    b2 = b - Ej- y[i]*(alphas[i]-alphaIold)*self.kernel(X[i,:], X[j,:]) - y[j]*(alphas[j]-alphaJold)*self.kernel(X[j,:], X[j,:])
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
        self.alphas = alphas
        self.b = b
        

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

    def f(self, x, alphas, b):
        m = self.data.shape[0]
        inter = np.zeros((1, m))
        for j in np.arange(m):
            inter[0,j]=self.kernel(x, self.data[j, :])
        return inter@(alphas*self.labels) + b


    def predict(self, Xtest, prob = False):
        m = self.data.shape[0]
        n = Xtest.shape[0]
        
        if n == 1:
            predLabel = 0
            inter = np.zeros((1, m))
            for j in np.arange(m):
                inter[0,j]=self.kernel(Xtest, self.data[j, :])
            f = inter@(self.alphas*self.labels) + self.b
            # end calculation of f
            if prob == True:
                return f
            if f >= 0:
                return 1
            else:
                return -1 

        predLabels = np.zeros((n,1))
        for i in np.arange(n):
            inter = np.zeros((1, m))
            for j in np.arange(m):
                inter[0,j]=self.kernel(Xtest[i, :], self.data[j, :])
            f = inter@(self.alphas*self.labels) + self.b
            # end calculation of f
            if f >= 0:
                predLabels[i, 0] = 1
            else:
                predLabels[i, 0] = -1
        return predLabels

    def kernel_matrix(self, X):
        """
            Return a matrix M such that M[i,j] = K(X[i,:], X[j,:])
        """
        m = X.shape[0]
        KerMat = np.zeros(m**2).reshape((m,m))
        for i in np.arange(m):
            for j in np.arange(m):
                KerMat[i, j] = self.kernel(X[i,:], X[j,:])
        return KerMat 
        
    def accuracy(self, trueLabels, predLabels):
        return float(sum(trueLabels == predLabels))/ float(len(trueLabels))

    def linear(self, u, v):
        return u@(v.T)

    def poly(self, u, v):
        """
            polynomial kernel (1 + u \cdot v)^d where d is degree. u and v are row vectors.
        """
        return (self.coef0 + self.gamma * u@(v.T))**self.degree

    def rbf(self, u, v):
        return math.exp(-self.gamma * np.dot((u-v),(u-v).T))

    def sigmoid(self, u, v):
        return math.tanh(self.gamma * u@v + self.coef0)

# In[25]:




