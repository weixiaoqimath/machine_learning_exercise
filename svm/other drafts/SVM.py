#!/usr/bin/env python
# coding: utf-8

# In[24]:

import math
import numpy as np
import matplotlib.pyplot as plt

def clip(alpha, L, H):
    ''' 
        clip alpha to lie within range [L, H].
    '''
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha
    
def select_j(i, m):
    """
        Select a number in np.arange(m) that doesn't equal to i.
    """
    j = np.random.choice(m-1, 1)
    if j == i:
        j = j + 1
    return j

def get_w(alphas, dataset, target):
    """
        alphas is a column vector. w is a column vector. dataset is not extended by ones.
        f = x*w + b where x is a row of the dataset.
    """
    w = (dataset.T)@(alphas*target)
    return w

def predict(w, b, X):
    """
        X is data.
    """
    m = X.shape[0]
    labels = np.zeros((m,1))
    for i in np.arange(m):
        if f(w, b, X[i,:])>= 0:
            labels[i, 0] = 1
        else:
            labels[i, 0] = -1
    return labels

def predict_poly(X, y, alphas, b, d):
    m = X.shape[0]
    labels = np.zeros((m,1))
    for i in np.arange(m):
        if f_poly(X, y, alphas, b, X[i, :], d) >= 0:
            labels[i, 0] = 1
        else:
            labels[i, 0] = -1
    return labels

def poly(u, v, d, coef0 = 0):
    """
        polynomial kernel (1 + u \cdot v)^d where d is degree. u and v are row vectors.
    """
    return (coef0+u@(v.T))**d

def kernel_matrix(X, kernel, d):
    """
        Return a matrix M such that M[i,j] = K(X[i,:], X[j,:])
    """
    m = X.shape[0]
    KerMat = np.zeros(m**2).reshape((m,m))
    if kernel == "poly":
        for i in np.arange(m):
            for j in np.arange(m):
                KerMat[i, j] = poly(X[i,:], X[j,:], d)
        return KerMat 

def f(w, b, x):
    return x@w + b

def f_poly(X, y, alphas, b, x, d):
    """
        X is data. 
        y is target
        x is a point. 
        d is the degree of poly kernel.
    """
    m = X.shape[0]
    inter = np.zeros((1, m))
    for i in np.arange(m):
        inter[0, i]=poly(x, X[i, :], d)
    return inter@(alphas*y) + b

def accuracy(trueLabels, predLabels):
    return float(sum(trueLabels == predLabels))/ float(len(trueLabels))

# In[25]:


def simple_smo(X, y, C, tol, iterations):
    """
        C: regularization parameter. See PRML (7.33) p333.
        tol: numerical tolerance
        X: dataset without extended column of ones 
        y: target or label.
        This function computes alphas (Lagrange mulipliers) and 
        return the corresponding w, b.
    """
    (m, n) = X.shape
    if m != y.shape[0]:
        print("Error of simple_smo: Dimension of data and target don't match.")
        return None
    alphas = np.zeros((m, 1))
    b = 0
    passes = 0
    while (passes < iterations):
        num_changed_alphas = 0
        for i in np.arange(m):
            Ei = f(get_w(alphas, X, y), b, X[i, :]) - y[i]
            if ((y[i, 0]*Ei < -tol) and (alphas[i, 0] < C)) or ((y[i, 0]*Ei > tol) and (alphas[i, 0] > 0)):
                j = select_j(i, m)
                Ej = f(get_w(alphas, X, y), b, X[j, :]) - y[j]
                alphaIold = np.copy(alphas[i]); # prevent alphaIold being modified unexpectedly
                alphaJold = np.copy(alphas[j]);
                if (y[i] != y[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H:
                    continue # continue to next i
                eta = 2.0 * X[i,:]@(X[j,:].T) - X[i,:]@(X[i,:].T) - X[j,:]@(X[j,:].T)
                if eta >= 0:
                    continue
                # Compute and clip new value for alphas[j]
                alphas[j] = alphas[j] - y[j]*(Ei - Ej)/eta  
                alphas[j] = clip(alphas[j], L, H)
                if abs(alphaJold - alphas[j]) < 0.00001:
                    continue 
                alphas[i] = alphas[i] + y[i]*y[j]*(alphaJold - alphas[j])
                # compute b1 and b2
                b1 = b - Ei- y[i]*(alphas[i]-alphaIold)*X[i,:]@(X[i,:].T) - y[j]*(alphas[j]-alphaJold)*X[i,:]@(X[j,:].T)
                b2 = b - Ej- y[i]*(alphas[i]-alphaIold)*X[i,:]@(X[j,:].T) - y[j]*(alphas[j]-alphaJold)*X[j,:]@(X[j,:].T)
                # compute b
                if (0 < alphas[i]) and (alphas[i] < C):
                    b = b1
                elif (0 < alphas[j]) and (alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2)/2 
                num_changed_alphas += 1
        if (num_changed_alphas == 0): 
            passes += 1
        else: 
            passes = 0
    return [get_w(alphas, X, y), b]


def simple_smo_poly(X, y, C, tol, iterations, d):
    """
        Polynomial kernel is used.
        C: regularization parameter. See PRML (7.33) p333.
        tol: numerical tolerance
        X: dataset without extended column of ones 
        y: target or label.
        This function computes alphas (Lagrange mulipliers) and 
        return alphas, b.
    """
    m = X.shape[0]
    if m != y.shape[0]:
        print("Error of simple_smo: Dimension of data and target don't match.")
        return None
    alphas = np.zeros((m, 1))
    b = 0
    passes = 0
    KerMat = kernel_matrix(X, "poly", d) # All product of xi and xj
    while (passes < iterations):
        num_changed_alphas = 0
        for i in np.arange(m):
            Ei = KerMat[i,:]@(alphas*y) + b - y[i]
            if (y[i, 0]*Ei < -tol and alphas[i, 0] < C) or ((y[i, 0]*Ei > tol) and (alphas[i, 0] > 0)):
                j = select_j(i, m)
                Ej = KerMat[j,:]@(alphas*y) + b - y[j]
                alphaIold = np.copy(alphas[i]); # prevent alphaIold being modified unexpectedly
                alphaJold = np.copy(alphas[j]);
                if (y[i] != y[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H:
                    continue # continue to next i
                eta = 2.0 * KerMat[i,j] - KerMat[i,i] - KerMat[j,j]
                if eta >= 0:
                    continue
                # Compute and clip new value for alphas[j]
                alphas[j] = alphas[j] - y[j]*(Ei - Ej)/eta  
                alphas[j] = clip(alphas[j], L, H)
                if abs(alphaJold - alphas[j]) < 0.00001:
                    continue 
                alphas[i] = alphas[i] + y[i]*y[j]*(alphaJold - alphas[j])
                # compute b1 and b2
                b1 = b - Ei- y[i]*(alphas[i]-alphaIold)*KerMat[i,i] - y[j]*(alphas[j]-alphaJold)*KerMat[i,j]
                b2 = b - Ej- y[i]*(alphas[i]-alphaIold)*KerMat[i,j] - y[j]*(alphas[j]-alphaJold)*KerMat[j,j]
                # compute b
                if 0 < alphas[i] and alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] and alphas[j] < C:
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
    return [alphas, b]

