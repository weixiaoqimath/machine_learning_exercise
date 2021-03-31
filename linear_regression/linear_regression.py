import pandas as pd
import scipy
import numpy as np

def normalize(X):
    """
        feature normalization. Each row of X represents a point in R^d. 
        Substract by the mean of X and then divided by the std of X.
    """
    mean = np.mean(X, axis=0) # mean of each column vector
    std = np.std(X, axis=0) # std of each column vector
    return (X - mean)/std

def gradient_descent(X, y, theta, lr, num_iters):
    """
        Perform gradient descent for multivariable linear regression.
        X is the extended matrix of which the first column consists of ones.
        y is a 1d array.
    """
    m = X.shape[0] # number of training examples
    J_history = np.zeros(num_iters) # stores costs of gradient descent steps.
    
    i = 0
    print("Now we begin training. The learning rate is {} and the number of iteration is {}.".format(lr, num_iters))
    for i in range(num_iters):
        # Perform a single gradient step on the parameter vector theta. 
        theta = theta - lr/m * np.dot(X.T, np.dot(X, theta)-y)
        # Save the cost J in every iteration    
        J_history[i] = 1/(2*m)*np.dot(np.dot(X, theta) - y, np.dot(X, theta) - y)
        if (i+1) % 10 == 0:
            print("After {} iterations, the loss is {:.4f}.".format(i+1, J_history[i]))   
        
    return theta, J_history

def PCC(x, y):
    """
        x and y are 1d vectors having the same size.
        return the Pearson's coefficient.
    """   
    m = x.shape[0]
    A = x - np.mean(x)
    B = y - np.mean(y)
    return np.dot(A, B)/np.sqrt(np.dot(A, A)*np.dot(B, B))

def RMSE(x, y):
    """
        x and y are 1d vectors having the same size.
    """
    m = x.shape[0]
    return np.sqrt(np.dot(x-y, x-y)/m)

# load data into dataframe
Xtrain = pd.read_csv("airfoil_self_noise_X_train.csv")
ytrain = pd.read_csv("airfoil_self_noise_y_train.csv")
Xtest = pd.read_csv("airfoil_self_noise_X_test.csv")
ytest = pd.read_csv("airfoil_self_noise_y_test.csv")

Xtrain = Xtrain.values
ytrain = ytrain.values
Xtest = Xtest.values
ytest = ytest.values

print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)
# reshape y
ytrain = ytrain.flatten()
ytest = ytest.flatten()

# rescale Xtrain. Make sure features are on a similar scale.
normalized_Xtrain = normalize(Xtrain)

# Add one column of ones to Xtrain
X = np.concatenate((np.ones((Xtrain.shape[0],1)), normalized_Xtrain), axis=1) 

# Perform gradient descent
lr = 0.1
num_iters = 200
theta = np.zeros(X.shape[1]) # initialize theta
theta, loss_history = gradient_descent(X, ytrain, theta, lr, num_iters)

# Add a column of ones to rescaled npXtest
extended_Xtest = np.concatenate((np.ones((Xtest.shape[0],1)), normalize(Xtest)), axis=1)
ypred = np.dot(extended_Xtest, theta)

# Compute pcc
print("The Pearson correlation coefficient is {:.4f}.".format(PCC(ypred, ytest)))

# Compute rmse
print("The root mean square error is {:.4f}".format(RMSE(ypred, ytest)))

# print theta
print("The coefficients given by the gradient descent method are {}".format(theta))

theta_normal_equation = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), ytrain)
print("The coefficients given by the normal equation are {}".format(theta_normal_equation))

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, ytrain)
print("The coefficients given by the sklearn model are {} and {}".format(reg.intercept_, reg.coef_))

#import matplotlib.pyplot as plt
#x = np.arange(1, num_iters+1)
#plt.plot(x, loss_history[x-1])  
#plt.xlabel('iterations')
#plt.ylabel('loss')
#plt.legend()



