#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import costFunction
from sigmoid import sigmoid

def gradientDescent(X, y, theta, alpha, num_iters):
    """
        Perform gradient descent for logistic regression.
    """
    m = X.shape[0] # number of training examples
    loss_history = np.zeros(num_iters);
    
    i = 0
    print("Now we begin training. The learning rate is {} and number of iteration is {}.".format(alpha, num_iters))
    while i < num_iters:
        # Perform a single gradient step on the parameter vector theta. 
        theta = theta - alpha/m * np.transpose(X)@(sigmoid(X@theta)-y);
        # Save the cost J in every iteration    
        loss_history[i] = costFunction.costFunction(X, y, theta)
        #if (i+1) % 10 == 0:
        #    print("After {} iterations, the loss is {:.4f}.".format(i+1, loss_history[i]))   
        i += 1
        
    return [theta, loss_history]

