#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from gradient import gradient
from cost import cost

def subGD(data, target, iterations, alpha, lam):
    """
        data: extended data. 
        alpha: learning rate.
        lam: regularization parameter
        c: parameter of linear classifier.
    """
    c = np.ones((data.shape[1],1))
    # subgradient descent
    for epoch in range(1, iterations+1): 
        c = c - alpha * gradient(data, target, c, lam)
        #if (epoch % 100) == 0:
            #print("After {} iterations, cost is {}".format(epoch, cost(data, target, c, lam)))
    return c

