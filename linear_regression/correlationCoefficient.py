#!/usr/bin/env python
# coding: utf-8

# In[1]:

from math import sqrt
import numpy as np
def correlationCoefficient(x, y):
    """
        x and y are column vectors having the same shape.
    """
    xmean = np.mean(x, axis = 0)
    ymean = np.mean(y, axis = 0)
    
    m = x.shape[0]
    A = x - np.ones((m,1))*xmean
    B = y - np.ones((m,1))*ymean
    return (np.transpose(A)@B)/(sqrt(np.transpose(A)@A)*sqrt(np.transpose(B)@B))


# In[ ]:




