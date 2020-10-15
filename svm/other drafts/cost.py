#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math
def cost(X, y, c, lam):
    """
        c is a column parameter vector. 
        X is extended data matrix. 
        Y is target.
        lam is regularization parameter.
    """
    # hingeloss
    distances = 1 - y * (X@c) 
    distances[distances < 0] = 0 # max(0, distances[i])
    cost = (c[1:].T)@c[1:] + lam * sum(distances[distances < 0])
    return cost


# In[ ]:




