#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
def R2(x, y):
    """
        x is the predicted column vector. y is the observed column vector.
    """
    m = y.shape[0]
    ymean = np.mean(y, axis = 0)
    B = y - np.ones((m,1))*ymean
    return 1 - (np.transpose(x-y)@(x-y)/(np.transpose(B)@B))

