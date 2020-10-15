#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
def gradient(X, y, c, lam):
    """
        
    """
    distances = 1 - y * (X@c) # hinge loss
    grad = np.zeros((len(c),1))
    for ind, d in enumerate(distances):
        if max(0, d) == 0:
            di = np.zeros((len(c),1))
        else:
            di = - lam * (y[ind] * X[ind, :]).T
            di = di.reshape((-1,1))
        grad += di
    grad += 2 * np.vstack(([[0]], c[1:]))
    return grad


# In[4]:



# In[ ]:




