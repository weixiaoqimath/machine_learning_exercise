#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from math import sqrt

def RMSE(x, y):
    """
        x and y are column vector having the same size.
    """
    m = x.shape[0]
    return sqrt(np.transpose(x-y)@(x-y)/m)

