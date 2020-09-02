#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
from distance import distance

def neighbors(a, B, k):
    """
        a is a row vector and B is a matrix. Return indices of k nearest neighbors.
    """
    m = B.shape[0]
    dist = np.zeros((m,1))
    for i, row in enumerate(B):
        dist[i] = distance(a, row)
    sorted_ind = dist.argsort(axis=0) # quicksort
    return sorted_ind[:k]
    
    


# In[5]:


#A=np.array([[4],[3],[2],[1]])
#A.argsort(axis = 0)


# In[ ]:




