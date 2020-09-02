#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math

def distance(a, b):
    """
        Return the euclidean distance between two row vectors a and b.
    """
    return math.sqrt(np.sum((a - b)**2))


# In[ ]:




