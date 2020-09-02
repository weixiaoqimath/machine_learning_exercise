#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from neighbors import neighbors

def predict(trainX, trainY, testX, k):
    """
        Return the prediction for testX
    """
    m = testX.shape[0]
    pred_testX = np.zeros((m,1))
    for i, x in enumerate(testX):
        ind = neighbors(x, trainX, k) # indices of k nearest neighbors
        prob = np.round(np.mean(trainY[ind])) # probability of being class 1
        if prob > 0.5:
            pred_testX[i] = 1
        elif prob < 0.5:
            pred_testX[i] = 0
        elif prob == 0.5: # When there is a tie, pick the index of the closest point. Note that there might be many closest points.
            pred_testX[i] = trainY[ind[0]]
    return pred_testX


# In[6]:


#A = np.arange(8).reshape((4,2))


# In[7]:


#for i in A:
#    print(i)


# In[ ]:




