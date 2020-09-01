import numpy as np
def featureNormalize(X, A):
    """
        A is a row vector. For each row of X, it substract A. 
    """
    return X - np.ones((X.shape[0],1))@A
    


# In[29]:


#A = np.array([[1,2],[7,9]])
#print(A)
#featureNormalize(A)


# In[ ]:




