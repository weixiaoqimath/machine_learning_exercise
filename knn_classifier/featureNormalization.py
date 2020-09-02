import numpy as np
def featureNormalize(X, A, B):
    """
        A and B are two row vectors. For each row of X, it substract A and divide the resulting row by B elementwise. 
    """
    return (X - np.ones((X.shape[0],1))@A)/(np.ones((X.shape[0],1))@B)
    


# In[29]:


#A = np.array([[1,2],[7,9]])
#print(A)
#featureNormalize(A)


# In[ ]:




