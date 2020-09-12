#!/usr/bin/env python
# coding: utf-8

# In[30]:


import tensorflow as tf
import numpy as np
(_, _), (test_data, test_target) = tf.keras.datasets.mnist.load_data()


# In[31]:


#print("The shape of test_data is {}".format(test_data.shape))
#print("The shape of test_data is {}".format(test_target.shape))


# In[37]:


data = test_data.reshape(test_data.shape[0], -1)
#print("The shape of data is {}".format(data.shape))


# In[38]:


target = test_target


# In[36]:


#test_data[0,:]


# In[39]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
#import mglearn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import fowlkes_mallows_score

np.random.seed(0)

# feature scaling
num_classes = 10

num_features = data.shape[1]
scaler = StandardScaler() 
scaler.fit(data)
X_scaled = scaler.transform(data)
num_init = 20

# In[40]:


kmeans = KMeans(n_clusters=num_classes, random_state=0, n_init=num_init) 
kmeans.fit(X_scaled)

#plt.figure(figsize=(8, 8))
#plt.gca().set_aspect("auto")
#plt.xlabel("First feature")
#plt.ylabel("Second feature")
#mglearn.discrete_scatter(X_scaled[:, 0], X_scaled[:, 1], kmeans.labels_, markers='o') 
#mglearn.discrete_scatter(    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], np.arange(num_classes), markers='^', markeredgewidth=2)


# In[41]:


# Apply pca to scaled data

pca = PCA()
pca.fit(X_scaled)
X_scaled_pca = pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape))) 
print("Reduced shape: {}".format(str(X_scaled_pca.shape)))
#print("Explained variance ratio: {}".format(pca.explained_variance_ratio_))

# plot the first and second PCs
#plt.figure(figsize=(8, 8))
#mglearn.discrete_scatter(X_scaled_pca[:, 0], X_scaled_pca[:, 1], target) 
#plt.legend(target_names, loc="best") 
#plt.gca().set_aspect("auto")
#plt.xlabel("First principal component")
#plt.ylabel("Second principal component")


# In[42]:


adRand = np.zeros((49, 1))
#FM = np.zeros((num_features, 1))
for i in np.arange(49):
    # build the clustering model for the first i PCs
    pca_kmeans = KMeans(n_clusters=num_classes, random_state=0, n_init = num_init) 
    pca_kmeans.fit(X_scaled_pca[:, :(i+1)*16])
    adRand[i] = adjusted_rand_score(target, pca_kmeans.labels_)
    #FM[i] = fowlkes_mallows_score(target, pca_kmeans.labels_) 

np.savetxt('adRand.csv', adRand, delimiter=',')
# In[ ]:


plt.figure(figsize=(16, 10))
plt.ylim(0,1)
plt.xlim(1, 49)
plt.gca().set_aspect("auto")
plt.xlabel("Number of principal components")
plt.ylabel("Adjusted Rand index")
plt.hlines(adjusted_rand_score(target, kmeans.labels_), 1, 49, color = "y",linestyle='--')
plt.plot(np.arange(1, 50), adRand, linestyle='--', marker='o', linewidth=2)
x_major_locator = plt.MultipleLocator(1)
y_major_locator = plt.MultipleLocator(0.1)
plt.gca().xaxis.set_major_locator(x_major_locator)
plt.gca().yaxis.set_major_locator(y_major_locator)
plt.savefig("adRand.png")
print(adjusted_rand_score(target, kmeans.labels_))

