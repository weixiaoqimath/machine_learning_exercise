#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Simple smo algorithm for SVM. Polynomial kernel with coeff0=0 is used.
# The results of each running may be different due to random selection inside simple smo algorithm.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SVM import simple_smo_poly, f_poly, predict_poly, accuracy
from scaler import scaler 

Xtrain = pd.read_csv("MNIST_X_train.csv")
ytrain = pd.read_csv("MNIST_Y_train.csv")
Xtest = pd.read_csv("MNIST_X_test.csv")
ytest = pd.read_csv("MNIST_Y_test.csv")
npXtrain = Xtrain.to_numpy()
npytrain = ytrain.to_numpy()
npXtest = Xtest.to_numpy()
npytest = ytest.to_numpy()
print("The shape of Xtrain is {}".format(Xtrain.shape))
print("The shape of Xtest is {}".format(Xtest.shape))

# Feature scaling
mean = np.mean(npXtrain, axis=0).reshape((1, npXtrain.shape[1]))
std = np.std(npXtrain, axis=0).reshape((1, npXtrain.shape[1]))
scaled_Xtrain = scaler(npXtrain, mean, std)
scaled_Xtest = scaler(npXtest, mean, std)


# In[5]:


C = 1
tol = 0.001
iterations = 10
d = 3 # degree of polynomial kernel

# Extract samples corresponding to each class
dataList = [scaled_Xtrain[np.where(npytrain == i)[0]] for i in range(10)]       
# The elements of paraMat are going to be computed by simple_smo_poly
paraMat = np.zeros((10,10), dtype=object)
# one vs one approach
for i in range(10):
    for j in range(10):
        if j > i:
            targetI = np.ones(dataList[i].shape[0]).reshape((-1,1))
            targetJ = -np.ones(dataList[j].shape[0]).reshape((-1,1))
            data = np.vstack((dataList[i], dataList[j]))
            target = np.vstack((targetI, targetJ))
            paraMat[i,j] = simple_smo_poly(data, target, C, tol, iterations, d)
            # compute training accuracy
            predLabelsIJ = np.zeros((target.shape[0],1))
            for ind in np.arange(target.shape[0]):
                if f_poly(data, target, paraMat[i,j][0],                           paraMat[i,j][1], data[ind,:], d)>=0:
                    predLabelsIJ[ind] = 1
                else:
                    predLabelsIJ[ind] = -1
            scoreIJ = accuracy(target, predLabelsIJ)
            print("Training class {} vs class {} is complete. The training accuracy is {:.2f}%".format(i,j,scoreIJ*100))

predLabels = np.zeros(npytest.shape[0]).reshape((-1, 1))
# one vs one classification
for ind in np.arange(npytest.shape[0]):
    OvOlabel_list = np.zeros(10)
    for i in range(9):
        for j in range(10):
            if j > i:
                dataIJ = np.vstack((dataList[i], dataList[j]))
                targetI = np.ones(dataList[i].shape[0]).reshape((-1,1))
                targetJ = -np.ones(dataList[j].shape[0]).reshape((-1,1))
                targetIJ = np.vstack((targetI, targetJ))
                if f_poly(dataIJ, targetIJ, paraMat[i,j][0],                           paraMat[i,j][1], scaled_Xtest[ind,:], d)>=0:
                    OvOlabel_list[i] += 1
                else:
                    OvOlabel_list[j] += 1
    # end for
    predLabels[ind, 0] = np.argmax(OvOlabel_list) 

score = accuracy(npytest, predLabels)
print("The accuracy of multiclass classification is {:.2f}%".format(score*100))  


# In[ ]:




