# The iris data set only has two classes.

import numpy as np
import math
import pandas as pd
from predict import predict 
import matplotlib.pyplot as plt

# load data
df_trainX = pd.read_csv("Iris_X_train.csv")
df_trainY = pd.read_csv("Iris_y_train.csv")
df_testX = pd.read_csv("Iris_X_test.csv")
df_testY = pd.read_csv("Iris_y_test.csv")

trainX = df_trainX.to_numpy()
trainY = df_trainY.to_numpy()
testX = df_testX.to_numpy()
testY = df_testY.to_numpy()

k = 6

from featureNormalization import featureNormalize
mean = np.mean(trainX, axis=0).reshape((1,trainX.shape[1]))
std = np.std(trainX, axis=0).reshape((1,trainX.shape[1]))
newtrainX = featureNormalize(trainX, mean, std)
newtestX = featureNormalize(testX, mean, std)

predY = predict(newtrainX, trainY, newtestX, k)
accuracy = np.mean(predY == testY)

print("When k is set to be {}, the accuracy is {:.2f}%".format(k, 100*accuracy))

#x = np.arange(1, 21)
#y = np.zeros(20)
#for i in x:
#    y[i-1] = np.mean(predict(newtrainX, trainY, newtestX, i) == testY)
#    print(y[i-1])

#plt.figure()
#plt.plot(x, y) 
#plt.xticks(x) 
#plt.xlabel('k')
#plt.ylabel('accuracy')
#plt.show()



# In[ ]:




