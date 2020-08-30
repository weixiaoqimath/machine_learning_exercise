import pandas as pd
import scipy
import numpy as np

# load data into dataframe
trainX = pd.read_csv("airfoil_self_noise_X_train.csv")
trainY = pd.read_csv("airfoil_self_noise_y_train.csv")
testX = pd.read_csv("airfoil_self_noise_X_test.csv")
testY = pd.read_csv("airfoil_self_noise_y_test.csv")

nptrainX = trainX.to_numpy()
nptrainY = trainY.to_numpy()
nptestX = testX.to_numpy()
nptestY = testY.to_numpy()

# rescale nptrainX
import featureNormalization as fN
mean = np.mean(nptrainX, axis=0).reshape((1,nptrainX.shape[1]))
std = np.std(nptrainX, axis=0).reshape((1,nptrainX.shape[1]))
normalized_trainX = fN.featureNormalize(nptrainX, mean, std)

# Add one column of ones to nptrainX
X = np.hstack((np.ones((nptrainX.shape[0],1)), normalized_trainX)) 

# Perform gradient descent
import gradientDescentMulti as gDM
alpha = 0.02;
num_iters = 400;
theta = np.zeros((X.shape[1],1))
[theta, loss_history] = gDM.gradientDescentMulti(X, nptrainY, theta, alpha, num_iters)

#import matplotlib.pyplot as plt
#x = np.arange(1, num_iters+1)
#plt.plot(x, loss_history[x-1])  
#plt.xlabel('iterations')
#plt.ylabel('loss')
#plt.legend()

# Add a column of ones to rescaled nptestX
newtestX = np.hstack((np.ones((nptestX.shape[0],1)),fN.featureNormalize(nptestX, mean, std)))
predY = newtestX@theta

# Compute pcc
import correlationCoefficient as cC
print("The Pearson correlation coefficient is {:.4f}.".format(cC.correlationCoefficient(predY, nptestY)[0][0]))

# Compute rmse
import RMSE
print("The root mean square error is {:.4f}".format(RMSE.RMSE(predY, nptestY)))





