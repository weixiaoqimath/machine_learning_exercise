# -*- coding: utf-8 -*-
'''
Author: Xiaoqi Wei
'''
import math
import time
import random
import numpy as np
import pandas as pd

np.random.seed(7)
tic = time.perf_counter()

from nn.RNNLayers_Xiaoqi import rnn_cell_forward, rnn_forward
from nn.RNNLayers_Xiaoqi import rnn_cell_backward, rnn_backward
from nn.activations import tanh, sigmoid, softmax


'''
In this file, we will use dictionary to combine all the seperate layers together
'''


def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values # convert values in dataframe to numpy array (features)
    y = df_y.values # convert values in dataframe to numpy array (label)
    return X, y


def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler #import libaray
    scaler = StandardScaler() # call an object function
    scaler.fit(X_train) # calculate mean, std in X_train
    X_train_norm1 = scaler.transform(X_train) # apply normalization on X_train
    X_test_norm1 = scaler.transform(X_test) # we use the same normalization on X_test
    X_train_norm = np.reshape(X_train_norm1,(-1,28,28)) # reshape X to be a 3-D array
    X_test_norm = np.reshape(X_test_norm1,(-1,28,28))
    return X_train_norm, X_test_norm


def one_hot_encoder(y_train, y_test):
    ''' convert label to a vector under one-hot-code fashion '''
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_test_ohe = lb.transform(y_test)
    return y_train_ohe, y_test_ohe


X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

print(X_train_norm.shape)
print(X_test_norm.shape)
print(y_train_ohe.shape)
print(y_test_ohe.shape)


class RNN():
    def __init__(self, X, y, H = 128, lr = 0.01):
        self.X = X
        self.y = y
        self.lr = lr
        # self.T is time
        self.N, self.D, self.T = self.X.shape
        self.H = H
        self.M = self.y.shape[1]

        self.Wx = np.random.randn(self.D, self.H)
        self.Wh = np.random.randn(self.H, self.H)
        self.Wy = np.random.randn(self.H, self.M)
        self.bh = np.zeros((1, self.H)) # broadcasting rules in python  a = [2,1,1] b = [2,2,2] a*b = [4, 2, 2]. c = 2, a*c = [4,2,2]
        self.by = np.zeros((1, self.M))

        self.h0 = np.random.randn(self.N, self.H)
        self.parameters = {"Wh": self.Wh, "Wx": self.Wx, "Wy": self.Wy, "bh": self.bh, "by": self.by}



    def forward(self):
        self.h, self.y_hat, self.caches = rnn_forward(self.X, self.h0, self.parameters)

    def backward(self):
        self.loss, dy = cross_entropy_loss(self.y_hat[:,:,27], self.y)  #index 27 means the timestep = 28
        dWy = np.dot(self.h[:,:,27].T, dy)
        dby = np.sum(dy, axis=0, keepdims=True)
        dh = np.zeros((self.N, self.H, self.T))
        for t in range(self.T):
            dh[:, :, t] = dy@(self.Wy.T)
        gradients = rnn_backward(dh, self.caches)


        self.Wx = self.Wx - self.lr * gradients['dWx']
        self.Wh = self.Wh - self.lr * gradients['dWh']
        self.bh = self.bh - self.lr * gradients['dbh']
        self.Wy = self.Wy - self.lr * dWy
        self.by = self.by - self.lr * dby
        self.parameters = {"Wh": self.Wh, "Wx": self.Wx, "Wy": self.Wy, "bh": self.bh, "by": self.by}        


    def predict(self, X_test):
        h, y_hat_test_all, caches = rnn_forward(X_test, self.h0, self.parameters)
        y_hat_test = y_hat_test_all[:,:,27]
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_test_samples = X_test.shape[0]
        ypred = np.zeros(num_test_samples, dtype=int)
        for i in range(num_test_samples):
            ypred[i] = labels[np.argmax(y_hat_test[i,:])]
        return ypred

        


def cross_entropy_loss(y_hat, y):
    """
    Cross entropy loss
    y_hat: predict y after softmax, shape:(M,d), M is the #of samples
    y: shape(M,d)
    """
    # loss = np.sum(- y * np.log(y_hat))
    loss = np.mean(np.sum(- y * np.log(y_hat), axis=-1))
    dy = y_hat - y
    return loss, dy

def accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))


myRNN = RNN(X_test_norm, y_test_ohe, H = 128, lr= 0.003)
epoch_num = 300
for i in range(epoch_num):
    myRNN.forward()
    myRNN.backward()
    if ((i + 1) % 20 == 0):
        print('epoch = %d, current loss = %.5f' % (i+1, myRNN.loss))

y_pred = myRNN.predict(X_test_norm)
# print(y_pred)
# print(y_test.ravel())
print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))

toc = time.perf_counter()
print('Totol time:' + str((toc-tic))+ 's')
print('===============================Finish===================================')

print('The epoch number is set to be {}'.format(epoch_num))

# best accuracy:0.908