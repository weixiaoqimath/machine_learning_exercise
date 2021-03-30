#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__      = "Rui Wang"
__email__ = "wangru25@msu.edu"

import math
import numpy as np
import pandas as pd
import time

tic = time.perf_counter()

from nn.convlayers import conv_forward, conv_backward
from nn.poolinglayers import max_pooling_forward, max_pooling_backward
from nn.flattenlayers import flatten_forward, flatten_backward
from nn.fclayers import fc_forward, fc_backward
from nn.activations import relu_forward, relu_backward

np.random.seed(7)


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
    X_train_norm = np.reshape(X_train_norm1,(-1,1,28,28)) # reshape X to be a 4-D array 
    X_test_norm = np.reshape(X_test_norm1,(-1,1,28,28))
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


class CNNLayer:

    def __init__(self, X, y, filters = 1, weights_scale = 1e-2, fc_nuerons = 100, lr=0.01, Lambda = 0.01, batch_size = 16):
        '''
        : Structure of this CNN layer: conv1--->pooling--->flatten--->fullyconnect2--->fullyconnect
                                             |                                      |
                                            relu                                   relu
        : X : input, with shape (N, C, H, W), in this project, (N,1,28,28)
              N: #of Sampels
              C: input channels
              H: Height of input figure
              W: Width of input figure
        : K : filter, with shape (C, D, k1, k2), in this project, (1,filters, 3,3)
              C: input channels
              D: output channels, actually #of filters
              k1: height of filter
              k2: width of filter
        : b : bias, with shape (D,)
              D: output channels, actually #of filters
        '''
        self.X = X
        self.y = y
        self.filters = filters
        self.weights_scale = weights_scale
        self.fc_nuerons = fc_nuerons
        self.lr = lr
        self.Lambda = Lambda
        self.batch_size = batch_size
        self.N, self.C, self.H, self.W = X.shape
        # Initialize filters, weights and bias
        self.K1 = self.weights_scale * np.random.randn(1, self.filters, 3, 3).astype(np.float64)
        self.b1 = np.zeros(self.filters).astype(np.float64)
        self.W2 = self.weights_scale * np.random.randn(self.filters*13*13, self.fc_nuerons).astype(np.float64)
        self.b2 = np.zeros(self.fc_nuerons).astype(np.float64)
        self.W3 = self.weights_scale * np.random.randn(self.fc_nuerons, 10).astype(np.float64)
        self.b3 = np.zeros(10).astype(np.float64)

    def forward(self):
        self.n_conv1 = conv_forward(self.X.astype(np.float64), self.K1, self.b1)
        self.n_conv1_relu = relu_forward(self.n_conv1)
        self.n_pool1 = max_pooling_forward(self.n_conv1_relu.astype(np.float64), pooling = (2,2))
        self.n_flatten = flatten_forward(self.n_pool1)
        self.n_fc2 = fc_forward(self.n_flatten, self.W2, self.b2)
        self.n_fc2_relu = relu_forward(self.n_fc2)
        self.n_fc = fc_forward(self.n_fc2_relu, self.W3, self.b3)
        self.y_hat = softmax(self.n_fc)

    def backward(self):
        # dy = self.y_hat - self.y
        loss, dy = cross_entropy_loss(self.y_hat, self.y)
        # add regularization in case of overfitting
        self.loss = loss + 0.5 * self.Lambda * (np.sum(self.K1**2)+np.sum(self.W2**2)+np.sum(self.W3**2))
        g_W3, g_b3, g_fc2_relu = fc_backward(dy, self.W3, self.n_fc2_relu)
        g_fc2 = relu_backward(g_fc2_relu, self.n_fc2)
        g_W2, g_b2, g_flatten = fc_backward(g_fc2, self.W2, self.n_flatten)
        g_pool1 = flatten_backward(g_flatten, self.n_pool1)
        # print('g_pool1', g_pool1.shape)
        g_conv1_relu = max_pooling_backward(g_pool1.astype(np.float64), self.n_conv1_relu.astype(np.float64),pooling=(2,2))
        g_conv1 = relu_backward(g_conv1_relu, self.n_conv1)
        # print('g_conv1', g_conv1.shape)
        g_K1, g_b1, _ = conv_backward(g_conv1, self.K1, self.X)

        self.K1 = self.K1 - self.lr * g_K1 - self.Lambda * self.K1
        self.b1 = self.b1 - self.lr * g_b1
        self.W2 = self.W2 - self.lr * g_W2 - self.Lambda * self.W2
        self.b2 = self.b2 - self.lr * g_b2
        self.W3 = self.W3 - self.lr * g_W3 - self.Lambda * self.W3
        self.b3 = self.b3 - self.lr * g_b3

    def mini_batch_gradient(self):
        X = self.X
        y = self.y
        self.X, self.y = mini_batch(self.X, self.y, self.batch_size)
        self.forward()
        self.backward()
        self.X = X
        self.y = y

    def predict(self, X_test):
        n_conv1 = conv_forward(X_test.astype(np.float64), self.K1, self.b1)
        n_conv1_relu = relu_forward(n_conv1)
        n_pool1 = max_pooling_forward(n_conv1_relu.astype(np.float64), pooling = (2,2))
        n_flatten = flatten_forward(n_pool1)
        n_fc2 = fc_forward(n_flatten, self.W2, self.b2)
        n_fc2_relu = relu_forward(n_fc2)
        n_fc = fc_forward(n_fc2_relu, self.W3, self.b3)
        y_hat_test = softmax(n_fc)
        # return ypred
        # the rest is similar to the logistic regression
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_test_samples = X_test.shape[0]
        # find which index gives us the highest probability
        ypred = np.zeros(num_test_samples, dtype=int)
        for i in range(num_test_samples):
            ypred[i] = labels[np.argmax(y_hat_test[i,:])]
        return ypred

def softmax(z):
    exp_value = np.exp(z-np.max(z, axis = 1, keepdims=True)) # for stablility
    # keepdims = True means that the output's dimension is the same as of z
    softmax_scores = exp_value / np.sum(exp_value, axis = 1, keepdims=True)
    return softmax_scores

def cross_entropy_loss(y_hat, y):
    """
    Cross entropy loss
    y_hat: predict y after softmax, shape:(M,d), M is the #of samples
    y: shape(M,d)
    """
    loss = np.mean(np.sum(- y * np.log(y_hat), axis=-1))
    dy = y_hat - y
    return loss, dy

def mini_batch(X, y, batch_size):
    N = X.shape[0]
    idx = np.random.choice(N, batch_size)
    return X[idx], y[idx]

def accuracy(ypred, yexact):
    # p = np.array(ypred == yexact, dtype = int)
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))

# Main
epoch_num = 2000
myCNN = CNNLayer(X_train_norm, y_train_ohe, weights_scale = 1e-2, filters = 4 , fc_nuerons = 256, lr=0.1, Lambda = 0.01, batch_size = 64)
for i in range(epoch_num):
    myCNN.mini_batch_gradient()
    if ((i+1)%20 == 0):
        print('epoch = %d, current loss = %.5f' % (i+1, myCNN.loss))

y_pred = myCNN.predict(X_test_norm)
print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))

toc = time.perf_counter()
print('Totol time:' + str((toc-tic)/60)+ 'min')
print('===============================Finish===================================')
