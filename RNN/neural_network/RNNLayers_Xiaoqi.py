# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2021-02-23 00:00:36
LastModifiedBy: Rui Wang
LastEditTime: 2021-02-23 19:05:23
Email: wangru25@msu.edu
FilePath: /RNNTest/nn/RNNLayers.py
Description: 
'''
import math
import numpy as np
import pandas as pd
from nn.activations import tanh, softmax, sigmoid


def rnn_cell_forward(Xt, h_prev, parameters):
    '''
    RNN Cell:
    Input: 
        - Xt: (N,D) N=2000 D=28
        - h_prev: (N,H) #of neurons in the hidden state. "prev" is actually for timestep "t-1"
        - parameters:
            : Wx: Weight matrix multiplying the input Xt, (D, H) 
            : Wh: Weight matrix multiplying the hidden state (H,H)
            : Wy: Weight matrix relating to the hidden-state. Shape is (H,M) # M = 10
            : bh: Bias, (1, H)
            : by: Bias, (1, M)
    Returns:
    - h_next: next hidden state (N, H)
    - yt_pred: prediction at timestep t, (N, M)
    - cache : tuple of values needed for the back-propagation part, has shape (h_next, h_prev, Xt, parameters)
    '''
    Wx = parameters["Wx"]
    Wh = parameters["Wh"]
    Wy = parameters["Wy"]
    bh = parameters["bh"]
    by = parameters["by"]

    # compute next activation state using the formula tanh(xxxx)
    h_next = tanh(np.dot(Xt,Wx) + np.dot(h_prev,Wh) + bh)
    yt_pred = softmax(np.dot(h_next, Wy) + by)
    cache = (h_next, h_prev, Xt, parameters)

    return h_next, yt_pred, cache



def rnn_forward(X, h0, parameters):
    '''
    Forward Layer of RNN
    Input:
        - X: Input data for every time-step. (N,D,T) # D=28, T=28
        - h0: Initial hidden state (N,H)
        - parameters:
            : Wx: Weight matrix multiplying the input, (D, H) 
            : Wh: Weight matrix multiplying the hidden state (H,H)
            : Wy: Weight matrix relating to the hidden-state. Shape is (H,M) # M = 10
            : bh: Bias, (1, H)
            : by: Bias, (1, M)
    Returns: 
        - h : Hidden states for all of the time steps. (N, H, T)
        - y_pred: Predictions that saves all of the yt_pred, The shape will be (N,M,T)
        - caches: tuple of values that needed for the back-propagation part, caches.append(cache)
    '''
    caches = []

    N, D, T = X.shape
    H, M = parameters['Wy'].shape

    # Initialize 'h' and 'y'
    h = np.zeros((N,H,T)) # h0 is not stored in h
    y_pred = np.zeros((N,M,T)) # M: the number of categories

    # Initialize h_next
    h_next = h0 
    for t in range(T):
        h_next, yt_pred, cache = rnn_cell_forward(X[:,:,t], h_next, parameters)
        h[:,:,t] = h_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
    caches.append(X)   

    # store values needed for backward propagation
    caches = (caches, X)

    #[cache0,cache2, ..., cache27,X]

    return h, y_pred, caches 



def rnn_cell_backward(dh_next, cache):
    '''
    Backward Layer of RNN cell:
    Input: 
        - dh_next : Upstream gradients of hidden state (Gradient of loss wrt hidden state)
        - cache: Output of the rnn_cell_forward
    Returns:
        - dXt: 
        - dh_prev
        - dWx
        - dWh
        - dbh
    '''

    (h_next, h_prev, Xt, parameters) = cache
    Wx = parameters["Wx"]
    Wh = parameters["Wh"]
    #Wy = parameters["Wy"]
    bh = parameters["bh"]
    #by = parameters["by"]

    dtanh = 1 - h_next**2 
    dWx = np.dot(Xt.T, dh_next*dtanh)
    dWh = np.dot(h_prev.T, dh_next*dtanh)
    dbh = np.sum(dtanh*dh_next, axis=0, keepdims=True)
    dh_prev = np.dot(dh_next*dtanh, Wh.T)
    #dXt = np.dot(dh_next*dtanh, Wx.T)

    #gradients = {'dXt': dXt, 'dh_prev': dh_prev, 'dWx': dWx,'dWh':dWh, 'dbh':dbh}  #keys and values in the Dictionary 
    gradients = {'dh_prev': dh_prev, 'dWx': dWx,'dWh':dWh, 'dbh':dbh}

    return gradients




def rnn_backward(dh, caches):
    '''
    Backward Layers 
    Input: 
        - dh : Upstream gradients of all hidden states.  (N,H,T)
        - caches: output of the rnn_forward
    Returns:
        - dh:  (N,H,T)
        - dh0: (N,H,T)
        - dWx
        - dWh
        - dbh
    '''
    (caches, X) = caches
    (h1, h0, X1, parameters) = caches[0]

    N, H, T = dh.shape
    N, D = X1.shape

    dWx = np.zeros((D,H))
    dWh = np.zeros((H,H))
    dbh = np.zeros((1,H))
    #dh0 = np.zeros((N,H))
    #dX = np.zeros((N,D,T))
    dh_prevt = np.zeros((N,H))

    for t in reversed(range(T)): 
        gradients = rnn_cell_backward(dh[:,:,t] + dh_prevt, caches[t])
        #dXt, dh_prev, dWxt, dWht, dbht = gradients['dXt'], gradients['dh_prev'], gradients['dWx'], gradients['dWh'], gradients['dbh']
        dh_prevt, dWxt, dWht, dbht = gradients['dh_prev'], gradients['dWx'], gradients['dWh'], gradients['dbh']

        #dX[:,:,t] = dXt

        dWx += dWxt  
        dWh += dWht
        dbh += dbht

    dh0 = dh_prevt

    #gradients = {'dX': dX, 'dh0': dh0, 'dWx': dWx, 'dWh': dWh, 'dbh': dbh}
    gradients = {'dWx': dWx,'dWh':dWh, 'dbh':dbh}

    return gradients




# def cross_entropy_loss(inPut):