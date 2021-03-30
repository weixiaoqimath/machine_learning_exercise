import numpy as np
import math



def relu(z):
    '''This is the finction for ReLU'''
    return z * (z > 0)

def relu_backward(next_dz, z):
    '''This is the derivative of ReLU'''
    dz = np.where(np.greater(z,0), next_dz, 0)
    return dz

def tanh(z):
    '''This is the finction for tanh'''
    return np.tanh(z)

def tanh_backward(dz):
    '''This is the derivative of tanh'''
    return 1. - np.tanh(dz) * np.tanh(dz)

def sigmoid(z):
    '''This is the finction for sigmoid'''
    return 1/(1+np.exp(-z))

def sigmoid_backward(dz):
    '''This is the derivative of sigmoid'''
    return np.exp(-dz)/ ( (1 + np.exp(-dz)) * (1 + np.exp(-dz)) )

def softmax(z):
    exp_value = np.exp(z-np.max(z, axis = 1, keepdims=True)) # for stablility
    # keepdims = True means that the output's dimension is the same as of z
    softmax_scores = exp_value / np.sum(exp_value, axis = 1, keepdims=True)
    return softmax_scores
