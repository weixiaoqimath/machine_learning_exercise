import math
import numpy as np
import pandas as pd
from nn.activations import tanh, softmax, sigmoid
'''Reference: Andrew Ng, Deep Learning, HW for week 3'''

# ===============================LSTM_Cell_Forward================================
def lstm_cell_forward(Xt, h_prev, c_prev, parameters):
    """
    Input:
        - Xt: Input data at timestep "t", shape: (N, D)
            : N : #of samples.
            : D : #of input examples. D = 28 in MNIST dataset
        - h_prev: Hidden state at timestep "t-1", shape: (N, H)
            : N : #of samples.
            : H : #of hidden neurans
        - c_prev: Memory state at timestep "t-1", shape: (N,H)
        - parameters: a dictionary containing:
            : Wf : Weight matrix of the forget gate, shape (H+D, H)
            : Wi : Weight matrix of the update gate, shape (H+D, H)
            : Wo : Weight matrix of the output gate, shape (H+D, H)
            : Wc : Weight matrix of the first "tanh", shape (H+D, H)
            : Wy : Weight matrix relating the hidden-state to the output, shape (H, M), M = 10 in MNIST dataset
            : bf  : Bias, shape (1, H)
            : bi  : Bias, shape (1, H)
            : bo  : Bias, shape (1, H)
            : bc  : Bias, shape (1, H)
            : by  : Bias, shape (1, M)
    Returns:
        - h_next : next hidden state, shape (N, H)
        - c_next : next memory state, shape (N, H)
        - yt_pred: prediction at timestep "t", shape (N, M)
        - cache  : tuple of values needed for the backward pass,
                   contains (h_next, c_next, h_prev, c_prev, Xt, parameters)

    Note:
        ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde), c stands for the memory value
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"]
    Wi = parameters["Wi"]
    Wo = parameters["Wo"]
    Wc = parameters["Wc"]
    Wy = parameters["Wy"]

    bf = parameters["bf"]
    bi = parameters["bi"]
    bo = parameters["bo"]
    bc = parameters["bc"]
    by = parameters["by"]

    # Retrieve dimensions from shapes of Xt and Wy
    N, D = Xt.shape
    H, M = Wy.shape

    # Concatenate h_prev and Xt
    concat = np.zeros((N, H+D))
    concat[:, :H] = h_prev
    concat[:, H:] = Xt

    # Compute values for ft, it, cct, c_next, ot, h_next
    ft = sigmoid(np.dot(concat, Wf) + bf)
    it = sigmoid(np.dot(concat, Wi) + bi)
    ot = sigmoid(np.dot(concat, Wo) + bo)
    cct = np.tanh(np.dot(concat, Wc) + bc)
    c_next = ft * c_prev + it * cct
    h_next = ot * np.tanh(c_next)

    # Compute prediction of the LSTM cell
    yt_pred = softmax(np.dot(h_next, Wy) + by)

    # store values needed for backward propagation in cache
    cache = (h_next, c_next, h_prev, c_prev, ft, it, cct, ot, Xt, parameters)

    return h_next, c_next, yt_pred, cache


def lstm_cell_backward(dh_next, dc_next, cache):
    """
    Backward of cell_RNN:

    Input:
        - dh_next : Gradient of next hidden state, shape (N, H)
        - dc_next : Gradient of next cell state, shape (N, H)
        - caches   : Output of lstm_cell_forward
    Returns:
        - dXt     : Gradient of the input data, shape (N, D)
        - dh_prev : Gradient of previous hidden state, shape (N, H)
        - dc_prev : Gradient of previous cell state, shape (N, H)
        - dWf     : Gradient w.r.t. the weight matrix of the forget gate, shape (H+D, H)
        - dWi     : Gradient w.r.t. the weight matrix of the update gate, shape (H+D, H)
        - dWo     : Gradient w.r.t. the weight matrix of the output gate, shape (H+D, H)
        - dWc     : Gradient w.r.t. the weight matrix of the memory gate, shape (H+D, H)
        - dbf     : Gradient w.r.t. biases of the forget gate, shape (1, H)
        - dbi     : Gradient w.r.t. biases of the update gate, shape (1, H)
        - dbo     : Gradient w.r.t. biases of the forget gate, shape (1, H)
        - dbf     : Gradient w.r.t. biases of the memory gate, shape (1, M)
    """

    # Retrieve information from "cache"
    (h_next, c_next, h_prev, c_prev, ft, it, cct, ot, Xt, parameters) = cache

    N, D = Xt.shape
    N, H = h_next.shape

    # Compute gates related derivatives
    dot = 
    dcct = 
    dit = 
    dft = 

    ##dit = None
    ##dft = None
    ##dot = None
    ##dcct = None
    concat = np.concatenate((h_prev, Xt), axis=1)

    # Compute parameters related derivatives
    dWf = np.dot(concat.T, dft)
    dWi = np.dot(concat.T, dit)
    dWc = np.dot(concat.T, dcct)
    dWo = np.dot(concat.T, dot)
    dbf = np.sum(dft, axis=0 ,keepdims = True)
    dbi = np.sum(dit, axis=0, keepdims = True)
    dbc = np.sum(dcct, axis=0,  keepdims = True)
    dbo = np.sum(dot, axis=0, keepdims = True)

    # Compute derivatives w.r.t previous hidden state, previous memory state and input
    dh_prev = 
    dc_prev = 
    dXt = 

    # Save gradients in dictionary
    gradients = {"dXt": dXt, "dh_prev": dh_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi, "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients

# ===============================LSTM_Forward====================================
def lstm_forward(X, h0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (3).

    Input:
        - X : Input data for every time-step, shape: (N, D, T)
            : N : #of samples.
            : D : #of input examples.
            : T : the length of the input sequence
        - h0: Initial hidden state, shape: (N, H)
            : N : #of samples.
            : H : #of hidden neurans.
        - parameters: a dictionary containing:
            : Wf : Weight matrix of the forget gate, shape (H+D, H)
            : Wi : Weight matrix of the update gate, shape (H+D, H)
            : Wo : Weight matrix of the output gate, shape (H+D, H)
            : Wc : Weight matrix of the first "tanh", shape (H+D, H)
            : Wy : Weight matrix relating the hidden-state to the output, shape (H,M), M = 10 in MNIST dataset
            : bf  : Bias, shape (1, H)
            : bi  : Bias, shape (1, H)
            : bo  : Bias, shape (1, H)
            : bc  : Bias, shape (1, H)
            : by  : Bias, shape (1, M)
    Returns:
    h -- Hidden states for every time-step, shape (N, H, T)
    y -- Predictions for every time-step, shape (N, M, T)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, X)
    """

    N, D, T = X.shape
    H, M = parameters["Wy"].shape

    caches = []
    h = np.zeros((N, H, T))
    c = h
    y = np.zeros((N, M, T))

    # Initialize h_next and c_next
    h_next = h0
    c_next = np.zeros(h_next.shape)

    for t in range(T):
        # Update next hidden state, next memory state, compute the prediction, get the cache
        h_next, c_next, yt, cache = lstm_cell_forward(X[:,:,t], h_next, c_next, parameters)
        # Save the value of the new "next" hidden state in h
        h[:,:,t] = h_next
        # Save the value of the prediction in y
        y[:,:,t] = yt
        # Save the value of the next cell state
        c[:,:,t]  = c_next
        # Append the cache into caches
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, X)

    return h, y, c, caches

def lstm_backward(dh, caches):
    """
    Backward Layers of LSTM:
    Input:
        - dh    : Gradients w.r.t the hidden states, shape (N, H, T)
                : N : #of samples.
                : H : #of hidden neurons.
                : T : the length of the input sequence
        - caches: Tuple containing information from the forward pass (lstm_forward)
    Returns:
        - dX  : Gradient of inputs, shape (N, D, T)
        - dh0 : Gradient w.r.t. the previous hidden state, shape (N, H, T)
        - dWf : Gradient w.r.t. the weight matrix of the forget gate, shape (H+D, H)
        - dWi : Gradient w.r.t. the weight matrix of the update gate, shape (H+D, H)
        - dWo : Gradient w.r.t. the weight matrix of the output gate, shape (H+D, H)
        - dWc : Gradient w.r.t. the weight matrix of the memory gate, shape (H+D, H)
        - dbf : Gradient w.r.t. biases of the forget gate, shape (1, H)
        - dbi : Gradient w.r.t. biases of the update gate, shape (1, H)
        - dbo : Gradient w.r.t. biases of the output gate, shape (1, H)
        - dbc : Gradient w.r.t. biases of the memory gate, shape (1, H)
    """

    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (h1, c1, h0, c0, f1, i1, cc1, o1, X1, parameters) = caches[0]

    ### START CODE HERE ###
    # Retrieve dimensions from dh's and X1's shapes
    N, H, T = dh.shape
    N, D = X1.shape

    # initialize the gradients with the right sizes
    dX = np.zeros((N, D, T))
    dh0 = np.zeros((N, H))
    dh_prevt = np.zeros(dh0.shape)
    dc_prevt = np.zeros(dh0.shape)
    dWf = np.zeros((H+D, H))
    dWi = np.zeros(dWf.shape)
    dWc = np.zeros(dWf.shape)
    dWo = np.zeros(dWf.shape)
    dbf = np.zeros((1, H))
    dbi = np.zeros(dbf.shape)
    dbc = np.zeros(dbf.shape)
    dbo = np.zeros(dbf.shape)

    # loop back over the whole sequence
    for t in reversed(range(T)):
        # Compute all gradients using lstm_cell_backward
        gradients = lstm_cell_backward(dh[:, :, t], dc_prevt, caches[t])
        # Store or add the gradient to the parameters' previous step's gradient
        dX[:,:,t] = gradients["dXt"]
        dWf += gradients["dWf"]
        dWi += gradients["dWi"]
        dWc += gradients["dWc"]
        dWo += gradients["dWo"]
        dbf += gradients["dbf"]
        dbi += gradients["dbi"]
        dbc += gradients["dbc"]
        dbo += gradients["dbo"]
    # Set the first activation's gradient to the backpropagated gradient dh_prev.
    dh0 = gradients["dh_prev"]

    # Store the gradients in a python dictionary
    gradients = {"dX": dX, "dh0": dh0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi, "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients
