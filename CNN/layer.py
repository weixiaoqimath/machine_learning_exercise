# can add layer to class ANN 
# The data set is mnist having the shape of (2000, 784)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def softmax(z):
    """
        compute the softmax of matrix z in a numerically stable way,
        by substracting each row with the max of each row. For more
        information refer to the following link:
        https://nolanbconaway.github.io/blog/2017/softmax-numpy
    """
    shift_z = z - np.amax(z, axis = 1, keepdims = 1)
    exp_z = np.exp(shift_z)
    softmax = exp_z / np.sum(exp_z, axis = 1, keepdims = 1)
    return softmax   

def leakyReLU(x):
    y1 = ((x > 0) * x)                                                 
    y2 = ((x <= 0) * x * 0.01)                                         
    return y1 + y2  

def leakyReLU_derivative(x):
    """
        find d_leakyReLU(x)
    """
    y1 = ((x > 0) * 1)
    y2 = ((x <= 0) * 0.01)
    return y1+y2


class linear():
    def __init__(self, input_nn, output_nn, activation):

        # Initialize weights
        self.W = np.random.randn(input_nn, output_nn) / np.sqrt(input_nn) # W_n
        self.b = np.zeros(output_nn) # b_n
        self.activation = activation
        self.input = None # f_{n-1}
        self.output = None # f_n
        self.z = None
        
    def feed_forward(self, input):
        if input.ndim == 4:
            input = input.reshape(input.shape[0], -1)
        self.input = input
        # z_n = f_{n-1}W_n + b_n
        self.z = np.dot(input, self.W) + self.b

        if self.activation == 'tanh':
            output = np.tanh(self.z) # f_n = tanh(z_n)
        
        if self.activation == 'leakyReLU':
            output = leakyReLU(self.z)

        self.output = output

        return output

        
    def back_propagation(self, gradient, lr=1e-3):
        """
            gradient: dL/df_n
        """
        if self.activation == 'tanh':
            # df_n/dz_n = 1 - f_n**2
            dfn_dzn = 1 - (self.output)**2 
        if self.activation == 'leakyReLU':
            dfn_dzn = leakyReLU_derivative(self.z)

        # dL/dW_n = f_{n-1}^T (dL/df_n * df_n/dz_n)
        self.W -= lr * np.dot(self.input.T, gradient * dfn_dzn)
        # dL/db_n = \sum_{axis=0} dL/df_n * df_n/dz_n
        self.b -= lr * np.sum(gradient * dfn_dzn, axis=0)

        return np.dot(gradient * dfn_dzn, self.W.T) # dL/df_{n-1}


class output_layer():
    def __init__(self, input_nn, output_nn, activation):

        # Initialize weights
        self.W = np.random.randn(input_nn, output_nn) / np.sqrt(input_nn) # W_n
        #self.W = np.random.uniform(-1.0, 1.0, (input_nn, output_nn)) / np.sqrt(input_nn)
        self.b = np.zeros(output_nn) # b_n
        self.activation = activation
        self.input = None # f_{n-1}
        self.output = None # f_n
        self.z = None

    def feed_forward(self, input):
        if input.ndim == 4:
            input = input.reshape(input.shape[0], -1)
        self.input = input
        # z_n = f_{n-1}W_n + b_n
        self.z = np.dot(input, self.W) + self.b

        if self.activation == 'softmax':
            output = softmax(self.z) # y_prob = softmax(z_n)

        self.output = output

        return output

    def back_propagation(self, target, lr):
        # dL/dW_n = f_{n-1}^T(y_prob - y)
        self.W -= lr * np.dot(self.input.T, self.output - target)
        # dL/db_n = \sum_{axis=0} y_prob - y
        self.b -= lr * np.sum(self.output - target, axis=0)

        return np.dot(self.output - target, self.W.T) # (y_prob - y)W_n^T 


class max_pooling():
    def __init__(self, pooling = (2,2), strides = (2,2)):
        self.pooling = pooling # pooling: Usually is the 2 by 2 matrix, just like filter.
        self.strides = strides 

    def feed_forward(self, input):
        """
            : Max Pooling Forward process
            
        """
        N, C, H, W = input.shape
        self.input_shape = input.shape

        s0 = self.strides[0]
        s1 = self.strides[1]
        out_H = (H - self.pooling[0]) // s0 + 1
        out_W = (W - self.pooling[1]) // s1 + 1

        self.output_shape = (N, C, out_H, out_W)
        output = np.zeros((N, C, out_H, out_W))
        self.max_indices = np.zeros((N, C, out_H, out_W), dtype=tuple)

        for n in np.arange(N):
            for c in np.arange(C):
                for i in np.arange(out_H):
                    for j in np.arange(out_W):
                        region = input[n, c, i * s0 : i*s0 + self.pooling[0], j*s1 : j*s1 + self.pooling[1]]
                        output[n, c, i, j] = np.max(region)
                        # this is the max indices of the region [n, i, j]. it is a np array

                        self.max_indices[n, c, i, j] = np.unravel_index(region.argmax(), region.shape)
        return output

    def back_propagation(self, gradient, lr):
        """
            gradient dL/d_output is of the shape (N, out_H, out_W)
            lr is not used
        """    
        if gradient.ndim == 2:
            gradient = gradient.reshape(self.output_shape)

        dL_d_input = np.zeros(self.input_shape)
        N, C, out_H, out_W = self.output_shape
        for n in np.arange(N):
            for c in np.arange(C):
                for i in np.arange(out_H):
                    for j in np.arange(out_W):
                        dL_d_input[n, c, self.max_indices[n, c, i, j][0]+i, self.max_indices[n, c, i, j][1]+j] = gradient[n, c, i, j]
        
        return dL_d_input


class convolutional():
    def __init__(self, shape = (1, 1, 5, 5), strides = (1,1)):
        # K is the filter (only one output channel in default case)
        self.K = np.random.randn(shape[0], shape[1], shape[2], shape[3])
        self.strides = strides

    def feed_forward(self, input):
        self.input = input
        _, out_C, k1, k2 = self.K.shape 
        N, C, H, W = input.shape # C is the number of input channel
        output = np.zeros((N, out_C, 1 + (H - k1), 1 + (W - k2))) / 20.0
        #for n in np.arange(N):
        #    for d in np.arange(out_C):
        #        for h in np.arange(H - k1 + 1):
        #            for w in np.arange(W - k2 + 1):
        #                output[n, d, h , w] = np.sum(input[n, :, h: h + k1, w: w + k2] * self.K[:, d, :,:])
        for n in np.arange(N):
            for h in np.arange(H - k1 + 1):
                for w in np.arange(W - k2 + 1):
                    output[n, 0, h , w] = np.sum(input[n, :, h: h + k1, w: w + k2] * self.K[:, 0, :,:])
        return output

    def back_propagation(self, gradient, lr):
        """
            I am only dealing with the case of 1 input channel 
        """
        dL_dK = np.zeros(self.K.shape)
        _, out_C, k1, k2 = self.K.shape
        #for d in np.arange(out_C):
        #    for x in np.arange(k1):
        #        for y in np.arange(k2):
        #            # dL/dK(0, d, x, y) = \sum_{n,i,j} dL/d_output(n, d, i, j) * d_output(n, d, i, j)/dK(1, d, x, y)
        #            dL_dK[0, d, x, y] = np.sum(gradient[:, d, :, :] * self.input[:, 0, x:x+gradient.shape[2], y:y+gradient.shape[3]])
        for x in np.arange(k1):
            for y in np.arange(k2):
                # dL/dK(1, d, x , y) = \sum_{n,i,j} dL/d_output(n, d, i, j) * d_output(n, d, i, j)/dK(1, d, x, y)
                dL_dK[0, 0, x, y] = np.sum(gradient[:, 0, :, :] * self.input[:, 0, x:x+gradient.shape[2], y:y+gradient.shape[3]])

        self.K -= lr * dL_dK

        #return dL_dK



    