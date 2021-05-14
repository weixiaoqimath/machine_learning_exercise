# accuracy 95%

import math
import time
import random
import numpy as np
import pandas as pd

np.random.seed(10)
tic = time.perf_counter()

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

def sigmoid(z):
    """
        This is the finction for sigmoid
    """
    return 1/(1+np.exp(-z))

def dsigmoid(f):
    """
        f = sigmoid(z)
    """
    return f*(1-f)

def dtanh(f):
    """
        f = tanh(z)
    """
    return 1 - f**2

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

def accuracy(ytest, ypred):
    return float(np.sum(ytest == ypred))/ float(len(ytest))

class cell():
    def __init__(self, Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by):
        self.Wf = Wf
        self.Wu = Wu 
        self.Wc = Wc
        self.Wo = Wo 
        self.Wy = Wy 
        self.bf = bf 
        self.bu = bu 
        self.bc = bc
        self.bo = bo 
        self.by = by 

        self.H = Wy.shape[0] 
        self.h_input = None
        self.c_input = None 
        self.h_output = None 
        self.c_output = None

    def feed_forward(self, Xt, h_input, c_input):
        self.Xt = Xt 
        self.h_input = h_input 
        self.c_input = c_input

        concat = np.concatenate((h_input, Xt), axis=1)
        self.concat = concat

        Gamma_f = sigmoid(np.dot(concat, self.Wf) + self.bf)
        c_tilde = np.tanh(np.dot(concat, self.Wc) + self.bc)
        Gamma_u = sigmoid(np.dot(concat, self.Wu) + self.bu)
        Gamma_o = sigmoid(np.dot(concat, self.Wo) + self.bo)

        self.Gamma_f = Gamma_f
        self.c_tilde = c_tilde 
        self.Gamma_u = Gamma_u 
        self.Gamma_o = Gamma_o

        self.c_output = Gamma_f * c_input + Gamma_u * c_tilde 
        self.h_output = Gamma_o * np.tanh(self.c_output)
        return self.h_output, self.c_output

    def back_propagation(self, dh_output, dc_output):
        """
            the backpropagation of each cell.
        """
        # dL/dGamma_f = dL/dh^t dh^t/dGamma_f + dL/dc^t dc^t/dGamma_f. 
        dGamma_f = dc_output * self.c_input + dh_output * self.Gamma_o * dtanh(np.tanh(self.c_output)) * self.c_input 
        dc_tilde = dc_output * self.Gamma_u + dh_output * self.Gamma_o * dtanh(np.tanh(self.c_output)) * self.Gamma_u
        dGamma_u = dc_output * self.c_tilde + dh_output * self.Gamma_o * dtanh(np.tanh(self.c_output)) * self.c_tilde
        dGamma_o = dh_output * np.tanh(self.c_output)

        # dL/dGamma_f dGamma_f/dh^{t-1} + dL/dGamma_o dGamma_o/dh^{t-1} + dL/dGamma_u dGamma_u/dh^{t-1} + dL/dc_tilde dc_tilde/dh^{t-1}
        dh_input = np.dot(dsigmoid(self.Gamma_o) * dGamma_o, self.Wo.T)[:, :self.H] + \
                   np.dot(dsigmoid(self.Gamma_f) * dGamma_f, self.Wf.T)[:, :self.H] + \
                   np.dot(dsigmoid(self.Gamma_u) * dGamma_u, self.Wu.T)[:, :self.H] + \
                   np.dot(dtanh(self.c_tilde) * dc_tilde, self.Wc.T)[:, :self.H]

        # dL/dc^{t-1} = dL/dh^t dh^t/dc^{t-1} + dL/dc^t dc^t/dc^{t-1}
        dc_input = dc_output * self.Gamma_f + dh_output * self.Gamma_o * dtanh(np.tanh(self.c_output)) * self.Gamma_f

        # dWo = dL/dGamma_o dGamma_o/dWo
        self.dWo = np.dot(self.concat.T, dsigmoid(self.Gamma_o) * dGamma_o)
        self.dWu = np.dot(self.concat.T, dsigmoid(self.Gamma_u) * dGamma_u)
        self.dWf = np.dot(self.concat.T, dsigmoid(self.Gamma_f) * dGamma_f)
        self.dWc = np.dot(self.concat.T, dtanh(self.c_tilde) * dc_tilde)

        self.dbo = np.sum(dsigmoid(self.Gamma_o) * dGamma_o, axis=0)
        self.dbu = np.sum(dsigmoid(self.Gamma_u) * dGamma_u, axis=0)
        self.dbf = np.sum(dsigmoid(self.Gamma_f) * dGamma_f, axis=0)
        self.dbc = np.sum(dtanh(self.c_tilde) * dc_tilde, axis=0) 

        return dh_input, dc_input

    def predict(self):
        """Return y_prob"""
        if self.h_output is None:
            print("Please run feed forward beforehand.")
        else:
            # y^hat = softmax(h^t W_hy + b_y)
            y_prob = softmax(np.dot(self.h_output, self.Wy) + self.by)
        return y_prob


class LSTM():
    def __init__(self, X, y, H=128, lr=0.0001):
 
        self.cells = []
        # X is of shape (n_samples, height, width)
        self.n_samples, self.n_cells, self.D = X.shape
        self.n_classes = y.shape[1]
        self.H = H 

        Wf = np.random.randn(self.H + self.D, self.H)
        Wu = np.random.randn(self.H + self.D, self.H)
        Wo = np.random.randn(self.H + self.D, self.H)
        Wc = np.random.randn(self.H + self.D, self.H)
        Wy = np.random.randn(self.H, self.n_classes)
        bf = np.zeros(self.H)
        bu = np.zeros(self.H)
        bo = np.zeros(self.H)
        bc = np.zeros(self.H)
        by = np.zeros(self.n_classes)
   
        self.h_init = np.random.randn(self.n_samples, self.H)
        self.c_init = np.random.randn(self.n_samples, self.H)
        # current
        i = 0
        while i < self.n_cells:
            self.cells.append(cell(Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by))
            i += 1

        self.X = X
        self.y = y
        self.lr = lr

    def feed_forward(self):
        """
            Xt here is X[:, i, :] 
        """
        h = self.h_init
        c = self.c_init
        for i, cell in enumerate(self.cells):
            h, c = cell.feed_forward(self.X[:, i, :], h, c)
        self.y_prob = self.cells[-1].predict()


    def back_propagation(self):
        # 
        dWy = np.dot(self.cells[-1].h_output.T, self.y_prob - self.y)
        # 
        dby = np.sum(self.y_prob - self.y, axis=0) 
        # 
        dh_last = np.dot(self.y_prob - self.y, self.cells[-1].Wy.T)
        # L is irrelevant to c^{last}
        dc_last = np.zeros((self.n_samples, self.H))

        dh, dc = dh_last, dc_last
        for cell in reversed(self.cells):
            dh, dc = cell.back_propagation(dh, dc)

        dWo = 0
        dWu = 0 
        dWf = 0 
        dWc = 0
        dbo = 0
        dbu = 0
        dbf = 0
        dbc = 0

        # 
        for cell in self.cells:
            dWo += cell.dWo
            dWu += cell.dWu
            dWf += cell.dWf
            dWc += cell.dWc
            dbo += cell.dbo
            dbu += cell.dbu
            dbf += cell.dbf
            dbc += cell.dbc 

        for cell in self.cells:
            cell.Wo -= self.lr * dWo
            cell.Wu -= self.lr * dWu
            cell.Wf -= self.lr * dWf
            cell.Wc -= self.lr * dWc
            cell.Wy -= self.lr * dWy 
            cell.bo -= self.lr * dbo 
            cell.bu -= self.lr * dbu 
            cell.bf -= self.lr * dbf 
            cell.bc -= self.lr * dbc 
            cell.by -= self.lr * dby

    def predict(self, Xtest):
        h = self.h_init
        c = self.c_init
        for i, cell in enumerate(self.cells):
            h, c = cell.feed_forward(Xtest[:,i,:], h, c)
        y_prob = self.cells[-1].predict()
        return np.argmax(y_prob, axis=1)

    def cross_entropy_loss(self):
        """
            only update loss.
        """
        #  $L = -\sum_n\sum_{i\in C} y_{n, i}\log(\hat{y}_{n, i})$
        # calculate y_prob
        self.feed_forward()
        return -np.sum(self.y*np.log(self.y_prob + 1e-6))

X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

#print(X_train_norm.shape)
#print(X_test_norm.shape)
#print(y_train_ohe.shape)
#print(y_test_ohe.shape)

myLSTM = LSTM(X_test_norm, y_test_ohe, H = 128, lr= 0.001)
epoch_num = 400
for i in range(epoch_num):
    myLSTM.feed_forward()
    myLSTM.back_propagation()
    if ((i + 1) % 20 == 0):
        y_pred = myLSTM.predict(X_test_norm)
        print('epoch = {}, current loss = {}, test accuracy = {:.2f}%'.format(i+1, myLSTM.cross_entropy_loss(), 100*accuracy(y_pred, y_test.ravel())))

#y_pred = myRNN.predict(X_test_norm)
# print(y_pred)
# print(y_test.ravel())
#print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))

toc = time.perf_counter()
print('Totol time: {:.2f}s'.format(toc-tic))
print('===============================Finish===================================')
