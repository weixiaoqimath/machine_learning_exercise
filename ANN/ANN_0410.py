# can add layer to class ANN

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

def accuracy(ytest, ypred):
    return float(np.sum(ytest == ypred))/ float(len(ytest))

def read_dataset(feature_file, label_file):
    """
        Read data set in *.csv to data frame in Pandas.
    """
    X = pd.read_csv(feature_file).values
    y = pd.read_csv(label_file).values
    return X, y

def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler #import libaray
    scaler = StandardScaler() # call an object function
    scaler.fit(X_train) # calculate mean, std in X_train
    X_train_norm = scaler.transform(X_train) # apply normalization on X_train
    X_test_norm = scaler.transform(X_test) # we use the same normalization on X_test
    return X_train_norm, X_test_norm

def one_hot_encoder(y_train, y_test):
    """
        convert label to a vector under one-hot-code fashion.
    """
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_test_ohe = lb.transform(y_test)
    return y_train_ohe, y_test_ohe

class ANN():
    def __init__(self, X, y, lr=0.01):
        self.X = X # features
        self.y = y # labels (targets) in one-hot-encoder
        self.lr = lr # learning rate
        # Initialize weights
        self.Ws = []
        self.bs = []
        self.functions = [] 
        self.fs = [] # store X, f_1, f_2, ...
        
        self.n_hidden_layer = -1
        
    def add_layer(self, input_nn, output_nn):
        self.Ws.append(np.random.randn(input_nn, output_nn) / np.sqrt(input_nn))  
        self.bs.append(np.zeros(output_nn))    
        self.n_hidden_layer += 1

    def add_activation(self, func='tanh'):
            self.functions.append(func) 
        
    def feed_forward(self):
        f = self.X 
        for idx in range(self.n_hidden_layer+1):
            # z_i = f_{i-1}W_i + b_i
            z = np.dot(f, self.Ws[idx]) + self.bs[idx] 
            if self.functions[idx] == 'tanh':
                f = np.tanh(z) 
            if self.functions[idx] == 'softmax':
                f = softmax(z)
            self.fs.append(f) # len(f) is n_hidden_layer + 1

        self.y_prob = self.fs[-1] # store y_prob
        
    def back_propagation(self):
        dzs = np.arange(self.n_hidden_layer+1, dtype=object) # store dL/dz_n, dL/dz_{n-1}, ..., dL/dz_1
        dzs[-1] = self.y_prob - self.y
        for i in np.arange(2, self.n_hidden_layer+2): # from 2 to n_hidden_layer+1
            dzs[-i] = np.dot(dzs[-(i-1)], self.Ws[-(i-1)].T) * (1 - self.fs[-i]**2)
        
        dWs = np.arange(self.n_hidden_layer+1, dtype=object)
        dbs = np.arange(self.n_hidden_layer+1, dtype=object)
        for i in np.arange(1, self.n_hidden_layer+1):
            # dL/dW_i = dL/dz_i dz_i/dW_i = f_{i-1}^T dL/dz_i
            dWs[-i] = np.dot(self.fs[-(i+1)].T, dzs[-i])
            dbs[-i] = np.sum(dzs[-i], axis=0)
        
        dWs[0] = np.dot(self.X.T, dzs[0])
        dbs[0] = np.sum(dzs[0], axis=0)

        for i in range(self.n_hidden_layer+1):
            self.Ws[i] -= self.lr*dWs[i]
            self.bs[i] -= self.lr*dbs[i]
        
    def cross_entropy_loss(self):
        """
            only update loss.
        """
        #  L = -\sum_n \sum_{i\in C} y_{n, i}\log(y_prob_{n, i})
        # calculate y_prob
        self.feed_forward()
        self.loss = -np.sum(self.y * np.log(self.y_prob + 1e-6)) 
        
    def predict(self, X_test):
        f = X_test
        for idx in range(self.n_hidden_layer+1):
            z = np.dot(f, self.Ws[idx]) + self.bs[idx]
            if self.functions[idx] == 'tanh':
                f = np.tanh(z)
            if self.functions[idx] == 'softmax':
                f = softmax(z)

        y_pred = np.argmax(f, axis=1)
        return y_pred

if __name__ == '__main__':
    import time        
 
    # load data
    X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
    X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')

    #print(X_train.shape)
    #print(X_test.shape)

    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

    # initialize network
    myANN = ANN(X_train_norm, y_train_ohe, lr=0.001)  
    myANN.add_layer(X_train_norm.shape[1], 300)
    myANN.add_activation('tanh')
    myANN.add_layer(300, 100)
    myANN.add_activation('tanh')
    myANN.add_layer(100, y_train_ohe.shape[1])
    myANN.add_activation('softmax')

    start = time.time()

    epoch_num = 200
    for i in range(epoch_num):
        myANN.feed_forward()
        myANN.back_propagation()
        myANN.cross_entropy_loss()
        y_pred = myANN.predict(X_test_norm)
        if (i+1) % 20 == 0: 
            print('epoch = {}, current loss = {:.2f}, test accuracy = {:.2f}%'.format(i+1, myANN.loss, 100*accuracy(y_pred, y_test.ravel())))  
              

    end = time.time()
    y_pred = myANN.predict(X_test_norm)
    print('Accuracy of my ANN model is {:.2f}%'.format(accuracy(y_pred, y_test.ravel())*100))
    print('Takes {:.2f}s'.format(end-start))
