import numpy as np
import pandas as pd
import time

np.random.seed(7)
tic = time.perf_counter()

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

class cell():
    """
        An RNN cell.
    """
    def __init__(self, W_hx, W_hh, W_hy, bh, by):
        self.W_hx = W_hx
        self.W_hh = W_hh
        self.W_hy = W_hy
        self.bh = bh
        self.by = by 
        self.output = None
        self.input = None

    def feed_forward(self, Xt, h_input):
        self.Xt = Xt
        self.input = h_input
        # h^t = tanh(x^t W_hx + h^{t-1}W_hh + b_h)
        h_output = np.tanh(np.dot(Xt, self.W_hx) + np.dot(h_input, self.W_hh) + self.bh)
        self.output = h_output
        return h_output 

    def predict(self):
        """Return y_prob"""
        if self.output is None:
            print("Please run feed forward beforehand.")
        else:
            # y^hat = softmax(h^t W_hy + b_y)
            y_prob = softmax(np.dot(self.output, self.W_hy) + self.by)
        return y_prob

    def back_propagation(self, gradient):
        """
            gradient = dL/dh_t
        """
        dtanh = 1 - self.output**2
        # dL/dh^t dh^t/dW_hx = np.dot(X^t.T, dL/dh^t * (1- h^t **2))
        self.dW_hx = np.dot(self.Xt.T, gradient * dtanh)
        # dL/dh^t dh^t/W_hh = np.dot(h^{t-1}.T, dL/dh^t * (1- h^t **2))
        self.dW_hh = np.dot(self.input.T, gradient * dtanh)
        self.dbh = np.sum(gradient* dtanh, axis=0)
        # dL/dh^{t-1} = dL/dh^t dh^t/dh^{t-1}
        self.dh_input = np.dot(gradient* dtanh, self.W_hh.T)

        return self.dh_input 

class RNN():
    def __init__(self, X, y, H=128, lr=0.0001):
        i = 0
        self.cells = []
        # X is of shape (n_samples, height, width)
        self.n_samples, self.n_cells, self.D = X.shape
        self.n_classes = y.shape[1]
        self.H = H 

        W_hx = np.random.randn(self.D, self.H)
        W_hh = np.random.randn(self.H, self.H)
        W_hy = np.random.randn(self.H, self.n_classes)
        bh = np.zeros(self.H) 
        by = np.zeros(self.n_classes)
        self.h_init = np.random.randn(self.n_samples, self.H)

        while i < self.n_cells:
            self.cells.append(cell(W_hx, W_hh, W_hy, bh, by))
            i += 1

        self.X = X
        self.y = y
        self.lr = lr

    def feed_forward(self):
        """
            Xt here is X[:, i, :] 
        """
        h = self.h_init
        for i, cell in enumerate(self.cells):
            h = cell.feed_forward(self.X[:, i, :], h)
        self.y_prob = self.cells[-1].predict()


    def back_propagation(self):
        # dL/dW_hy = np.dot(h_last.T, y_prob - y)
        self.dL_dW_hy = np.dot(self.cells[-1].output.T, self.y_prob - self.y)
        # dL/db_y = \sum_axis=0 y_prob - y
        self.dL_dby = np.sum(self.y_prob - self.y, axis=0) 
        # dL/dh_last = np.dot(y_prob - y, W_hy.T)
        self.dL_dh_last = np.dot(self.y_prob - self.y, self.cells[-1].W_hy.T)

        gradient = self.dL_dh_last
        for cell in reversed(self.cells):
            gradient = cell.back_propagation(gradient)

        self.dL_dW_hx = 0
        self.dL_dW_hh = 0 
        self.dL_dbh = 0 
        # dL/dW_hx = dL/dh_0 dh_0/dW_hx + dL/dh_1 dh_1/dW_hx + ... + dL/dh_{n_cells-1} dh_{n_cells-1}/dW_hx
        for cell in self.cells:
            self.dL_dW_hx += cell.dW_hx
            self.dL_dW_hh += cell.dW_hh
            self.dL_dbh += cell.dbh

        for cell in self.cells:
            cell.W_hx -= self.lr * self.dL_dW_hx
            cell.W_hh -= self.lr * self.dL_dW_hh
            cell.W_hy -= self.lr * self.dL_dW_hy
            cell.bh -= self.lr * self.dL_dbh
            cell.by -= self.lr * self.dL_dby

    def predict(self, Xtest):
        h = self.h_init
        for i, cell in enumerate(self.cells):
            h = cell.feed_forward(Xtest[:,i,:], h)
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

#======================================

X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

#print(X_train_norm.shape)
#print(X_test_norm.shape)
#print(y_train_ohe.shape)
#print(y_test_ohe.shape)

myRNN = RNN(X_test_norm, y_test_ohe, H = 128, lr= 0.0001)
epochs = 600
for i in range(epochs):
    myRNN.feed_forward()
    myRNN.back_propagation()
    if ((i + 1) % 20 == 0):
        y_pred = myRNN.predict(X_test_norm)
        print('epoch = {}, current loss = {}, test accuracy = {:.2f}%'.format(i+1, myRNN.cross_entropy_loss(), 100*accuracy(y_pred, y_test.ravel())))

#y_pred = myRNN.predict(X_test_norm)
# print(y_pred)
# print(y_test.ravel())
#print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))

toc = time.perf_counter()
print('Totol time: {:.2f}s'.format(toc-tic))
print('===============================Finish===================================')

#print('The epoch number is set to be {}'.format(epochs))

     