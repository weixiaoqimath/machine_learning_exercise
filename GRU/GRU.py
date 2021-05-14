"""
    Reference: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    Our formulation is slightly different from the formulation in the reference.
"""
import numpy as np
import pandas as pd
import time

# 94-95%

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

class cell():
    """
        An GRU cell.
    """
    def __init__(self, W, Wz, Wr, Wy, by):
        self.W = W
        self.Wz = Wz
        self.Wr = Wr
        self.Wy = Wy
        self.by = by
        self.H = Wy.shape[0] 
        self.h_output = None
        self.h_input = None


    def feed_forward(self, Xt, h_input):
        self.Xt = Xt
        self.h_input = h_input

        concat = np.concatenate((h_input, Xt), axis=1)
        # z^t = sigmoid([h^{t-1}, x^t] Wz)
        z = sigmoid(np.dot(concat, self.Wz))
        # r^t = sigmoid([h^{t-1}, x^t] Wr)
        r = sigmoid(np.dot(concat, self.Wr))

        concat_ = np.concatenate((r * h_input, Xt), axis=1)
        # h^tilde^t = tanh([r^t * h^{t-1}, x^t] W)
        h_tilde = np.tanh(np.dot(concat_, self.W))
        # h^t = (1-z^t)*h^{t-1} + z^t*h^{tilde}^t
        h_output = (1-z) * h_input + z * h_tilde

        self.concat = concat
        self.concat_ = concat_
        self.z = z 
        self.r = r 
        self.h_tilde = h_tilde
        self.h_output = h_output
        return h_output 

    def predict(self):
        """Return y_prob"""
        if self.h_output is None:
            print("Please run feed forward beforehand.")
        else:
            # y^hat = softmax(h^t Wy + by)
            y_prob = softmax(np.dot(self.h_output, self.Wy) + self.by)
        return y_prob

    def back_propagation(self, dh_output):
        """
            dh_output = dL/dh_t. 
        """
        # dL/dh^t dh^t/dz^t
        dz = dh_output * (-self.h_input + self.h_tilde)
        # dL/dh^t dh^t/dh^{tilde}^t
        dh_tilde = dh_output * self.z 
        # dL/dh^{tilde}^t dh^{tilde}^t/dr^t
        dr = np.dot(dh_tilde * dtanh(self.h_tilde), self.W.T)[:, :self.H] * self.h_input
        # dL/dh^{t-1} = dL/dh^t dh^t/dh^{t-1} + dL/dz^t dz^t/dh^{t-1} + dL/dr^t dr^t/dh^{t-1} + dL/dh^{tilde}^t dh^{tilde}^t/dh^{t-1}
        dh_input = dh_output * (1-self.z) + np.dot(dz * dsigmoid(self.z), self.Wz.T)[:, :self.H] + \
                   np.dot(dr * dsigmoid(self.r), self.Wr.T)[:, :self.H] + \
                   np.dot(dh_tilde * dtanh(self.h_tilde), self.W.T)[:, :self.H] * self.r 
        # 
        self.dW = np.dot(self.concat_.T, dh_tilde * dtanh(self.h_tilde))
        self.dWz = np.dot(self.concat_.T, dz * dsigmoid(self.z))
        self.dWr = np.dot(self.concat_.T, dr * dsigmoid(self.r))

        return dh_input 

class GRU():
    def __init__(self, X, y, H=128, lr=0.0001):
   
        self.cells = []
        # X is of shape (n_samples, height, width)
        self.n_samples, self.n_cells, self.D = X.shape
        self.n_classes = y.shape[1]
        self.H = H 

        W = np.random.randn(self.H + self.D, self.H)
        Wr = np.random.randn(self.H + self.D, self.H)
        Wz = np.random.randn(self.H + self.D, self.H)
        Wy = np.random.randn(self.H, self.n_classes)
        by = np.zeros(self.n_classes)
        self.h_init = np.random.randn(self.n_samples, self.H)

        i = 0
        while i < self.n_cells:
            self.cells.append(cell(W, Wz, Wr, Wy, by))
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
        # dL/dWy = np.dot(h_last.T, y_prob - y)
        dWy = np.dot(self.cells[-1].h_output.T, self.y_prob - self.y)
        # dL/dby = \sum_axis=0 y_prob - y
        dby = np.sum(self.y_prob - self.y, axis=0) 
        # dL/dh_last = np.dot(y_prob - y, W_hy.T)
        dh_last = np.dot(self.y_prob - self.y, self.cells[-1].Wy.T)

        dh_output = dh_last
        for cell in reversed(self.cells):
            dh_output = cell.back_propagation(dh_output)

        dW = 0
        dWr = 0 
        dWz = 0 
        # 
        for cell in self.cells:
            dW += cell.dW
            dWr += cell.dWr
            dWz += cell.dWz

        for cell in self.cells:
            cell.W -= self.lr * dW
            cell.Wr -= self.lr * dWr
            cell.Wz -= self.lr * dWz
            cell.Wy -= self.lr * dWy
            cell.by -= self.lr * dby

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

X_train, y_train = read_dataset('../RNN/MNIST_X_train.csv', '../RNN/MNIST_y_train.csv')
X_test, y_test = read_dataset('../RNN/MNIST_X_test.csv', '../RNN/MNIST_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

#print(X_train_norm.shape)
#print(X_test_norm.shape)
#print(y_train_ohe.shape)
#print(y_test_ohe.shape)

myGRU = GRU(X_test_norm, y_test_ohe, H = 128, lr= 0.0001)
epochs = 500
for i in range(epochs):
    myGRU.feed_forward()
    myGRU.back_propagation()
    if ((i + 1) % 20 == 0):
        y_pred = myGRU.predict(X_test_norm)
        print('epoch = {}, current loss = {}, test accuracy = {:.2f}%'.format(i+1, myGRU.cross_entropy_loss(), 100*accuracy(y_pred, y_test.ravel())))

#y_pred = myGRU.predict(X_test_norm)
# print(y_pred)
# print(y_test.ravel())
#print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))

toc = time.perf_counter()
print('Totol time: {:.2f}s'.format(toc-tic))
print('===============================Finish===================================')

#print('The epoch number is set to be {}'.format(epochs))

     