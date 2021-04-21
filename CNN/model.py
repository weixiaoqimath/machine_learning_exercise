import pandas as pd
import numpy as np
from layer import *

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
    X_train_norm = np.reshape(X_train_norm1,(-1,1,28,28)) # reshape X to be a 3-D array 
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

class Network:
    def __init__(self, lr):
        self.layers = []
        self.lr = lr
        self.y_prob = None
        #self.X = X
        #self.y = y
        self.loss = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def feed_data(self, X, y):
        self.X = X
        self.y = y

    def feed_forward(self):                # feed forward 
        input = self.X
        for layer in self.layers:
            input = layer.feed_forward(input)
        self.y_prob = input
        return input

    def back_propagation(self):                # back propagate
        gradient = self.y # for the output layer, the input gradient is indeed target.
        for layer in reversed(self.layers):
            gradient = layer.back_propagation(gradient, self.lr) 

    def cross_entropy_loss(self):
        """
            only update loss.
        """
        #  $L = -\sum_n\sum_{i\in C} y_{n, i}\log(\hat{y}_{n, i})$
        # calculate y_hat
        self.feed_forward()
        self.loss = -np.sum(self.y*np.log(self.y_prob + 1e-6)) 

    def predict(self, X_test):
        input = X_test
        for layer in self.layers:
            #if plot_feature_maps:
            #    image = (image * 255)[0, :, :]
            #    plot_sample(image, None, None)
            input = layer.feed_forward(input)
        return np.argmax(input, axis=1)     

    def mini_batch_descent(self, X, y, batch_size=50):
        """
            Note that y is a 2d array. 
        """
        shuffled_idx = np.arange(X.shape[0])
        np.random.shuffle(shuffled_idx)
        n_batches = X.shape[0] // batch_size 
        
        for i in range(n_batches):
            self.feed_data(X[shuffled_idx[i*batch_size:(i+1)*batch_size]], y[shuffled_idx[i*batch_size:(i+1)*batch_size]])
            self.feed_forward()
            self.back_propagation()
        if X.shape[0] % batch_size != 0:
            self.feed_data(X[shuffled_idx[n_batches*batch_size:]], y[shuffled_idx[n_batches*batch_size:]])
            self.feed_forward()
            self.back_propagation()


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
    myANN = Network(lr=0.0001) 
    myANN.add_layer(convolutional(shape=(1,1,3,3)))
    myANN.add_layer(max_pooling())
    #myANN.add_layer(linear(784, 300, 'tanh'))
    myANN.add_layer(linear(169, 100, 'leakyReLU'))
    #myANN.add_layer(linear(300, 100, 'tanh'))
    myANN.add_layer(output_layer(100, y_train_ohe.shape[1], 'softmax'))

    start = time.time()

    #myANN.feed_data(X_train_norm, y_train_ohe)
    epoch_num = 40
    for i in range(epoch_num):
        myANN.mini_batch_descent(X_train_norm, y_train_ohe, batch_size=100)
        myANN.cross_entropy_loss()
        y_pred = myANN.predict(X_test_norm)
        print('epoch = {}, current loss = {:.2f}, test accuracy = {:.2f}%'.format(i+1, myANN.loss, 100*accuracy(y_pred, y_test.ravel())))      
        #print('epoch = {}, current loss = {:.2f}, test accuracy = {:.2f}%'.format(i+1, myANN.loss, 100*accuracy(y_pred, y_test.ravel())))      

    end = time.time()

    y_pred = myANN.predict(X_test_norm)
    print('Accuracy of my ANN model is {:.2f}%'.format(accuracy(y_pred, y_test.ravel())*100))
    print('Takes {:.2f}s'.format(end-start))



    