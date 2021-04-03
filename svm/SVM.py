# written by Jiahui Chen
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, StandardScaler

def printImage(image):
    fig = plt.figure
    plt.imshow(image, cmap='Blues')
    plt.show()

def accuracy(ypred, yreal):
    return np.sum(ypred==yreal)/float(len(yreal))

class SVM():
    def __init__(self, X, y, kernel='linear', lr=0.01, lmd=0.01, epoch=2000):
        '''
        - X: shape (M, N)
        - y: shape (M, P), P = 10
        - W: shape (N, P)
        '''
        self.scaler = StandardScaler()
        self.scaler.fit(X)

        self.X     = self.scaler.transform(X)
        self.y     = y
        self.M     = X.shape[0]
        self.N     = X.shape[1]
        self.P     = y.shape[1]
        self.W     = np.random.randn(self.M, self.P)
        self.b     = np.zeros((1, self.P))
        self.lr    = lr
        self.lmd   = lmd # lambda
        self.epoch = epoch

        self.kernels = {'linear': self.linear, 'poly': self.poly,
                        'rbf': self.rbf, 'sigmoid': self.sigmoid}
        self.kernel = self.kernels[kernel]
        self.gamma  = 1/self.N
        self.KerMat = self.kernel_matrix(self.X)

    def fit(self):
        for _ in range(self.epoch):
            y_hat = np.dot(self.KerMat, self.W) + self.b
            cond  = 1 - self.y*y_hat
            y     = np.where(cond>0, self.y, 0)

            Wnorm = np.linalg.norm(self.W, axis=0) # (1, P)
            dW    = (1/Wnorm)*self.W - self.lmd*np.dot(self.KerMat.T, y)
            db    = -np.sum(self.lmd*y, axis=0)

            self.W -= self.lr*dW
            self.b -= self.lr*db

    def predict(self, X_test):
        X = self.scaler.transform(X_test)
        KerMat = self.kernel_matrix(X)
        y_hat_test = np.dot(KerMat, self.W) + self.b
        labels = [x for x in range(10)]
        ypred = np.zeros(X_test.shape[0], dtype=int)
        for i in range(X_test.shape[0]):
            ypred[i] = labels[np.argmax(y_hat_test[i, :])]
        return ypred

    def kernel_matrix(self, X):
        M = X.shape[0]
        KerMat = np.zeros((M, self.M))
        for i in range(M):
            for j in range(self.M):
                KerMat[i, j] = self.kernel(X[i,:], self.X[j,:])
        return KerMat

    # Xi: training, Xj: landmarker
    def linear(self, Xi, Xj):
        return np.dot(Xi, Xj)

    def poly(self, Xi, Xj):
        return (self.gamma * np.dot(Xi, Xj))**2

    def rbf(self, Xi, Xj):
        return np.exp(-self.gamma*np.dot(Xi-Xj, (Xi-Xj)))

    def sigmoid(self, Xi, Xj):
        return np.tanh(self.gamma * np.dot(Xi, Xj))

if __name__ == '__main__':
    X_train = pd.read_csv('MNIST_X_train.csv').values
    y_train = pd.read_csv('MNIST_y_train.csv').values
    X_test  = pd.read_csv('MNIST_X_test.csv').values
    y_test  = pd.read_csv('MNIST_y_test.csv').values
    #print(len(y_test))
    #printImage(X_train[1].reshape((28, 28)))

    lb = LabelBinarizer(neg_label=-1)
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_test_ohe  = lb.transform(y_test)

    mySVM = SVM(X_train, y_train_ohe, kernel='poly', lr=0.01)
    mySVM.fit()
    y_pred = mySVM.predict(X_test)
    print('Accuracy:', accuracy(y_pred, y_test.ravel()))
