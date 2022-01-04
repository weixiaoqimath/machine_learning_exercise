
import pandas as pd
import sklearn.tree
import numpy as np
from DT_regressor import *

def accuracy(ytest, ypred):
    return float(np.sum(ytest == ypred))/ float(len(ytest))

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

class GBT_classifier():
    def __init__(self, n_estimators = 10, min_samples_split = 2, lr = 0.01, max_depth = None, max_features = None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.lr = lr # shrinkage coefficient
        self.max_depth = max_depth
        self.max_features = max_features
        self.gammas = []
        self.regressors = {}
        #for i in range(self.n_estimators):
        #    self.regressors.append(DT_regressor(max_depth = self.max_depth, min_samples_split = self.min_samples_split, max_features = self.max_features))

    def fit(self, X, y):
        """
            y has the shape (n_instance, n_classes)
        """
        
        self.n_classes = y.shape[1]
        
        y_pred = np.zeros((y.shape[0],y.shape[1]))
        y_prob = np.full(y.shape, 1/y.shape[1]) # result of softmax
        residue = y - y_prob # negative gradients
        
        for m in range(self.n_estimators):
            self.regressors[m] = []
            for k in range(self.n_classes):
                self.regressors[m].append(sklearn.tree.DecisionTreeRegressor(max_depth=self.max_depth))

        for m in range(self.n_estimators):
            print(m)
            for k in range(self.n_classes):
                self.regressors[m][k].fit(X, residue[:, k])
                single_tree_pred = self.regressors[m][k].predict(X)
                y_pred[:, k] += self.lr*single_tree_pred
            
            y_prob = softmax(y_pred)
            residue = y - y_prob 
            
    def predict(self, Xtest):
        """
            Return array of shape (Xtest.shape[0]).
        """
        y_pred = np.zeros((Xtest.shape[0], self.n_classes))

        for m in range(self.n_estimators):
            for k in range(self.n_classes):
                y_pred[:, k] += self.lr*self.regressors[m][k].predict(Xtest)     
        y_prob = softmax(y_pred)
        
        return np.argmax(y_prob, axis=1)

if __name__ == '__main__':

    from sklearn.preprocessing import LabelBinarizer, StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    import time
    Xtrain = pd.read_csv("MNIST_X_train.csv").values
    ytrain = pd.read_csv("MNIST_y_train.csv").values
    Xtest = pd.read_csv("MNIST_X_test.csv").values
    ytest = pd.read_csv("MNIST_y_test.csv").values
    ytrain, ytest = ytrain.flatten(), ytest.flatten()

    lb = LabelBinarizer(neg_label=0)
    lb.fit(ytrain)
    ytrain_ohe = lb.transform(ytrain)
    ytest_ohe  = lb.transform(ytest)

    start = time.time()
      
    GBTC = GBT_classifier(n_estimators=100, max_depth = 3, lr = 0.5) 
    GBTC.fit(Xtrain, ytrain_ohe)
    ypred = GBTC.predict(Xtest)
    # learning_rate=0.1, n_estimators=10, max_depth=4, max_features=20, around 60s, around 83%


    end = time.time()
    score = accuracy(ytest, ypred)
    print("The accuracy of multiclass classification is {:.2f}%".format(score*100))
    print("Takes {:.2f} seconds.".format(end - start))

    gbc = GradientBoostingClassifier(learning_rate=0.5, n_estimators=100, max_depth=3, max_features=2)
    gbc.fit(Xtrain, ytrain)
    score=gbc.score(Xtest, ytest) 
    print(score) # learning_rate=0.1, n_estimators=10, max_depth=4, max_features=20, around 81.00%



