
import pandas as pd
import scipy
import numpy as np
from DT_regressor import *
#from sklearn.ensemble import RandomForestClassifier as SKRFL
import sys

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
        self.regressors = []
        #for i in range(self.n_estimators):
        #    self.regressors.append(DT_regressor(max_depth = self.max_depth, min_samples_split = self.min_samples_split, max_features = self.max_features))

    def fit(self, X, y):
        """
            y has the shape (n_instance, n_classes)
        """
        
        self.n_classes = y.shape[1]
        
        y_pred = np.zeros((y.shape[0],y.shape[1]))
        y_prob = np.full(y.shape, 1/y.shape[1]) # 
        residue = y - y_prob
        
        for m in range(self.n_estimators):
            for k in range(self.n_classes):
                Reg = DT_regressor(max_depth = self.max_depth, min_samples_split = self.min_samples_split, max_features = self.max_features)
                Reg.fit(X, residue[:, k])
                #print(k, m)
                node_labels = get_node_labels(Reg.tree, X)
                J = np.unique(node_labels)
                for node in J:
                    gamma = (self.n_classes - 1)/self.n_classes * np.sum(residue[:, k][node_labels==node])/ np.sum(np.abs(residue[:, k][node_labels==node])*(1-np.abs(residue[:, k][node_labels==node])))
                    #print(gamma)
                    update_node(Reg.tree, str(node), gamma)
                y_pred[:, k] = y_pred[:, k] + self.lr*Reg.predict(X)
                self.regressors.append(Reg) 
            y_prob = softmax(y_pred)
            residue = y - y_prob 
            
    def predict(self, Xtest):
        """
            Return array of shape (Xtest.shape[0]).
        """
        y_pred = np.zeros((Xtest.shape[0], self.n_classes))

        for k in range(self.n_classes):
            regs = self.regressors[k::self.n_classes]
            for reg in regs:
                y_pred[:, k] = y_pred[:, k] + self.lr*reg.predict(Xtest)     
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
      
    GBTC = GBT_classifier(n_estimators=10, max_depth = 4, lr = 0.1, max_features=20) # around 60s, around 83%
    GBTC.fit(Xtrain, ytrain_ohe)
    ypred = GBTC.predict(Xtest)


    end = time.time()
    score = accuracy(ytest, ypred)
    print("The accuracy of multiclass classification is {:.2f}%".format(score*100))
    print("Takes {:.2f} seconds.".format(end - start))

    gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=10, max_depth=4, max_features=20)
    gbc.fit(Xtrain, ytrain)
    score=gbc.score(Xtest, ytest) 
    print(score) # around 81.00%



