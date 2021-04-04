

import numpy as np
import pandas as pd
import time

def PCC(x, y):
    """
        x and y are 1d vectors having the same size.
        return the Pearson's coefficient.
    """   
    A = x - np.mean(x)
    B = y - np.mean(y)
    return np.dot(A, B)/np.sqrt(np.dot(A, A)*np.dot(B, B))

def RMSE(x, y):
    """
        x and y are 1d vectors having the same size.
    """
    m = x.shape[0]
    return np.sqrt(np.dot(x-y, x-y)/m)

def rss(y):
    """residue square error
    """
    if y.size==0 or y is None:
        return 0
    return np.sum((y-np.mean(y))**2)

def partition(X, y, feature, threshold):
    """
        split dataset with respect to a decision.
    """
    feature_values = X[:, feature]
    left_samples = X[feature_values <= threshold]
    left_target = y[feature_values <= threshold]
    right_samples = X[feature_values > threshold]
    right_target = y[feature_values > threshold]
    
    return left_samples, left_target, right_samples, right_target

def best_split_one_feature(X, y, feature):
    """
        Compute the best split along a certain column.
    """
    thresholds = np.unique(X[:, feature])
    gain_list = np.zeros(thresholds.size)
    best_gain = 0
    best_threshold = None
    if len(thresholds) == 1:
        return best_gain, best_threshold

    for threshold in thresholds:
        left_target = y[X[:, feature] <= threshold]
        right_target = y[X[:, feature] > threshold]
        new_gain = rss(y) - rss(left_target) - rss(right_target)
        if new_gain > best_gain:
            best_threshold = threshold
            best_gain = new_gain
        
    return best_gain, best_threshold

def tree_propagation(x, tree):
    """
        Predict x using a tree
    """
    if tree.predict != None:
        return tree.predict
    elif x[tree.feature] <= tree.threshold:
        branch = tree.left_branch
    else:
        branch = tree.right_branch
    
    return tree_propagation(x, branch)

class DT_regressor:
    
    def __init__(self, min_samples_split=2, max_depth = None, max_features=None):
        self.min_samples_split = min_samples_split
        self.tree = None
        self.max_depth = max_depth
        self.max_features=None
        
    def fit(self, X, y):
        if self.max_features== None or self.max_features > X.shape[1]:
            self.max_features = X.shape[1]
        self.tree = self.build_tree(X, y, self.min_samples_split)   

    def build_tree(self, X, y, min_samples_split, depth=1):
        """
            Create a decision tree recursively.
        """
        if X.shape[0] <= min_samples_split:
            return decision_node(predict = np.mean(y))
        elif self.max_depth == depth: # if max_depth is reached, build a leaf node.
            return decision_node(predict = np.mean(y))

        best_gain, best_feature, best_threshold = self.best_split_all(X, y)
        if best_gain == 0: # no need to further build branch from it
            return decision_node(predict = np.mean(y))

        left_samples, left_targets, right_samples, right_targets = partition(X, y, best_feature, best_threshold)
        left_branch = self.build_tree(left_samples, left_targets, min_samples_split, depth+1)
        right_branch = self.build_tree(right_samples, right_targets, min_samples_split, depth+1)

        return decision_node(feature = best_feature, threshold = best_threshold, left_branch = left_branch, right_branch = right_branch, depth=depth)

    def best_split_all(self, X, y):
        """Compute the best split among all columns.
        Return
        ------
        left_samples, left_targets, right_samples, right_targets, col, threshold
        """
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        if self.max_features == X.shape[1]:
            for idx in range(X.shape[1]):
                new_gain, new_threshold = best_split_one_feature(X, y, idx)
                if new_gain > best_gain:
                    best_gain = new_gain
                    best_threshold = new_threshold
                    best_feature = idx   
        else:
            sampled_features = np.random.choice(X.shape[1], self.max_features, replace=False)
            for idx in sampled_features:
                new_gain, new_threshold = best_split_one_feature(X, y, idx)
                if new_gain > best_gain:
                    best_gain = new_gain
                    best_threshold = new_threshold
                    best_feature = idx
                    
        return best_gain, best_feature, best_threshold

    def predict(self, Xtest):
        if Xtest.ndim == 1:
            return tree_propagation(Xtest, self.tree)
        # if Xtest.ndim == 2
        pred = np.zeros(Xtest.shape[0])
        for i in np.arange(Xtest.shape[0]):
            pred[i] = tree_propagation(Xtest[i], self.tree)
        return pred
        
class decision_node:
    """
        A Decision Node asks a question.
        This holds a reference to the question, and to the two child nodes.
    """
    def __init__(self, feature = -1, threshold = None, left_branch = None, right_branch = None, predict = None, depth=1):
        self.feature = feature
        self.threshold = threshold
        self.depth = depth
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.predict = predict # if self.predict is not None, then this node is indeed a leaf.

if __name__ == '__main__':
    import time

    Xtrain = pd.read_csv("airfoil_self_noise_X_train.csv").values
    ytrain = pd.read_csv("airfoil_self_noise_y_train.csv").values
    Xtest = pd.read_csv("airfoil_self_noise_X_test.csv").values
    ytest = pd.read_csv("airfoil_self_noise_y_test.csv").values

    ytrain, ytest = ytrain.flatten(), ytest.flatten()

    start = time.time()
    reg = DT_regressor(min_samples_split=2)
    reg.fit(Xtrain, ytrain)
    ypred = reg.predict(Xtest)
    rmse = RMSE(ypred, ytest)
    pcc = PCC(ypred, ytest)

    end = time.time()
    print("The RMSE is {:.2f} and the PCC is {:.2f}".format(rmse, pcc)) # 2.86 and 0.91
    print("Takes {:.2f} seconds".format(end-start))

    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor()
    regressor.fit(Xtrain, ytrain)
    ypred = regressor.predict(Xtest)
    rmse = RMSE(ypred, ytest)
    pcc = PCC(ypred, ytest)
    print("The RMSE is {:.2f} and the PCC is {:.2f}".format(rmse, pcc)) # 2.86 and 0.91

    start = time.time()
    reg = DT_regressor(min_samples_split=2, max_features=20)
    reg.fit(Xtrain, ytrain)
    ypred = reg.predict(Xtest)
    rmse = RMSE(ypred, ytest)
    pcc = PCC(ypred, ytest)

    end = time.time()
    print("The RMSE is {:.2f} and the PCC is {:.2f}".format(rmse, pcc)) # 2.86 and 0.91
    print("Takes {:.2f} seconds".format(end-start))





