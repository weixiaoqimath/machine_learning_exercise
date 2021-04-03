import numpy as np
import pandas as pd
import time

def accuracy(ytest, ypred):
    return float(np.sum(ytest == ypred))/ float(len(ytest))

def majority_class(y):
    """
        Compute the majority class
    """
    unique, counts = np.unique(y, return_counts=True)
    return unique[np.argmax(counts)]

def gini(y):
    """
        y is a 1d array.
        Return the gini index
    """
    unique, counts = np.unique(y, return_counts=True)
    return 1 - np.dot(counts/y.size, counts/y.size)

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
    best_idx = None

    for i in np.arange(thresholds.size-1):
        left_target = y[X[:, feature] <= thresholds[i]]
        right_target = y[X[:, feature] > thresholds[i]]
        p = left_target.size / y.size
        new_gain = gini(y) - p * gini(left_target) - (1-p) * gini(right_target)
        if new_gain > best_gain:
            best_idx = i
            best_gain = new_gain
        
    return best_gain, thresholds[best_idx]

def best_split_all(X, y):
    """
        Compute the best split among all columns.
        Return
        ------
        left_samples, left_targets, right_samples, right_targets, best_feature, best_threshold
    """
    best_gain = 0
    best_threshold = None
    best_feature = None
    for i in np.arange(X.shape[1]):
        new_gain, new_threshold = best_split_one_feature(X, y, i)
        if new_gain > best_gain:
            best_threshold = new_threshold
            best_feature = i
            best_gain = new_gain

    left_samples, left_targets, right_samples, right_targets = partition(X, y, best_feature, best_threshold) 
        
    return left_samples, left_targets, right_samples, right_targets, best_feature, best_threshold

def predictor(x, tree):
    """
        Predict x using a tree
    """
    if tree.predict != None:
        return tree.predict
    elif x[tree.feature] <= tree.threshold:
        branch = tree.left_branch
    else:
        branch = tree.right_branch
    
    return predictor(x, branch)

class DT_classifier:
    
    def __init__(self, min_samples_split=2, max_depth = None):
        self.min_samples_split = min_samples_split
        self.tree = None
        self.max_depth = max_depth
        
    def fit(self, X, y):
        self.tree = self.build_tree(X, y, self.min_samples_split)   

    def build_tree(self, X, y, min_samples_split, depth=1):
        """
            Create a decision tree recursively.
        """
        unique = np.unique(y)
        if unique.size == 1:
            return decision_node(predict = y[0])
        elif X.shape[0] <= min_samples_split:
            return decision_node(predict = majority_class(y))
        elif self.max_depth == depth: # if max_depth is reached, build a leaf node.
            return decision_node(predict = majority_class(y))

        left_samples, left_targets, right_samples, right_targets, feature, threshold = best_split_all(X, y)
        left_branch = self.build_tree(left_samples, left_targets, min_samples_split, depth+1)
        right_branch = self.build_tree(right_samples, right_targets, min_samples_split, depth+1)

        return decision_node(feature = feature, threshold = threshold, left_branch = left_branch, right_branch = right_branch, depth=depth)

    def predict(self, Xtest):
        if Xtest.shape[0] == 1:
            return predictor(Xtest, self.tree)
        pred = np.zeros(Xtest.shape[0])
        for i in np.arange(Xtest.shape[0]):
            pred[i] = predictor(Xtest[i], self.tree)
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

import time

Xtrain = pd.read_csv("MNIST_X_train.csv").values
ytrain = pd.read_csv("MNIST_y_train.csv").values
Xtest = pd.read_csv("MNIST_X_test.csv").values
ytest = pd.read_csv("MNIST_y_test.csv").values

ytrain, ytest = ytrain.flatten(), ytest.flatten()

start = time.time()
clf = DT_classifier(min_samples_split=3, max_depth = 10)
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
#print(ypred)
score = accuracy(ypred, ytest)
end = time.time()
print("The accuracy of my decision tree classifier is {:.2f}%".format(score*100))
print("Takes {:.2f} seconds".format(end-start))