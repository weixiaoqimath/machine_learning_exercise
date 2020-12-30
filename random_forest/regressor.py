#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

def accuracy(trueLabels, predLabels):
    return float(sum(trueLabels == predLabels.reshape(-1,1)))/ float(len(trueLabels))

def pcc(x, y):
    """Pearson coefficient.
    """
    xmean = np.mean(x, axis = 0)
    ymean = np.mean(y, axis = 0)
    
    m = x.shape[0]
    A = x - xmean
    B = y - ymean
    return (A.T@B)/np.sqrt((A.T@A)*(B.T@B))

def rss(y):
    """residue square error
    """
    if y.size == 0:
        return 0
    return np.sum((y-np.mean(y))**2)

def partition(X, y, col, threshold):
    """split dataset with respect to decision.
    """
    col_values = X[:, col] 
    bools = (col_values <= threshold)
    left_samples = X[bools]
    left_target = y[bools]
    right_samples = X[np.logical_not(bools)]
    right_target = y[np.logical_not(bools)]
    
    return left_samples, left_target, right_samples, right_target

def best_split_col(X, y, col):
    """Compute the best split along a certain column.
    """
    best_gain = 0
    best_threshold = None
    values = np.unique(X[:, col])
    # Be careful when the column only has a single value.
    if values.size == 1:
        return best_gain, best_threshold

    for i in np.arange(values.size-1):
        bools = (X[:, col] <= values[i])
        left_target = y[bools]
        right_target = y[np.logical_not(bools)]
        current_gain = rss(y) - rss(left_target) - rss(right_target)
        if current_gain > best_gain:
            best_gain = current_gain
            best_threshold = values[i]
        
    return best_gain, best_threshold

def predictor(x, tree):
    """Predict x using tree
    """
    if tree.predict != None:
        return tree.predict

    elif x[tree.col] <= tree.threshold:
        branch = tree.left_branch
    else:
        branch = tree.right_branch
    
    return predictor(x, branch)


# In[3]:


class RandomForestRegressor():
    
    def __init__(self, n_estimators=10, min_samples_split=2, max_features = "auto", max_depth = None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.max_depth = max_depth
        self.regressors = np.arange(self.n_estimators, dtype = object)

        
    def fit(self, X, y):
        # bagging
        for i in np.arange(self.n_estimators):
            sampled_indices = np.random.choice(X.shape[0], X.shape[0])
            X_sampled = X[sampled_indices, :]
            y_sampled = y[sampled_indices, :]
            self.regressors[i] = DecisionTreeRegressor(min_samples_split = self.min_samples_split, max_features = self.max_features, max_depth = self.max_depth)
            self.regressors[i].fit(X_sampled, y_sampled)  
            
    def predict(self, Xtest):
        predict_values = np.zeros((Xtest.shape[0], self.n_estimators))
        for i in np.arange(Xtest.shape[0]):
            for j in np.arange(self.n_estimators):
                predict_values[i, j] = predictor(Xtest[i], self.regressors[j].tree)
        pred = np.mean(predict_values, axis = 1)
        return pred.reshape(-1, 1)


# In[4]:


class DecisionTreeRegressor:
    
    def __init__(self, min_samples_split=2, max_depth=None, max_features = 'auto'):
        """
        max_features: 'auto', 'sqrt', 'log2'
            number of features used for each recursion.
        """
        self.min_samples_split = min_samples_split
        self.tree = None
        self.max_depth = max_depth
        self.max_features = max_features
        
    def fit(self, X, y):
        if self.max_features == 'auto':
            self.n_features = X.shape[1]
        if self.max_features == 'sqrt':
            self.n_features = int(np.sqrt(X.shape[1]))
        if self.max_features == 'log2':
            self.n_features = int(np.log2(X.shape[1]))
            
        self.tree = self.build_tree(X, y, self.min_samples_split)   

    def predict(self, Xtest):
        """Return an numpy array.
        """
        pred = np.zeros(Xtest.shape[0])
        for i in np.arange(Xtest.shape[0]):
            pred[i] = predictor(Xtest[i], self.tree)
        return pred.reshape(-1,1)

    def build_tree(self, X, y, min_samples_split, depth=1):
        """Create a decision tree.
        """
        if X.shape[0] <= min_samples_split:
            return decision_node(predict = np.mean(y))
        elif self.max_depth == depth:
            return decision_node(predict = np.mean(y))
        
        best_gain, best_col, best_threshold = self.best_split_all(X, y)
        if best_gain == 0:
            return decision_node(predict = np.mean(y))

        left_samples, left_targets, right_samples, right_targets = partition(X, y, best_col, best_threshold)
        left_branch = self.build_tree(left_samples, left_targets, min_samples_split, depth+1)
        right_branch = self.build_tree(right_samples, right_targets, min_samples_split, depth+1)

        return decision_node(col = best_col, threshold = best_threshold, left_branch = left_branch, right_branch = right_branch)
        
    def best_split_all(self, X, y):
        """Compute the best split among all columns.
        Return
        ------
        left_samples, left_targets, right_samples, right_targets, col, threshold
        """
        best_gain = 0
        best_col = None
        best_threshold = None
        
        if self.n_features == X.shape[1]:
            for ind in np.arange(X.shape[1]):
                current_gain, current_threshold = best_split_col(X, y, ind)
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_threshold = current_threshold
                    best_col = ind    
        else:
            sampled_features = np.random.choice(X.shape[1], self.n_features, replace=False)
            for ind in sampled_features:
                current_gain, current_threshold = best_split_col(X, y, ind)
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_threshold = current_threshold
                    best_col = ind
                    
        return best_gain, best_col, best_threshold
    
class decision_node:
    """A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """
    def __init__(self, col = -1, threshold = None, left_branch = None, right_branch = None, predict = None):
        self.col = col
        self.threshold = threshold
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.predict = predict # if self.predict is not None, then this node is indeed a leaf.




