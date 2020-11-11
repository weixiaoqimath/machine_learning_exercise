


import numpy as np
from scipy.spatial import distance 


# In[69]:


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
    values = np.unique(X[:,col])
    num = values.size
    rss_list = np.zeros(num)

    for i in np.arange(num):
        bools = (X[:, col] <= values[i])
        left_target = y[bools]
        right_target = y[np.logical_not(bools)]
        rss_list[i] = rss(left_target) + rss(right_target)
    
    ind = np.argmin(rss_list)
        
    return rss_list[ind], values[ind]

def best_split_all(X, y):
    """Compute the best split among all columns.
    Return
    ------
    left_samples, left_targets, right_samples, right_targets, col, threshold
    """
    rss_list = np.zeros(X.shape[1])
    thresholds = np.zeros(X.shape[1])
    for i in np.arange(X.shape[1]):
        rss_list[i], thresholds[i] = best_split_col(X, y, i)
    col = np.argmin(rss_list)
    threshold = thresholds[col]
    left_samples, left_targets, right_samples, right_targets = partition(X, y, col, threshold) 
        
    return left_samples, left_targets, right_samples, right_targets, col, threshold

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


class DecisionTreeRegressor:
    
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.tree = None
        self.max_depth = max_depth
        
    def fit(self, X, y):
        self.tree = self.build_tree(X, y, self.min_samples_split)   

    def predict(self, Xtest):
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

        left_samples, left_targets, right_samples, right_targets, col, threshold = best_split_all(X, y)
        left_branch = self.build_tree(left_samples, left_targets, min_samples_split, depth+1)
        right_branch = self.build_tree(right_samples, right_targets, min_samples_split, depth+1)

        return decision_node(col = col, threshold = threshold, left_branch = left_branch, right_branch = right_branch)
        
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

        
        
        


