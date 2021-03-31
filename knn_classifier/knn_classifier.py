# The iris data set only has two classes.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class normalizer():
    def __init__(self):
        self.mean = 0
        self.std = 0

    def fit(self, X):
        self.mean = np.mean(X, axis=0) # mean of each column vector
        self.std = np.std(X, axis=0) # std of each column vector

    def transform(self, X):
        """
            feature normalization. Each row of X represents a point in R^d. 
            Substract by the mean of X and then divided by the std of X.
        """
        return (X - self.mean)/self.std

def k_neighbors(v, M, k):
    """
        v is a 1d vector and M is 2d array where each row is a point in R^d. 
        Return indices of k nearest neighbors.
    """
    distances = np.sum((M - v)**2, axis=1) # distances from a to points
    sorted_idx = distances.argsort() # quicksort
    return sorted_idx[:k]

def knn(Xtrain, ytrain, Xtest, k=6):
    """
        Return the knnion for Xtest
    """
    m = Xtest.shape[0]
    ypred = np.zeros(m)
    for idx, x in enumerate(Xtest):
        indices = k_neighbors(x, Xtrain, k) # indices of k nearest neighbors
        prob = np.mean(ytrain[indices]) # probability of being class 1
        if prob > 0.5:
            ypred[idx] = 1
        elif prob < 0.5:
            ypred[idx] = 0
        elif prob == 0.5: # When there is a tie, pick the index of the closest point. Note that there might be many closest points.
            ypred[idx] = ytrain[indices[0]]
    return ypred

# load data
Xtrain = pd.read_csv("Iris_X_train.csv")
ytrain = pd.read_csv("Iris_y_train.csv")
Xtest = pd.read_csv("Iris_X_test.csv")
ytest = pd.read_csv("Iris_y_test.csv")

Xtrain = Xtrain.values
ytrain = ytrain.values
Xtest = Xtest.values
ytest = ytest.values

ytrain, ytest = ytrain.flatten(), ytest.flatten()

k = 4
# rescale data
scaler = normalizer()
scaler.fit(Xtrain)
normalized_Xtrain = scaler.transform(Xtrain)
normalized_Xtest = scaler.transform(Xtest)

ypred = knn(normalized_Xtrain, ytrain, normalized_Xtest, k)
accuracy = np.mean(ypred == ytest)

print("When k is set to be {}, the accuracy is {:.2f}%".format(k, 100*accuracy))

from sklearn.neighbors import KNeighborsClassifier

for i in np.arange(1, 10):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(normalized_Xtrain, ytrain)
    print("When k is set to be {}, the accuracy calculated by sklearn is {:.2f}%".format(i, clf.score(normalized_Xtest, ytest)*100))

#x = np.arange(1, 21)
#y = np.zeros(20)
#for i in x:
#    y[i-1] = np.mean(knn(normalized_Xtrain, ytrain, normalized_Xtest, i) == ytest)
#    print(y[i-1])

#plt.figure()
#plt.plot(x, y) 
#plt.xticks(x) 
#plt.xlabel('k')
#plt.ylabel('accuracy')
#plt.show()



# In[ ]:




