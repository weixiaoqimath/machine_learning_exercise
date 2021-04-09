# up to date version

from DT_classifier import *
from DT_regressor import * 
from scipy import stats

class random_forest_classifier:
    def __init__(self, n_estimators=10, max_features=None, min_samples_split=2, max_depth=None):
        self.n_estimators = n_estimators
        self.forest = []
        for i in range(self.n_estimators):
            self.forest.append(DT_classifier(max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split))

        # feature values
        self.X = None
        self.y = None
        self.N = 0
        self.M = 0

    def fit(self, X, y):
        '''bagging'''
        self.X = X
        self.y = y
        self.N = X.shape[1]
        self.M = X.shape[0]

        self.trees_idx = np.random.randint(0, self.M, size=(self.n_estimators, self.M))
        for i, itree in enumerate(self.forest):
            itree.fit(self.X[self.trees_idx[i]], self.y[self.trees_idx[i]])
            print("The {}th tree is built".format(i))

    def predict(self, Xtest):
        n_test = Xtest.shape[0]
        ypred = np.zeros((self.n_estimators, n_test))
        for i, itree in enumerate(self.forest):
            ypred[i, :] = itree.predict(Xtest)
        return (stats.mode(ypred, axis=0)[0]).ravel() # stats.mode get the most common element along a certain axis

if __name__ == '__main__':
    import time
    Xtrain = pd.read_csv('MNIST_X_train.csv').values
    ytrain = pd.read_csv('MNIST_y_train.csv').values
    Xtest  = pd.read_csv('MNIST_X_test.csv').values
    ytest  = pd.read_csv('MNIST_y_test.csv').values
    ytrain, ytest = ytrain.flatten(), ytest.flatten()

    start = time.time()
    RF = random_forest_classifier(n_estimators=10, max_features=8) # n_estimator=10, max_features=None, max_depth=8, takes about 900s to run, accuracy is 84.40%
    # n_estimators=10, max_features=8, accuracy is around 81%, takes 17s to run.
    RF.fit(Xtrain, ytrain)
    ypred = RF.predict(Xtest)
    score = accuracy(ypred, ytest)
    end = time.time()
    print("The accuracy of my decision tree classifier is {:.2f}%".format(score*100))
    print("Takes {:.2f} seconds".format(end-start))
    
    from sklearn.ensemble import RandomForestClassifier
    skRF = RandomForestClassifier(max_features=8, n_estimators=10)
    skRF.fit(Xtrain, ytrain)

    ypred = skRF.predict(Xtest)
    print(accuracy(ypred, ytest.ravel())) 
