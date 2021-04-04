from DT_classifier import *
from DT_regressor import * 

class random_forest_regressor:
    def __init__(self, n_estimators=10, max_features=None):
        self.n_estimators = n_estimators
        self.forest = []
        for i in range(self.n_estimators):
            self.forest.append(DT_regressor(max_depth=8, max_features=max_features))

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
        return np.mean(ypred, axis=0) # stats.mode get the most common element along a certain axis

if __name__ == '__main__':
    import time
    Xtrain = pd.read_csv('airfoil_self_noise_X_train.csv').values
    ytrain = pd.read_csv('airfoil_self_noise_y_train.csv').values
    Xtest  = pd.read_csv('airfoil_self_noise_X_test.csv').values
    ytest  = pd.read_csv('airfoil_self_noise_y_test.csv').values
    ytrain, ytest = ytrain.flatten(), ytest.flatten()


    start = time.time()
    RF = random_forest_regressor(n_estimators=10)
    RF.fit(Xtrain, ytrain)
    ypred = RF.predict(Xtest)
    rmse = RMSE(ypred, ytest)
    pcc = PCC(ypred, ytest)
    end = time.time()
    print("The RMSE is {:.2f} and the PCC is {:.2f}".format(rmse, pcc)) # 2.75 and 0.92
    print("Takes {:.2f} seconds".format(end-start))
    
    from sklearn.ensemble import RandomForestRegressor
    skRF = RandomForestRegressor()
    skRF.fit(Xtrain, ytrain)

    ypred = skRF.predict(Xtest)
    rmse = RMSE(ypred, ytest)
    pcc = PCC(ypred, ytest)
    print("The RMSE is {:.2f} and the PCC is {:.2f}".format(rmse, pcc)) # 1.98 and 0.96

