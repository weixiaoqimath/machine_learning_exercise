# https://en.wikipedia.org/wiki/Gradient_boosting




from DT_regressor import *

class GBT_regressor():
    def __init__(self, n_estimators = 10, min_samples_split = 2, lr = 0.01, max_depth = None, max_features = None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.lr = lr # shrinkage coefficient
        self.max_depth = max_depth
        self.max_features = max_features
        self.gradient_coeffs = []
        self.regressors = []
        for i in range(self.n_estimators):
            self.regressors.append(DT_regressor(max_depth = self.max_depth, min_samples_split = self.min_samples_split, max_features = self.max_features))

    def fit(self, X, y):
        y_pred = np.full(y.size, np.mean(y)) # y_pred will be updated
        self.y_pred0 = np.mean(y) # initial prediction is y_mean for any x_i
        residue = y - y_pred
        
        for i in np.arange(self.n_estimators):
            self.regressors[i].fit(X, residue)
            ipred = self.regressors[i].predict(X) 
            # compute gradient coeff alpha, multiplier \gamma in wiki page.
            # \gamma = arg min_{gamma} \sigma_{i=1}^n 1/2 (y_i - F_{m-1}(x_i) - \gamma h_m(x_i))^2 
            #  Taking derivative, we get 
            #  \gamma * \sigma_{i=1}^n h_m^2(x_i) = \sigma_{i=1}^n h_m(x_i) (y_i-F_{m-1})
            # here y_i - F_{m-1}(x_i) is the residue, h_m is the ipred, \gamma is the alpha
            alpha = np.sum(ipred*residue)/np.sum(ipred**2) 
            self.gradient_coeffs.append(alpha) 
            
            y_pred = y_pred + self.lr * alpha * ipred 
            residue = y - y_pred
            
    def predict(self, Xtest):
        """Return array of shape (Xtest.shape[0]).
        """
        y_pred = np.full(Xtest.shape[0], self.y_pred0)
        for i in np.arange(self.n_estimators):
            y_pred += self.lr*self.gradient_coeffs[i]*(self.regressors[i].predict(Xtest))
        return y_pred

if __name__ == '__main__':
    import time
    Xtrain = pd.read_csv('airfoil_self_noise_X_train.csv').values
    ytrain = pd.read_csv('airfoil_self_noise_y_train.csv').values
    Xtest  = pd.read_csv('airfoil_self_noise_X_test.csv').values
    ytest  = pd.read_csv('airfoil_self_noise_y_test.csv').values
    ytrain, ytest = ytrain.flatten(), ytest.flatten()


    start = time.time()

    GBTR = GBT_regressor(n_estimators = 20, max_depth = None, min_samples_split=2, lr =0.1, max_features = None) 
    GBTR.fit(Xtrain, ytrain)
    ypred = GBTR.predict(Xtest)

    rmse = RMSE(ypred, ytest)
    pcc = PCC(ypred, ytest) 
    end = time.time()
    print("The RMSE is {:.2f} and the PCC is {:.2f}".format(rmse, pcc)) 
    # n_estimators = 20, max_depth = None, min_samples_split=2, lr =0.1, max_features = None, The RMSE is 2.36 and the PCC is 0.94
    # n_estimators = 100, max_depth = 3, min_samples_split=2, lr =0.1, max_features = None, rmse = 2.70, pcc=0.92
    print("Takes {:.2f} seconds".format(end-start)) # Takes 10-15s


    
        






