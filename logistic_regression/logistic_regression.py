import pandas as pd
import scipy
import numpy as np

def sigmoid(X):
    return 1/(1 + np.exp(-X))

class normalizer():
    def __init__(self):
        self.mean = 0
        self.std = 0

    def fit(self, X):
        self.mean = np.mean(X, axis=0) # mean of each column vector
        self.std = np.std(X, axis=0) # std of each column vector
        self.std[self.std <= 1e-5] = 1 # avoid division by 0

    def transform(self, X):
        """
            feature normalization. Each row of X represents a point in R^d. 
            Substract by the mean of X and then divided by the std of X.
        """
        return (X - self.mean)/self.std

def loss(X, y, theta):
    """
        theta is a 1d vector. X is the extended matrix.
    """
    m = X.shape[0]
    return 1/m*(-np.dot(y, np.log(sigmoid(np.dot(X, theta))))-np.dot(1-y, np.log(1 - sigmoid(np.dot(X, theta)))))

def gradient_descent(X, y, theta, lr, num_iters):
    """
        Perform gradient descent for logistic regression.
    """
    m = X.shape[0] # number of training examples
    loss_history = np.zeros(num_iters)
    
    i = 0
    print("Now we begin training. The learning rate is {} and the number of iterations is {}.".format(lr, num_iters))
    for i in range(num_iters):
        # Perform a single gradient descent on the parameter vector theta. 
        theta = theta - lr/m * np.dot(X.T, sigmoid(np.dot(X, theta))-y)
        # Save the loss of every iteration    
        loss_history[i] = loss(X, y, theta)
        #if (i+1) % 10 == 0:
        #    print("After {} iterations, the loss is {:.4f}.".format(i+1, loss_history[i]))   
        
    return [theta, loss_history]

def predict(X, theta):
    pred = np.zeros(X.shape[0])
    pred[sigmoid(np.dot(X, theta))<0.5] = 0
    pred[sigmoid(np.dot(X, theta))>=0.5] = 1
    return pred

Xtrain = pd.read_csv("Iris_X_train.csv")
ytrain = pd.read_csv("Iris_y_train.csv")
Xtest = pd.read_csv("Iris_X_test.csv")
ytest = pd.read_csv("Iris_y_test.csv")

Xtrain = Xtrain.values
ytrain = ytrain.values
Xtest = Xtest.values
ytest = ytest.values

print(Xtrain.shape)
print(ytrain.shape)
print(Xtest.shape)
print(ytest.shape)
print("We have labels {}".format(np.unique(ytrain)))

ytrain, ytest=ytrain.flatten(), ytest.flatten()

# normalize Xtrain, Xtest
scaler = normalizer()
scaler.fit(Xtrain)
normalized_Xtrain = scaler.transform(Xtrain)
normalized_Xtest = scaler.transform(Xtest)

X = np.concatenate((np.ones((Xtrain.shape[0],1)), normalized_Xtrain), axis=1) 

# Training
lr = 0.3
num_iters = 200
theta = np.zeros(X.shape[1])
theta, loss_history = gradient_descent(X, ytrain, theta, lr, num_iters)
print("The theta is {}".format(theta))

ytrain_pred = predict(X, theta)
score_train = float(sum(ytrain_pred == ytrain))/ float(len(ytrain))
print("The accuracy on training set is {:.2f}%.".format(score_train*100))

extended_normalized_Xtest = np.concatenate((np.ones((Xtest.shape[0],1)), normalized_Xtest), axis=1)
ypred_test = predict(extended_normalized_Xtest, theta)
score = float(sum(ypred_test == ytest))/ float(len(ytest))

print("The accuracy on test set is {:.2f}%, and the loss on testing set is {:.4f}.".format(score*100, loss(extended_normalized_Xtest, ytest, theta)))

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(normalized_Xtrain, ytrain)


print("The accuracy on test set calculated by sklearn model is {:.2f}%".format(clf.score(normalized_Xtest, ytest)*100))

import matplotlib.pyplot as plt
import seaborn as sns

slope = -(theta[1] / theta[2])
intercept = -(theta[0] - (np.dot(np.mean(Xtrain, axis=0), theta[1:])))/ theta[2]

plt.figure()
plt.subplot(121)
sns.set_style('white')
sns.scatterplot(Xtrain[:,0],Xtrain[:,1],hue=ytrain.reshape(-1))
plt.title('Training Set')
plt.axis("square")

ax = plt.gca()
ax.autoscale(False)
x_vals = np.array(ax.get_xlim())
y_vals = intercept + (slope * x_vals)
plt.plot(x_vals, y_vals, c="k");

plt.subplot(122)
sns.set_style('white')
sns.scatterplot(Xtest[:,0],Xtest[:,1],hue=ytest.reshape(-1));
plt.title("Testing Set")
plt.axis("square")

ax = plt.gca()
ax.autoscale(False)
x_vals = np.array(ax.get_xlim())
y_vals = intercept + (slope * x_vals)
plt.plot(x_vals, y_vals, c="k");
plt.show()
