import pandas as pd
import scipy
import numpy as np

df_trainX = pd.read_csv("Iris_X_train.csv")
df_trainY = pd.read_csv("Iris_y_train.csv")
df_testX = pd.read_csv("Iris_X_test.csv")
df_testY = pd.read_csv("Iris_y_test.csv")

trainX = df_trainX.to_numpy()
trainY = df_trainY.to_numpy()
testX = df_testX.to_numpy()
testY = df_testY.to_numpy()


# In[7]:
from featureNormalization import featureNormalize

mean = np.mean(trainX, axis = 0).reshape((1, trainX.shape[1]))
normalized_trainX = featureNormalize(trainX, mean)

X = np.hstack((np.ones((trainX.shape[0],1)), normalized_trainX)) 
import gradientDescent as gD
alpha = 0.3;
num_iters = 500;
theta = np.zeros((X.shape[1],1))
[theta, loss_history] = gD.gradientDescent(X, trainY, theta, alpha, num_iters)

from predict import predict
from costFunction import costFunction

predY_train = predict(X, theta)
score_train = float(sum(predY_train == trainY))/ float(len(trainY))
print("The accuracy on training set is {:.2f}%.".format(score_train*100))

normalized_testX = featureNormalize(testX, mean)
newtestX = np.hstack((np.ones((testX.shape[0],1)), normalized_testX))
predY = predict(newtestX, theta)
score = float(sum(predY == testY))/ float(len(testY))

print("The accuracy on testing set is {:.2f}%, and the loss on testing set is {:.4f}.".format(score*100, costFunction(newtestX, testY, theta)[0][0]))

# In[ ]:
import matplotlib.pyplot as plt
import seaborn as sns

slope = -(theta[1] / theta[2])
intercept = -(theta[0] - (mean@theta[1:])[0])/ theta[2]

plt.figure()
plt.subplot(121)
sns.set_style('white')
sns.scatterplot(trainX[:,0],trainX[:,1],hue=trainY.reshape(-1));
plt.title('Training Set')
plt.axis("square")

ax = plt.gca()
ax.autoscale(False)
x_vals = np.array(ax.get_xlim())
y_vals = intercept + (slope * x_vals)
plt.plot(x_vals, y_vals, c="k");

plt.subplot(122)
sns.set_style('white')
sns.scatterplot(testX[:,0],testX[:,1],hue=testY.reshape(-1));
plt.title("Testing Set")
plt.axis("square")

ax = plt.gca()
ax.autoscale(False)
x_vals = np.array(ax.get_xlim())
y_vals = intercept + (slope * x_vals)
plt.plot(x_vals, y_vals, c="k");
plt.show()