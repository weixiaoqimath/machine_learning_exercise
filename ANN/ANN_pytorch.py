'''
Author: Xiaoqi Wei
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time 

def accuracy(ytest, ypred):
    return float(np.sum(ytest == ypred))/ float(len(ytest))

def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    X = pd.read_csv(feature_file).values
    y = pd.read_csv(label_file).values
    return X, y

def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler #import libaray
    scaler = StandardScaler() # call an object function
    scaler.fit(X_train) # calculate mean, std in X_train
    X_train_norm = scaler.transform(X_train) # apply normalization on X_train
    X_test_norm = scaler.transform(X_test) # we use the same normalization on X_test
    return X_train_norm, X_test_norm

def one_hot_encoder(y_train, y_test):
    ''' convert label to a vector under one-hot-code fashion '''
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_test_ohe = lb.transform(y_test)
    return y_train_ohe, y_test_ohe

       

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Initialize weights
        self.linear1 = nn.Linear(784, 300)
        nn.init.xavier_uniform_(self.linear1.weight)

        self.linear2 = nn.Linear(300, 100)
        nn.init.xavier_uniform_(self.linear2.weight)

        self.linear3 = nn.Linear(100, 10)
        nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, X):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        X = self.linear1(X)  
        X = torch.tanh(X)
        X = self.linear2(X)
        X = torch.tanh(X)
        X = self.linear3(X)
        y_hat = F.log_softmax(X, dim=1)
        return y_hat

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): # the length of train_loader is n_samples / batch_size
        data, target = data.to(device), target.to(device)
        output = model(data)
        # loss = nn.MSELoss()(output, target)  # For regression
        loss = F.nll_loss(output, target)      # For classification
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset), loss.item())

    if (epoch+1)%20 == 0:
        print(F.nll_loss(output, target, reduction='sum'))


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.mse_loss(output, target, reduction='sum').item() # sum up batch loss. For regression
            # pcc = PCC(output, target)[0]  # For regression
            # rmse = RMSE(output, target)   # For regression
            test_loss += F.nll_loss(output, target, reduction='sum').item() # reduction = 'sum': sum up batch loss
            pred = output.argmax(dim=1) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if (epoch + 1) % 20 == 0:
        print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        # print("[test_loss: {:.4f}] [PCC: {:.4f}] [RMSE: {:.4f}] [ST] ".format(test_loss, pcc, rmse)) # For regression


# load data
X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)


train_data = torch.from_numpy(X_train_norm).float()
test_data = torch.from_numpy(X_test_norm).float()

trainset = torch.utils.data.TensorDataset(train_data, torch.from_numpy(y_train.ravel()))
testset = torch.utils.data.TensorDataset(test_data, torch.from_numpy(y_test.ravel()))
# Define data loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 1e-3
batch_size = 2000
epochs = 200

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=2000, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=500, shuffle=False)

ANN = NeuralNetwork().to(device)

#optimizer = torch.optim.SGD(ANN.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(ANN.parameters(), lr=learning_rate, eps=1e-08, weight_decay=0, amsgrad=False) # converge much faster than SGD

for epoch in range(epochs):
    #lr_adjust.step()
    train(ANN, device, train_loader, optimizer, epoch)
    test(ANN, device, test_loader, epoch)