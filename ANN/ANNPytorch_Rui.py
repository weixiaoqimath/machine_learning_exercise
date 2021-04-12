# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2021-02-10 22:28:11
LastModifiedBy: Rui Wang
LastEditTime: 2021-02-11 19:09:58
Email: wangru25@msu.edu
FilePath: /Pytorch/ANNPytorch.py
Description: 
'''
from __future__ import print_function
import argparse
import sys
import numpy as np
import pandas as pd
import time
import random
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

tic = time.perf_counter()

#===============================Parameters=====================================
cut = 12.0
radius_lower = 2.0
radius_upper = 6.25
radius_step = 0.25
#====================================Functions==================================
def RMSE(ypred, yexact):
    return torch.sqrt(torch.sum((ypred-yexact)**2)/ypred.shape[0])

def PCC(ypred, yexact):
    from scipy import stats
    a = (Variable(yexact).data).cpu().numpy().ravel()
    b = (Variable(ypred).data).cpu().numpy().ravel()
    pcc = stats.pearsonr(a,b)
    return pcc
#===============================Data Preprocessing==============================
def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values
    y = df_y.values
    return X, y

def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm

#======================================Classes==================================
class NNLayerNet(nn.ModuleList):
    def __init__(self, D_in, H1, H2, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(NNLayerNet, self).__init__()
        # Initialize weights
        self.linear1 = nn.Linear(D_in, H1, bias = True)
        nn.init.xavier_uniform_(self.linear1.weight)

        self.linear2 = nn.Linear(H1, H2, bias = True)
        nn.init.xavier_uniform_(self.linear2.weight)

        self.linear3 = nn.Linear(H2, D_out, bias = True)
        nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, X):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        X = self.linear1(X).clamp(min=0)  #.clamp is the relu function
        X = self.linear2(X).clamp(min=0)
        X = self.linear3(X)
        y_hat = F.log_softmax(X, dim=1)
        # print(y_hat.shape)
        return y_hat

#=================================Training & Testing============================
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        # loss = nn.MSELoss()(output, target)  # For regression
        loss = F.nll_loss(output, target)      # For classification
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))


def test(args, model, device, epoch, test_loader):
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
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if epoch % args.log_interval == 0:
        print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        # print("[test_loss: {:.4f}] [PCC: {:.4f}] [RMSE: {:.4f}] [ST] ".format(test_loss, pcc, rmse)) # For regression


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test MNIST')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 50)')  
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--momentum', type=float, default=0.3, metavar='M',
                        help='SGD momentum (default: 0.005)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='M',
                        help='SGD momentum (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cuda")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    #=================================Load Data=================================
    X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
    X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    print(X_train_norm.shape)
    print(y_train.shape)

    #==================================Pack Data================================
    train_data = torch.from_numpy(X_train_norm).float()
    test_data = torch.from_numpy(X_test_norm).float()

    trainset = torch.utils.data.TensorDataset(train_data, torch.from_numpy(y_train.ravel()))
    testset = torch.utils.data.TensorDataset(test_data, torch.from_numpy(y_test.ravel()))
    # Define data loader
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    #=================================Design Net================================
    layers = [300, 100]
    D_in = 784
    H1 = layers[0]
    H2 = layers[1]
    D_out = 10
    model = NNLayerNet(D_in, H1, H2, D_out).to(device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    lr_adjust = optim.lr_scheduler.StepLR(optimizer, step_size = 500, gamma = 0.1, last_epoch = -1)

    for epoch in range(1, args.epochs + 1):
        lr_adjust.step()
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, epoch, test_loader)
    # print("[lr: {:.4f}] [momentum: {:.4f}] [weight_decay: {:.4f}] [epochs: {:d}] [batch_size: {:d}] [seed: {:d}] [Hidden: {:d}] [node: {:d}] ".format(args.lr, args.momentum, args.weight_decay, args.epochs, args.batch_size, args.seed, len(layers), H1))

    if (args.save_model):
        torch.save(model.state_dict(),"ANN_MNIST.pt")

    # params = model.state_dict()
    # np.save("linear1_weight.npy",(Variable(params['linear1.weight']).data).cpu().numpy())
    # np.save("linear1_bias.npy",(Variable(params['linear1.bias']).data).cpu().numpy())

if __name__ == '__main__':
    main()

toc = time.perf_counter()
print(("Elapsed time: %.1f [min]" % ((toc-tic)/60)))
print("==============================Finish=====================================")