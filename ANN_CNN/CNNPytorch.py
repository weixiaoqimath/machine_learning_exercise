from __future__ import print_function
import argparse
import sys
import numpy as np
import pandas as pd
import time
import random
from sklearn import preprocessing
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

tic = time.perf_counter()

#======================================Classes==================================
class CNNLayerNet(nn.ModuleList):
    def __init__(self, C_in, H0, H1, H2, K1, P1, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        H0: shape[1] of Matrix after flatten layer
        """
        super(CNNLayerNet, self).__init__()
        self.conv1 = nn.Conv2d(C_in, H1, K1, bias=True)
        nn.init.xavier_uniform_(self.conv1.weight)

        self.pool1 = nn.MaxPool2d(P1)

        self.fc = nn.Linear(H0, H2, bias = True)
        nn.init.xavier_uniform_(self.fc.weight)

        self.linear1 = nn.Linear(H2, D_out, bias = True)
        nn.init.xavier_uniform_(self.linear1.weight)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.lrelu = nn.LeakyReLU()
        self.prelu = nn.PReLU()

    def forward(self, X):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        X = self.conv1(X)
        X = self.relu(X)
        X = self.pool1(X)
        # X = (N,C, H, W)
        X = X.view(X.size(0),-1)
        # X = (N, C*H*W)
        X = self.fc(X)
        X = self.relu(X)
        X = self.linear1(X)
        y_hat = F.log_softmax(X, dim=1)
        return y_hat

#=================================Training & Testing============================
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))

def test(args, model, device, epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # pcc = PCC(output, target)[0]
            # rmse = RMSE(output, target)
    test_loss /= len(test_loader.dataset)
    if epoch % args.log_interval == 0:
        print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

        # print("[test_loss: {:.4f}] [PCC: {:.4f}] [RMSE: {:.4f}] [Epoch: {:d}] [2DCNN] ".format(test_loss, pcc, rmse, epoch))



def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values # convert values in dataframe to numpy array (features)
    y = df_y.values # convert values in dataframe to numpy array (label)
    return X, y


def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler #import libaray
    scaler = StandardScaler() # call an object function
    scaler.fit(X_train) # calculate mean, std in X_train
    X_train_norm1 = scaler.transform(X_train) # apply normalization on X_train
    X_test_norm1 = scaler.transform(X_test) # we use the same normalization on X_test
    X_train_norm = np.reshape(X_train_norm1,(-1,1,28,28)) # reshape X to be a 4-D array
    X_test_norm = np.reshape(X_test_norm1,(-1,1,28,28))
    return X_train_norm, X_test_norm

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 50)')   # train itself 9221, test 3767
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--momentum', type=float, default=0.3, metavar='M',
                        help='SGD momentum (default: 0.005)')
    parser.add_argument('--weight_decay', type=float, default=0, metavar='M',
                        help='SGD momentum (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}


    #=================================Load Data=================================
    X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
    X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
    X_train, X_test = normalize_features(X_train, X_test)

    print('Trian:', X_train.shape)
    print('Test:', X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    #==================================Pack Data================================
    train_data = torch.from_numpy(X_train).float()
    test_data = torch.from_numpy(X_test).float()

    trainset = torch.utils.data.TensorDataset(train_data, torch.from_numpy(y_train.ravel()))
    testset = torch.utils.data.TensorDataset(test_data, torch.from_numpy(y_test.ravel()))

    # Define data loader
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    #=================================Design Net================================
    C_in = 1
    H1 = 4
    H2 = 256
    K1 = (3,3)
    P1 = 2
    H0 = H1*13*13
    D_out = 10
    model = CNNLayerNet(C_in, H0, H1, H2, K1, P1, D_out).to(device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    lr_adjust = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5, last_epoch = -1)

    for epoch in range(1, args.epochs + 1):
        lr_adjust.step()
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, epoch, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

if __name__ == '__main__':
    main()

toc = time.perf_counter()
print(("Elapsed time: %.1f [min]" % ((toc-tic)/60)))
print("==============================Finish=====================================")