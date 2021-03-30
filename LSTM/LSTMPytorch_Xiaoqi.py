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

# epoch = 40 

tic = time.perf_counter()

#======================================Classes==================================
class LSTMNet(nn.Module):
    def __init__(self, D_in, T, H, D_out):
        """
        D_in  : input size, 28
        T     : time step, 28
        H     : The number of features in the hidden state h
        D_out : #of classes 10
        """
        super(LSTMNet, self).__init__()
        # DRNN
        self.lstm = nn.LSTM(D_in, H, num_layers=1, batch_first=True) # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        self.linear = nn.Linear(H, D_out)

    def forward(self, X):
        """
        X: (N, T, D_in)
        r_out: (N, T, D_out)
        h_n: (n_layers, N, H)
        """
        r_out, h_n = self.lstm(X, None)   # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.linear(r_out[:, -1, :])     # [:,:,-1]
        print(out.shape)
        y_hat = F.log_softmax(out, dim=1)
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
        # if epoch % args.log_interval == 0:
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
    X_train_norm = np.reshape(X_train_norm1,(-1,28,28)) # reshape X to be a 3-D array
    X_test_norm = np.reshape(X_test_norm1,(-1,28,28))
    return X_train_norm, X_test_norm


def one_hot_encoder(y_train, y_test):
    ''' convert label to a vector under one-hot-code fashion '''
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_test_ohe = lb.transform(y_test)
    return y_train_ohe, y_test_ohe


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100000, metavar='N',
                        help='input batch size for testing (default: 50)')   # train itself 9221, test 3767
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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
    train_data = torch.from_numpy(X_train).float()  # numpy to tensor 
    test_data = torch.from_numpy(X_test).float()

    trainset = torch.utils.data.TensorDataset(train_data, torch.from_numpy(y_train.ravel()))
    testset = torch.utils.data.TensorDataset(test_data, torch.from_numpy(y_test.ravel()))
    # Define data loader
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    #=================================Design Net================================
    D_in = 28
    T = 28
    H = 64
    D_out = 10
    # 
    model = LSTMNet(D_in, T, H, D_out).to(device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    lr_adjust = optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma = 0.5, last_epoch = -1)

    for epoch in range(1, args.epochs + 1):
        lr_adjust.step()
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, epoch, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_rnn.pt")

if __name__ == '__main__':
    main()

toc = time.perf_counter()
print(("Elapsed time: %.1f [min]" % ((toc-tic)/60)))
print("==============================Finish=====================================")
print('The epoch number is set to be {}'.format(40))