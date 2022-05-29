from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def getPartData(data_loader):
    smallData = None
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx > 0:
            return smallData
        else:
            smallData = [(d,t) for d,t in zip(data,target)]

import pickle
results = {'train_loss':[],'train_prec':[],
            'test_loss':[],'test_prec':[]}
import math

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            return loss,output
        loss,output = optimizer.step(closure)
        train_loss += loss*len(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    
    train_loss /= len(train_loader.dataset)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    results['train_loss'].append(float(train_loss))
    results['train_prec'].append(float(correct/len(train_loader.dataset)))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    results['test_loss'].append(float(test_loss))
    results['test_prec'].append(float(correct/len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-part-size', type=int, default=1000, metavar='N',
                        help='part size for training (default: 1000)')
    parser.add_argument('--test-part-size', type=int, default=100, metavar='N',
                        help='part size for testing (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dump-data',default='output.ser',type=str,metavar='PATH',
                        help='path to save loss\correct data (default: output.ser)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random
    import numpy as np
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if use_cuda else "cpu")
 
    train_part_kwargs = {'batch_size': args.train_part_size,'shuffle': True}
    test_part_kwargs = {'batch_size': args.test_part_size,'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_part_kwargs.update(cuda_kwargs)
        test_part_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_part_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_part_kwargs)

    train_kwargs = {'batch_size': args.batch_size,'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size,'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    part_dataset1 = getPartData(train_loader)
    part_dataset2 = getPartData(test_loader)

    train_loader1 = torch.utils.data.DataLoader(part_dataset1, **train_kwargs)
    test_loader1 = torch.utils.data.DataLoader(part_dataset2, **test_kwargs)

    train_loader = train_loader1
    test_loader = test_loader1

    model = Net().to(device)
    sgd = optim.SGD(model.parameters(),lr=args.lr,weight_decay=0,momentum=0)
    #adagrad = optim.Adagrad(model.parameters(),lr=args.lr)      #lr=0.01
    #rmsprop = optim.RMSprop(model.parameters(),lr=args.lr)      #lr=0.001

    adasam = optim.AdaSAM(sgd,period=1,hist_length=20,gamma=0.9,damp=1e-4,precision=1)
    #padasam = optim.pAdaSAM(rmsprop,period=1,hist_length=20,gamma=0,damp=1e-4,precision=1)

#    ram = optim.RAM(sgd,period=1,hist_length=20,damp=1e-6,gamma=0)
    optimizer = adasam

    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        #scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    
    pickle.dump(results,open(args.dump_data,'wb'))

if __name__ == '__main__':
    main()
