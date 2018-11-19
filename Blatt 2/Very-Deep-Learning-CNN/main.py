from __future__ import print_function
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

from dataset import dataset
from AlexNet import AlexNet
from train_test import start_train_test

trainloader, testloader, outputs, inputs = dataset('dog-breed')
# trainloader, testloader, outputs, inputs = dataset('mnist')

def main():
    print ('Output classes: {}\nInput channels: {}'.format(outputs, inputs))

    net = AlexNet(num_classes = outputs,inputs=inputs)
    file_name = 'alexnet-'

    if use_cuda:
        print("Cuda will be used:")
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    train_loss, test_loss = start_train_test(net, trainloader, testloader, criterion)

    plt.plot(train_loss)
    plt.ylabel('Train Loss')
    plt.show()

    plt.plot(test_loss)
    plt.ylabel('Test Loss')
    plt.show()

if __name__ == '__main__':
    main()