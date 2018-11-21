from __future__ import print_function
import torch
import torch.optim as optim
from torch.autograd import Variable
import config as cf
import time
import numpy as np
import platform


import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

from dataset import dataset
from AlexNet import AlexNet
import torch.utils.model_zoo as model_zoo

torch.manual_seed(42)

use_cuda = torch.cuda.is_available()

best_acc = 0

systemName = platform.system()
if systemName == 'Linux':
    dirSlash = '/'
else:
    dirSlash = '\\'

def train(epoch, net, trainloader, criterion, optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    train_loss_stacked = np.array([0])

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.lr))
    for batch_idx, (inputs_value, targets, paths) in enumerate(trainloader):
        print("Iteration:", batch_idx)
        if use_cuda:
            inputs_value, targets = inputs_value.cuda(), targets.cuda()    # GPU settings

        optimizer.zero_grad()
        inputs_value, targets = Variable(inputs_value), Variable(targets)
        outputs = net(inputs_value)               # Forward Propagation
        # print("outputs.shape =", outputs.shape)
        # print("targets =", targets)
        # print("targets.shape =", targets.shape)
        # print("targets.shape =", targets[:,0].shape)
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        train_loss_stacked = np.append(train_loss_stacked, loss.data[0].cpu().numpy())
    print ('| Epoch [%3d/%3d] \t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, cf.num_epochs, loss.data[0], 100.*correct/total))

    return train_loss_stacked


def test(epoch, net, testloader, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_loss_stacked = np.array([0])
    for batch_idx, (inputs_value, targets, paths) in enumerate(testloader):
        print(batch_idx)
        if use_cuda:
            inputs_value, targets = inputs_value.cuda(), targets.cuda()    # GPU settings

        with torch.no_grad():
            inputs_value, targets = Variable(inputs_value), Variable(targets)
        outputs = net(inputs_value)
        loss = criterion(outputs, targets)
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        test_loss_stacked = np.append(test_loss_stacked, loss.data[0].cpu().numpy())


    # Save checkpoint when best model
    acc = 100. * correct / total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" % (epoch, loss.data[0], acc))



    if acc > best_acc:
        best_acc = acc
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))

    return test_loss_stacked



def start_train_test(net,trainloader, testloader, criterion):
    elapsed_time = 0

    # optimizer = optim.Adam(net.parameters(), lr=cf.lr, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=cf.lr, weight_decay=5e-4)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_loss = []
    test_loss = []

    for epoch in range(cf.start_epoch, cf.start_epoch + cf.num_epochs):
        start_time = time.time()

        scheduler.step()
        train_loss.extend(train(epoch, net, trainloader, criterion, optimizer))
        test_loss.extend(test(epoch, net, testloader, criterion))


        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    return train_loss, test_loss



def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s





def main():
    trainloader, testloader, outputs, inputs = dataset('dog-breed')
    # trainloader, testloader, outputs, inputs = dataset('mnist')

    print ('Output classes: {}\nInput channels: {}'.format(outputs, inputs))
    
    pretrained = True
    iter_per_epoch_train = trainloader.__len__()
    iter_per_epoch_test = testloader.__len__()

    print("iter_per_epoch_train =", iter_per_epoch_train)


    if pretrained:
        net = models.alexnet(pretrained=True)
        #net = models.resnet101(pretrained=True)
    else:
        net = AlexNet(num_classes=outputs, inputs=inputs)

    print("cf.batch_size =", cf.batch_size)
    file_name = 'alexnet-'

    if use_cuda:
        print("Cuda will be used:")
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MultiMarginLoss()

    train_loss, test_loss = start_train_test(net, trainloader, testloader, criterion)

    new_train_loss = []
    new_test_loss = []
    num_epochs_train = []
    num_epochs_test = []

    print("cf.batch_size =", cf.batch_size)
    counter = 0
    for i in enumerate(train_loss):
        # print(i)
        # print("i[0] =", i[0])
        # print("i[1] =", i[1])
        if (i[0] % iter_per_epoch_train == 0 and not(i[0] == 0)) or i[0] == 1:
            # print("i[1] =", i[1], "wird angehangen")
            new_train_loss.append(i[1])
            counter += 1
            num_epochs_train.append(counter)


    counter = 0
    for i in enumerate(test_loss):
        # print(i)
        # print("i[0] =", i[0])
        # print("i[1] =", i[1])
        if (i[0] % iter_per_epoch_test == 0 and not (i[0] == 0)) or i[0] == 1:
            # print("i[1] =", i[1], "wird angehangen")
            new_test_loss.append(i[1])
            counter += 1
            num_epochs_test.append(counter)

    print("new_train_loss =", new_train_loss)
    print("new_test_loss =", new_test_loss)
    torch.save(net, '.' + dirSlash + 'test.pt')

    """
    num_epochs_train = [1,2,3]
    num_epochs_test = [1,2,3]
    new_train_loss = [6.4614146406413, 3.3310906887054443, 2.825629711151123]
    new_test_loss = [6.105416494, 2.7234768390655518, 1.5601584911346436]
    """

    # Beginn des plottens:
    plt.style.use('seaborn-whitegrid')
    plt.subplot(121)


    plt.subplot(121)

    # Plot Axes-Labeling
    plt.xlabel('Epochs')

    plt.plot(num_epochs_train, new_train_loss)
    plt.ylabel('Train Loss')



    plt.subplot(122)
    # Plot Axes-Labeling
    plt.xlabel('Epochs')

    plt.plot(num_epochs_test, new_test_loss)
    plt.ylabel('Test Loss')
    plt.show()

    """
    plt.plot(train_loss)
    plt.ylabel('Train Loss')
    plt.show()

    plt.plot(test_loss)
    plt.ylabel('Test Loss')
    plt.show()
    """

if __name__ == '__main__':
    main()

