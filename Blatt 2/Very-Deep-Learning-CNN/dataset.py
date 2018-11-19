import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import random_split
import torchvision
from PIL import Image
import os
import platform
import sys
import zipfile

import config as cf

from transform import transform_training, transform_testing
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import transforms


cur_dir = os.getcwd()
systemName = platform.system()
if systemName == 'Linux':
    dirSlash = '/'
else:
    dirSlash = '\\'
IMG_PATH = cur_dir + dirSlash + 'dog-breed-identification-datase' + dirSlash + 'train' + dirSlash
IMG_EXT = '.jpg'
TRAIN_DATA = cur_dir + dirSlash + 'dog-breed-identification-dataset' + dirSlash + 'labels.csv'




class ConvertDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['id'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
            "Some images referenced in the CSV file were not found"

        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.num_classes = len(list(set(tmp_df['breed'])))

        self.X_train = tmp_df['id']

        dic = dict()
        seen = []
        max_class = 0
        for elem in tmp_df['breed']:
            if not elem in dic.keys():
                dic[elem] = max_class
                seen.append(max_class)
                max_class += 1
            else:
                seen.append(dic[elem])


        self.y_train = np.array(seen, dtype=np.int64)



    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)




        # label = torch.from_numpy(self.y_train[index])
        label = self.y_train[index]

        return img, label

    def __len__(self):
        return len(self.X_train.index)





def dataset(dataset_name):

    if (dataset_name == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 10
        inputs = 3
    
    elif (dataset_name == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 100
        inputs = 3
    
    elif (dataset_name == 'mnist'):
        print("| Preparing MNIST dataset...")
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_testing())
        print(type(trainset))
        outputs = 10
        inputs = 1
    
    elif (dataset_name == 'fashionmnist'):
        print("| Preparing FASHIONMNIST dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 10
        inputs = 1
    elif (dataset_name == 'stl10'):
        print("| Preparing STL10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.STL10(root='./data',  split='train', download=True, transform=transform_training())
        testset = torchvision.datasets.STL10(root='./data',  split='test', download=False, transform=transform_testing())
        outputs = 10
        inputs = 3
    elif (dataset_name == 'dog-breed'):
        print("| Preparing dog-breed dataset...")
        sys.stdout.write("| ")

        if os.path.isfile("." + dirSlash + "dog-breed-identification-dataset" + dirSlash + "labels.csv.zip"):
            print("labels.csv has already been downloaded")
        else:
            print("Download labels.csv")
            os.system("kaggle competitions download dog-breed-identification -f labels.csv.zip -p ." + dirSlash + "dog-breed-identification-dataset" + dirSlash)
            zip_ref = zipfile.ZipFile("." + dirSlash + "dog-breed-identification-dataset" + dirSlash + "labels.csv.zip", 'r')
            zip_ref.extractall("." + dirSlash + "dog-breed-identification-dataset" + dirSlash)
            zip_ref.close()
            print("labels.csv download finished")
        if os.path.isfile("." + dirSlash + "dog-breed-identification-dataset" + dirSlash + "test.zip"):
            print("Test folder has already been downloaded")
        else:
            print("Download test folder")
            os.system("kaggle competitions download dog-breed-identification -f test.zip -p ." + dirSlash + "dog-breed-identification-dataset" + dirSlash)
            zip_ref = zipfile.ZipFile("." + dirSlash + "dog-breed-identification-dataset" + dirSlash + "test.zip", 'r')
            zip_ref.extractall("." + dirSlash + "dog-breed-identification-dataset" + dirSlash)
            zip_ref.close()
            print("Test folder download finished")
        
        if os.path.isfile("." + dirSlash + "dog-breed-identification-dataset" + dirSlash + "train.zip"):
            print("Train folder has already been downloaded")
        else:
            print("Download train folder")
            os.system("kaggle competitions download dog-breed-identification -f train.zip -p ." + dirSlash + "dog-breed-identification-dataset" + dirSlash)
            zip_ref = zipfile.ZipFile("." + dirSlash + "dog-breed-identification-dataset" + dirSlash + "train.zip", 'r')
            zip_ref.extractall("." + dirSlash + "dog-breed-identification-dataset" + dirSlash)
            zip_ref.close()
            print("Train folder download finished")
        outputs = 10
        inputs = 3

        transformations = transforms.Compose([transforms.Resize(size=(224,224)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])

        complete_set = ConvertDataset(TRAIN_DATA, IMG_PATH, IMG_EXT, transformations)

        train_length = int(len(complete_set)/3) * 2
        test_length = len(complete_set) - train_length
        trainset, testset = random_split(complete_set, (train_length,test_length))



        # trainset = torch.utils.data.DataLoader(trainset,batch_size=256, shuffle=True,num_workers=4)
        # testset = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=4)
        # trainset = trainset.transform_training()
        # testset = testset.transform_training()


        # Number of possible classes
        outputs = complete_set.num_classes

        # Input Channel: Since RGB Images => 3 Channels
        inputs = 3
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cf.batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cf.batch_size, shuffle=False, num_workers=1)


    return trainloader, testloader, outputs, inputs

def main():
    dataset('dog-breed')

if __name__ == '__main__':
    main()


    """
        if dataset_name == 'dog-breed':
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=cf.batch_size, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cf.batch_size, shuffle=False, num_workers=1)

        dset = torch.utils.data.Dataset()

        for elem in trainloader:
            print("elem[0].shape =", elem[0].shape)
            print("elem[1].shape =", elem[1].shape)
            # dset.__add__(elem[:, 0])

        trainloader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=True, num_workers=1)

        # trainloader = [(batch_idx, (inputs_value, targets[:,0])) for batch_idx, (inputs_value, targets) in enumerate(trainloader)]
        # testloader = [(batch_idx, (inputs_value, targets[:,0])) for batch_idx, (inputs_value, targets) in enumerate(testloader)]
        
        for elem in trainloader:
            print(elem[0].shape)
            print(elem[1].shape)
            elem = elem[:,0]
        

    """
