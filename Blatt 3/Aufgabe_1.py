#import all required modules
from download import download
import zipfile
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
from torch.utils.data import Dataset
import collections
import os
import torch
import torch.nn as nn
import argparse
import datetime
import pytz
import math
import tqdm
import torch.nn.functional as F
import shutil
import scipy
from google_drive_downloader import GoogleDriveDownloader as gdd
from torchvision import transforms

from fcn_utils import utils

# TODO after downloading delete the files 1.png and 2.png from the mask folder and name 0.png to 2.png and 0.jpg in JPEG images to 2.jpg

# extract the dataset
import os

def download_dataset():
    print("Hallo")
    # path_to_zip_file = './data/bags_data.zip'
    pycharm_mag_mich_nicht = os.getcwd() + '\\data\\'
    path_to_zip_file = pycharm_mag_mich_nicht + 'bags_data.zip'
    print(path_to_zip_file)
    if not os.path.isdir(pycharm_mag_mich_nicht):
        os.makedirs(pycharm_mag_mich_nicht)

    # download the bag dataset
    url = "https://drive.google.com/file/d/1P4bdP6nSUOqQLhGXvZ1z4z3hNncJFrCs/view?usp=sharing"
    gdd.download_file_from_google_drive(file_id='1P4bdP6nSUOqQLhGXvZ1z4z3hNncJFrCs',
                                        dest_path=path_to_zip_file,
                                        unzip=True)

    directory_to_extract_to = pycharm_mag_mich_nicht
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    print(directory_to_extract_to)
    zip_ref.close()

def viszulize_dataset():
    # visualizing an image in the dataset
    image = PIL.Image.open('data/bags_data/JPEGImages/15.jpg')
    label = PIL.Image.open('data/bags_data/segmentation_mask/15.png')

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image)
    axarr[1].imshow(label)
    # plt.show()


### TO DO: Implement your own custom dataset class, which can be used to load train and validation images in batches.
### Hint: The given dataset is in standard VOC datset format, for which implemented Dataset class can be easily found on Internet
### You max refer to the original repository to get further help.
class BagDataset(Dataset):
    mean_bgr = np.array([0.485, 0.456, 0.406])
    std_bgr = np.array([0.229, 0.224, 0.225])
    class_names = np.array([
        'background',
        'bag',
    ])

    # overridden function
    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform
        dataset_dir = os.path.join(self.root, 'bags_data')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = os.path.join(dataset_dir, 'imagesets/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = os.path.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                label_file = os.path.join(dataset_dir, 'segmentation_mask/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': label_file
                })

    # overidden function
    def __len__(self):
        ###TO DO###
        '''
        Implement code to return the size of the dataset
        '''
        return len(self.files["train"]) + len(self.files["val"])

    # overridden function
    def __getitem__(self, index):
        index = index + 2  # Because of faulty dataset
        ###TO DO###
        '''
        Write code to load RGB image and label annotation image
        Resize both Images and labels to size 200x200 pixels for faster computation on Bag Dataset 
        '''
        toTensor = transforms.ToTensor()
        # TODO Images like 1.jpg do not exist!
        img = PIL.Image.open(os.path.join(self.root, 'bags_data', 'JPEGImages', str(index) + '.jpg'))
        img = np.array(img.resize((200, 200)))
        lbl = PIL.Image.open(os.path.join(self.root, 'bags_data', 'segmentation_mask', str(index) + '.png'))
        lbl = np.array(lbl.resize((200, 200)))

        if self._transform:
            img, lbl = self.transform(img, lbl)
            return img, lbl  # toTensor(img), toTensor(lbl)
        else:
            return img, lbl  # toTensor(img), toTensor(lbl)

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img /= self.std_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    @staticmethod
    def untransform(img, lbl):
        toPIL = transforms.ToPILImage()
        img, lbl = img, lbl  # toPIL(img), toPIL(lbl)
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img *= BagDataset.std_bgr
        img += BagDataset.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl


def visualize_array(image, label):
    # visualizing numpy arrays of images and labels
    image = PIL.Image.fromarray(image)
    label = label.astype(dtype=np.uint8)
    label = PIL.Image.fromarray(label)
    label_color_indexed = label.convert('RGB').convert('L', palette=PIL.Image.ADAPTIVE, colors=2)
    label_color_indexed.putpalette([
    0, 0, 0, # index 0 is black backgroun
    0, 0, 255, # index 1 is blue
    ])
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image)
    axarr[1].imshow(label_color_indexed)
    # plt.show()


import torchvision

def load_something():
    # root = os.path.expanduser('data') # use the path to parent dir of dataset
    root = os.getcwd() + '\\data'
    print("root =", root)
    train_loader = torch.utils.data.DataLoader(
        BagDataset(root, split='train', transform=True),
        batch_size=1, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        BagDataset(root, split='val', transform=True),
        batch_size=1, shuffle=True)

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    image = torch.squeeze(images)
    label = torch.squeeze(labels)
    image, label = BagDataset.untransform(image, label)
    visualize_array(image, label)




from fcn_models.fcn32s import FCN32s
from fcn_models.fcn8 import FCN8s, FCN8sAtOnce

from trainer import Trainer






def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        #FCN32s,
        FCN8s
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))







def nerv():
    here = os.getcwd()
    model = 'FCN32s'
    #model = 'FCN8s'
    #git_hash = git_hash()
    gpu = 0
    resume = None
    max_iteration = 8000
    lr = 1.0e-10
    weight_decay = 0.0005
    momentum = 0.99

    now = datetime.datetime.now()
    out = os.path.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))
    print(out)
    os.makedirs(out)
    # with open(os.path.join(out, 'config.yaml'), 'w') as f:
    #    yaml.safe_dump(__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    root = os.path.expanduser('data')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        BagDataset(root, split='train', transform=True),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        BagDataset(root, split='val', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    # model = FCN32s(n_class=2)
    model = FCN8sAtOnce(n_class=2)
    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = torchvision.models.vgg16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
        print(model)
    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': lr * 2, 'weight_decay': 0},
        ],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay)
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        max_iter=max_iteration,
        interval_validate=4000,
    )


    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


def do_something():

    ###To DO###
    '''
    Write the code to load the trained model in variable "net", and set it to evaluation mode so predictions can be obtained.
    '''
    lr = 1.0e-10
    weight_decay = 0.0005
    momentum = 0.99
    # net = FCN32s(n_class=2)
    net = FCN8s(n_class=2)
    optimizerEval = torch.optim.SGD(
        [
            {'params': get_parameters(net, bias=False)},
            {'params': get_parameters(net, bias=True),
             'lr': lr * 2, 'weight_decay': 0},
        ],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay)
    # TODO specify the path accordingly to your own system
    checkpoint = torch.load("./logs/20181201_143909.078664/model_best.pth.tar")
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizerEval.load_state_dict(checkpoint['optim_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    net.eval()


if __name__ == '__main__':
    # download_dataset()
    load_something()
    nerv()