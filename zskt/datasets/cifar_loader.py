
import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset

from utils.config import DATA_PATHS

from .gtsrb import GTSRB


def fetch_dataloader(train, batch_size, subset_percent=1., do_aug=True, data_name='CIFAR10'):
    if data_name == 'CIFAR10':
        DATA_CLASS = torchvision.datasets.CIFAR10
        IMG_MEAN, IMG_STD = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    elif data_name == 'GTSRB':
        DATA_CLASS = GTSRB
        IMG_MEAN, IMG_STD = (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)

    # using random crops and horizontal flip for train set
    if do_aug:
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize()])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD)])

    if train:
        trainset = DATA_CLASS(DATA_PATHS[data_name], train=True,
            download=True, transform=train_transformer)

        train_len = len(trainset)
        indices = list(range(train_len))
        train_len = int(np.floor(subset_percent * train_len))
        np.random.seed(230)
        np.random.shuffle(indices)

        dl = torch.utils.data.DataLoader(Subset(trainset, indices[:train_len]), batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True)
    else:
        devset = DATA_CLASS(DATA_PATHS[data_name], train=False,
            download=True, transform=dev_transformer)
            
        dl = torch.utils.data.DataLoader(devset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True)

    return dl
