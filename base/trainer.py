#!/usr/bin/env python3

import torch
import torch.optim as optim
import torch.nn as nn


import math
from datetime import datetime as dt
# from utils.localFourier import LFA, make_coeffs
from actor import Actor
# from data_loader.data_reader import
from data_loader.data_reader import *
from utils.settings import save_results, learning_rate

class Trainer(object):

    def __init__(self, args, net_args):
        self.configs = args
        self.net_args = net_args
        self.device = net_args['device']

    def datareader(self):
        if self.configs.dataset == 'Cifar100':
            return Cifar100DataReader
        elif self.configs.dataset == 'Cifar10':
            return Cifar10DataReader
        elif self.configs.dataset == 'FashionMNIST':
            return FashionMNISTDataReader
        elif self.configs.dataset == 'MNIST':
            return MNISTDataReader
        elif self.configs.dataset == 'ImageNet' or self.configs.dataset == 'TinyImageNet':
            return ImageNetDataReader


    def define_train_args(self, net):
        """
        Specifies arguments required for training and testing-
           - criterion
           - optimizer # todo: momentum and weightdecay in net_args
           - scheduler
        """

        if self.configs.dataset == 'FashionMNIST':
            print('data set is fashion mnist')
            # Define the optimizer
            # optimizer = optim.Adam(net.parameters(), lr=self.configs.lr)
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.configs.lr, momentum=0.9,
                                  weight_decay=0.0001)
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.configs.num_epochs)
            # Define the loss
            criterion = nn.CrossEntropyLoss()

        else:
            print('Define optimizer and scheduler')
            criterion = nn.CrossEntropyLoss()
            # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.configs.lr, momentum=0.9,
            optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=0.0001)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.configs.num_epochs)

        return {'criterion': criterion,
                'optimizer': optimizer,
                'scheduler': scheduler}