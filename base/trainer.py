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
            optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=0.0001)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.configs.num_epochs)

        return {'criterion': criterion,
                'optimizer': optimizer,
                'scheduler': scheduler}


    def imagenet_train_routine(self, net):
        for epoch in range(4):
            print('Epoch {}/{}'.format(epoch, 4 - 1))
            print('-' * 10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    net.train()  # Set model to training mode
                else:
                    net.eval()  # Set model to evaluate mode

                DataReader = self.datareader()

                dataloaders, class_names, dataset_sizes = DataReader.read_data(self.configs)

                train_args = self.define_train_args(net)
                optimizer = train_args['optimizer']
                criterion = train_args['criterion']
                scheduler = train_args['scheduler']

                save_results(net, self.configs)

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    if 1000 in labels:
                        print('fuck it')
                    # print(idx)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = net(inputs)

                    with torch.set_grad_enabled(phase == 'train'):

                        _, preds = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            # statistics

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                        if phase == 'train':
                            scheduler.step()

                        iter_loss = running_loss / dataset_sizes[phase]
                        iter_acc = running_corrects / dataset_sizes[phase]

                        print(f'{phase} Loss: {iter_loss:.4f} Acc: {iter_acc:.4f}')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    def train_routine(self, net):
        train_args = self.define_train_args(net)

        if self.configs.resume:

            checkpoint = torch.load('/path/to/checkpoint.pth')
            net.load_state_dict(checkpoint['net_state_dict'])
            train_args['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
            train_args['scheduler'].load_state_dict(checkpoint['scheduler_state_dict'])

            pre_epoch = checkpoint['epoch']
            print('loaded checkpoint from epoch ', pre_epoch)
            print('Resume training {}/{}'.format(pre_epoch + 1, self.configs.num_epochs))
            print('Resume training with lr {}'.format(train_args['scheduler'].optimizer.param_groups[0]['lr']))

            loss = checkpoint['loss']
            best_acc = checkpoint['acc']
            print('acc', best_acc)

        else:
            pre_epoch = 0
            save_results(net, self.configs)

        DataReader = self.datareader()

        if self.configs.dataset in ['ImageNet', 'TinyImageNet']:
            print('reading Datalaoders for', self.configs.dataset)
            dataloaders, classes, dataset_sizes = DataReader.read_data(self.configs)
            train_loader = dataloaders['train']
            test_loader = dataloaders['val']
            print('Dataloaders are ready!')

        else:
            train_loader, test_loader, classes = DataReader.read_data(self.configs)

        actor = Actor(configs=self.configs,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      classes=classes,
                      device=self.device,
                      net=net,
                      **train_args)

        if not self.configs.resume:
            print('\nInitial Inference: %d' % 0)
            actor.test_step(0, net, test_loader, self.device)

        print('overall epochs:', self.configs.num_epochs)

        for epoch in range(self.configs.num_epochs):
            if epoch + 1 > pre_epoch:
                start = dt.now()
                actor.train_step(epoch + 1, net, train_loader, self.device)

                train_args['scheduler'].step()
                end = dt.now()
                print('Train step duration: {}'.format(end - start))

                actor.test_step(epoch + 1, net, test_loader, self.device)
