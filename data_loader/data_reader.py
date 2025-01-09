import torch
import argparse
import numpy as np
import os

from torch.utils.data import DataLoader, RandomSampler, Subset
import torchvision
from torchvision import datasets, transforms
from utils.autoaugment import CIFAR10Policy
import matplotlib.pyplot as plt

import scipy.io

args = None

def main(_):
    datareader_dict = {
        'CIFAR10': Cifar10DataReader,
        'CIFAR100': Cifar100DataReader,
        'MNIST': MNISTDataReader,
        'FashionMNIST': FashionMNISTDataReader,
        'ImageNet': ImageNetDataReader,
        'TinyImageNet': ImageNetDataReader,
    }

    DataReader = datareader_dict[args.dataset]

    if args.dataset == 'ImageNet' or args.dataset =='TinyImagNet':

        dataloaders, class_names, dataset_sizes = DataReader.read_data(args)

        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        DataReader.imshow(out, title=[class_names[x] for x in classes])

    else:
     train_loader, val_loader, classes = DataReader.read_data(args)


     for batch_idx, (data, labels) in enumerate(train_loader):
            DataReader.imshow(img=data[0], classes=classes, labels=labels, batch_idx=batch_idx)

    def imshow(img, classes, labels, batch_idx):
                img = img / 2 + 0.5  # unnormalize
                npimg = img.numpy()
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
                plt.title(classes[labels[0]])
                plt.show()


class ImageNetDataReader(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    @staticmethod
    def read_data(configs):
        if configs.dataset == 'TinyImageNet':
            data_dir = './../../datasets/tiny-imagenet-200'
            test = 'testing'
            if not os.path.exists(data_dir + '/' + test):
                print('Validation set is not formatted correctly, please run \' val_format.py \' first.')

            data_transforms = {'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=np.array([0.4914, 0.4822, 0.4465]),
                                     std=np.array([0.2023, 0.1994, 0.2010])),
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),

                                  ]),
                            test: transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.RandomCrop(64),
                transforms.Normalize(mean=np.array([0.4914, 0.4822, 0.4465]),
                                     std=np.array([0.2023, 0.1994, 0.2010]))
                                  ])
            }
            tiny = True

        else:
            test = 'val'
            data_dir = './../../datasets/ImageNet_download_2207'
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                      data_transforms[x])
                              for x in ['train', test]}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=configs.batch_size,
                                                          shuffle=True, num_workers=8, pin_memory=True)
                           for x in ['train', test]}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', test]}
        class_names = image_datasets['train'].classes

        # if tiny:
        if configs.dataset == 'TinyImageNet':
            dataloaders['val'] = dataloaders['testing']
            del dataloaders['testing']

        return dataloaders, class_names, dataset_sizes


    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


class MNISTDataReader(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    @staticmethod
    def read_data(configs: object) -> object:
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        kwargs = {
            'batch_size': configs.batch_size,
            'shuffle': True,
        }

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        train_set = datasets.MNIST('../data', train=True, download=True,
                                   transform=transform)
        test_set = datasets.MNIST('../data', train=False,
                                  transform=transform)

        train_loader = DataLoader(train_set, **kwargs)
        test_loader = DataLoader(test_set, **kwargs)

        return train_loader, test_loader, classes

    @staticmethod
    def imshow(img, classes, labels, batch_idx):
        npimg = img.numpy()
        plt.imshow(npimg[0])
        plt.suptitle('batch:_'.join('%5s' % batch_idx))
        plt.show()


class FashionMNISTDataReader(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    @staticmethod
    def read_data(configs: object) -> object:
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        labels_map = {
            0: "T-Shirt",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot",
        }
        kwargs = {
            'batch_size': configs.batch_size,
            'shuffle': True,
        }

        # transforms
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])

        # datasets
        trainset = torchvision.datasets.FashionMNIST('data',
                                                     download=True,
                                                     train=True,
                                                     transform=transform)
        testset = torchvision.datasets.FashionMNIST('data',
                                                    download=True,
                                                    train=False,
                                                    transform=transform)

        # dataloaders
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                        shuffle=True, num_workers=4)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                       shuffle=False, num_workers=4)
        classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                   'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


        return train_loader, test_loader, classes

    @staticmethod
    def imshow(img, classes, labels, batch_idx):
        npimg = img.numpy()
        plt.imshow(npimg[0])
        plt.suptitle('batch:_'.join('%5s' % batch_idx))
        plt.show()


class Cifar100DataReader(object):
    """Loads and reads CIFAR10 dataset """

    def __init__(self, batch_size):
        """
        Args:
            batch_size (int): number of samples in train batch
        """
        self.batch_size = batch_size

    @staticmethod
    def read_data(configs):
        """ read and preprocess CIFAR100 dataset  """
        classes = []

        stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # todo:  transforms.RandomRotation(15),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])

        train_dataset = datasets.CIFAR100(root=configs.data_path,
                                          train=True,
                                          transform=train_transform,
                                          download=True)

        test_dataset = datasets.CIFAR100(root=configs.data_path,
                                         train=False,
                                         transform=test_transform,
                                         download=True)
        train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=8)
        test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=2)

        return train_loader, test_loader, classes


class Cifar10DataReader(object):
    """Loads and reads CIFAR10 dataset """

    def __init__(self, batch_size):
        """
        Args:
            batch_size (int): number of samples in train batch
        """
        self.batch_size = batch_size

    @staticmethod
    def read_data(configs):
        """ download and read train und torchmodels_resnet18_parameters_1000 data from torchvision datasets """

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
             transforms.RandomHorizontalFlip(), CIFAR10Policy(),
             transforms.ToTensor(),
             # Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_dataset = datasets.CIFAR10(root=configs.data_path,
                                         train=True,
                                         transform=transform,
                                         download=True)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_dataset = datasets.CIFAR10(root=configs.data_path,
                                        train=False,
                                        transform=transform_test,
                                        download=True)

        train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

        return train_loader, test_loader, classes

    @staticmethod
    def imshow(img, classes, labels, batch_idx):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(classes[labels[0]])
        plt.suptitle('batch:_'.join('%5s' % batch_idx))
        plt.show()

    # TODO: INCL. EXPERIMENT ( @ex.capture)
    def get_dataloader(self, train_dataset, test_dataset, batch_size):
        train_loader = DataLoader(train_dataset, 128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

        return train_loader, test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./../data', nargs='+')
    parser.add_argument('--dataset', type=str, default='', nargs='+')
    parser.add_argument('--num_epochs', type=int, default='1', nargs='+')
    parser.add_argument('--batch_size', type=int, default='1', nargs='+')

    args = parser.parse_args()

    args.dataset = 'MNIST'

    # if args.dataset == 'TinyImageNet':
    #     args.data_path = './../../../datasets/tiny-imagenet-200'
    # elif args.dataset == 'ImageNet':
    #     args.data_path = './../../../datasets/ImageNet_download_2207'

    main(args)
