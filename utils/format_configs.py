import argparse
import warnings
import numpy as np
from utils.settings import *

def format_args(BASE, args, cfg):
    if not args.dataset in ('Cifar10', 'FashionMNIST', 'MNIST', 'ImageNet') and args.net_args['num_classes'] != 10:
        warnings.warn('Number of classes is corrected from {} to 10.'.format(args.net_args['num_classes']))
        args.net_args['num_classes'] = 10

    elif args.dataset == 'Cifar100' and args.net_args['num_classes'] != 100:
        warnings.warn('Number of classes is corrected from {} to 100.'.format(args.net_args['num_classes']))
        args.net_args['num_classes'] = 100
        warnings.warn('Number of classes'.format(args.net_args['num_classes']))

    elif args.dataset == "ImageNet":
        args.data_path = '.~/datasets/ImageNet_download_2207'
        args.net_args['num_classes'] = 1000


    elif args.dataset == "TinyImageNet":
        args.data_path = '.~/datasets/tiny-imagenet-200'
        args.net_args['num_classes'] = 200

    if BASE == 'MGiaD':
        args.lr = args.lr

    elif BASE != 'ResNet':
        args.lr = learning_rate(cfg['net_args'])
        args.lr = 0.1 * 1 / cfg['net_args']['smoothings'][0] * 2

    if not args.channel_scaling == 1:
        print('scale')
        if type(args.channel_scaling) == str:
            factor = args.channel_scaling.split('-')[1]
            args.channel_scaling = 0.5 * np.sqrt(int(factor))

        args.net_args['in_channels'] = [int(args.net_args['in_channels'][i] * args.channel_scaling)
                                        for i in range(len(args.net_args['in_channels']))]
        args.net_args['out_channels'] = [int(args.net_args['out_channels'][i] * args.channel_scaling)
                                         for i in range(len(args.net_args['out_channels']))]
