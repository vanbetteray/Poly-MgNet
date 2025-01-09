"""

 Helper script to define set-up and other utility functions

  - learingrate: returns a fixed learning rate dependend on the number of smoothings
  - config_device: specifies device
  - get_cfg: read configs.json containing hyperparameters for network specification and training setting
  - name_ex: generator for experiment name including relevant information about architecture and hyperparameters, incl. date
  - count_parameter: returns the number of parameters that requires gradient
  - save_results: creates directory to store results in,
                  saves number of parameters and model summary
  - count_flops: # todo: function that count FLOPs and/or GMACs

"""

import logging
import os
from easydict import EasyDict as edict
import pandas as pd
import torch

import json
import warnings

from datetime import date


# from utils.plots import print_graph
from ptflops import get_model_complexity_info
# from fvcore.nn.flop_count import flop_count
# from slowfast.datasets.utils import pack_pathway_outpu
def learning_rate(net_args, lr=0.1):

    smoothings = net_args["smoothings"][0]
    return lr*1/smoothings*2


def get_cfg(name):
    if name == 'ResNet':
        with open('resnet-configs.json'.format()) as cfg_file:
            cfg = json.load(cfg_file)
            print('{} configs loaded.'.format(cfg['net_name']))
    elif name == 'MgNetpoly':
        with open('configs-poly.json'.format()) as cfg_file:
            cfg = json.load(cfg_file)
            print('Polynomial {} configs loaded.'.format(cfg['net_name']))
    else:
        with open('configs.json'.format()) as cfg_file:
            cfg = json.load(cfg_file)
            print('{} configs loaded.'.format(cfg['net_name']))

    print(cfg)
    return cfg


def config_device(cpu_only: bool, idx):
    """
    Set main torch device for model and dataloader based on configuration settings.
    Args:
        cpu_only (bool): Flag indicating whether to exclusively use CPU.
    Returns:
        (torch.device).
    """
    if not cpu_only:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            logging.warning('CUDA device is not available. Will use CPU')
    else:
        use_cuda = False
    _device = torch.device("cuda:" + str(idx) if use_cuda else "cpu")
    return _device


def get_date():
    today = date.today()
    return today.strftime("%d%m")

# Summary parameter count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_results(net, args):
    saver_path = args.saver_path

    if not os.path.exists(saver_path):
        os.makedirs(saver_path)
        print('Results will be stored in {}'.format(saver_path))

    if not net._get_name() == 'ResNet':
        print(net._get_name())
        if args.dataset in ('MNIST', 'FashionMNIST'):
            gmacs, params = get_model_complexity_info(net, (1, 32, 32))

        else:
            pass

    params = count_parameters(net)
    print(params)
    gmacs = ""
    print(gmacs)

    for key, item in enumerate(args.net_args):
        if type(args.net_args[item]) not in [str, int]:
            args.net_args[item] = str(args.net_args[item])

    with open(saver_path + "/configs.json", "w") as outfile:
        json.dump(vars(args), outfile)

    f = open(saver_path + "/results.txt", "a")
    f.write('\n' + str(args) + '\n')
    f.write("----------------------------------------------------------------\n")
    f.write("----------------------------------------------------------------\n")
    f.write("\n Initial number of parameters: " + str(params) + '')
    f.write("\n Number of GMacs: " + str(gmacs) + '\n\n')
    f.write("----------------------------------------------------------------\n")
    f.write("----------------------------------------------------------------\n")


