import torch
import torch.nn as nn

from utils.convolutions import convkxk

import math
import numpy as np


class BaseMgBlock(nn.Module):
    """
    Module that provides the basic functions
        :arg
        in_channels: number of input channels
        out_channels: number of output channels
        num_smoothings: number of smoothing steps on each resolution, usually the same amount on each resolution
        num_layer: layers are enumerated, due to implementation details # todo: tbc
        device: number of GPU
    """

    def __init__(self, num_layer: int, in_channels, out_channels, num_smoothings: int, device, block_args, Ablock_args,
                 Bblock_args, *args):
        super(BaseMgBlock, self).__init__()
        self.us = None
        self.imag_max = None
        self.real_max = None
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_smoothings = num_smoothings
        self.num_layer = num_layer
        self.A_weights = []
        self.activation = nn.ReLU()

        block_args['num_layer'] = num_layer
        self.block_args = block_args
        self.Ablock_args = Ablock_args
        self.Bblock_args = Bblock_args

        Ablock_args.update(block_args)
        Bblock_args.update(block_args)

        if self._get_name() in ['MgBlock_poly', 'MgBlock_polyquadratic', 'MgBlock_real_polyquadratic',
                                'MgBlock_polysmoother']:
            self.Ablock = self.make_block('A', out_channels, self.num_smoothings, **Ablock_args)
            self.Bblock = self.make_block('B', out_channels, self.num_smoothings, **Bblock_args)
            self.batchnorms = self.make_batchnorms(self.num_smoothings, out_channels, **block_args)

    @staticmethod
    def make_batchnorms(num_smoothings: int, out_channels: int, **kwargs) -> {}:
        """
            freq: if number of blocks > 2 on one resolution level and A only partially shared, e.g. 4 blocks and two shared A1A1 A2A2
                  default freq=1
        """
        freq = kwargs['frequency'][kwargs['num_layer'] - 1]
        keys = []
        for s in range(2 * num_smoothings * freq):
            keys.append('bn' + str(s))

        return nn.ModuleDict({key: nn.BatchNorm2d(out_channels) for key in keys})
    def keys(self, i):
        """
            i: smoothing iteration/ block number
        """
        if self.Bblock_args['shared']:
            keyB = 'B'
        else:
            keyB = 'B' + str(i)

        if self.Ablock_args['shared']:
            freq = self.block_args['frequency'][self.num_layer - 1]
            if freq > 1:
                keyA = 'A' + str(i // freq)
            else:
                keyA = 'A'
        else:
            keyA = 'A' + str(i)

        return keyA, keyB

    def bn_keys(self, i, s):
        key1 = 'bn' + str(2 * i * s)
        key2 = 'bn' + str(2 * i * s + 1)

        return key1, key2

    def _get_smoothing_batchnorms(self, i, v, s, postsmoothing: bool):
        """
        returns batchnormalization for smoothing operation based in level index
        if postsmoothing:
            new batchnorm operations, not implemented yet
        """
        key1, key2 = self.bn_keys(i, 1)
        bn1, bn2 = self.batchnorms[key1], self.batchnorms[key2]

        return bn1, bn2

    def get_operations(self, i):
        if self.Bblock_args['block'] == 'polynomial':
            keyA, _ = self.keys(i)
            if not self.Bblock_args['shared']:
                return self.Ablock[keyA], self.Bblock[i]
            else:
                return self.Ablock[keyA], self.Bblock[0]
        else:
            keyA, keyB = self.keys(i)
            return self.Ablock[keyA], self.Bblock[keyB]

    def smoothing_step(self, f, u0, i):
        """
        f data
        u0 feature
        smoothing step
        """

        bn1, bn2 = self._get_smoothing_batchnorms(i, 0, 0, False)
        A, B = self.get_operations(i)
        r = f - A(u0)
        r = bn1(r)
        r = self.activation(r)

        u = B(r)
        u = bn2(u)
        u = self.activation(u)

        return u + u0, A.weight

    def smoothing(self, f, u, s):
        """
        f: data
        u: features
        s: placeholder
        """
        for i in range(self.num_smoothings):
            u, self.A_weights = self.smoothing_step(f, u, i)
        return u, self.A_weights


    def groups(self, **kwargs):
        """in case of grouped convolutions"""
        groups_size = kwargs['groups_size']

        if not bool(groups_size):
            return kwargs['groups']
        else:
            return self.out_channels // int(groups_size)

    def make_block(self, name, out_channels, smoothings, **kwargs):
        """
        designs different blocks for operations A and B
            polynomial of degree n
            richardson: trainable or non-trainable scalar, initial value derived from LFA
            shared: module reused
        """
        block = kwargs['block']
        frequency = kwargs['frequency'][self.num_layer - 1]
        shared = True if kwargs['shared'] == 'True' else False
        k = int(kwargs['k'])
        kwargs.update({'num_smoothings': smoothings})
        kwargs.update({'out_channels': out_channels})
        groups = self.groups(**kwargs)

        if block in ['polynomial', 'polynomialquad']:
            perspective = kwargs['perspective']
            if shared or perspective == "regular":
                B = torch.nn.ParameterList(None)
                for s in range(smoothings * frequency):
                    B.append(self.make_polynomial(**kwargs))
                return B
            else:
                B = torch.nn.ParameterList(None)
                for s in range(smoothings * frequency):
                    B.append(self.make_polynomial_block(s, **kwargs))
                return B
        if shared and frequency == 1:
            return nn.ModuleDict({name: convkxk(out_channels, out_channels, groups, k)})

        elif shared and frequency > 1:
            keys = []
            for s in range(frequency):
                keys.append(name + str(s))
            return nn.ModuleDict({key: convkxk(out_channels, out_channels, groups, k) for key in keys})
        else:
            keys = []
            for s in range(smoothings):
                keys.append(name + str(s))
            return nn.ModuleDict({key: convkxk(out_channels, out_channels, groups, k) for key in keys})

    def make_polynomial_block(self, s, **kwargs):
        perspective = kwargs['perspective']
        out_channels = kwargs['out_channels']
        if perspective in ['regular', 'quadratic']:
            deg_poly = kwargs['num_smoothings']
        else:
            deg_poly = kwargs['degree'] * kwargs['frequency'][0]

        train_coeff = kwargs['training']

        init_values = str.split(kwargs['pol-init'], ',')
        init = init_values[0]

        a = init_values[1]
        b = init_values[2]

        if init == 'EV':
            raise "Not implemented yet"

        elif init == 'Xavier':
            vals = torch.empty(1)
            if s == 0:
                self.us = torch.Tensor(deg_poly).uniform_(-1, 1) * math.sqrt(6. / (out_channels + out_channels))
                self.min = self.us[0]
                self.max = self.us[-1]
                vals[0] = self.min

            elif s == kwargs['num_smoothings'] - 1:
                vals[0] = self.max
            else:
                print('only implemented for g6')
                vals[0] = self.us[1]

        elif init == 'U':
            vals = torch.empty(1)
            self.us = torch.empty(deg_poly).uniform_(int(a), int(b))

            self.min = self.us[0]
            self.max = self.us[-1]

            if s == 0:
                vals[0] = self.min
            elif s == kwargs['num_smoothings'] - 1:
                vals[deg_poly - 1] = self.min
                vals[-1] = self.max
            else:
                vals[1] = self.us[1]

        if train_coeff == 1:
            alpha = torch.nn.Parameter(1 / vals, requires_grad=True)
        else:
            alpha = torch.nn.Parameter(1 / vals, requires_grad=False)
        return alpha

    def _get_real_evs(self, s, deg):
        vals = torch.empty(1)
        if self.Bblock_args['linear_type'] in ['double', 'double-norelu', 'double-outside', 'double-relu-outside']:
            if deg == 2:
                z = max(np.abs(np.real(self.ev_min)), np.abs(np.real(self.ev_max)))
                vals[0] = z**2
            else:
                if s == 0:
                    vals[0] = np.real(self.ev_min**2)
                elif s == 1:
                    vals[0] = np.real(self.ev_max**2)
        else:
            if s == 0:
                vals[0] = np.real(self.ev_min)
            elif s == 1:
                vals[0] = np.real(self.ev_max)
        return vals

    def forward(self, x):
        pass
