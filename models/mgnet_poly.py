import torch
import torch.nn as nn
import numpy as np
import csv

from utils.convolutions import conv1x1, conv3x3, convkxk
from models.smoothingstepvariants import *  # PolySmoothingStep, PolySmoothingStep_bnreluoutside

# from utils.eigenvalpues import calc_ev

from base.base_mgnet import BaseMgBlock

class MgBlock_poly(BaseMgBlock):
    """ Mg Block with predefined filters in Restriction
       :argument

        filters: optional, if true weights of restriction and projection are set to

                           [[0.25, 0.5, 0.25]
                    0.25 *  [0.5,  1.0,  0.5]
                            [0.25, 0.5, 0.25]]

       train_filters: optional, if true preset weights (filters) are updated

       """

    # def __init__(self, in_channels: int, out_channels: int, num_smoothings: int,
    #              num_layer: int, device: str,
    #              block_args: dict, Ablock_args: dict, Bblock_args:dict):
    def __init__(self, *args):
        super(MgBlock_poly, self).__init__(*args)
        in_channels, out_channels = args[1], args[2]
        self.linear_type = args[-1]['linear_type']
        self.residual_type = args[-1]['residual_type']

        self.convR = convkxk(in_channels, out_channels, stride=2, groups=in_channels, k=3)
        self.convI = convkxk(in_channels, out_channels, stride=2, groups=in_channels, k=3)

    def linear_smoothing(self, f, u, s):
        freq = self.block_args['frequency'][self.block_args['num_layer'] - 1]
        if self.linear_type == 'regular':
            for i in range(self.num_smoothings * freq):
                u, self.A_weights = self.regularsmoothing_step(f, u, i)
            return u, self.A_weights

        elif self.linear_type in ['outside', 'double-outside']:
            for i in range(self.num_smoothings * freq):
                u, self.A_weights, bn2 = self.outside_smoothing_step(f, u, i)
                u = bn2(u)
                u = self.activation(u)
            return u, self.A_weights

        elif self.linear_type in ['relu-outside', 'relu-outside-nobn', 'double-relu-outside']:
            for i in range(self.num_smoothings * freq):
                u, self.A_weights = self.relu_outside_smoothing_step(f, u, i)
            u = self.activation(u)

        return u, self.A_weights

    def smoothing(self, f, u, s):
        """
        f: data
        u: features
        s: placeholder
        """
        # -------------------------------------------------------------------------
        return self.linear_smoothing(f, u, None)

    def regularsmoothing_step(self, f, u0, i):
        bn1, bn2 = self._get_smoothing_batchnorms(i, 0, 0, False)
        A, alphas = self.get_operations(i)
        residual_type = self.Bblock_args['residual_type']

        if len(alphas.shape) == 1:
            alpha = alphas[0]
        else:
            alpha = alphas[1:][i]

        ##########################################
        # regular smoothing step
        #######################################
        if self.linear_type == 'regular':
            u = LinearPolyStep((A, alpha), (bn1, bn2), True, residual_type)((f, u0))
        else:
            #u = LinearPolyStepDouble((self.A, alpha), (bn1, bn2), '')((f, u0))
            u = LinearPolyStep((A, alpha), (bn1, bn2), '', residual_type)((f, u0))
        return u, self.A_weights

    def outside_smoothing_step(self, f, u0, i):
        bn1, bn2 = self._get_smoothing_batchnorms(i, 0, 0, False)
        A, alphas = self.get_operations(i)
        residual_type = self.Bblock_args['residual_type']

        if len(alphas.shape) == 1:
            alpha = alphas[0]
        else:
            alpha = alphas[1:][i]
            # alpha = alphas[0][i]
        if self.linear_type.split('-')[0] == 'double':
            u, bn = LinearPolyStep_bnrelu_outside((A, alpha), (bn1, bn2), '', residual_type)((f, u0))
        else:
            u, bn = LinearPolyStep_bnrelu_outside((A, alpha), (bn1, bn2), True, residual_type)((f, u0))
        return u, self.A_weights, bn

    def relu_outside_smoothing_step(self, f, u0, i):
        bn1, bn2 = self._get_smoothing_batchnorms(i, 0, 0, False)
        # self.A, alphas = self.get_operations2(i)
        A, alphas = self.get_operations(i)
        if len(alphas.shape) == 1:
            alpha = alphas[0]
        else:
            alpha = alphas[1:][i]
            # alpha = alphas[0][i]
        if self.linear_type.split('-')[0] == 'double':
            u, bn = LinearPolyStep_relu_outside((A, alpha), (bn1, bn2), '', self.residual_type)((f, u0))
        elif self.linear_type.split('-')[1] != 'outside': # elf.linear_type.split('-')[2]  ==bn
            u, bn = LinearPolyStep_relu_outside_nobn((A, alpha), (bn1, bn2), True, self.residual_type)((f, u0))
        else:
            u, bn = LinearPolyStep_relu_outside((A, alpha), (bn1, bn2), True, self.residual_type)((f, u0))
        return u, self.A_weights

    def forward(self, x):

        f = x[0]
        u = x[1]

        self.A_weights = []
        if self.num_layer > 1:
            u = self.convI(u)
            key = list(self.Ablock)[-1]
            f = self.convR(f) + self.Ablock[key](u)

        u, self.A_weights = self.smoothing(f, u, 1)

        v = self.Ablock.A(u)
        f = f - v
        f = self.activation(f)

        return f, u
