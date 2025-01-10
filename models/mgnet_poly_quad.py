import torch
import torch.nn as nn
import numpy as np
import csv

from utils.convolutions import conv1x1, conv3x3, convkxk
# from utils.eigenvalpues import calc_ev
from models.mgnet_poly import MgBlock_poly
from base.base_mgnet import BaseMgBlock
from models.smoothingstepvariants import *


class MgBlock_polyquadratic(MgBlock_poly):
    def __init__(self, *args):
        super(MgBlock_polyquadratic, self).__init__(*args)
        in_channels, out_channels = args[1], args[2]
        self.quad_type = args[-1]['quad_type']

        self.convR = convkxk(in_channels, out_channels, stride=2, groups=in_channels, k=3)
        self.convI = convkxk(in_channels, out_channels, stride=2, groups=in_channels, k=3)
        # self.activation = nn.ReLU()

    def linear_smoothing(self, f, u, s):
        ''' similiar to smoothing in mgnet_poly '''

        if self.linear_type in ['regular', 'double']:
            for i in range(s):
                u, self.A_weights = self.regularsmoothing_step(f, u, i)

        elif self.linear_type in ['outside', 'double-outside']:
            for i in range(s):
                u, self.A_weights, bn2 = self.outside_smoothing_step(f, u, i)
            # linear blocks only
            if self.num_smoothings == 2:
                u = self.activation(bn2(u))

        elif self.linear_type in ['relu-outside', 'relu-outside-nobn', 'double-relu-outside']:
            for i in range(s):
                u, self.A_weights = self.relu_outside_smoothing_step(f, u, i)
            if self.num_smoothings == 2:
                u = self.activation(u)

        return u, self.A_weights

    def linsmoothing_test(self, i, f, u0):
        bn1, bn2 = self._get_smoothing_batchnorms(i, 0, 0, False)
        self.A, alphas = self.get_operations(i)

        r = f - self.A(u0)
        r = bn1(r)
        r = self.activation(r)

        W = torch.mul(r, self.Bblock[-1])
        W = W.type(torch.float)
        y = self.A(W)

        u = bn2(y)
        u = self.activation(u)

        return u + u0

    def smoothing(self, f, u0, s):
        bn0, bn1 = self._get_smoothing_batchnorms(0, 0, 0, False)
        self.A, alphas = self.get_operations(0)

        r = f - self.A(u0)
        r = bn0(r)
        r = self.activation(r)

        w = torch.mul(r, alphas)
        w = w.type(torch.float)
        y = self.A(w)

        u = bn1(y)
        u = self.activation(u)
        u0 = u + u0
        # -------------------
        bn2, bn3 = self._get_smoothing_batchnorms(1, 0, 0, False)
        _, alphas = self.get_operations(1)

        r = f - self.A(u0)
        r = bn2(r)
        r = self.activation(r)

        w = torch.mul(r, alphas)
        w = w.type(torch.float)
        y = self.A(w)

        u = bn3(y)
        u = self.activation(u)
        u0 = u + u0
        ###################################################
        ## ----
        bn4, bn5 = self._get_smoothing_batchnorms(2, 0, 0, False)
        _, alpha = self.get_operations(2)

        a = torch.abs(np.real(alpha))
        b = torch.abs(np.imag(alpha))

        r = f - self.A(u0)
        r = bn4(r)
        r = self.activation(r)

        u = (2 * a) / (a ** 2 + b ** 2) * r
        u -= 1 / (a ** 2 + b ** 2) * self.A(r)
        # u = bn5(u)
        # u = self.activation(u)

        return u + u0, self.A_weights

    def smoothing2(self, f, u, s):
        # two linear smoothing step
        # s = 2
        sl = self.num_smoothings - 1
        u, _ = self.linear_smoothing(f, u, sl)

        u, bn = self.quad_smoothing_step(f, u, sl)

        # --------------------------------------------

        if self.quad_type == 'outside':
            u = self.activation(bn(u))

        elif self.quad_type == 'norelu':
            u = bn(u)

        elif self.quad_type == 'relu-outside':
            u = self.activation(u)

        return u, self.A_weights

    def quad_smoothing_step(self, f, u0, s):
        bn1, bn2 = self._get_smoothing_batchnorms(s, 0, 0, False)
        A, alpha = self.get_operations(s)

        residual_type = self.Bblock_args['residual_type']

        alpha = alpha.to(dtype=torch.cfloat)
        a = torch.abs(np.real(alpha))
        b = torch.abs(np.imag(alpha))

        if self.quad_type == 'regular':
            ''' (') regular block '''
            return QuadraticPolyStep((A, alpha), (bn1, bn2), residual_type)((f, u0, a, b)), bn2
        elif self.quad_type in ['plain', 'outside', 'norelu']:
            ''' ('')  '''
            return QuadraticPolyStep_outside((A, alpha), (bn1, bn2), residual_type)((f, u0, a, b)), bn2
        elif self.quad_type in ['relu-outside', 'norelu-bninside']:
            ''' (''‡) '''
            return QuadraticPolyStep_relu_outside((A, alpha), (bn1, bn2), residual_type)((f, u0, a, b)), bn2
        elif self.quad_type in ['norelu-bninside-out']:
            ''' (''‡) '''
            return QuadraticPolyStep_relu_outside_bninside_out((A, alpha), (bn1, bn2), residual_type)(
                (f, u0, a, b)), bn2
        elif self.quad_type == 'inside':
            ''' ('' ') '''
            return QuadraticPolyStep_inside((A, alpha), (bn1, bn2), residual_type)((f, u0, a, b))
        elif self.quad_type == 'relu-outside-bnin':
            ''' ('' ') '''
            return QuadraticPolyStep_outbninside((A, alpha), (bn1, bn2), residual_type)((f, u0, a, b))

    def _make_identity(self):
        # bs, c, n = x.shape[0], x.shape[1], x.shape[2]
        n = self.A.weight.size()[-1]
        c = self.A.weight.size()[1]
        I = torch.eye(n)
        I = I.reshape(1, 1, n, n)
        I = I.repeat(c, c, 1, 1).to(self.device)
        return I

    def forward(self, x):

        f = x[0]
        u = x[1]

        self.A_weights = []
        if self.num_layer > 1:
            u = self.convI(u)
            # f = self.convR(f) + self.Ablock.A(u)
            # print('Attention this is for two diffferent block blocks')
            key = list(self.Ablock)[-1]
            f = self.convR(f) + self.Ablock[key](u)

            # self.A_weights.append(self.Ablock.A.weight)

        # u, self.A_weights = self.smoothing(f, u, 1)
        u, self.A_weights = self.smoothing2(f, u, 1)

        v = self.Ablock.A(u)
        # v = self.A(u)
        f = f - v

        f = self.activation(f)

        return f, u
