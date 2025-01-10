import torch
import torch.nn as nn
import numpy as np
import csv

from utils.convolutions import conv1x1, conv3x3, convkxk
# from utils.eigenvalpues import calc_ev
from models.mgnet_poly import MgBlock_poly
from base.base_mgnet import BaseMgBlock
from models.smoothingstepvariants import *


class MgBlock_real_polyquadratic(MgBlock_poly):
    def __init__(self, *args):
        super(MgBlock_real_polyquadratic, self).__init__(*args)
        in_channels, out_channels = args[1], args[2]
        self.quad_type = args[-1]['quad_type']

        self.convR = convkxk(in_channels, out_channels, stride=2, groups=in_channels, k=3)
        self.convI = convkxk(in_channels, out_channels, stride=2, groups=in_channels, k=3)

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

    def smoothing(self, f, u, s):
        # two linear smoothing step
        # s = 2
        sl = self.num_smoothings
        u, _ = self.linear_smoothing(f, u, sl)

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
