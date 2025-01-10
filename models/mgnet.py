import torch
import torch.nn as nn


class MgNet(nn.Module):
    def __init__(self, block, device, num_gpus, num_classes, in_channels, out_channels,
                 smoothings,
                 block_args, Ablock_args, Bblock_args,
                 dataset):
        super(MgNet, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.panel = in_channels[0]

        max_layer = len(in_channels)

        if dataset == 'ImageNet':  # or 'TinyImageNet':
            self.ImageNet = True
        else:
            self.ImageNet = False

        self.activation = nn.ReLU(inplace=True)

        self.bn = nn.BatchNorm2d(self.panel)

        if dataset == 'ImageNet':  # or 'TinyImageNet':
            self.conv1 = nn.Conv2d(3, self.panel, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1),
                                   bias=False)
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        elif dataset in ('FashionMNIST', 'MNIST'):
            self.conv1 = nn.Conv2d(1, self.panel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                   bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.panel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                   bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[max_layer - 1], self.num_classes)

        self.mg_block1 = self.make_mg_block(block, 1, in_channels[0], out_channels[0], smoothings[0], self.device,
                                            block_args, Ablock_args, Bblock_args)

        self.mg_block2 = self.make_mg_block(block, 2, in_channels[1], out_channels[1], smoothings[1], self.device,
                                            block_args, Ablock_args, Bblock_args)

        self.mg_block3 = self.make_mg_block(block, 3, in_channels[2], out_channels[2], smoothings[2], self.device,
                                            block_args, Ablock_args, Bblock_args)

        self.mg_block4 = self.make_mg_block(block, 4, in_channels[3], out_channels[3], smoothings[3], self.device,
                                            block_args, Ablock_args, Bblock_args)



    def make_mg_block(self, block, num_layer, *args):
        return block(num_layer, *args)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn(out)
        out = self.activation(out)
        if self.ImageNet:
            out = self.max_pool(out)

        # ---------------------------------------
        u0 = torch.zeros(out.shape, device=self.device)
        x = (out, u0)
        out, u = self.mg_block1(x)

        x = (out, u)
        out, u = self.mg_block2(x)

        x = (out, u)
        out, u = self.mg_block3(x)

        x = (out, u)
        out, u = self.mg_block4(x)

        out = self.avg_pool(u)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out, u0
