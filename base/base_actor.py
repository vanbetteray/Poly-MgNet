import csv
import torch.backends.cudnn

class BaseActor(object):

    def __init__(self, configs, train_loader, test_loader, net, device, classes,
                 criterion, optimizer, scheduler):
        self.configs = configs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.net = net
        self.classes = classes
        self.device = device
        self.acc = 0
        self.best_acc = 0
        self.best_train_acc = 0
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler


    def train_step(self, epoch, net, trainloader, device):
        pass

    def test_step(self, epoch, net, testloader, device):
        pass