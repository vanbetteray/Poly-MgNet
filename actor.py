import torch
import torch.backends.cudnn

from time import time
from base.base_actor import BaseActor
from easydict import EasyDict as edict

class Actor(BaseActor):

    def __init__(self, configs, train_loader, test_loader, net, device, classes,
                 criterion, optimizer, scheduler):
        super(Actor, self).__init__(configs, train_loader, test_loader, net, device, classes,
                                    criterion, optimizer, scheduler)


    def train_step(self, epoch, net, trainloader, device):

        global batch_idx
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        visualisation_args = []

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs, targets = inputs.cuda(), targets.cuda()
            # nur bei rnn
            # inputs = inputs.reshape(-1, 32, 32*3).to(device)
            # torch.cuda.synchronize(self.device)
            self.optimizer.zero_grad()
           #  outputs = net(inputs)
            if net._get_name() in ['ResNet', 'MobileNetV2']:
                outputs = net(inputs)
            else:
                outputs, _ = net(inputs)

            if (epoch == self.configs.num_epochs - 1 and batch_idx == 0):
                start_time = time()
                # self.calc_ev(self.net.mg_block1.convA, self.net.mg_block1.convB, xshape)
                end_time = time()
                duration = end_time - start_time
                print(duration)
                # self.save_ev(evA, duration)

            loss = self.criterion(outputs, targets)
            loss.backward()

            self.optimizer.step()
            train_loss += loss.item()  #+ inputs.size()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    def test_step(self, epoch, net, testloader, device):
        """
        epoch: int
        """
        print('epoch test step', epoch)
        global best_acc

        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):

                inputs, targets = inputs.to(device), targets.to(device)

                if net._get_name() in ['ResNet', 'MobileNetV2']:
                    outputs = net(inputs)
                else:
                    outputs, _ = net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print(f'Epoch test acc:{correct}')
