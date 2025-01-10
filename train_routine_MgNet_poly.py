import argparse
import torch.cuda
import torch.nn as nn

from utils.settings import *
from utils.format_configs import format_args
from base.trainer import Trainer
from torchvision import *

BASE = 'MgNetpoly'
print(BASE)

from models.mgnet_poly import MgBlock_poly as Block
from models.mgnet import MgNet as Net

cfg = get_cfg(BASE)


# torch.manual_seed(9)
def main(_):
    # get the pid for the current process
    pid = os.getpid()
    print('PID:', pid)
    print(args)
    if BASE == 'ResNet':
        num_classes = args.net_args['num_classes']
        net = models.resnet18(pretrained=False)
        net.fc = torch.nn.Linear(512, num_classes)
        # net.fc = torch.nn.Linear(256, num_classes)
        if not args.dataset in ['ImageNet']:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                  bias=False)
            # net.maxpool = nn.MaxPool2d(1, dilation=1)
    else:
        net = Net(**args.net_args)
    print(net)
    # if args.net_args['num_gpus'] > 1:
    #     net = nn.DataParallel(net)
    #     net.cuda()
    # else:

    print('available', torch.cuda.is_available())
    print('counts', torch.cuda.device_count())
    print('current device', torch.cuda.current_device())
    print('\n' + 20 * '--')

    net.to(args.net_args['device'])

    print('Net on cuda .... ')

    trainer = Trainer(args, args.net_args)
    trainer.train_routine(net)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')

    parser.add_argument('--load_json',
                        help='Load settings from file in json format. Command line options override values in file.',
                        type=bool, default=True, nargs='+')
    parser.add_argument('--number', default='1', nargs='+')

    args = parser.parse_args()
    argparse_dict = vars(args)
    argparse_dict.update(cfg)

    format_args(BASE, args, cfg)

    if BASE == 'ResNet':
        Block = 'res'
    args.net_args.update({'block': Block})
    args.net_args.update({'dataset': args.dataset})
    if BASE != 'ResNet' and args.net_args['Bblock_args'] == 'polynomial' and args.net_args['block'] != 'MgBlock_poly':
        print('this does not match')

    print(20 * "-" + "\n")

    args.net_args['smoothings'] = [2, 2, 2, 2]
    if not args.net_args['Bblock_args']['perspective'] == 'linear':
        args.net_args['Bblock_args']['perspective'] = 'linear'
    print(args.net_args['Bblock_args']['perspective'])

    # args.net_args['in_channels'] = [22, 22, 44, 88]
    # args.net_args['out_channels'] = [22, 44, 88, 88]

    # args.net_args['in_channels'] = [45, 45, 90, 180]
    #args.net_args['out_channels'] = [45, 90, 180, 180]

    # args.net_args['in_channels'] = [90, 90, 180, 360]
    # args.net_args['out_channels'] = [90, 180, 360, 360]

    # args.net_args['in_channels'] = [180, 180, 360, 720]
    # args.net_args['out_channels'] = [180, 360, 720, 720]

    # args.net_args['in_channels'] = [182, 182, 364, 728]
    # args.net_args['out_channels'] = [182, 364, 728, 728]

    # args.net_args['in_channels'] = [192, 192, 384, 768]
    # args.net_args['out_channels'] = [192, 384, 768, 768]

    '''
    choose type from: [regular, outside , relu-outside] 
    
       regular        (*): u + σ◦bn 1/λ (σ◦bn(f − Au))
       outside       (**): σ◦bn(... u + 1/λ (σ ◦ bn(f − Au))) # relu and bn before resolution coarsening
       relu-outside (**‡): σ(... u + bn 1/λ (σ ◦ bn(f − Au))) # relu before resolution coarsening
       relu-outside-nobn  (**‡-): σ(... u + 1/λ (σ ◦ bn(f − Au))) # relu before resolution coarsening no bn after coeff
    '''
    # args.net_args['Bblock_args'].update({'linear_type': 'regular'})
    # args.net_args['Bblock_args'].update({'linear_type': 'relu-outside-nobn'})
    # args.net_args['Bblock_args'].update({'linear_type': 'relu-outside'})
    args.net_args['Bblock_args'].update({'linear_type': 'outside'})

    '''
           regular: σ ◦ bn(f − Au)
           bn_only: bn(f − Au)
           plain:   f − Au
    '''
    # args.net_args['Bblock_args'].update({'residual_type': 'plain'})
    args.net_args['Bblock_args'].update({'residual_type': 'bn_only'})
    # args.net_args['Bblock_args'].update({'residual_type': 'regular'})

    args.net_args['device'] = config_device(cpu_only=False, idx=0)
    args.net_args['num_gpus'] = 1
    args.debug = False
    # "pol-init": "LFA,min,max",
   #  args.net_args['Bblock_args']['pol-init'] = 'Xavier,,'
    # args.num_epochs = 6
    args.net_args['Bblock_args']['pol-init'] = 'Xavier,,'
    if args.resume:
        print('Please specify experiment to resume')
        args.experiment_name = ''
    else:
        # ----------------------------------------------------------------------------------------
        # # *
        # if args.net_args['Bblock_args']['linear_type'] == 'regular':
        #     args.experiment_name = name_ex(BASE, '--(...u+bnreluB(f-bnreluAu))_degree2', args)
        # # ----------------------------------------------------------------------------------------
        # # **
        # elif args.net_args['Bblock_args']['linear_type'] == 'outside':
        #     args.experiment_name = name_ex(BASE, '--bnrelu(...u+B(f-bnreluAu))_degree2', args)
        # # ----------------------------------------------------------------------------------------
        # # **‡
        # elif args.net_args['Bblock_args']['linear_type'] == 'relu-outside':
        #     args.experiment_name = name_ex(BASE, '--relu(...u+bnB(f-bnreluAu))_degree2', args)

        name = 'residual-' + args.net_args['Bblock_args']['residual_type'] + '_' \
               'linear-' + args.net_args['Bblock_args']['linear_type'] + '_'

        name += '/' + str(args.net_args['out_channels'])
        name += '/xavier'

        args.experiment_name = name_ex(BASE, '-' + name + '-', args)

        # args.experiment_name += '-1'
        if args.debug == True:
            args.experiment_name += '-' + str(args.number[0])
        else:
            args.experiment_name += '-' + str(args.number[0])




    args.saver_path = os.path.join('results', args.experiment_name)
    print('data saved in', args.saver_path)
    main(args)
