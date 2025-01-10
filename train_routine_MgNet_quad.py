import argparse
import torch.cuda
import torch.nn as nn

from utils.settings import *
from utils.format_configs import format_args
from base.trainer import Trainer
from torchvision import *

BASE = 'MgNetpoly'
print(BASE)

from models.mgnet_poly_quad import MgBlock_polyquadratic as Block
# from models.mgnet_poly_quad_real import MgBlock_real_polyquadratic as Block

from models.mgnet import MgNet as Net

cfg = get_cfg(BASE)

def main(_):
    # get the pid for the current process
    pid = os.getpid()
    print('PID:', pid)
    print(args)
    if BASE == 'ResNet':
        num_classes = args.net_args['num_classes']
        net = models.resnet18(pretrained=False)
        net.fc = torch.nn.Linear(512, num_classes)
        if not args.dataset in ['ImageNet']:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                  bias=False)

    else:
        net = Net(**args.net_args)
    print(net)

    print('available', torch.cuda.is_available())
    print('counts', torch.cuda.device_count())
    print('current device', torch.cuda.current_device())
    print('\n' + 20 * '--')

    net.to(args.net_args['device'])

    print('Net on cuda ... ')

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

    args.channel_scaling = 1
    if args.channel_scaling != 1:
        sc = args.channel_scaling
        args.net_args['in_channels'] = [int(sc * c) for c in args.net_args['in_channels']]

    print(20 * "-" + "\n")
    args.net_args['Bblock_args']['pol-init'] = 'Xavier,,'

    if not args.net_args['Bblock_args']['perspective'] == 'quadratic':
        if Block.__name__ == "MgBlock_real_polyquadratic":
            args.net_args['Bblock_args']['perspective'] = 'quadratic-real'
        else:
            args.net_args['Bblock_args']['perspective'] = 'quadratic'
    print(args.net_args['Bblock_args']['perspective'])
    args.net_args["block_args"].update({'degree_quad': 1})

    '''
       choose linear_type from: [regular, outside, relu-outside, double, double-norelu] 

          regular         (*): u + σ◦bn 1/λ (σ◦bn(f − Au))
          outside        (**): σ◦bn(... u + 1/λ (σ ◦ bn(f − Au))) # relu and bn before resolution coarsening
          relu-outside  (**‡): σ(... u + bn 1/λ (σ ◦ bn(f − Au))) # relu before resolution coarsening
          relu-outside-nobn  (**‡): σ(... u + 1/λ (σ ◦ bn(f − Au))) # relu before resolution coarsening
          double             : u + σ◦bn A * 1/sqrt(λ) (σ◦bn(f − Au))    # regular with an additional A 
          double-outside     : σ◦bn(... u + 1/sqrt(λ) (σ ◦ bn(f − Au)))   # bn before resolution coarsening
          double-relu-outside: σ◦(... u + bn A* 1/sqrt(λ) (σ ◦ bn(f − Au)))   # bn before resolution coarsening
    '''

    # args.net_args['Bblock_args'].update({'linear_type': 'relu-outside'})
    # args.net_args['Bblock_args'].update({'linear_type': 'outside'})
    # args.net_args['Bblock_args'].update({'linear_type': 'relu-outside-nobn'})
    # args.net_args['Bblock_args'].update({'linear_type': 'regular'})
    # args.net_args['Bblock_args'].update({'linear_type': 'double-relu-outside'})
    # args.net_args['Bblock_args'].update({'linear_type': 'double-outside'})
    #  args.net_args['Bblock_args'].update({'linear_type': 'double-norelu'})
    args.net_args['Bblock_args'].update({'linear_type': 'double'})

    '''
        regular: σ ◦ bn(f − Au)
        bn_only: bn(f − Au)
        plain:   f − Au
    '''
    # args.net_args['Bblock_args'].update({'residual_type': 'plain'})
    # args.net_args['Bblock_args'].update({'residual_type': 'bn_only'})
    args.net_args['Bblock_args'].update({'residual_type': 'regular'})

    # ---------------------------------------------------------------------------------
    '''
           choose quad_type from: [regular, outside , relu-outside, norelu] 
              regular          ('): u + σ◦bn 1/(a2+b2)(2a − A)(σ ◦ bn(r)) # regular
              outside         (''): σ◦bn(... u + 1/(a2+b2) (2a − A)(σ ◦ bn(r))) # relu and bn before resolution coarsening
              plain                   : (2a/(a2+b2) r − (1/a2+b2 A)r
              norelu          ( ) : bn(u + 2a/(a2+b2) r − (1/a2+b2 A)(σ ◦ bn(r)))
              relu-outside   : σ (...u + bn ◦ 2a/(a2+b2) r −  1/(a2+b2)A(σ ◦ bn(r)) # relu before resolution coarsening
              norelu-bninside ( ) : (u +  2a/(a2+b2) r − bn ◦ (1/a2+b2 A)(σ ◦ bn(r)))
               
              relu-outside-bnin  (''‡): σ (...u + 2a/(a2+b2) r − bn ◦ 1/(a2+b2)A(σ ◦ bn(r)) # relu before resolution coarsening
              inside       ('' ') : u + 2a/(a2+b2) r − (σ ◦ bn 1/a2+b2 A)(σ ◦ bn(r)) 
              norelu-bninside-out ( ) : (u + bn ◦ ( 2a/(a2+b2) r − (1/a2+b2 A)(σ ◦ bn(r))))
    '''

    if Block.__name__ == "MgBlock_polyquadratic":
        # args.net_args['Bblock_args'].update({'quad_type': 'regular'})
        args.net_args['Bblock_args'].update({'quad_type': 'outside'})
        # args.net_args['Bblock_args'].update({'quad_type': 'norelu'})
        # args.net_args['Bblock_args'].update({'quad_type': 'relu-outside'})
        # args.net_args['Bblock_args'].update({'quad_type': 'relu-outside-bn'})
        #args.net_args['Bblock_args'].update({'quad_type': 'norelu-bninside'})
        #args.net_args['Bblock_args'].update({'quad_type': 'norelu-bninside-out'})
        # args.net_args['Bblock_args'].update({'quad_type': 'plain'})
        # args.net_args['Bblock_args'].update({'quad_type': 'inside'})
        args.net_args['smoothings'] = [3, 3, 3, 3]
        # args.net_args['smoothings'] = [4, 4, 4, 4]
        # args.net_args['smoothings'] = [5, 5, 5, 5]
    else:
        args.net_args['smoothings'] = [2, 2, 2, 2]
        args.net_args['Bblock_args'].update({'quad_type': ''})

    args.net_args['device'] = config_device(cpu_only=False, idx=1)
    args.net_args['num_gpus'] = 1
    args.debug = False
    # args.num_epochs = 2

    if args.resume:
        print('Please specify experiment to resume')
        args.resume_name = 'experiment_name'
        d = date.today()
        args.experiment_name = args.resume_name + '_' + str(d)

    else:
        args.experiment_name = 'experiment_name'
        if args.debug == True:
            args.experiment_name += '-' + '2'
        else:
            args.experiment_name += '-' + str(args.number[0])

    args.saver_path = os.path.join('results', args.experiment_name)
    print('data saved in', args.saver_path)
    main(args)
