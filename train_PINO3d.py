import yaml
from argparse import ArgumentParser
import random
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os

from train_utils import Adam
from train_utils.datasets import NSLoader
from train_utils.data_utils import data_sampler
from train_utils.losses import get_forcing
from train_utils.train_3d import train
from train_utils.distributed import setup, cleanup
from train_utils.utils import requires_grad

from models import FNO3d, FNO2d


def subprocess_fn(rank, args):
    if args.distributed:
        setup(rank, args.num_gpus)
    print(f'Running on rank {rank}')

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    # construct dataloader
    data_config = config['data']

    seed = data_config['seed']
    print(f'Seed :{seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    if 'datapath2' in data_config:
        loader = NSLoader(datapath1=data_config['datapath'], datapath2=data_config['datapath2'],
                          nx=data_config['nx'], nt=data_config['nt'],
                          sub=data_config['sub'], sub_t=data_config['sub_t'],
                          N=data_config['total_num'],
                          t_interval=data_config['time_interval'])
    else:
        loader = NSLoader(datapath1=data_config['datapath'],
                          nx=data_config['nx'], nt=data_config['nt'],
                          sub=data_config['sub'], sub_t=data_config['sub_t'],
                          N=data_config['total_num'],
                          t_interval=data_config['time_interval'])
    if args.start != -1:
        config['data']['offset'] = args.start
    # trainset = loader.make_dataset(data_config['n_sample'],
    #                            start=data_config['offset'])
    trainset, testset = loader.split_dataset(data_config['n_sample'], data_config['offset'], data_config['test_ratio'])
    train_loader = DataLoader(trainset, batch_size=config['train']['batchsize'],
                              shuffle=False,
                              sampler=data_sampler(trainset,
                                                   shuffle=data_config['shuffle'],
                                                   distributed=args.distributed),
                              drop_last=True)

    # construct model
    model = FNO3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers']).to(rank)

    start_epoch = 0
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        if ckpt_path is not None:
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location={'cuda:%d' % 0: 'cuda:%d' % rank})
                
                # Check and load model state dict if it exists
                if 'model' in ckpt:
                    model.load_state_dict(ckpt['model'])
                    print('Model state loaded from %s' % ckpt_path)
                # Update start epoch if it exists
                if 'epoch' in ckpt:
                    start_epoch = ckpt['epoch'] + 1
                    print('Starting epoch updated to %d' % start_epoch)
            else:
                print('Checkpoint file does not exist at %s' % ckpt_path)
        else:
            print('Checkpoint path is None in the config')
    else:
        print('Checkpoint path is not provided in the config')

    if args.distributed:
        model = DDP(model, device_ids=[rank])

    if 'twolayer' in config['train'] and config['train']['twolayer']:
        requires_grad(model, False)
        requires_grad(model.sp_convs[-1], True)
        requires_grad(model.ws[-1], True)
        requires_grad(model.fc1, True)
        requires_grad(model.fc2, True)
        params = []
        for param in model.parameters():
            if param.requires_grad == True:
                params.append(param)
    else:
        params = model.parameters()

    optimizer = Adam(params, betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    forcing = get_forcing(loader.S).to(rank)
    train(model,
          loader, train_loader,
          optimizer, scheduler,
          forcing, config,
          rank,
          log=args.log,
          project=config['log']['project'],
          group=config['log']['group'],
          distributed=args.distributed,
          start_epoch=start_epoch)

    if args.distributed:
        cleanup()
    print(f'Process {rank} done!...')

if __name__ == '__main__':
    # seed = random.randint(1, 10000)
    # print(f'Random seed :{seed}')
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = True
    parser =ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--num_gpus', type=int, help='Number of GPUs', default=1)
    parser.add_argument('--start', type=int, default=-1, help='start index')
    args = parser.parse_args()
    args.distributed = args.num_gpus > 1

    if args.distributed:
        mp.spawn(subprocess_fn, args=(args, ), nprocs=args.num_gpus)
    else:
        subprocess_fn(0, args)

