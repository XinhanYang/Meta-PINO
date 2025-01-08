import yaml
from argparse import ArgumentParser
from baselines.train_ns import train_deeponet_cp
from baselines.test import test_deeponet_ns, test_deeponet_darcy
from baselines.train_darcy import train_deeponet_darcy
import torch
import numpy as np
import random
from train_utils.distributed import setup, cleanup
import os


def subprocess_fn(args):
    if args.distributed:
        rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
    if args.distributed:
        setup() 
    print(f'Running on rank {rank}')

    seed = 42
    print(f'Random seed :{seed}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    if args.mode == 'train':
        print('Start training DeepONet Cartesian Product')
        if 'name' in config['data'] and config['data']['name'] == 'Darcy':
            train_deeponet_darcy(config)
        else:
            train_deeponet_cp(config, args)
    else:
        print('Start testing DeepONet Cartesian Product')
        if 'name' in config['data'] and config['data']['name'] == 'Darcy':
            test_deeponet_darcy(config)
        else:
            test_deeponet_ns(config)
    print('Done!')

if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--mode', type=str, default='train', help='Train or test')
    parser.add_argument('--num_gpus', type=int, default='1', help='Train or test')
    args = parser.parse_args()

    args.distributed = args.num_gpus > 1

    if args.distributed:
        subprocess_fn(args)
    else:
        subprocess_fn(args)