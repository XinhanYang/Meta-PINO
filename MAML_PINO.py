import yaml
from argparse import ArgumentParser
from time import time
from timeit import default_timer
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import json

try:
    import wandb
except ImportError:
    wandb = None
    

import numpy as np
import torch
import torch.nn.functional as F

import torchopt

from torch.utils.data import DataLoader
from train_utils.datasets import NSLoader
from train_utils.data_utils import data_sampler
from train_utils.losses import get_forcing
from train_utils.distributed import setup, reduce_loss_dict
from train_utils.losses import LpLoss, PINO_loss3d, get_forcing
from train_utils.utils import save_checkpoint_meta
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.optim as optim

from models import FNO3d, FNO2d

import warnings

warnings.filterwarnings("ignore", message=".*functorch.vjp.*")
warnings.filterwarnings("ignore", message=".*functorch.grad.*")


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
    torch.backends.cudnn.benchmark = True

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
    
    trainset, testset = loader.split_dataset(data_config['n_sample'], data_config['offset'], data_config['test_ratio'])
    train_loader = DataLoader(trainset, batch_size=config['train']['batchsize'],
                              sampler=data_sampler(trainset,
                                                   shuffle=data_config['shuffle'],
                                                   distributed=args.distributed),
                              drop_last=True)

    # construct model
    meta_net = FNO3d(modes1=config['model']['modes1'],
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
                    meta_net.load_state_dict(ckpt['model'])
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
        meta_net = DDP(meta_net, device_ids=[rank], broadcast_buffers=False)

    forcing = get_forcing(loader.S).to(rank)

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_net.train()
    meta_lr = config['train']['meta_lr']
    meta_opt = optim.Adam(meta_net.parameters(), lr=meta_lr)

    train(meta_net,
        loader,
        train_loader,
        meta_opt,
        forcing, 
        config,
        rank,
        log=args.log,
        start_epoch = start_epoch)

def train(meta_net,
        loader,
        train_loader,
        meta_opt,
        forcing,
        config,
        rank,
        log,
        start_epoch=0,
        use_tqdm=True,
        profile=False):

    # data parameters
    v = 1 / config['data']['Re']
    S, T = loader.S, loader.T
    t_interval = config['data']['time_interval']

    # training settings
    batch_size = config['train']['batchsize']
    inner_ic_weight = config['train']['inner_ic_loss']
    inner_f_weight = config['train']['inner_f_loss']
    inner_lr = config['train']['inner_lr']
    data_weight = config['train']['data_loss']
    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']

    n_inner_iter = config['train']['inner_steps']
    loss_fn = LpLoss(size_average=True)

    pbar = range(start_epoch, config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.05)

    cumulative_time = 0  # Initialize cumulative time

    log_file = config['log']['logfile'] 

    if rank == 0:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{config['log']['logfile']}_{current_time}.log"
        with open(log_file, 'w') as f:
            # Convert config dictionary to a pretty-printed string and write it to the file
            config_str = json.dumps(config, indent=4)
            print(f"Configuration:\n{config_str}\n", file=f)
    
    min_l2_loss = 1000

    for ep in pbar:
        epoch_start_time = time()  # Start time for the epoch
        loss_dict = {'total_loss': 0.0,
                     'loss_ic': 0.0,
                     'loss_f': 0.0,
                     'loss_l2': 0.0}
        log_dict = {}
        if rank == 0 and profile:
                torch.cuda.synchronize()
                t1 = default_timer()
       
        inner_opt = torchopt.MetaAdam(meta_net, lr=inner_lr)

        instance_inner_losses = {'total_loss': [0.0] * n_inner_iter,
                             'loss_ic': [0.0] * n_inner_iter,
                             'loss_f': [0.0] * n_inner_iter,
                             'loss_l2': [0.0] * n_inner_iter}


        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(rank), y_batch.to(rank)

            total_losses = 0

            meta_opt.zero_grad()
            net_state_dict = torchopt.extract_state_dict(meta_net, by='reference', detach_buffers=True)
            optim_state_dict = torchopt.extract_state_dict(inner_opt, by='reference')

            for i in range(batch_size):
                # Extract individual samples from the batch
                x_instance = x_batch[i].unsqueeze(0)
                y_instance = y_batch[i].unsqueeze(0)
                x_in = F.pad(x_instance, (0, 0, 0, 5), "constant", 0)
                x_instance = x_instance[:, :, :, 0, -1]

                for inner_iter in range(n_inner_iter):
                    out_instance = meta_net(x_in).reshape(1, S, S, T + 5)
                    out = out_instance[..., :-5]

                    loss_l2 = loss_fn(out.view(1, S, S, T), y_instance.view(1, S, S, T))
                    loss_ic, loss_f = PINO_loss3d(out.view(1, S, S, T), x_instance, forcing, v, t_interval)

                    total_loss = loss_f * inner_f_weight + loss_ic * inner_ic_weight
                    
                    instance_inner_losses['loss_l2'][inner_iter] += loss_l2.item()
                    instance_inner_losses['loss_ic'][inner_iter] += loss_ic.item()
                    instance_inner_losses['loss_f'][inner_iter] += loss_f.item()
                    instance_inner_losses['total_loss'][inner_iter] += total_loss.item()

                    inner_opt.step(total_loss)

                    torch.cuda.empty_cache()

                    if inner_iter == 1:
                        total_losses += loss_l2
                
                out_instance = meta_net(x_in).reshape(1, S, S, T + 5)
                out = out_instance[..., :-5]
                loss_l2 = loss_fn(out.view(1, S, S, T), y_instance.view(1, S, S, T))

                loss_ic, loss_f = PINO_loss3d(out.view(1, S, S, T), x_instance, forcing, v, t_interval)

                total_loss = loss_l2 * data_weight + loss_ic * ic_weight + loss_f * f_weight

                total_losses += total_loss
                
                loss_dict['total_loss'] += total_loss
                loss_dict['loss_l2'] += loss_l2
                loss_dict['loss_f'] += loss_f
                loss_dict['loss_ic'] += loss_ic

                torchopt.recover_state_dict(meta_net, net_state_dict)
                torchopt.recover_state_dict(inner_opt, optim_state_dict)

            total_losses.backward()
            meta_opt.step()

            del out_instance, out, loss_l2, loss_ic, loss_f, total_loss, total_losses
            torch.cuda.empty_cache()

        epoch_time = time() - epoch_start_time
        cumulative_time += epoch_time

        if rank == 0 and profile:
            torch.cuda.synchronize()
            t2 = default_timer()
            log_dict['Time cost'] = t2 - t1

        loss_reduced = reduce_loss_dict(loss_dict)

        loss_ic = loss_reduced['loss_ic'].item() / (len(train_loader)*batch_size)
        loss_f = loss_reduced['loss_f'].item() / (len(train_loader)*batch_size)
        total_loss = loss_reduced['total_loss'].item() / (len(train_loader)*batch_size)
        loss_l2 = loss_reduced['loss_l2'].item() / (len(train_loader)*batch_size)
        avg_instance_loss = {
            f'avg_total_loss_iter_{i+1}': instance_inner_losses['total_loss'][i] / (len(train_loader) * batch_size)
            for i in range(n_inner_iter)
        }
        avg_instance_loss.update({
            f'avg_loss_ic_iter_{i+1}': instance_inner_losses['loss_ic'][i] / (len(train_loader) * batch_size)
            for i in range(n_inner_iter)
        })
        avg_instance_loss.update({
            f'avg_loss_f_iter_{i+1}': instance_inner_losses['loss_f'][i] / (len(train_loader) * batch_size)
            for i in range(n_inner_iter)
        })
        avg_instance_loss.update({
            f'avg_loss_l2_iter_{i+1}': instance_inner_losses['loss_l2'][i] / (len(train_loader) * batch_size)
            for i in range(n_inner_iter)
        })
        
        # Log dictionary structured for JSON
        log_dict = {
            'epoch': ep + 1,
            'train_total_loss': total_loss,
            'train_l2_error': loss_l2,
            'train_ic_loss': loss_ic,
            'train_f_loss': loss_f,
            'avg_instance_losses': avg_instance_loss,
            'epoch_time': str(timedelta(seconds=epoch_time)),
            'cumulative_time': str(timedelta(seconds=cumulative_time))
        }

        if rank == 0:
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Epoch: {ep+1}; '
                        f'Total error: {total_loss:.5f}; l2 error: {loss_l2:.5f} '
                        f'Train f error: {loss_f:.5f}; Ic error: {loss_ic:.5f}. '
                    )
                )

                
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_dict, indent=4) + '\n')

        if wandb and log:
            wandb.log(log_dict)

        if rank == 0 and loss_l2 < min_l2_loss:
            min_l2_loss = loss_l2
            save_checkpoint_meta(ep,
                config['train']['save_dir'],
                config['train']['save_name'],
                meta_net, meta_opt)

if __name__ == '__main__':
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