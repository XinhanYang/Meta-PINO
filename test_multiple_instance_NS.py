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

from train_utils import Adam
from torch.utils.data import DataLoader
from train_utils.datasets import NSLoader
from train_utils.data_utils import data_sampler
from train_utils.losses import get_forcing
from train_utils.distributed import setup, cleanup, reduce_loss_dict
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

    test_loader = DataLoader(testset, batch_size=config['train']['batchsize'],
                             sampler=data_sampler(testset,
                                                  shuffle=False,
                                                  distributed=args.distributed),
                             drop_last=False)


    # construct model
    meta_net = FNO3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers']).to(rank)
    
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location={'cuda:%d' % 0: 'cuda:%d' % rank})
            meta_net.load_state_dict(ckpt['model'])
            print('Checkpoint loaded from %s' % ckpt_path)
        else:
            print('No checkpoint found at %s' % ckpt_path)

    if args.distributed:
        meta_net = DDP(meta_net, device_ids=[rank], broadcast_buffers=False)

    forcing = get_forcing(loader.S).to(rank)

    test(meta_net,
         loader,
         test_loader,
         config,
         rank,
         forcing,
         use_tqdm=True)

def test(meta_net, loader, test_loader, config, rank, forcing, use_tqdm=True):
    S, T = loader.S, loader.T
    loss_fn = LpLoss(size_average=True)
    inner_lr = config['train']['inner_lr']
    n_inner_iter = config['train']['epochs']
    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']
    xy_weight = config['train']['data_loss']
    reg_weight = config['train']['reg_loss']
    t_interval = config['data']['time_interval']
    v = 1 / config['data']['Re']
    zero = torch.tensor(0.0, device=rank)

    # Initialize log file
    if rank == 0:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{config['log']['logfile']}_{current_time}.log"
        with open(log_file, 'w') as f:
            config_str = json.dumps(config, indent=4)
            f.write(f"Configuration:\n{config_str}\n\n")

    all_loss_logs = {
        'total_loss': [0.0] * (n_inner_iter + 1),
        'loss_l2': [0.0] * (n_inner_iter + 1),
        'loss_ic': [0.0] * (n_inner_iter + 1),
        'loss_f': [0.0] * (n_inner_iter + 1),
        'iter_time': [0.0] * (n_inner_iter + 1),  # Start with 0 at index 0
        'regularization_loss': [0.0] * (n_inner_iter + 1),
        'cumulative_time': [0.0] * (n_inner_iter + 1)
    }
    total_samples = 0

    if use_tqdm:
        pbar = tqdm(test_loader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = test_loader

    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(rank), y_batch.to(rank)
        batch_size = x_batch.size(0)

        for i in range(batch_size):
            x_instance = x_batch[i].unsqueeze(0)
            y_instance = y_batch[i].unsqueeze(0)

            model = FNO3d(
                modes1=config['model']['modes1'],
                modes2=config['model']['modes2'],
                modes3=config['model']['modes3'],
                fc_dim=config['model']['fc_dim'],
                layers=config['model']['layers']
            ).to(rank)

            if 'ckpt' in config['train']:
                ckpt_path = config['train']['ckpt']
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location={'cuda:%d' % 0: 'cuda:%d' % rank})
                    model.load_state_dict(ckpt['model'])
                    print(f'Checkpoint loaded from {ckpt_path}')
                else:
                    print(f'No checkpoint found at {ckpt_path}')
            else:
                model.load_state_dict(meta_net.state_dict())

            optimizer = torch.optim.Adam(model.parameters(), lr=inner_lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=config['train']['milestones'],
                gamma=config['train']['scheduler_gamma']
            )

            # Error log including both losses and times
            error_log = {
                'total_loss': [],
                'loss_l2': [],
                'loss_ic': [],
                'loss_f': [],
                'regularization_loss': [],
                'iter_time': [],
                'cumulative_time': []
            }
            cumulative_time = 0.0

            # Perform inner loop optimization
            for iter in range(n_inner_iter):
                start_time = time()

                optimizer.zero_grad()
                x_in = F.pad(x_instance, (0, 0, 0, 5), "constant", 0)
                out = model(x_in).reshape(1, S, S, T + 5)
                out = out[..., :-5]
                x_instance_padded = x_instance[:, :, :, 0, -1]

                loss_l2 = loss_fn(out.view(1, S, S, T), y_instance.view(1, S, S, T))

                if ic_weight != 0 or f_weight != 0:
                    loss_ic, loss_f = PINO_loss3d(out.view(1, S, S, T), x_instance_padded, forcing, v, t_interval)
                else:
                    loss_ic, loss_f = zero, zero

                total_loss = loss_l2 * xy_weight + loss_f * f_weight + loss_ic * ic_weight

                # Adding regularization loss
                regularization_loss = 0
                for p1, p2 in zip(model.parameters(), meta_net.parameters()):
                    diff = p1 - p2
                    diff_norm = torch.norm(diff)
                    regularization_loss += 0.5 * reg_weight * diff_norm**2

                total_loss += regularization_loss

                # Record loss before parameter update
                error_log['total_loss'].append(total_loss.item())
                error_log['loss_l2'].append(loss_l2.item())
                error_log['loss_ic'].append(loss_ic.item())
                error_log['loss_f'].append(loss_f.item())
                error_log['regularization_loss'].append(regularization_loss.item())

                # Update model parameters
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                iter_time = time() - start_time
                cumulative_time += iter_time

                # Record time after parameter update
                error_log['iter_time'].append(iter_time)
                error_log['cumulative_time'].append(cumulative_time)

            # Calculate and record final loss after the last iteration
            with torch.no_grad():
                out = model(x_in).reshape(1, S, S, T + 5)
                out = out[..., :-5]
                final_loss_l2 = loss_fn(out.view(1, S, S, T), y_instance.view(1, S, S, T))

                if ic_weight != 0 or f_weight != 0:
                    final_loss_ic, final_loss_f = PINO_loss3d(out.view(1, S, S, T), x_instance_padded, forcing, v, t_interval)
                else:
                    final_loss_ic, final_loss_f = zero, zero

                final_total_loss = final_loss_l2 * xy_weight + final_loss_f * f_weight + final_loss_ic * ic_weight

                # Record final loss at n_inner_iter + 1
                all_loss_logs['total_loss'][-1] += final_total_loss.item()
                all_loss_logs['loss_l2'][-1] += final_loss_l2.item()
                all_loss_logs['loss_ic'][-1] += final_loss_ic.item()
                all_loss_logs['loss_f'][-1] += final_loss_f.item()

            # Accumulate loss and time logs for all samples at each inner iteration
            for key in all_loss_logs:
                if key not in ['iter_time', 'cumulative_time']:  # Time logs are handled separately
                    all_loss_logs[key][0:n_inner_iter] = [
                        all_loss_logs[key][j] + error_log[key][j] for j in range(0, n_inner_iter)
                    ]

            # Accumulate time logs for all samples at each inner iteration
            all_loss_logs['iter_time'][1:n_inner_iter+1] = [
                all_loss_logs['iter_time'][j] + error_log['iter_time'][j-1] for j in range(1, n_inner_iter+1)
            ]
            all_loss_logs['cumulative_time'][1:n_inner_iter+1] = [
                all_loss_logs['cumulative_time'][j] + error_log['cumulative_time'][j-1] for j in range(1, n_inner_iter+1)
            ]

            total_samples += 1

    # Compute average loss and time
    avg_loss_logs = {key: [val / total_samples for val in all_loss_logs[key]] for key in all_loss_logs}

    # Save average loss log to file
    if rank == 0:
        with open(log_file, 'a') as f:
            f.write(f"Average Loss Logs:\n{json.dumps(avg_loss_logs, indent=4)}\n")

    # Print and return average L2 loss and all error logs
    initial_l2_loss = avg_loss_logs['loss_l2'][0]
    print(f'Initial L2 Loss: {initial_l2_loss:.5f}')

    final_l2_loss = avg_loss_logs['loss_l2'][-1]
    print(f'Final Test L2 Loss: {final_l2_loss:.5f}')

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