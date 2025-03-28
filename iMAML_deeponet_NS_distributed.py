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
from torchopt.diff.implicit import ImplicitMetaGradientModule

from torch.utils.data import DataLoader
from baselines.data import DeepONetCPNS
from train_utils.data_utils import data_sampler
from train_utils.losses import get_forcing
from train_utils.distributed import setup, cleanup, reduce_loss_dict
from train_utils.losses import LpLoss, PINO_loss3d, get_forcing
from train_utils.utils import save_checkpoint_meta
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from baselines.model import DeepONetCP

import warnings

warnings.filterwarnings("ignore", message=".*functorch.vjp.*")
warnings.filterwarnings("ignore", message=".*functorch.grad.*")

class InnerNet(
    ImplicitMetaGradientModule,
    linear_solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0),
):
    def __init__(self,
                meta_net,
                dataset,
                loss_fn,
                inner_lr,
                n_inner_iter,
                reg_param,
                batch_size,
                S,
                T,
                rank,
                forcing,
                t_interval,
                v,
                inner_ic_weight,
                inner_f_weight
                ):
        super().__init__()
        self.meta_net = meta_net
        self.net = torchopt.module_clone(meta_net, by='deepcopy', detach_buffers=True)
        self.loss_fn = loss_fn
        self.inner_lr = inner_lr
        self.n_inner_iter = n_inner_iter
        self.reg_param = reg_param
        self.batch_size = batch_size
        self.S = S
        self.T = T
        self.rank = rank
        self.forcing = forcing
        self.t_interval = t_interval
        self.v = v
        self.ic_weight = inner_ic_weight
        self.f_weight = inner_f_weight
        self.current_iter = 0  # Initialize the inner loop counter
        self.grid = dataset.xyt
        self.reset_parameters()

        self.instance_inner_losses = {
            'loss_l2': [0.0] * self.n_inner_iter,
            'loss_ic': [0.0] * self.n_inner_iter,
            'loss_f': [0.0] * self.n_inner_iter,
            'pinn_loss' : [0.0] * self.n_inner_iter,
            'regularization_loss': [0.0] * self.n_inner_iter,
            'total_loss': [0.0] * self.n_inner_iter,
        }

    def reset_parameters(self):
        with torch.no_grad():
            for p1, p2 in zip(self.parameters(), self.meta_parameters()):
                p1.data.copy_(p2.data)
                p1.detach_().requires_grad_()

        self.instance_inner_losses = {
            'loss_l2': [0.0] * self.n_inner_iter,
            'loss_ic': [0.0] * self.n_inner_iter,
            'loss_f': [0.0] * self.n_inner_iter,
            'pinn_loss' : [0.0] * self.n_inner_iter,
            'regularization_loss': [0.0] * self.n_inner_iter,
            'total_loss': [0.0] * self.n_inner_iter,
        }

    def forward(self, x, grid):
        return self.net(x, grid)

    def objective(self, x, grid, y):
        out = self(x, grid)
        out = out.reshape(1, self.S, self.S, self.T)
        x = x.reshape(1, self.S, self.S)
    
        loss_ic, loss_f = PINO_loss3d(out, x, self.forcing, self.v, self.t_interval)

        pinn_loss = loss_f * self.f_weight + loss_ic * self.ic_weight
        loss_l2 = self.loss_fn(out, y.reshape(1, self.S, self.S, self.T))

        regularization_loss = 0
        for p1, p2 in zip(self.parameters(), self.meta_parameters()):
            diff = p1 - p2
            diff_norm = torch.norm(diff)
            regularization_loss += 0.5 * self.reg_param * diff_norm**2

        total_loss = pinn_loss + regularization_loss
        self.instance_inner_losses['loss_ic'][self.current_iter] += loss_ic.item()
        self.instance_inner_losses['loss_f'][self.current_iter] += loss_f.item()
        self.instance_inner_losses['loss_l2'][self.current_iter] += loss_l2.item()
        self.instance_inner_losses['pinn_loss'][self.current_iter] += pinn_loss.item()
        self.instance_inner_losses['total_loss'][self.current_iter] += total_loss.item()
        self.instance_inner_losses['regularization_loss'][self.current_iter] += regularization_loss.item()

        return total_loss

    def solve(self, x, grid, y):
        params = tuple(self.parameters())
        inner_optim = torch.optim.Adam(params, lr=self.inner_lr)
        with torch.enable_grad():
            for self.current_iter in range(self.n_inner_iter):
                loss = self.objective(x, grid, y)
                inner_optim.zero_grad()
                loss.backward(inputs=params)
                inner_optim.step()
        return self

def subprocess_fn(args):

    if args.distributed:
        rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
    if args.distributed:
        setup() 
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

    dataset = DeepONetCPNS(datapath=data_config['datapath'],
                           nx=data_config['nx'], nt=data_config['nt'],
                           sub=data_config['sub'], sub_t=data_config['sub_t'],
                           offset=data_config.get('offset', 0),
                           num=data_config['n_sample'],
                           t_interval=data_config['time_interval'])
    
    trainset, valset, testset = dataset.split_dataset(data_config['n_sample'], 
                                                    offset=data_config['offset'], 
                                                    test_ratio=data_config['test_ratio'],
                                                    val_ratio=data_config.get('val_ratio', 0.1))
    
    train_loader = DataLoader(trainset, batch_size=config['train']['batchsize'],
                              sampler=data_sampler(trainset,
                                                   shuffle=data_config['shuffle'],
                                                   distributed=args.distributed),
                              drop_last=True)
    
    val_loader = DataLoader(valset, batch_size=config['train']['batchsize'],
                            sampler=data_sampler(valset,
                                                shuffle=False,
                                                distributed=args.distributed),
                            drop_last=False)
                            
    test_loader = DataLoader(testset, batch_size=config['train']['batchsize'],
                             sampler=data_sampler(testset,
                                                  shuffle=False,
                                                  distributed=args.distributed),
                             drop_last=False)

    # construct model
    u0_dim = dataset.S ** 2
    activation = config['model']['activation']
    normalize = config['model']['normalize']
    meta_net = DeepONetCP(branch_layer=[u0_dim] + config['model']['branch_layers'],
                          trunk_layer=[3] + config['model']['trunk_layers'],
                          nonlinearity = activation,
                          normalize=normalize).to(rank)

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

    forcing = get_forcing(dataset.S).to(rank)

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_net.train()
    meta_lr = config['train']['meta_lr']
    meta_opt = torchopt.Adam(meta_net.parameters(), lr=meta_lr)

    train(meta_net,
        dataset,
        train_loader,
        val_loader,
        meta_opt,
        forcing, 
        config,
        rank,
        start_epoch = start_epoch)
    
    # test(meta_net,
    #      dataset,
    #      test_loader,
    #      config,
    #      rank,
    #      use_tqdm=True)

def train(meta_net,
        dataset,
        train_loader,
        val_loader,
        meta_opt,
        forcing,
        config,
        rank,
        start_epoch=0,
        use_tqdm=True,
        profile=False):

    # data parameters
    v = 1 / config['data']['Re']
    S, T = dataset.S, dataset.T
    t_interval = config['data']['time_interval']

    # training settings
    batch_size = config['train']['batchsize']
    data_weight = config['train']['data_loss']
    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']
    inner_ic_weight = config['train']['inner_ic_loss']
    inner_f_weight = config['train']['inner_f_loss']
    inner_lr = config['train']['inner_lr']

    n_inner_iter = config['train']['inner_steps']
    reg_param = config['train']['reg_params']
    loss_fn = LpLoss(size_average=True)

    if rank == 0 and use_tqdm:
        pbar = tqdm(range(start_epoch, config['train']['epochs']), dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = range(start_epoch, config['train']['epochs'])
    
    cumulative_time = 0  # Initialize cumulative time

    log_file = config['log']['logfile'] 

    if rank == 0:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{config['log']['logfile']}_{current_time}.log"
        with open(log_file, 'w') as f:
            # Convert config dictionary to a pretty-printed string and write it to the file
            config_str = json.dumps(config, indent=4)
            print(f"Configuration:\n{config_str}\n", file=f)
    
    # Calculate error before training.
    min_val_loss = float('inf')
    loss_dict = {'total_loss': 0.0,
                     'loss_ic': 0.0,
                     'loss_f': 0.0,
                     'loss_l2': 0.0}
    
    # Initialize log_dict
    log_dict = {
        'epoch': 0,
        'train_total_loss': 0.0,
        'train_l2_error': 0.0,
        'train_ic_loss': 0.0,
        'train_f_loss': 0.0,
        'val_total_loss': 0.0,
        'val_l2_error': 0.0,
        'val_ic_loss': 0.0,
        'val_f_loss': 0.0,
        'avg_instance_losses': None,
        'epoch_time': 0,
        'cumulative_time': 0
    }
    
    inner_nets = [InnerNet(meta_net,
            dataset,
            loss_fn,
            inner_lr,
            n_inner_iter,
            reg_param,
            batch_size,
            S,
            T,
            rank,
            forcing,
            t_interval,
            v,
            inner_ic_weight,
            inner_f_weight) for _ in range(batch_size)]
        
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(rank), y.to(rank)
            grid = dataset.xyt.to(rank)
            out = meta_net(x, grid).reshape(batch_size, S, S, T)
            x = x.reshape(batch_size, S, S)

            loss_l2 = loss_fn(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T))

            loss_ic, loss_f = PINO_loss3d(out.view(batch_size, S, S, T), x, forcing, v, t_interval)

            total_loss = loss_l2 * data_weight + loss_f * f_weight + loss_ic * ic_weight

            loss_dict['total_loss'] += total_loss
            loss_dict['loss_l2'] += loss_l2
            loss_dict['loss_f'] += loss_f
            loss_dict['loss_ic'] += loss_ic
        
        loss_reduced = reduce_loss_dict(loss_dict)

        loss_start_ic = loss_reduced['loss_ic'].item() / (len(val_loader))
        loss_start_f = loss_reduced['loss_f'].item() / (len(val_loader))
        start_total_loss = loss_reduced['total_loss'].item() / (len(val_loader))
        loss_start_l2 = loss_reduced['loss_l2'].item() / (len(val_loader))

    log_dict.update({
        'epoch': 0,
        'val_total_loss': start_total_loss,
        'val_l2_error': loss_start_l2,
        'val_ic_loss': loss_start_ic,
        'val_f_loss': loss_start_f,
        'avg_instance_losses': None,
        'epoch_time': 0,
        'cumulative_time': 0
    })

    for ep in pbar:
        epoch_start_time = time()  # Start time for the epoch
        train_loss_dict = {'total_loss': 0.0,
                     'loss_ic': 0.0,
                     'loss_f': 0.0,
                     'loss_l2': 0.0}
        val_loss_dict = {'total_loss': 0.0,
                     'loss_ic': 0.0,
                     'loss_f': 0.0,
                     'loss_l2': 0.0}
        log_dict = {}
        if rank == 0 and profile:
                torch.cuda.synchronize()
                t1 = default_timer()

        inner_loss_dict = {
            'loss_l2': [0.0] * n_inner_iter,
            'loss_ic': [0.0] * n_inner_iter,
            'loss_f': [0.0] * n_inner_iter,
            'regularization_loss': [0.0] * n_inner_iter,
            'pinn_loss': [0.0] * n_inner_iter,
            'total_loss': [0.0] * n_inner_iter,
        }
        
        inner_nets = [InnerNet(meta_net,
            dataset,
            loss_fn,
            inner_lr,
            n_inner_iter,
            reg_param,
            batch_size,
            S,
            T,
            rank,
            forcing,
            t_interval,
            v,
            inner_ic_weight,
            inner_f_weight) for _ in range(batch_size)]
        
        # Training phase
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(rank), y_batch.to(rank)
            grid = dataset.xyt.to(rank)

            total_losses = 0

            meta_opt.zero_grad()

            for i in range(batch_size):
                # Extract individual samples from the batch
                x_instance = x_batch[i].unsqueeze(0)
                y_instance = y_batch[i].unsqueeze(0)
                inner_net = inner_nets[i]
                inner_net.reset_parameters()
                optimal_inner_net = inner_net.solve(x_instance, grid, y_instance)

                out_instance = optimal_inner_net(x_instance, grid).reshape(1, S, S, T)
                x_instance = x_instance.reshape(1, S, S)
                loss_l2 = loss_fn(out_instance, y_instance.reshape(1, S, S, T))

                loss_ic, loss_f = PINO_loss3d(out_instance, x_instance, forcing, v, t_interval)

                total_loss = loss_l2 * data_weight + loss_ic * ic_weight + loss_f * f_weight
                
                total_losses += total_loss

                train_loss_dict['total_loss'] += total_loss
                train_loss_dict['loss_l2'] += loss_l2
                train_loss_dict['loss_f'] += loss_f
                train_loss_dict['loss_ic'] += loss_ic

                for j in range(n_inner_iter):
                    inner_loss_dict['loss_l2'][j] += inner_net.instance_inner_losses['loss_l2'][j]
                    inner_loss_dict['loss_ic'][j] += inner_net.instance_inner_losses['loss_ic'][j]
                    inner_loss_dict['loss_f'][j] += inner_net.instance_inner_losses['loss_f'][j]
                    inner_loss_dict['regularization_loss'][j] += inner_net.instance_inner_losses['regularization_loss'][j]
                    inner_loss_dict['pinn_loss'][j] += inner_net.instance_inner_losses['pinn_loss'][j]
                    inner_loss_dict['total_loss'][j] += inner_net.instance_inner_losses['total_loss'][j]
                
                torch.cuda.synchronize()

            total_losses /= batch_size
            total_losses.backward()
            meta_opt.step()

        # Validation phase
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(rank), y_batch.to(rank)
                grid = dataset.xyt.to(rank)
                out = meta_net(x_batch, grid).reshape(batch_size, S, S, T)
                x = x_batch.reshape(batch_size, S, S)

                loss_l2 = loss_fn(out.view(batch_size, S, S, T), y_batch.view(batch_size, S, S, T))

                loss_ic, loss_f = PINO_loss3d(out.view(batch_size, S, S, T), x, forcing, v, t_interval)

                total_loss = loss_l2 * data_weight + loss_f * f_weight + loss_ic * ic_weight

                val_loss_dict['total_loss'] += total_loss
                val_loss_dict['loss_l2'] += loss_l2
                val_loss_dict['loss_f'] += loss_f
                val_loss_dict['loss_ic'] += loss_ic
        
        epoch_time = time() - epoch_start_time
        cumulative_time += epoch_time

        if rank == 0 and profile:
            torch.cuda.synchronize()
            t2 = default_timer()
            log_dict['Time cost'] = t2 - t1

        train_loss_reduced = reduce_loss_dict(train_loss_dict)
        val_loss_reduced = reduce_loss_dict(val_loss_dict)
        inner_loss_dict_reduced = reduce_loss_dict(inner_loss_dict)

        train_loss_ic = train_loss_reduced['loss_ic'].item() / (len(train_loader)*batch_size)
        train_loss_f = train_loss_reduced['loss_f'].item() / (len(train_loader)*batch_size)
        train_total_loss = train_loss_reduced['total_loss'].item() / (len(train_loader)*batch_size)
        train_loss_l2 = train_loss_reduced['loss_l2'].item() / (len(train_loader)*batch_size)

        val_loss_ic = val_loss_reduced['loss_ic'].item() / (len(val_loader)*batch_size)
        val_loss_f = val_loss_reduced['loss_f'].item() / (len(val_loader)*batch_size)
        val_total_loss = val_loss_reduced['total_loss'].item() / (len(val_loader)*batch_size)
        val_loss_l2 = val_loss_reduced['loss_l2'].item() / (len(val_loader)*batch_size)

        avg_instance_loss = {
            f'avg_total_loss_iter_{j+1}': inner_loss_dict_reduced['total_loss'][j] / (len(train_loader) * batch_size)
            for j in range(n_inner_iter)
        }
        avg_instance_loss.update({
            f'avg_loss_ic_iter_{j+1}': inner_loss_dict_reduced['loss_ic'][j] / (len(train_loader) * batch_size)
            for j in range(n_inner_iter)
        })
        avg_instance_loss.update({
            f'avg_loss_f_iter_{j+1}': inner_loss_dict_reduced['loss_f'][j] / (len(train_loader) * batch_size)
            for j in range(n_inner_iter)
        })
        avg_instance_loss.update({
            f'avg_regularization_loss_iter_{j+1}': inner_loss_dict_reduced['regularization_loss'][j] / (len(train_loader) * batch_size)
            for j in range(n_inner_iter)
        })
        avg_instance_loss.update({
            f'avg_pinn_loss_iter_{j+1}': inner_loss_dict_reduced['pinn_loss'][j] / (len(train_loader) * batch_size)
            for j in range(n_inner_iter)
        })
        avg_instance_loss.update({
            f'avg_loss_l2_iter_{j+1}': inner_loss_dict_reduced['loss_l2'][j] / (len(train_loader) * batch_size)
            for j in range(n_inner_iter)
        })

        log_dict.update({
            'epoch': ep + 1,
            'train_total_loss': train_total_loss,
            'train_l2_error': train_loss_l2,
            'train_ic_loss': train_loss_ic,
            'train_f_loss': train_loss_f,
            'val_total_loss': val_total_loss,
            'val_l2_error': val_loss_l2,
            'val_ic_loss': val_loss_ic,
            'val_f_loss': val_loss_f,
            'avg_instance_losses': avg_instance_loss,
            'epoch_time': str(timedelta(seconds=epoch_time)),
            'cumulative_time': str(timedelta(seconds=cumulative_time))
        })
        
        if rank == 0:
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Epoch: {ep+1}; '
                        f'Train Total: {train_total_loss:.5f}; Val Total: {val_total_loss:.5f}; '
                        f'Train L2: {train_loss_l2:.5f}; Val L2: {val_loss_l2:.5f}'
                    )
                )

            with open(log_file, 'a') as f:
                f.write(json.dumps(log_dict, indent=4) + '\n')


        if rank == 0 and val_total_loss < min_val_loss:
            min_val_loss = val_total_loss
            save_checkpoint_meta(ep,
                config['train']['save_dir'],
                config['train']['save_name'],
                meta_net, meta_opt)

def test(meta_net,
         dataset,
         test_loader,
         config,
         rank,
         use_tqdm=True):

    S, T = dataset.S, dataset.T

    batch_size = config['test']['batchsize']
    loss_fn = LpLoss(size_average=True)

    total_l2_loss = 0.0
    total_samples = 0

    if use_tqdm:
        pbar = tqdm(test_loader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = test_loader

    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(rank), y_batch.to(rank)
        grid = dataset.xyt.to(rank)
        batch_size = x_batch.size(0)

        with torch.no_grad():
            out_batch = meta_net(x_batch, grid).reshape(batch_size, S, S, T)
            x_batch = x_batch.reshape(batch_size, S, S)

            loss_l2 = loss_fn(out_batch.view(batch_size, S, S, T), y_batch.view(batch_size, S, S, T))
            total_l2_loss += loss_l2.item() * batch_size
            total_samples += batch_size

    final_l2_loss = total_l2_loss / total_samples
    print(f'Final Test L2 Loss: {final_l2_loss:.5f}')
    return final_l2_loss

if __name__ == '__main__':
    parser =ArgumentParser(description='iMAML with DeepONet')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--num_gpus', type=int, help='Number of GPUs', default=1)
    parser.add_argument('--start', type=int, default=-1, help='start index')
    args = parser.parse_args()
    args.distributed = args.num_gpus > 1

    subprocess_fn(args)