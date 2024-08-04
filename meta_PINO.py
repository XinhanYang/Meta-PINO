import yaml
from argparse import ArgumentParser
from time import time
from timeit import default_timer
from datetime import datetime, timedelta
from tqdm import tqdm
import csv
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
from train_utils.datasets import NSLoader
from train_utils.data_utils import data_sampler
from train_utils.losses import get_forcing
from train_utils.distributed import setup, cleanup, reduce_loss_dict
from train_utils.losses import LpLoss, PINO_loss3d, get_forcing
from train_utils.utils import save_checkpoint_meta
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from models import FNO3d, FNO2d

import warnings

warnings.filterwarnings("ignore", message=".*functorch.vjp.*")
warnings.filterwarnings("ignore", message=".*functorch.grad.*")

class InnerNet(
    ImplicitMetaGradientModule,
    linear_solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0),
):
    def __init__(self,
                meta_net,
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
                xy_weight,
                ic_weight,
                f_weight
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
        self.xy_weight = xy_weight
        self.ic_weight = ic_weight
        self.f_weight = f_weight
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p1, p2 in zip(self.parameters(), self.meta_parameters()):
                p1.data.copy_(p2.data)
                p1.detach_().requires_grad_()

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 5), "constant", 0)
        return self.net(x)

    def objective(self, x, y):
        zero = torch.zeros(1).to(self.rank)
        out = self(x).reshape(1, self.S, self.S, self.T + 5)
        out = out[..., :-5]
        x = x[:, :, :, 0, -1]
        loss_l2 = self.loss_fn(out.view(1, self.S, self.S, self.T), y.view(1, self.S, self.S, self.T))

        if self.ic_weight != 0 or self.f_weight != 0:
            loss_ic, loss_f = PINO_loss3d(out.view(1, self.S, self.S, self.T), x, self.forcing, self.v, self.t_interval)
        else:
            loss_ic, loss_f = zero, zero

        total_loss = loss_l2 * self.xy_weight + loss_f * self.f_weight + loss_ic * self.ic_weight

        regularization_loss = 0
        for p1, p2 in zip(self.parameters(), self.meta_parameters()):
            diff = p1 - p2
            diff_norm = torch.norm(diff)
            regularization_loss += 0.5 * self.reg_param * diff_norm**2
        return total_loss + regularization_loss

    def solve(self, x, y):
        params = tuple(self.parameters())
        inner_optim = torchopt.SGD(params, lr=self.inner_lr)
        with torch.enable_grad():
            # Temporarily enable gradient computation for conducting the optimization
            for _ in range(self.n_inner_iter):
                loss = self.objective(x, y)
                inner_optim.zero_grad()
                loss.backward(inputs=params)
                inner_optim.step()
        return self



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
        ckpt = torch.load(ckpt_path)
        meta_net.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    if args.distributed:
        meta_net = DDP(meta_net, device_ids=[rank], broadcast_buffers=False)

    forcing = get_forcing(loader.S).to(rank)

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_net.train()
    meta_lr = config['train']['meta_lr']
    meta_opt = torchopt.Adam(meta_net.parameters(), lr=meta_lr)

    train(meta_net,
        loader,
        train_loader,
        meta_opt,
        forcing, config,
        rank,
        log=args.log,
        )
    
    test(meta_net,
         loader,
         test_loader,
         config,
         rank,
         use_tqdm=True)

def train(meta_net,
        loader,
        train_loader,
        meta_opt,
        forcing,
        config,
        rank,
        log,
        use_tqdm=True,
        profile=False):

    # data parameters
    v = 1 / config['data']['Re']
    S, T = loader.S, loader.T
    t_interval = config['data']['time_interval']

    # training settings
    batch_size = config['train']['batchsize']
    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']
    xy_weight = config['train']['xy_loss']
    inner_lr = config['train']['inner_lr']

    batch_size = config['train']['batchsize']  # Assuming batch_size is defined in the config
    n_inner_iter = config['train']['inner_steps']
    reg_param = config['train']['reg_params']
    loss_fn = LpLoss(size_average=True)

    pbar = range(config['train']['epochs'])
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


    zero = torch.zeros(1).to(rank)

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
        
        inner_nets = [InnerNet(meta_net,
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
            xy_weight,
            ic_weight,
            f_weight) for _ in range(batch_size)]

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(rank), y_batch.to(rank)

            total_losses = 0

            meta_opt.zero_grad()

            for i in range(batch_size):
                # Extract individual samples from the batch
                x_instance = x_batch[i].unsqueeze(0)
                y_instance = y_batch[i].unsqueeze(0)
                inner_net = inner_nets[i]
                inner_net.reset_parameters()
                optimal_inner_net = inner_net.solve(x_instance, y_instance)

                out_instance = optimal_inner_net(x_instance).reshape(1, S, S, T + 5)
                out = out_instance[..., :-5]
                x_instance = x_instance[:, :, :, 0, -1]
                loss_l2 = loss_fn(out.view(1, S, S, T), y_instance.view(1, S, S, T))

                if ic_weight != 0 or f_weight != 0:
                    loss_ic, loss_f = PINO_loss3d(out.view(1, S, S, T), x_instance, forcing, v, t_interval)
                else:
                    loss_ic, loss_f = zero, zero

                total_loss = loss_l2 * xy_weight + loss_f * f_weight + loss_ic * ic_weight
                
                total_losses += total_loss

                loss_dict['total_loss'] += total_loss
                loss_dict['loss_l2'] += loss_l2
                loss_dict['loss_f'] += loss_f
                loss_dict['loss_ic'] += loss_ic

            total_losses.backward()
            meta_opt.step()
        
        epoch_time = time() - epoch_start_time
        cumulative_time += epoch_time

        if rank == 0 and profile:
            torch.cuda.synchronize()
            t2 = default_timer()
            log_dict['Time cost'] = t2 - t1

        loss_reduced = reduce_loss_dict(loss_dict)

        loss_ic = loss_reduced['loss_ic'].item() / len(train_loader)
        loss_f = loss_reduced['loss_f'].item() / len(train_loader)
        total_loss = loss_reduced['total_loss'].item() / len(train_loader)
        loss_l2 = loss_reduced['loss_l2'].item() / len(train_loader)
        log_dict = {
            'Train f error': loss_f,
            'Train L2 error': loss_ic,
            'Total loss': total_loss,
            'Train L2 error': loss_l2
            }
        
        if rank == 0:
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Epoch: {ep+1}; '
                        f'Train f error: {loss_f:.5f}; Ic error: {loss_ic:.5f}. '
                        f'Total error: {total_loss:.5f}; l2 error: {loss_l2:.5f}'
                    )
                )

            with open(log_file, 'a') as f:
                print(
                f"Epoch: {ep+1}; ",
                f"Train IC Loss: {loss_ic:.5f}; ",
                f"Train F Loss: {loss_f:.5f}; ",
                f"Train Loss: {total_loss:.5f}; ",
                f"Train L2 Error: {loss_l2:.5f}; ",
                f"Epoch Time: {str(timedelta(seconds=epoch_time))}; ",
                f"Cumulative Time: {str(timedelta(seconds=cumulative_time))}",
                file=f
            )
        if wandb and log:
            wandb.log(log_dict)

    if rank == 0:
        save_checkpoint_meta(config['train']['save_dir'],
                        config['train']['save_name'],
                        meta_net, meta_opt)

def test(meta_net,
         loader,
         test_loader,
         config,
         rank,
         use_tqdm=True):

    # 数据参数
    S, T = loader.S, loader.T

    # 测试设置
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
        batch_size = x_batch.size(0)

        with torch.no_grad():
            x_in = F.pad(x_batch, (0, 0, 0, 5), "constant", 0)
            out_batch = meta_net(x_in).reshape(batch_size, S, S, T + 5)
            out_batch = out_batch[..., :-5]
            x_batch = x_batch[:, :, :, 0, -1]

            loss_l2 = loss_fn(out_batch.view(batch_size, S, S, T), y_batch.view(batch_size, S, S, T))
            total_l2_loss += loss_l2.item() * batch_size
            total_samples += batch_size

    final_l2_loss = total_l2_loss / total_samples
    print(f'Final Test L2 Loss: {final_l2_loss:.5f}')
    return final_l2_loss

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