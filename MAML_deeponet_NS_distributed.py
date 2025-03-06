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

import torch
import torchopt

from torch.utils.data import DataLoader
from baselines.data import DeepONetCPNS
from train_utils.data_utils import data_sampler
from train_utils.losses import LpLoss, PINO_loss3d, get_forcing
from train_utils.distributed import setup, cleanup, reduce_loss_dict
from train_utils.utils import save_checkpoint_meta
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

from baselines.model import DeepONetCP

import warnings

warnings.filterwarnings("ignore", message=".*functorch.vjp.*")
warnings.filterwarnings("ignore", message=".*functorch.grad.*")


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

    data_config = config['data']
    seed = data_config['seed']
    print(f'Seed: {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    dataset = DeepONetCPNS(datapath=data_config['datapath'],
                           nx=data_config['nx'], nt=data_config['nt'],
                           sub=data_config['sub'], sub_t=data_config['sub_t'],
                           offset=data_config.get('offset', 0),
                           num=data_config['n_sample'],
                           t_interval=data_config['time_interval'])

    train_dataset, val_dataset, test_dataset = dataset.split_dataset(data_config['n_sample'], 
                                                    offset=data_config['offset'], 
                                                    test_ratio=data_config['test_ratio'],
                                                    val_ratio=data_config.get('val_ratio', 0.1))

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batchsize'],
                              sampler=data_sampler(train_dataset,
                                                   shuffle=data_config['shuffle'],
                                                   distributed=args.distributed),
                              drop_last=True)

    grid = dataset.xyt.to(rank)  # (S*S*T, 3)

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
                if 'model' in ckpt:
                    meta_net.load_state_dict(ckpt['model'])
                    print('Model state loaded from %s' % ckpt_path)
                if 'meta_epoch' in ckpt:
                    start_epoch = ckpt['meta_epoch'] + 1
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

    meta_net.train()
    meta_lr = config['train']['meta_lr']
    meta_opt = optim.Adam(meta_net.parameters(), lr=meta_lr)

    train(meta_net,
        dataset, 
        train_loader, 
        meta_opt, 
        forcing, 
        grid, 
        config, 
        rank, 
        start_epoch=start_epoch)

    if args.distributed:
        cleanup()
    print(f'Process {rank} done!...')


def train(meta_net, 
          dataset, 
          train_loader, 
          meta_opt, 
          forcing, 
          grid, 
          config, 
          rank=0, 
          start_epoch=0, 
          use_tqdm=True, 
          profile=False):
    v = 1 / config['data']['Re']
    S, T = dataset.S, dataset.T
    t_interval = config['data']['time_interval']
    batch_size = config['train']['batchsize']
    inner_ic_weight = config['train']['inner_ic_loss']
    inner_f_weight = config['train']['inner_f_loss']
    inner_lr = config['train']['inner_lr']
    data_weight = config['train']['data_loss']
    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']

    n_inner_iter = config['train']['inner_steps']
    loss_fn = LpLoss(size_average=True)

    if rank == 0 and use_tqdm:
        pbar = tqdm(range(start_epoch, config['train']['epochs']), dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = range(start_epoch, config['train']['epochs'])

    cumulative_time = 0
    log_file = config['log']['logfile']

    if rank == 0:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{config['log']['logfile']}_{current_time}.log"
        with open(log_file, 'w') as f:
            config_str = json.dumps(config, indent=4)
            f.write(f"Configuration:\n{config_str}\n")

    min_l2_loss = 1e6

    for ep in pbar:
        epoch_start_time = time()
        loss_dict = {'total_loss': 0.0,
                     'loss_ic': 0.0,
                     'loss_f': 0.0,
                     'loss_l2': 0.0}
        instance_inner_losses = {'total_loss': [0.0] * n_inner_iter,
                                 'loss_ic': [0.0] * n_inner_iter,
                                 'loss_f': [0.0] * n_inner_iter,
                                 'loss_l2': [0.0] * n_inner_iter}

        if rank == 0 and profile:
            torch.cuda.synchronize()
            t1 = default_timer()

        inner_opt = torchopt.MetaAdam(meta_net, lr=inner_lr)

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(rank), y_batch.to(rank)
            total_losses = 0
            meta_opt.zero_grad()

            net_state_dict = torchopt.extract_state_dict(meta_net, by='reference', detach_buffers=True)
            optim_state_dict = torchopt.extract_state_dict(inner_opt, by='reference')

            for i in range(x_batch.size(0)):
                # x: (1, u0_dim)；y: (1, S*S*T)
                x_instance = x_batch[i].unsqueeze(0)
                y_instance = y_batch[i].unsqueeze(0)
                x_ic = x_instance.reshape(1, S, S)

                for inner_iter in range(n_inner_iter):
                    pred = meta_net(x_instance, grid)
                    loss_l2 = loss_fn(pred.view(1, S, S, T), y_instance.view(1, S, S, T))

                    loss_ic, loss_f = PINO_loss3d(pred.view(1, S, S, T), x_ic, forcing, v, t_interval)
                    inner_loss = loss_f * inner_f_weight + loss_ic * inner_ic_weight
                    instance_inner_losses['loss_l2'][inner_iter] += loss_l2.item()
                    instance_inner_losses['loss_ic'][inner_iter] += loss_ic.item()
                    instance_inner_losses['loss_f'][inner_iter] += loss_f.item()
                    instance_inner_losses['total_loss'][inner_iter] += inner_loss.item()

                    inner_opt.step(inner_loss)

                    # if inner_iter == 1:
                    #     total_losses += loss_l2

                pred = meta_net(x_instance, grid)
                loss_l2 = loss_fn(pred.view(1, S, S, T), y_instance.view(1, S, S, T))
                loss_ic, loss_f = PINO_loss3d(pred.view(1, S, S, T), x_ic, forcing, v, t_interval)
                meta_loss = loss_l2
                total_losses += meta_loss

                loss_dict['total_loss'] += meta_loss.item()
                loss_dict['loss_l2'] += loss_l2.item()
                loss_dict['loss_ic'] += loss_ic.item()
                loss_dict['loss_f'] += loss_f.item()

                torchopt.recover_state_dict(meta_net, net_state_dict)
                torchopt.recover_state_dict(inner_opt, optim_state_dict)

            total_losses.backward()
            meta_opt.step()
            torch.cuda.empty_cache()

        epoch_time = time() - epoch_start_time
        cumulative_time += epoch_time

        if rank == 0 and profile:
            torch.cuda.synchronize()
            t2 = default_timer()

        loss_reduced = reduce_loss_dict(loss_dict)
        instance_inner_losses_reduce = reduce_loss_dict(instance_inner_losses)

        avg_loss_ic = loss_reduced['loss_ic'] / (len(train_loader) * batch_size)
        avg_loss_f = loss_reduced['loss_f'] / (len(train_loader) * batch_size)
        avg_total_loss = loss_reduced['total_loss'] / (len(train_loader) * batch_size)
        avg_loss_l2 = loss_reduced['loss_l2'] / (len(train_loader) * batch_size)
        avg_instance_loss = {
            f'avg_total_loss_iter_{i+1}': instance_inner_losses_reduce['total_loss'][i] / (len(train_loader) * batch_size)
            for i in range(n_inner_iter)
        }
        avg_instance_loss.update({
            f'avg_loss_ic_iter_{i+1}': instance_inner_losses_reduce['loss_ic'][i] / (len(train_loader) * batch_size)
            for i in range(n_inner_iter)
        })
        avg_instance_loss.update({
            f'avg_loss_f_iter_{i+1}': instance_inner_losses_reduce['loss_f'][i] / (len(train_loader) * batch_size)
            for i in range(n_inner_iter)
        })
        avg_instance_loss.update({
            f'avg_loss_l2_iter_{i+1}': instance_inner_losses_reduce['loss_l2'][i] / (len(train_loader) * batch_size)
            for i in range(n_inner_iter)
        })

        log_dict = {
            'epoch': ep + 1,
            'train_total_loss': avg_total_loss,
            'train_l2_error': avg_loss_l2,
            'train_ic_loss': avg_loss_ic,
            'train_f_loss': avg_loss_f,
            'avg_instance_losses': avg_instance_loss,
            'epoch_time': str(timedelta(seconds=epoch_time)),
            'cumulative_time': str(timedelta(seconds=cumulative_time))
        }

        if rank == 0:
            if use_tqdm:
                pbar.set_description(
                    f"Epoch: {ep+1}; Total error: {avg_total_loss:.5f}; l2 error: {avg_loss_l2:.5f}; "
                    f"F loss: {avg_loss_f:.5f}; IC loss: {avg_loss_ic:.5f}."
                )
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_dict, indent=4) + '\n')

        if rank == 0 and avg_loss_l2 < min_l2_loss:
            min_l2_loss = avg_loss_l2
            save_checkpoint_meta(ep,
                                 config['train']['save_dir'],
                                 config['train']['save_name'],
                                 meta_net, meta_opt)


if __name__ == '__main__':
    parser = ArgumentParser(description='Meta Learning with DeepONet Example')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--num_gpus', type=int, help='Number of GPUs', default=1)
    parser.add_argument('--start', type=int, default=-1, help='start index')
    args = parser.parse_args()
    args.distributed = args.num_gpus > 1

    subprocess_fn(args)