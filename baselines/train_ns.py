from tqdm import tqdm
from datetime import datetime, timedelta
from time import time
import json
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from train_utils.distributed import reduce_loss_dict

from baselines.model import DeepONet, DeepONetCP
from baselines.data import DeepOnetNS, DeepONetCPNS
from train_utils.utils import save_checkpoint
from train_utils.data_utils import sample_data

from train_utils.losses import LpLoss, PINO_loss3d, get_forcing

from torch.nn.parallel import DistributedDataParallel as DDP

from train_utils.data_utils import data_sampler

def train_deeponet_cp(config, args):
    '''
    Train Cartesian product DeepONet
    Args:
        config:

    Returns:
    '''
    if args.distributed:
        rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0

    data_config = config['data']
    batch_size = config['train']['batchsize']
    dataset = DeepONetCPNS(datapath=data_config['datapath'],
                           nx=data_config['nx'], nt=data_config['nt'],
                           sub=data_config['sub'], sub_t=data_config['sub_t'],
                           offset=data_config['offset'], num=data_config['n_sample'],
                           t_interval=data_config['time_interval'])
    train_loader = DataLoader(dataset, batch_size=config['train']['batchsize'],
                            shuffle=data_config['shuffle'],
                            sampler=data_sampler(dataset,
                                                shuffle=data_config['shuffle'],
                                                distributed=args.distributed),
                            drop_last=True)
    v = 1 / config['data']['Re']
    S, T = dataset.S, dataset.T
    t_interval = config['data']['time_interval']

    # Training settings
    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']
    data_weight = config['train']['data_loss']
    forcing = get_forcing(dataset.S).to(rank)

    u0_dim = dataset.S ** 2
    model = DeepONetCP(branch_layer=[u0_dim] + config['model']['branch_layers'],
                       trunk_layer=[3] + config['model']['trunk_layers']).to(rank)
    
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
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
    scheduler = MultiStepLR(optimizer, milestones=config['train']['milestones'],
                            gamma=config['train']['scheduler_gamma'])
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    myloss = LpLoss(size_average=True)
    model.train()

    zero = torch.zeros(1).to(rank)

    # Initialize logging
    cumulative_time = 0
    min_l2_loss = float('inf')
    log_file = None
    if rank == 0:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{config['log']['logfile']}_{current_time}.log"
        with open(log_file, 'w') as f:
            config_str = json.dumps(config, indent=4)
            f.write(f"Configuration:\n{config_str}\n")

    for e in pbar:
        epoch_start_time = time()
        loss_dict = {'total_loss': 0.0,
                     'loss_ic': 0.0,
                     'loss_f': 0.0,
                     'loss_l2': 0.0}

        for x, y in train_loader:
            x = x.to(rank)  # initial condition, (batchsize, u0_dim)
            grid = dataset.xyt
            grid = grid.to(rank)  # grid value, (SxSxT, 3)
            y = y.to(rank)  # ground truth, (batchsize, SxSxT)

            pred = model(x, grid)

            # Reshape tensors
            x = x.reshape(batch_size, S, S)
            y = y.reshape(batch_size, S, S, T)
            pred = pred.reshape(batch_size, S, S, T)

            # Compute losses
            loss_l2 = myloss(pred.view(batch_size, S, S, T), y.view(batch_size, S, S, T))
            if ic_weight != 0 or f_weight != 0:
                loss_ic, loss_f = PINO_loss3d(pred.view(batch_size, S, S, T), x, forcing, v, t_interval)
            else:
                loss_ic, loss_f = zero, zero

            total_loss = loss_l2 * data_weight + loss_f * f_weight + loss_ic * ic_weight

            # Backpropagation
            model.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            loss_dict['loss_l2'] += loss_l2.item() * y.shape[0]
            loss_dict['loss_ic'] += loss_ic.item() * y.shape[0]
            loss_dict['loss_f'] += loss_f.item() * y.shape[0]
            loss_dict['total_loss'] += total_loss.item() * y.shape[0]

        # Reduce losses across distributed processes
        if args.distributed:
            loss_reduced = reduce_loss_dict(loss_dict)
            if rank == 0:
                loss_dict = {k: v.item() / len(dataset) for k, v in loss_reduced.items()}
        else:
            loss_dict = {k: v / len(dataset) for k, v in loss_dict.items()}

        # Scheduler step
        scheduler.step()
        epoch_time = time() - epoch_start_time
        cumulative_time += epoch_time

        if rank == 0:
            # Prepare log dictionary
            log_dict = {
                'epoch': e + 1,
                'train_total_loss': loss_dict['total_loss'],
                'train_l2_error': loss_dict['loss_l2'],
                'train_ic_loss': loss_dict['loss_ic'],
                'train_f_loss': loss_dict['loss_f'],
                'epoch_time': str(timedelta(seconds=epoch_time)),
                'cumulative_time': str(timedelta(seconds=cumulative_time))
            }

           
            pbar.set_description(
                (
                    f"Epoch: {e + 1}; Total loss: {loss_dict['total_loss']:.5f}; "
                    f"L2 loss: {loss_dict['loss_l2']:.5f}; "
                    f"IC loss: {loss_dict['loss_ic']:.5f}; F loss: {loss_dict['loss_f']:.5f}"
                )
            )

            # Write to log file
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_dict, indent=4) + '\n')

            # Save checkpoint if this epoch achieves the minimum L2 loss
            if loss_dict['loss_l2'] < min_l2_loss:
                min_l2_loss = loss_dict['loss_l2']
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', f'_best.pt'),
                                model, optimizer)


def train_deeponet(config):
    '''
    train plain DeepOnet
    Args:
        config:

    Returns:

    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    dataset = DeepOnetNS(datapath=data_config['datapath'],
                         nx=data_config['nx'], nt=data_config['nt'],
                         sub=data_config['sub'], sub_t=data_config['sub_t'],
                         offset=data_config['offset'], num=data_config['n_sample'],
                         t_interval=data_config['time_interval'])
    train_loader = DataLoader(dataset, batch_size=config['train']['batchsize'], shuffle=False)

    u0_dim = dataset.S ** 2
    model = DeepONet(branch_layer=[u0_dim] + config['model']['branch_layers'],
                     trunk_layer=[3] + config['model']['trunk_layers']).to(device)
    optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
    scheduler = MultiStepLR(optimizer, milestones=config['train']['milestones'],
                            gamma=config['train']['scheduler_gamma'])

    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    myloss = LpLoss(size_average=True)
    model.train()
    loader = sample_data(train_loader)
    for e in pbar:
        u0, x, y = next(loader)
        u0 = u0.to(device)
        x = x.to(device)
        y = y.to(device)
        pred = model(u0, x)
        loss = myloss(pred, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_description(
            (
                f'Epoch: {e}; Train loss: {loss.item():.5f}; '
            )
        )
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')
