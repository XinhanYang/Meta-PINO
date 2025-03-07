import yaml
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from timeit import default_timer
from tqdm import tqdm

from baselines.model import DeepONetCP
from baselines.data import DeepONetCPNS
from train_utils.data_utils import data_sampler
from train_utils.losses import LpLoss, PINO_loss3d, get_forcing

def eval_ns_deeponet(model, dataset, dataloader, forcing, config, device, use_tqdm=True):
    """
    Evaluate the DeepONet model on the NS dataset by computing the relative L2 error,
    initial condition (IC) error, and equation residual error.
    """
    # Fluid parameter
    v = 1 / config['data']['Re']
    S, T = dataset.S, dataset.T
    t_interval = config['data']['time_interval']
    batch_size = config['test']['batchsize']

    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    loss_dict = {'loss_f': 0.0,
                 'loss_data': 0.0,
                 'loss_ic': 0.0}
    start_time = default_timer()
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            b = x.size(0)
            # DeepONet model requires an additional trunk input.
            # Assume the dataset provides grid information as dataset.xyt with shape (S*S*T, 3)
            grid = dataset.xyt.to(device)
            pred = model(x, grid)
            pred = pred.reshape(b, S, S, T)
            loss_l2 = myloss(pred, y.reshape(b, S, S, T))
            # Reshape the branch input to (b, S, S) for IC error computation
            x_ic = x.reshape(b, S, S)
            loss_ic, loss_f = PINO_loss3d(pred, x_ic, forcing, v, t_interval)

            loss_dict['loss_f'] += loss_f
            loss_dict['loss_data'] += loss_l2
            loss_dict['loss_ic'] += loss_ic

    end_time = default_timer()
    avg_l2 = loss_dict['loss_data'].item() / len(dataloader)
    avg_f = loss_dict['loss_f'].item() / len(dataloader)
    avg_ic = loss_dict['loss_ic'].item() / len(dataloader)
    print(f'== Averaged relative L2 data error: {avg_l2} ==')
    print(f'== Averaged IC error: {avg_ic} ==')
    print(f'== Averaged equation error: {avg_f} ==')
    print(f'Time cost: {end_time - start_time} s')

def test_deeponet_ns(config):
    device = 0 if torch.cuda.is_available() else 'cpu'
    data_config = config['data']
    dataset = DeepONetCPNS(datapath=data_config['datapath'],
                           nx=data_config['nx'], nt=data_config['nt'],
                           sub=data_config['sub'], sub_t=data_config['sub_t'],
                           offset=data_config['offset'], num=data_config['n_sample'],
                           t_interval=data_config['time_interval'])
    train_dataset, val_dataset, test_dataset = dataset.split_dataset(data_config['n_sample'], 
                                                    offset=data_config['offset'], 
                                                    test_ratio=data_config['test_ratio'],
                                                    val_ratio=data_config.get('val_ratio', 0.1))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['test']['batchsize'],
        sampler=data_sampler(train_dataset, shuffle=False, distributed=False),
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['test']['batchsize'],
        sampler=data_sampler(val_dataset, shuffle=False, distributed=False),
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['test']['batchsize'],
        sampler=data_sampler(test_dataset, shuffle=False, distributed=False),
        drop_last=False
    )

    # Determine the branch network input dimension (S^2) from the dataset dimensions
    u0_dim = dataset.S ** 2
    activation = config['model']['activation']
    normalize = config['model']['normalize']
    model = DeepONetCP(branch_layer=[u0_dim] + config['model']['branch_layers'],
                          trunk_layer=[3] + config['model']['trunk_layers'],
                          nonlinearity = activation,
                          normalize=normalize).to(device)

    # Load pretrained model weights for testing if provided in the configuration
    if 'ckpt' in config['test']:
        ckpt_path = config['test']['ckpt']
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            if 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
                print('Weights loaded from %s' % ckpt_path)

    # Get forcing (e.g., via get_forcing) and move it to the device
    forcing = get_forcing(dataset.S).to(device)

    print('Train set evaluation:\n')
    eval_ns_deeponet(model, dataset, train_loader, forcing, config, device)

    print('Validation set evaluation:\n')
    eval_ns_deeponet(model, dataset, val_loader, forcing, config, device)

    print('Test set evaluation:\n')
    eval_ns_deeponet(model, dataset, test_loader, forcing, config, device)

if __name__ == '__main__':
    parser = ArgumentParser(description='Test DeepONet on NS Dataset')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    # construct dataloader
    seed = config['data']['seed']
    print(f'Seed :{seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    test_deeponet_ns(config)
