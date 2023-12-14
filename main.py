import yaml
import torch
import argparse
from tqdm import tqdm
from torch import nn

from .dataloader import MIDIDataset, get_dataloaders
from .training_testing import train_network
from .model import CNN, RNN


def main(config_path: str) -> nn.Module:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    if config['model class'] == 'CNN':
        model = CNN(config)
    elif config['model class'] == 'RNN':
        model = RNN(config)
    else:
        raise AssertionError(
            f'The model class \'{config["model class"]}\' is not known'
        )
    model = model.to(device)

    label_path = config['label file']
    data_path = config['train folder']
    pred_path = config['prediction folder']
    dataset = MIDIDataset(
        data_path, label_path, config, desc='Loading Dataset'
    )
    prediction_set = MIDIDataset(
        pred_path, None, config, desc='Loading Testset'
    )

    lr = config['learning rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, optimizer = train_network(
        model, optimizer, dataset, prediction_set, config
    )
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="Path to the config file",
        type=str,
        default="config.yaml",
    )
    args = parser.parse_args()
    main(args.config)
