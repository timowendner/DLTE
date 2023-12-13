import yaml
import torch
import argparse
from tqdm import tqdm

from .dataloader import MIDIDataset, get_dataloaders
from .training_testing import train_network
from .model import CNN, RNN


def main():
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    x = MIDIDataset()

    if config['model class'] == 'CNN':
        model = CNN(config)
    elif config['model class'] == 'RNN':
        model = RNN(config)
    else:
        raise AssertionError(
            f'The model class \'{config["model class"]}\' is not known'
        )
    model = model.to(device)

    lr = config['learning rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, optimizer = train_network(model, optimizer, config)
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
    main()
