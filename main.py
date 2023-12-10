import yaml
import torch

from dataloader import MIDIDataset, get_dataloaders
from utils import train_network
from model import CNN


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    lr = config['learning rate']
    model = CNN(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, optimizer = train_network(model, optimizer, config)


if __name__ == '__main__':
    main()
