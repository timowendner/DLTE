import yaml
import torch

from dataloader import MIDIDataset, get_dataloaders


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    dataset = MIDIDataset(config)
    train_loader, test_loader = get_dataloaders(dataset, config)


if __name__ == '__main__':
    main()
