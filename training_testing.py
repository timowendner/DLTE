import torch
import time
import datetime
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


from .dataloader import get_dataloaders, MIDIDataset, rnn_collate_fn
from .utils import TempoLoss, write_file


def train_network(model: nn.Module, optimizer: torch.optim, config: dict):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    with open(config['label file'], 'r') as f:
        labels = f.readlines()
        labels = sorted(labels)

    dataset = MIDIDataset(config)
    train_loader, test_loader = get_dataloaders(dataset, config)

    model.train()
    tempoLoss = TempoLoss()

    start_time = time.time()
    num_epoch = config['train epochs']
    for epoch in range(1, num_epoch+1):

        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        for i, (path, key, num, denom, bpm, datastream) in tqdm(
            enumerate(train_loader), desc=f'{time_now} Starting Epoch {epoch:>3}', total=len(train_loader)
        ):
            outputs = model(datastream)
            bpm = bpm.float().to(config['device'])
            loss = tempoLoss(outputs.squeeze(), bpm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        n_iter = config['write and test every n iteration']
        if epoch % n_iter == 0:
            lr = optimizer.param_groups[0]['lr']
            error = test_network(model, test_loader, config)
            print(f'\t current error: {error:.4f}, lr: {lr}')
            # write_file(model, config)
    return model, optimizer


class TempoLoss(nn.Module):
    def __init__(self):
        super(TempoLoss, self).__init__()

    def forward(self, predictions, targets):
        tempo_error = torch.mean(torch.abs(predictions - targets))
        half_tempo_error = torch.mean(torch.abs(0.5 * predictions - targets))
        double_tempo_error = torch.mean(torch.abs(2 * predictions - targets))
        return torch.min(torch.min(tempo_error, half_tempo_error), double_tempo_error)


def compute_tempo_error(model: nn.Module, dataloader: DataLoader):
    model.eval()
    tempoLoss = TempoLoss()
    error = torch.tensor([])
    for path, key, num, denom, bpm, datastream in dataloader:
        pred = model(datastream)
        current = tempoLoss(pred.squeeze(), bpm.float())
        error = torch.cat([error, current])
    model.train()
    return float(torch.mean(error))


def test_network(model: nn.Module, dataloader: DataLoader, config: dict):
    collate = rnn_collate_fn if config['model class'] == 'RNN' else None
    dataloader = DataLoader(
        dataloader.dataset, batch_size=1, collate_fn=collate
    )
    model.eval()
    tempoLoss = TempoLoss()
    errors = []
    for path, key, num, denom, bpm, datastream in tqdm(dataloader, desc='\t Testing Network'):
        pred = model(datastream)
        error = tempoLoss(pred.squeeze(), bpm.float())
        errors.append(float(error))

    model.train()
    return float(np.mean(errors))
