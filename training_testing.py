import torch
import time
import datetime
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


from .dataloader import get_dataloaders, MIDIDataset, rnn_collate_fn
from .utils import TempoLoss, FileWriter


def train_network(model: nn.Module, optimizer: torch.optim, config: dict):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    dataset = MIDIDataset(config)
    train_loader, test_loader = get_dataloaders(dataset, config)

    model.train()
    tempoLoss = TempoLoss()

    n_iter = config['write every n iteration']
    if n_iter != 0:
        fileWriter = FileWriter(config)

    start_time = time.time()
    num_epoch = config['train epochs']
    for epoch in range(1, num_epoch+1):

        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        for i, (path, key, num, denom, bpm, datastream) in tqdm(
            enumerate(train_loader), desc=f'{time_now} Starting Epoch {epoch:>3}',
            ncols=75, total=len(train_loader)
        ):
            outputs = model(datastream)
            bpm = bpm.float().to(config['device'])
            loss = tempoLoss(outputs.squeeze(), bpm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr = optimizer.param_groups[0]['lr']
        error = test_network(model, test_loader, config)
        print(f'\t current error: {error:.4f}, lr: {lr}\n')
        if n_iter != 0 and (epoch % n_iter == 0 or epoch == num_epoch+1):
            fileWriter.write(model)
    return model, optimizer


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
    for path, key, num, denom, bpm, datastream in tqdm(
        dataloader, desc='\t Testing Network', ncols=75, leave=False, total=len(dataloader)
    ):
        pred = model(datastream)
        error = tempoLoss(pred.squeeze(), bpm.float())
        errors.append(float(error))

    model.train()
    return float(np.mean(errors))
