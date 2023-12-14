import torch
import time
import datetime
import numpy as np
from torch import nn
from tqdm import tqdm
from itertools import product
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer


from .dataloader import get_dataloaders, MIDIDataset, rnn_collate_fn
from .utils import TempoLoss, write_file


def train_network(
    model: nn.Module, 
    optimizer: Optimizer, 
    dataset: Dataset, 
    prediction_set: Dataset, 
    config: dict
) -> tuple[nn.Module, Optimizer]:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    train_loader, test_loader = get_dataloaders(
        dataset, config, split=config['train split']
    )
    pred_loader = get_dataloaders(prediction_set, config, split=None)

    model.train()
    tempoLoss = TempoLoss()

    n_iter = config['write every n iteration']
    num_epoch = config['train epochs']
    for epoch in range(1, num_epoch+1):
        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        for i, (path, key, num, denom, bpm, datastream) in tqdm(
            enumerate(train_loader), desc=f'{time_now} \tStarting Epoch {epoch:>3}',
            ncols=75, total=len(train_loader)
        ):
            outputs = model(datastream)
            bpm = bpm.float().to(config['device'])
            loss = tempoLoss(outputs.squeeze(), bpm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr = optimizer.param_groups[0]['lr']
        result, error = test_network(
            model, test_loader, config, desc='\tTesting Network'
        )
        print(f'\tcurrent error: {error:.4f}, lr: {lr}\n')
        if n_iter != 0 and (epoch % n_iter == 0 or epoch == num_epoch+1):
            prediction, error = test_network(
                model, pred_loader, config, desc='\tWriting Predictions'
            )
            write_file(prediction, config)
    return model, optimizer


def compute_error(predictions: float, targets: float) -> float:
    tempo_error = abs(predictions - targets)
    half_tempo_error = abs(0.5 * predictions - targets)
    double_tempo_error = abs(2 * predictions - targets)
    return min(tempo_error, half_tempo_error, double_tempo_error)


def test_network(model: nn.Module, dataloader: DataLoader, config: dict, desc: str = None) -> tuple[dict, float]:
    n = config['check n times']
    results = {}
    truth = {}
    tempoLoss = TempoLoss()
    model.eval()
    for i, (path, key, num, denom, bpm, datastream) in tqdm(
        product(range(n), dataloader), total=len(dataloader)*n, desc=desc, ncols=75
    ):
        pred = model(datastream).to('cpu')
        for name, prediction, gt in zip(path, pred.squeeze(), bpm):
            truth[name] = gt
            results[name] = results.get(name, 0) + float(prediction)
    errors = []
    for name, prediction in results.items():
        results[name] = prediction / n
        error = tempoLoss(
            torch.Tensor([results[name]]),
            torch.Tensor([truth[name]])
        )
        errors.append(error)
    model.train()
    return results, np.mean(errors)
