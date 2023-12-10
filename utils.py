import torch
import time
import datetime
from mido import MidiFile
from torch import nn
from tqdm import tqdm

from dataloader import get_dataloaders, MIDIDataset


def train_network(model: nn.Module, optimizer: torch.optim, config: dict):
    dataset = MIDIDataset(config)
    train_loader, test_loader = get_dataloaders(dataset, config)

    model.train()
    total_step = len(train_loader)
    mse = torch.nn.MSELoss()

    start_time = time.time()
    num_epoch = config['train epochs']
    for epoch in range(1, num_epoch+1):

        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        for i, (path, key, num, denom, bpm, datastream) in tqdm(
            enumerate(train_loader), desc=f'{time_now} Starting Epoch {epoch}', total=len(train_loader)
        ):
            outputs = model(datastream)
            bpm = bpm.float()
            loss = mse(outputs.squeeze(), bpm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"End Epoch: {epoch}/{num_epoch}",
              f"Loss: {loss.item():.4f}",
              f"    {time_now}",
              f"   (lr: {lr}) "
              )

        # # add the number of epochs
        # config.current_epoch += 1

        # # save the model if enough time has passed
        # if abs(time.time() - start_time) >= config.save_time or epoch == num_epoch:
        #     save_model(model, optimizer, config)
        #     start_time = time.time()
    return model, optimizer


def open_midi_file(path: str):
    midi_file = MidiFile(path)
    result = []
    open_notes = {}
    current = 0
    for msg in midi_file.tracks[0]:
        current += msg.time
        if msg.type == 'note_on':
            pitch = msg.note
            open_notes[pitch] = (current, msg.velocity)
        elif msg.type == 'note_off' and (pitch := msg.note) in open_notes:
            start, velocity = open_notes[pitch]
            del open_notes[pitch]
            result.append((start, current - start, velocity, pitch))
    return sorted(result)


def compute_tempo_error(pred_tempo, target_tempo):
    tempo_error = abs(pred_tempo - target_tempo)
    half_tempo_error = abs(0.5 * pred_tempo - target_tempo)
    double_tempo_error = abs(2 * pred_tempo - target_tempo)
    return min(tempo_error, half_tempo_error, double_tempo_error)
