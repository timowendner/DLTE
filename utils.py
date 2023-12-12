import torch
import time
import datetime
import os
import glob
from mido import MidiFile
from torch import nn
from tqdm import tqdm


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


class TempoLoss(nn.Module):
    def __init__(self):
        super(TempoLoss, self).__init__()

    def forward(self, predictions, targets):
        tempo_error = torch.mean(torch.abs(predictions - targets))
        half_tempo_error = torch.mean(torch.abs(0.5 * predictions - targets))
        double_tempo_error = torch.mean(torch.abs(2 * predictions - targets))
        return torch.min(torch.min(tempo_error, half_tempo_error), double_tempo_error)


def compute_tempo_error(model, dataloader):
    model.eval()
    tempoLoss = TempoLoss()
    error = torch.tensor([])
    for path, key, num, denom, bpm, datastream in dataloader:
        pred = model(datastream)
        current = tempoLoss(pred.squeeze(), bpm.float())
        error = torch.cat([error, current])
    model.train()
    return float(torch.mean(error))


def write_file(model, config):
    files = glob.glob(os.path.join(config['test folder'], '*.mid'))
    result = []
    for file in tqdm(files, desc='\t Writing Files', total=len(files)):
        dataset = open_midi_file(file)
