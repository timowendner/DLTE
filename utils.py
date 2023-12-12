import numpy as np
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


class FileWriter:
    def __init__(self, config) -> None:
        self.config = config
        files = glob.glob(os.path.join(config['test folder'], '*.mid'))
        files = sorted(files)
        if 'debug size' in config:
            files = files[:config['debug size']]
        results = []
        for file in tqdm(files, desc='Load Test Files', ncols=75):
            datastream = open_midi_file(file)
            datastream = torch.Tensor(datastream).float().T
            for i in range(config['n order differences']):
                diffs = torch.zeros(datastream[0].shape[0])
                diffs[1:] = torch.diff(datastream[0])
                datastream = torch.cat([diffs.unsqueeze(0), datastream], dim=0)
            results.append(((os.path.basename(file)), datastream))
        self.data = results

    def write(self, model):
        model.eval()
        results = []
        for file, datastream in self.data:
            size = datastream.shape[1]
            length = self.config['Data Length']
            r = np.random.randint(0, max(1, size-length))
            current = datastream[:, r:r+length]
            padding = torch.zeros((datastream.shape[0], length))
            padding[:, length-current.shape[1]:] = current
            datastream = padding.to(self.config['device']).unsqueeze(0)
            pred = float(model(datastream))
            results.append((file, pred))
        model.train()

        output_file = self.config['output file']
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.savetxt(
            output_file,
            np.array(results),
            fmt="%s",
            delimiter=",",
            comments="//",
            header="filename,ts_num,tempo(bpm)",
        )
