import numpy as np
import torch
import time
import datetime
import os
import glob
from mido import MidiFile
from torch import nn
from tqdm import tqdm


def open_midi_file(path: str) -> list[tuple[float, float, float, str]]:
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

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tempo_error = torch.mean(torch.abs(predictions - targets))
        return tempo_error ** 2
        half_tempo_error = torch.mean(torch.abs(0.5 * predictions - targets))
        double_tempo_error = torch.mean(torch.abs(2 * predictions - targets))
        return torch.min(torch.min(tempo_error, half_tempo_error), double_tempo_error) ** 2


def write_file(predictions: dict, config: dict) -> None:
    predictions = predictions.items()
    predictions = sorted(predictions)
    output_file = config['output file']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savetxt(
        output_file,
        np.array(predictions),
        fmt="%s",
        delimiter=",",
        comments="//",
        header="filename,ts_num,tempo(bpm)",
    )
