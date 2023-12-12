import numpy as np
import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from .utils import open_midi_file


class MIDIDataset(Dataset):
    def __init__(self, config):
        label_path = config['label file']
        with open(label_path, 'r') as f:
            labels = f.readlines()
            labels = sorted(labels)
            if 'debug size' in config:
                labels = labels[:config['debug size']+1]

        dataset = []
        for line in tqdm(labels[1:], desc='Loading Dataset', ncols=75):
            path, key, num, denom, bpm = line.replace('\n', '').split(',')
            path = os.path.join(config['train folder'], path)

            datastream = open_midi_file(path)
            datastream = torch.Tensor(datastream).float().T
            for i in range(config['n order differences']):
                diffs = torch.zeros(datastream[0].shape[0])
                diffs[1:] = torch.diff(datastream[0])
                datastream = torch.cat([diffs.unsqueeze(0), datastream], dim=0)
            dataset.append(
                (path, key, int(num), int(denom), float(bpm), datastream)
            )
        if len(dataset) == 0:
            raise AttributeError('Data path seems to be empty')

        self.device = config['device']
        self.model = config['model class']
        self.data_length = config['Data Length']
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, key, num, denom, bpm, datastream = self.dataset[idx]
        if self.model == 'CNN':
            size = datastream.shape[1]
            length = self.data_length
            r = np.random.randint(0, max(1, size-length))
            current = datastream[:, r:r+length]
            padding = torch.zeros((datastream.shape[0], length))
            padding[:, length-current.shape[1]:] = current
            datastream = padding
        elif self.model == 'RNN':
            datastream = datastream.T
        datastream = datastream.to(self.device)
        return path, key, num, denom, bpm, datastream


def rnn_collate_fn(batch):
    path, key, num, denom, bpm, inputs = zip(*batch)
    padded_inputs = pad_sequence(inputs)
    lengths = torch.tensor([len(seq) for seq in inputs])
    bpm = torch.tensor(bpm)
    return path, key, num, denom, bpm, (padded_inputs, lengths)


def get_dataloaders(dataset, config):
    train_size = int(config['train split'] * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )

    batch_size = config['batch size']
    collate = rnn_collate_fn if config['model class'] == 'RNN' else None
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, collate_fn=collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size, shuffle=False, collate_fn=collate
    )

    return train_loader, test_loader
