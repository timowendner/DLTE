# DLTE (Deep-Learning Tempo Estimation)
This is a Deep-Learning Neural Network for estimation of Tempo (bpm) with midi-files. Meaning we don't have the audio waveforms. The performance is unfortunately not good and therefore other tempo estimation methods are preferable.

## Setup
We first need a config file that is setting our model up. We have an example config file in the repository `config.yaml`.

To install the package use:
```
!pip install git+https://github.com/timowendner/DLTE
```

Then the model can be run like:
```
from DLTE import DLTE
model = DLTE.run('/path/to/config.yaml')
```

## Approach
The model opens the MIDI-files as a list of all keypresses. It stores this as a sorted array with `[start, duration, velocity, pitch]`. The n-order differences of the starting values are calculated and stored as well. Every time the Dataloader is called the tempo is called a subarray of specific size (in config file) is chosen at random from the sorted array. Then we only select the tempo and n-order differences to get as input for the model.

The model is ether a CNN or a RNN that then regresses to the bpm. The settings for the CNN is two convolutional layers followed by a max pooling. This is repeated for as many times as specified. Following this is a Linear model that is regressing to a single (tempo) value.

## Issues
Currently the RNN option is not well tested and is probably faulty.