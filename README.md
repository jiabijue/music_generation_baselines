# Music Generation Baselines
This project includes a set of self-implemented (midi) music generation baseline examples. Also great examples for generating polyphonic music of multiple tracks. 

## Models included

**RNN**

**(coming soon...)**

## Let's see some results first

### RNN

![Piano roll](https://github.com/BerylJia/music_generation_baselines/raw/master/docs/rnn_pianoroll.png)

<audio controls="controls" preload="preload" autoplay="autoplay">  
        <source src="https://github.com/BerylJia/music_generation_baselines/raw/master/docs/rnn_test.mp3" >  
        test-generate-midi  
</audio></span></span>

## Getting Started
`music_generation_baselines` should work in Windows, Linux and MacOS environments. Open a terminal and run:

```bash
# Install the dependencies
pip install -r requirements.txt
```

We would like to recommend that you use CUDA to accelerate training. But if you don't have a GPU, remove the line of installing `tensorflow-gpu` in the `requirements.txt` file and additionally run:

```bash
pip install tensorflow>=1.4.0
``` 

## Prepare training data

Using dataset `lpd_5_cleansed` from [Lakh Pianoroll Dataset](https://salu133445.github.io/lakh-pianoroll-dataset/dataset).

I cut each song into 400 time steps, removed the lowest 20 pitches and the highest 24 pitches, and made a .npy file called ``lpd_5_cleansed.npy`` (you can put the downloaded `lpd_5_cleansed` dataset in `/data/lpd_5_cleansed` and run the script `/data/dir_to_npy.np` to make this data file). Matrix shape is: 21425(song) &times; 400(time step) &times; 84(note) &times; 5(track)

## Run the code

### Traning a model

```bash
python train.py
```

### Generate midi

Once you've trained your model, you can generate MIDI file using `gen.py`

```bash
python gen.py
```

By default, this create one MIDI file using a model checkpoint which is specified by argument `--ckpt`. The MIDI file path and name is specified by the argument `--filename`)


## Command Line Arguments

### `train.py`

- `--model`: Which neural network model you want to train.

### `gen.py`

- `--model`: Which already-trained model you want to use to generate.
- `--ckpt`: The model checkpoint file.
- `--filename`: Generated MIDI filename.

## Demo

Check out this demo `demo.ipynb` to see how it works. 

 