# Self playing tetris
This repository contains files to train and test models to play tetris. this is a simple example of reinforcement learning that I did over the summer

## Installation
After cloning the repository all you need to do is run `pip install -r requirements.txt` in the terminal to install dependencies

## Usage

### Create a model
Once requirements are installed, run `train.py` to start training a model.
Models are automatically named `tetris_[epoch number]` where epoch number is the amount of iterations the model ahs been training for.
The model is then saved to a `saved_models` directory.

### Testing the model
To test a model rename a model in `trained_models` to `tetris` and then run `train.py`. 
This will show your model playing a game of tetris.
Upon either the game finishing or just simply stopping the program, an MP4 file named `output.mp4` fill will be created which stores a recording of the tetris game
