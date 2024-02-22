# UNet-Speech_Denoising

## Summary
This repository outlines the definition and training of a type of convolutional neural network (U-Net), for reducing background noise in audio samples. 

Broadly, this repo does the following things:
1) Clones a module for producing synthetic audio samples of peoples speech. The synethetic samples include both clean speech (no noise), and those with noise overlayed into the audio samples. Currently, because this is a toy problem, the only background noise added to the audio samples consists of humming from air conditioners. The Microsoft scalable noisy speech dataset was used for this purpose (github.com/microsoft/MS-SNSD). 
2) Produces spectrograms of the synthetic clean and noisy audio samples. The python code generate_spectrograms.py accomplishes this.
3) Outlines two types of U-Net networks (in the unet/ directory) one using Conv2d layers, and another implementing a time shift prior to each Conv2d layer.
4) Trains both types of U-Net models (using train.py), and saves these models in the model/ directory.
5) Has unit tests (in the tests/ directory) for the process used to generate the spectrograms, the special layer types in the model, and the accuracy of the trained models. 


## Getting started
A shell script, get_training_data.sh, clones the MS-SNSD repository, makes edits to that repos config file, and generates clean and noisy datasets. Following this, the shell script generates spectrograms from the clean and noisy generated WAV files and saves them into the data/ directory. The spectrograms are saved torch tensors, with the .sg file extension. This script also installs any repositories needed to run other code in this repo. 

