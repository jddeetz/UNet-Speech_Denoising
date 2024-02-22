#!/usr/bin/env python3
# Authors: jddeetz@gmail.com
"""
This is a simple, quick, dirty version of Stochastic Gradient Descent training of the U-Net models.

This code is intended to be used via command line to train UNet neural network models, as defined in ./unet/unet_models.py

TODO: Either load all of the files into memory, or a batch of them, to speed up training
TODO: Refactor training code to avoid repeating similar steps for training and testing
"""
import argparse
import os

import glob
import numpy as np
from torch import load, save
from torch.nn import MSELoss
from torch.optim import Adam

from unet import UNet, UNet_model_cfg

TEST_DATA_FRACTION = 0.3
LEARNING_RATE = 1e-3
N_CHANNELS = 1
NUM_EPOCHS = 10

def train_model(clean_directory: str, noisy_directory: str, model_type: str):
    """ This function takes an input directory as an argument, takes all wav files, and makes spectrograms.

    Args:
        clean_directory: the path of the clean spectrograms
        noisy_directory: the path of the noisy spectrograms
        model_type: the type of U-Net model to train
    """
    #### Step 1: How many spectrogram files are in the clean_directory and noisy_directory
    num_clean_files = len(glob.glob("{}/*.sg".format(clean_directory)))
    num_noisy_files = len(glob.glob("{}/*.sg".format(noisy_directory)))
    # Assert that the number of files for the input and output spectrograms must be the same
    assert num_clean_files == num_noisy_files, 'Number of clean and noisy spectrograms do not match'

    #### Step 2: Train/Test Split
    # Get range of integers for files, since data filenames are like clnsp1.sg, noisy1.sg
    file_nums = np.arange(1, num_clean_files + 1)
    # Randomly split files into training and test data
    is_test_data = (np.random.rand(num_clean_files) < TEST_DATA_FRACTION)
    test_file_nums = file_nums[is_test_data]
    train_file_nums = file_nums[~is_test_data]

    #### Step 3: Train the U-Net Model
    # Instantiate the model
    # This passes the model configuration as json data to the UNet class
    if model_type == "UNet_Conv2D":
        model = UNet(model_cfg=UNet_model_cfg, use_ts_conv=False)
    elif model_type == "UNet_TSConv":
        model = UNet(model_cfg=UNet_model_cfg, use_ts_conv=True)
    else:
        raise ValueError("model_type is not one of the supported types")
    
    # Define loss function and optimizer
    loss_fn = MSELoss(reduction='sum')
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        print("#"*20)
        print("STARTING EPOCH {}".format(epoch))
        print("#"*20)
        for file_num in train_file_nums:
            # Load clean and noisy spectrograms (This part is slow)
            clean_sg = load("{}/clnsp{}.sg".format(clean_directory, file_num))
            noisy_sg = load("{}/noisy{}.sg".format(noisy_directory, file_num))
            # Reshape these to be passed into network
            clean_sg = clean_sg.reshape(1, clean_sg.shape[0], clean_sg.shape[1], clean_sg.shape[2])
            noisy_sg = noisy_sg.reshape(1, noisy_sg.shape[0], noisy_sg.shape[1], noisy_sg.shape[2])
            # Forward pass: compute predicted clean spectrogram by passing noisy one to the model.
            clean_sg_pred = model(noisy_sg)

            # Compute and print loss.
            loss = loss_fn(clean_sg_pred, clean_sg)
            if file_num % 5 == 4:
                print("Trained on file {} of {}: MSE = {}".format(file_num, num_clean_files, loss.item()))

            # Zero gradients, because otherwise they accumulate
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Update parameters
            optimizer.step()

        # Evalate the NN Model against the test set
        # Calling model.eval to set batch normalization layers to evaluation mode
        model.eval()
        test_loss = []
        for file_num in test_file_nums:
            # Load clean and noisy spectrograms (This part is slow)
            clean_sg = load("{}/clnsp{}.sg".format(clean_directory, file_num))
            noisy_sg = load("{}/noisy{}.sg".format(noisy_directory, file_num))
            # Reshape these to be passed into network
            clean_sg = clean_sg.reshape(1, clean_sg.shape[0], clean_sg.shape[1], clean_sg.shape[2])
            noisy_sg = noisy_sg.reshape(1, noisy_sg.shape[0], noisy_sg.shape[1], noisy_sg.shape[2])
            # Forward pass: compute predicted clean spectrogram by passing noisy one to the model.
            clean_sg_pred = model(noisy_sg)
            # Compute and print loss.
            loss = loss_fn(clean_sg_pred, clean_sg)
            test_loss.append(loss.item())
        print("Mean loss of test set: MSE = {}".format(np.mean(test_loss)))
        model.train()

    #### Step 4: Save the NN Model
    # Create the models directory if it does not exist
    if not os.path.exists("models"):
        os.mkdir("models")
    print("Saving model to disk")
    save(model, "models/{}.pt".format(model_type))

def main() -> None:
    """ Main function for training neural networks.

    Example usage of this code is:
    python3 train.py --model_type UNet_Conv2D

    Where:
        clean_directory: directory of clean speech spectrograms, the default value is data/CleanSpeechSpectrograms
        noisy_directory: directory of noisy speech spectrograms, the default value is data/NoisySpeechSpectrograms
        model_type: the architecture of the U-Net model, either UNet_Conv2D or UNet_TSConv, default is UNet_Conv2D
    """
    # Add and Parse Arguments
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        '--clean_directory',
        action="store",
        default="data/CleanSpeechSpectrograms",
        help='What directory are the clean spectrogram files stored in?')
    parser.add_argument(
        '--noisy_directory',
        action="store",
        default="data/NoisySpeechSpectrograms",
        help='What directory are the noisy spectrogram files stored in?')
    parser.add_argument(
        '--model_type',
        action="store",
        default="UNet_Conv2D",
        help='What model type are we training?')
    args = parser.parse_args()

    train_model(args.clean_directory, args.noisy_directory, args.model_type)
            
if __name__ == '__main__':
    main()
