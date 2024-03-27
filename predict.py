#!/usr/bin/env python3
# Authors: jddeetz@gmail.com
"""
This is code for loading the trained unet model and using it to make predictions.
"""
import logging
import os

import glob
from torch import load, complex64
import torchaudio as ta

from unet import UNet, UNet_model_cfg

def load_spectrogram(filename: str):
    """ This function loads a spectrogram from disk and reshapes it into the dimensions for the U-Net.

    Args:
        filename: the path of the spectrogram

    Returns:
        pytorch tensor
    """
    # Load spectrogram (This part is slow)
    spectrogram = load(filename)
    return spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1], spectrogram.shape[2])

def main():
    # Init the logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler("unet_predict.log"), logging.StreamHandler()])

    # Get list of all noisy speech spectrograms
    noisy_speech_filenames = glob.glob('data/NoisySpeechSpectrograms/noisy*.sg')

    # Load the saved model from disk
    logging.info("Loading model from disk")
    model = UNet(model_cfg=UNet_model_cfg, use_ts_conv=False)
    model.load_state_dict(load("models/UNet_Conv2D_16_channels.pt"))
    model.eval()

    # Create the output_directory if it does not exist
    if not os.path.exists("data/FilteredSpeech"):
        os.mkdir("data/FilteredSpeech")

    # For each spectrogram, predict the clean speech spectrogram
    for filename in noisy_speech_filenames:
        # Load noisy spectrogram from disk
        logging.info("Loading spectrogtam {}".format(filename))
        noisy_sg = load_spectrogram(filename)

        # Predict clean spectrogram
        logging.info("Predicting clean spectrogram for {}".format(filename))
        clean_sg_pred = model(noisy_sg)

        # Convert spectrogram to wav file
        frequency_bins = clean_sg_pred.shape[2]
        win_length_s = 0.01
        sample_rate = 16000
        frames_per_window = int(sample_rate * win_length_s)
        transform = ta.transforms.InverseSpectrogram(n_fft=2*(frequency_bins - 1), hop_length=frames_per_window)
        waveform = transform(clean_sg_pred.type(complex64))
        waveform = waveform.reshape(1, waveform.shape[2])

        # Save filtered wav file
        before_period = os.path.basename(filename).split('.')[0]
        output_filename = "{}/{}".format("data/FilteredSpeech", before_period + '_filtered.wav')
        
        logging.info("Saving wav file for {}".format(output_filename))
        ta.save(output_filename, waveform, sample_rate, format="wav")




if __name__ == '__main__':
    main()