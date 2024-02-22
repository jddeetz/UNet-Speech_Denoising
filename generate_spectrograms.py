#!/usr/bin/env python3
# Authors: jddeetz@gmail.com
"""
This code is intended to be used via command line to generate spectrograms from wav files. See example below
"""

import argparse
import os
import shutil

import glob
import torch
import torchaudio as ta

def make_spectrogram(filename: str, frequency_bins: int, win_length_s: float):
    """ This function takes an wav files, and makes a spectrogram.

    Args:
        filename: the path of a wav file
        frequency_bins: the number of bins in which to disect the audio frequency into
        win_length_s: the time length of a window in the spectrogram, in seconds

    Returns:
        Spectrogram tensor
    """
    # Get the intensity (dB) and sample_rate for each file
    # The sample rate should be 16000 Hz, based on the dataset generated and the MS-SNSD config file
    # The waveform shape is 1 x n_frames
    waveform, sample_rate = ta.load(filename)

    # Calculate the number of frames per window (frames / window)
    # The sample rate is frames / s and win_length_s is s / window
    frames_per_window = int(sample_rate * win_length_s)

    # Get the spectrogram for the waveform
    # In order to actually get 256 frequency bins in the output, for some reason you need n_fft 2 * (256 - 1).
    # The spectrogram shape is 1 x n_frequency_bins x n_windows
    transform = ta.transforms.Spectrogram(n_fft=2*(frequency_bins - 1), hop_length = frames_per_window)
    spectrogram = transform(waveform)

    return spectrogram

def make_spectrograms(input_directory: str, output_directory: str, frequency_bins: int, win_length_s: float):
    """ This function takes an input directory as an argument, takes all wav files, and makes spectrograms.

    Args:
        input_directory: the path of the input wav data
        output_directory: the path to put the output spectrograms in
        frequency_bins: the number of bins in which to disect the audio frequency into
        win_length_s: the time length of a window in the spectrogram, in seconds
    """
    # Delete output_directory if it already exists
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)

    # Create the output_directory if it does not exist
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    # Get the files in the input directory
    filenames = glob.glob("{}/*.wav".format(input_directory))

    # For each wav file, load it and generate the spectrogram
    for filename in filenames:
        # Get the spectrogram for the wav file
        spectrogram = make_spectrogram(filename, frequency_bins, win_length_s)

        # Save the spectrogram in the output directory
        # To keep filenames simpler, only keep the first part of it.
        # noisy8_SNRdb_0.0_clnsp8.wav -> noisy8.sg
        # clean8.wav -> clean8.sg
        before_period = os.path.basename(filename).split('.')[0]
        before_underscore = before_period.split('_')[0]
        output_filename = before_underscore + '.sg'
        torch.save(spectrogram, '{}/{}'.format(output_directory, output_filename))

        print("{} -> {}/{}".format(filename, output_directory, output_filename))

def main() -> None:
    """ Main function for generating spectrograms from wav files.

    Example usage of this code is:
    python3 generate_spectrograms.py --input_directory INPUT_DIRECTORY --output_directory OUTPUT_DIRECTORY \
                                     --frequency_bins BINS --win_length WIN_LENGTH

    Where:
        input_directory: A postgres username to analytics_prod
        output_directory: A corresponding password for analytics_prod
        frequency_bins: The number of frequency bins in the spectrogram, default is 256.
        win_length: The time length of windows in the spectrogram, default is 0.01 seconds.
    """
    # Add and Parse Arguments
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        '--input_directory',
        action="store",
        help='What directory are the wav files stored in?')
    parser.add_argument(
        '--output_directory',
        action="store",
        help='What directory will the output spectrograms be saved in?')
    parser.add_argument(
        '--frequency_bins',
        action="store_const",
        const=True,
        default=256,
        help='The number of frequency bins in the spectrogram.')
    parser.add_argument(
        '--win_length',
        action="store_const",
        const=True,
        default=0.01,
        help='time length of windows in the spectrogram.')
    args = parser.parse_args()

    make_spectrograms(args.input_directory, args.output_directory, args.frequency_bins, args.win_length)
            
if __name__ == '__main__':
    main()