#!/usr/bin/env python3
# Authors: jddeetz@gmail.com
"""
Tests:
- Test to make sure the pytorch version has not changed.
- Test the shape and characteristics of the training data, to make sure it is consistent with the model and has not changed.
- For both trained models (UNet_Conv2d and UNet_TSConv) test output on a specific file to ensure it matches.
- Test the accuracy of both models for all test set samples. Make model accuracy explicit here.

Run this with: python3 -m tests.test_trained_models
"""
import unittest

import torch

class SpectrogramTests(unittest.TestCase):
    def test_torch_version(self):
        # Tests for versions of torch
        self.assertEqual(torch.__version__, '2.2.0')

    def test_torch_version(self):
        # I didn't finish this yet, but I have outlined the steps here
        # Defines unit tests for the training data.
        # We want to be sure that the definition of out training set is what we say it is.
        pass

    def test_unet_conv2d(self):
        # I didn't finish this yet, but I have outlined the steps here
        # Load noisy and clean spectrograms that were in the test set
        # Properly shape the spectrogram to match the input shape of UNet
        # Load the UNet model
        # Put the UNet in evaluation mode
        # Predict the clean spectrogram using the UNet
        # Perform tests to compare the predicted vs. real clean spectrograms
        pass

    def test_unet_tsconv(self):
        # Same as above, but with TSConv
        pass

    def test_unet_conv2d_accuracy(self):
        # I didn't finish this yet, but I have outlined the steps here
        # The basic principle here is to load the model and predict the entire test set.
        # Once the clean and noisy spectrograms in the test set are compared, we can unit test the models accuracy
        # This will ensure that the performance of our model doesn't accidentally change with code changes
        pass

    