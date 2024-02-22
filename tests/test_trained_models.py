#!/usr/bin/env python3
# Authors: jddeetz@gmail.com
"""
Tests:
- For both models (UNet_Conv2d and UNet_TSConv) test output on a specific file to ensure it matches

Run this with: python3 -m tests.test_models
"""
import unittest

import torch

# import predict

class SpectrogramTests(unittest.TestCase):
    def test_torch_version(self):
        # Tests for versions of torch
        self.assertEqual(torch.__version__, '2.2.0')

    def test_unet_conv2d(self):
        # I didn't finish this yet, but I have outlined the steps here
        # Load noisy and clean spectrograms that were in the test set
        # Properly shape the spectrogram to match the input shape of UNet
        # Load the UNet model
        # Put the UNet in evaluation mode
        # Predict the clean spectrogram using the UNet
        # Perform tests to compare the predicted vs. real clean spectrograms
        self.assertTrue(True)

    def test_unet_tsconv(self):
        # Same as above, but with TSConv
        self.assertTrue(True)

    def test

    