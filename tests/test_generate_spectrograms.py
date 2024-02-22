#!/usr/bin/env python3
# Authors: jddeetz@gmail.com
"""
Tests ../generate_spectrograms.py
- Tests module versions
- Tests functionality of generate_spectrogram

Run this with: python3 -m tests.test_generate_spectrograms
"""
import unittest

import torch
import torchaudio as ta

import generate_spectrograms as gs

class SpectrogramTests(unittest.TestCase):
    def test_torch_versions(self):
        # Tests for versions of torch and torch audio
        self.assertEqual(torch.__version__, '2.2.0')
        self.assertEqual(ta.__version__, '2.2.0')

    def test_generate_spectrogram(self):
        # Make a spectrogram from test data
        spectrogram = gs.make_spectrogram(filename='./tests/test.wav', frequency_bins=256, win_length_s=0.01)

        # Make sure its the right shape
        self.assertEqual(spectrogram.shape[0], 1)
        self.assertEqual(spectrogram.shape[1], 256)
        self.assertEqual(spectrogram.shape[2], 1178)

        # Validate spectrogram values compared to test data 
        expected_spectrogram = torch.load('./tests/test.sg')
        self.assertTrue(torch.equal(spectrogram, expected_spectrogram))
                   
if __name__ == '__main__':
    unittest.main()
