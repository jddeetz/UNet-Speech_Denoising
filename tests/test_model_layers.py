#!/usr/bin/env python3
# Authors: jddeetz@gmail.com
"""
Tests:
- Version of torch, to make sure it has not changed
- Test time shifting methodology to ensure it behaves as expected
- Dummy input to TSConv to ensure it behaves as expected

Run this with: python3 -m tests.test_model_layers
"""
import unittest

import torch

import unet.unet_models as um

class SpectrogramTests(unittest.TestCase):
    def test_torch_version(self):
        # Tests for versions of torch
        self.assertEqual(torch.__version__, '2.2.0')

    def test_TS(self):
        # Explicitly test the time shifting functionality of TSConv
        tsconv = um.TSConv(in_channels=4, out_channels=4)

        # Define input data and expected output
        input_tensor = torch.tensor([[[[ 0,  1,  2]],
                                      [[ 3,  4,  5]],
                                      [[ 6,  7,  8]],
                                      [[ 9, 10, 11]]]])
        expected_tensor = torch.tensor([[[[ 0,  1,  2]],
                                       [[ 3,  4,  5]],
                                       [[ 7,  8, 0]],
                                       [[ 0, 9, 10]]]])
        # Get the output tensor from the TS unction
        output_tensor = tsconv._do_ts(input_tensor)
        
        # Check to see if the output is the expected
        self.assertTrue(torch.equal(output_tensor, expected_tensor))

    def test_TSConv(self):
        # Tests to see if this layer has no errors, and returns correct shape
        tsconv = um.TSConv(in_channels=4, out_channels=8)
        input_tensor = torch.tensor([[[[ 0.,  1.,  2.]],
                                      [[ 3.,  4.,  5.]],
                                      [[ 6.,  7.,  8.]],
                                      [[ 9., 10., 11.]]]])
        output = tsconv.forward(input_tensor)
        
        self.assertEqual(output.shape[0], 1) # Expect that the number of samples has not changed
        self.assertEqual(output.shape[1], 8) # Expect that the number of channels is equal to output
        self.assertEqual(output.shape[2], 1) # Expect that the number of frequency bins has not changed
        self.assertEqual(output.shape[3], 3) # Expect that the number of time steps has not changed

if __name__ == '__main__':
    unittest.main()