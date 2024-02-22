#!/usr/bin/env python3
# Authors: jddeetz@gmail.com
"""
This is outlines the model configuration as json data to allow for hyperparameter tuning of the models.

The json data here will be passed to the model classes to determine the number of filters/channels in each layer.
"""

# The user can specify the number of channels heading into the network, and the number of filters/outputs from each layer
UNet_model_cfg = {"inc": [1, 4], # input = # of channels in the spectrogram
                  "down1": (4, 4),
                  "down2": (4, 4),
                  "down3": (4, 8),
                  "down4": (8, 8),
                  "down5": (8, 16),
                  # Note that because of concatenation, the channels in up layers is the sum of two prior layers
                  "up1": (16+8, 8), # input = output of down5 + down4
                  "up2": (8+8, 8), # input = output of up1 + down3
                  "up3": (8+4, 4), # input = output of up2 + down2
                  "up4": (4+4, 4), # input = output of up3 + down1
                  "up5": (4+4, 4), # input = output of up4 + inc                                 
                  "outc": (4, 1), # input = output of up5, output = # of channels in the spectrogram
                 }
