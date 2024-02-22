#!/usr/bin/env python3
# Authors: jddeetz@gmail.com
"""
Defines two implementations of UNet:
1) Using Conv2D layers (UNet w/ use_ts_conv == False) 
2) Using Time Shift Conv (TSConv) layers (UNet w/ use_ts_conv == True)

Most of this code is stolen from https://github.com/milesial/Pytorch-UNet/

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from unet.model_config import UNet_model_cfg

################################################################################
# LAYER PIECES
################################################################################
class TSConv(nn.Module):
    """ Implements timeshift convolution, which:
            1) Splits filter/channel dimension into two equal slices -> static and dynamic
            2) Among the dynamic slice, half of the channels will be shifted forward one time step, and half backward
            3) A Conv2d operation will be applied to the resulting tensor

            Ex of Timeshift)
            Input (time dimension --->, channel dimension vvvvvv):
            0, 1, 2
            3, 4, 5
            6, 7, 8
            9, 10, 11

            Output:
            0, 1, 2 # static
            3, 4, 5 # static
            7, 8, 0 # dynamic, moved time index backwards, zero padding added to end
            0, 9, 10 # dynamic, moved time index forwards, zero padding added to beginning
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(1,1), padding=(1,0), bias=False)

    def _do_ts(self, x):
        """ Given a tensor x, timeshift it according to the above example.
            The tensor has shape: n_samples, n_channels, n_frequency_bins, n_time_steps
            Several examples have been added to this function to illustrate the tensor operations on an input tensor

            Example x:
            tensor([[[[ 0,  1,  2]],
                    [[ 3,  4,  5]],
                    [[ 6,  7,  8]],
                    [[ 9, 10, 11]]]])
        """
        # Split the tensor into two equal slices: static, and dynamic
        static_tensor, dynamic_tensor = torch.split(x, split_size_or_sections=x.shape[1]//2, dim=1)
        backward_shift_tensor, forward_shift_tensor = torch.split(dynamic_tensor, 
                                                          split_size_or_sections=dynamic_tensor.shape[1]//2, dim=1)
        """
        static tensor:
        tensor([[[[0, 1, 2]],
                [[3, 4, 5]]]])

        backward_shift_tensor:
        tensor([[[[6, 7, 8]]]])

        forward_shift_tensor:
        tensor([[[[ 9, 10, 11]]]])
        """
        # Add zero padding to backward_shift_tensor and forward_shift_tensor
        backward_shift_tensor = torch.nn.functional.pad(backward_shift_tensor, (1,1))
        forward_shift_tensor = torch.nn.functional.pad(forward_shift_tensor, (1,1))
        """
        backward_shift_tensor:
        tensor([[[[0, 6, 7, 8, 0]]]])

        forward_shift_tensor:
        tensor([[[[ 0,  9, 10, 11,  0]]]])
        """
        # Shift tensor forward/backward in time dimension with periodic boundary conditions
        backward_shift_tensor = torch.roll(backward_shift_tensor, shifts=-1, dims=3)
        forward_shift_tensor = torch.roll(forward_shift_tensor, shifts=1, dims=3)
        """
        backward_shift_tensor:
        tensor([[[[6, 7, 8, 0, 0]]]])

        forward_shift_tensor:
        tensor([[[[ 0,  0,  9, 10, 11]]]])
        """
        # Crop tensors in time dimension. This can also be done with torchvision, but it didn't seem worth adding a dependency
        backward_shift_tensor = backward_shift_tensor[:,:,:,1:-1]
        forward_shift_tensor = forward_shift_tensor[:,:,:,1:-1]
        """
        backward_shift_tensor:
        tensor([[[[7, 8, 0]]]])

        forward_shift_tensor:
        tensor([[[[ 0,  9, 10]]]])
        """
        # Combine tensors and return output
        output_tensor = torch.cat((static_tensor, backward_shift_tensor, forward_shift_tensor), dim=1)
        """
        output_tensor:
        tensor([[[[ 0,  1,  2]],
                [[ 3,  4,  5]],
                [[ 7,  8,  0]],
                [[ 0,  9, 10]]]])
        """
        return output_tensor
    
    def forward(self, x):
        return self.conv(self._do_ts(x))
    
class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm2d => ReLU) * 2
    
    The argument use_ts_conv tells DoubleConv to use TSConv instead of Conv2d.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, use_ts_conv=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_ts_conv:
            self.conv1 = TSConv(in_channels, mid_channels)
            self.conv2 = TSConv(mid_channels, out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), stride=(1,1), padding=1, bias=False)
            self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), stride=(1,1), padding=1, bias=False)

        self.double_conv = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            self.conv2,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)

################################################################################
# UNET LAYERS
################################################################################
class Down(nn.Module):
    """Downscaling with maxpool then double conv
    
    The argument use_ts_conv tells DoubleConv to use TSConv instead of Conv2d.
    """

    def __init__(self, in_channels, out_channels, use_ts_conv=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 1)),
            DoubleConv(in_channels, out_channels, use_ts_conv=use_ts_conv)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv
    
    The argument use_ts_conv tells DoubleConv to use TSConv instead of Conv2d.
    """

    def __init__(self, in_channels, out_channels, use_ts_conv=False):
        super().__init__()
        # Using bilinear upsampling linearly interpolates to generate new locations.
        # align_corners preserves the values in the corners, rather than interpolating
        self.up = nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels, use_ts_conv=use_ts_conv)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Determines differences in the size of the 2nd (frequency bins) and 3rd (time step) dimensions
        diffF = x2.size()[2] - x1.size()[2]
        diffT = x2.size()[3] - x1.size()[3]

        # Pads x1 'None' values, to prepare for cat operation with x2
        x1 = F.pad(x1, [diffT // 2, diffT - diffT // 2,
                        diffF // 2, diffF - diffF // 2])
        
        # Concatenates x2 and x1, along the filter direction
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Upscaling then double conv
    
    The argument use_ts_conv tells OutConv to use TSConv instead of Conv2d.
    """
    def __init__(self, in_channels, out_channels, use_ts_conv=False):
        # Inherits the properties of nn.Module __init__
        super(OutConv, self).__init__()
        # Add padding in reflect mode, which just uses the edge values for the padding values
        # If no padding is added, then output values dimensions will be smaller than training data
        if use_ts_conv:
            self.conv = TSConv(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1,1), 
                                  padding=1, padding_mode="reflect")

    def forward(self, x):
        return self.conv(x)

################################################################################
# UNET DEFINIION
################################################################################
class UNet(nn.Module):
    """
    Definition of U-Net

    Args:
        model_cfg (dict): defines number of input channels, output filters in each layer. See unet/model_config.py
        use_ts_conv (bool): If True, use TSConv layers instead of Conv2d layers
    """
    def __init__(self, model_cfg, use_ts_conv=False):
        # Inherit all of the __init__ properties of nn.Module
        super(UNet, self).__init__()
        #The input data can have <4 channels, making TSConv impossible, use Conv2d in the input layer if input channels < 4
        if use_ts_conv and model_cfg["inc"][0] < 4:
            ts_conv_in_out = False
        else:
            ts_conv_in_out = use_ts_conv
        self.inc = (DoubleConv(model_cfg["inc"][0], model_cfg["inc"][1], use_ts_conv=ts_conv_in_out))

        # Defines Conv layers going down
        self.down1 = (Down(model_cfg["down1"][0], model_cfg["down1"][1], use_ts_conv))
        self.down2 = (Down(model_cfg["down2"][0], model_cfg["down2"][1], use_ts_conv))
        self.down3 = (Down(model_cfg["down3"][0], model_cfg["down3"][1], use_ts_conv))
        self.down4 = (Down(model_cfg["down4"][0], model_cfg["down4"][1], use_ts_conv))
        self.down5 = (Down(model_cfg["down5"][0], model_cfg["down5"][1], use_ts_conv))

        # Defines upsampling/concatenation layers going up in network
        self.up1 = (Up(model_cfg["up1"][0], model_cfg["up1"][1], use_ts_conv))
        self.up2 = (Up(model_cfg["up2"][0], model_cfg["up2"][1], use_ts_conv))
        self.up3 = (Up(model_cfg["up3"][0], model_cfg["up3"][1], use_ts_conv))
        self.up4 = (Up(model_cfg["up4"][0], model_cfg["up4"][1], use_ts_conv))
        self.up5 = (Up(model_cfg["up5"][0], model_cfg["up5"][1], use_ts_conv))

        # Defines output convolution
        self.outc = (OutConv(model_cfg["outc"][0], model_cfg["outc"][1], ts_conv_in_out))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        output = self.outc(x)
        return output

if __name__ == '__main__':
    # If running this as main, print a summary of each layer, and the number of parameters.
    # Run this with python3 -m unet.unet_models

    # Define the input size of the UNet models
    input_size = (1, 256, 100)

    # Print a summary of UNet using Conv2d (no TSConv)
    unet_conv2d = UNet(model_cfg=UNet_model_cfg, use_ts_conv=False)
    summary(unet_conv2d, input_size)

    # Print a summary of UNet using TSConv
    unet_conv2d = UNet(model_cfg=UNet_model_cfg, use_ts_conv=True)
    summary(unet_conv2d, input_size)