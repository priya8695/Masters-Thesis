#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:22:41 2023

@author: priya
"""

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, device):
        super(UNet, self).__init__()
        self.init_channels = 32
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.max_pool_2x2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Define the convolutional layers and transposed convolutional layers
        self.enc_conv1 = self.double_conv(1, self.init_channels)
        self.enc_conv2 = self.double_conv(self.init_channels, self.init_channels * 2)
        self.enc_conv3 = self.double_conv(self.init_channels * 2, self.init_channels * 4)
        self.enc_conv4 = self.double_conv(self.init_channels * 4, self.init_channels * 8)
        self.down_conv_bottle_neck = self.double_conv(self.init_channels * 8, self.init_channels * 16)
        self.up_trans1 = nn.ConvTranspose3d(self.init_channels * 16, self.init_channels * 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_conv1 = self.double_conv(self.init_channels * 16, self.init_channels * 8)
        self.up_trans2 = nn.ConvTranspose3d(self.init_channels * 8, self.init_channels * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_conv2 = self.double_conv(self.init_channels * 8, self.init_channels * 4)
        self.up_trans3 = nn.ConvTranspose3d(self.init_channels * 4, self.init_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_conv3 = self.double_conv(self.init_channels * 4, self.init_channels * 2)
        self.up_trans4 = nn.ConvTranspose3d(self.init_channels * 2, self.init_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_conv4 = self.double_conv(self.init_channels * 2, self.init_channels)
        self.out_conv = nn.Conv3d(self.init_channels, 1, kernel_size=1)

            
    def double_conv(self, in_c, out_c):
        conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),  # Add padding here
            nn.ReLU(inplace=True), nn.BatchNorm3d(out_c),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),  # Add padding here
            nn.BatchNorm3d(out_c),nn.ReLU(inplace=True))
        return conv
    

    # Function to compute encoder part of Unet
    def downsample_block(self, image):
        convolution_output_1 = self.enc_conv1(image)
        max_pool_output_1 = self.max_pool_2x2(convolution_output_1)
        convolution_output_2 = self.enc_conv2(max_pool_output_1)
        max_pool_output_2 = self.max_pool_2x2(convolution_output_2)
        convolution_output_3 = self.enc_conv3(max_pool_output_2)
        max_pool_output_3 = self.max_pool_2x2(convolution_output_3)
        convolution_output_4 = self.enc_conv4(max_pool_output_3)
        max_pool_output_4 = self.max_pool_2x2(convolution_output_4)
        return convolution_output_1, convolution_output_2, convolution_output_3, convolution_output_4, max_pool_output_4
        

    # Function to compute decoder part of Unet
    def upsample_block(self, input_upsample, convolution_output_1,convolution_output_2, convolution_output_3, convolution_output_4):
        upsample_output_1 = self.up_trans1(input_upsample)
        concat_output_1 = self.up_conv1(torch.cat([upsample_output_1, convolution_output_4], 1))
        upsample_output_2 = self.up_trans2(concat_output_1)
        concat_output_2 = self.up_conv2(torch.cat([upsample_output_2, convolution_output_3], 1))
        upsample_output_3 = self.up_trans3(concat_output_2)
        concat_output_3 = self.up_conv3(torch.cat([upsample_output_3, convolution_output_2], 1))
        upsample_output_4 = self.up_trans4(concat_output_3)
        concat_output_4 = self.up_conv4(torch.cat([ upsample_output_4, convolution_output_1], 1))
        return concat_output_4

    def forward(self, image):
        image = image.to(self.device)
        # encoder
        convolution_output_1, convolution_output_2, convolution_output_3, convolution_output_4, max_pool_output_4 = self.downsample_block(image)
        # bottleneck
        bottle_neck_output = self.down_conv_bottle_neck(max_pool_output_4)
        # decoder
        decoder_output = self.upsample_block(bottle_neck_output, convolution_output_1,convolution_output_2, convolution_output_3, convolution_output_4)
        # output
        final_layer = self.out_conv(decoder_output)
        # print(final_layer)
        # print(final_layer.size())
        return final_layer

if __name__ == "__main__":
    image = torch.rand((1, 1, 128, 128, 128))
    model = UNet()
    model.to(model.device)
    image = image.to(model.device)
    a = model(image)
