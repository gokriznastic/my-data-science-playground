import glob
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from sklearn.utils import shuffle
from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


def conv_seq(input_channel, output_channel, kernel_size=3, padding = 1):
    return nn.Sequential(

        nn.Conv3d(in_channels=input_channel,out_channels=output_channel,kernel_size=kernel_size,padding=padding),
        nn.Dropout(0.5),
        nn.InstanceNorm3d(output_channel),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels=output_channel, out_channels=output_channel, kernel_size=kernel_size, padding=padding),
        nn.Dropout(0.5),
        nn.InstanceNorm3d(output_channel),
        nn.ReLU(inplace=True),

    )


def conv_seq_without_dropout(input_channel, output_channel, kernel_size=3, padding = 1):
    return nn.Sequential(

        nn.Conv3d(in_channels=input_channel,out_channels=output_channel,kernel_size=kernel_size,padding=padding),
        nn.InstanceNorm3d(output_channel),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels=output_channel, out_channels=output_channel, kernel_size=kernel_size, padding=padding),
        nn.InstanceNorm3d(output_channel),
        nn.ReLU(inplace=True),

    )

def up_conv_seq(input_channel, output_channel, kernel_size=1, padding=0):
    return nn.Sequential(

        nn.Conv3d(in_channels=input_channel,out_channels=output_channel,kernel_size=kernel_size,padding=padding),
        nn.InstanceNorm3d(output_channel),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels=output_channel, out_channels=output_channel, kernel_size=kernel_size, padding=padding),
        nn.InstanceNorm3d(output_channel),
        nn.ReLU(inplace=True),

    )

def deep_sup(input_channel, output_channel, scale_factor):
    return nn.Sequential(

        nn.Conv3d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, stride=1, padding=0),
        nn.Upsample(scale_factor=scale_factor, mode='trilinear'),
        nn.Sigmoid(),

    )

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.maxpool(self.W_x(x))
        psi = self.relu(g1+x1)
        alpha = self.upsample(self.psi(psi))

        return x*alpha


class UNet(nn.Module):
    def __init__(self, input_channel, nclasses, dsv=False):
        super().__init__()

        self.dsv = dsv
        self.conv1 = conv_seq_without_dropout(input_channel, 16)
        self.conv2 = conv_seq_without_dropout(16, 32)
        self.conv3 = conv_seq(32, 64)
        self.conv4 = conv_seq(64, 128)
        self.conv5 = conv_seq(128, 256)

        self.conv6 = conv_seq(256,512)

        self.attn5 = Attention_block(F_g=512,F_l=256,F_int=256)
        self.upconv5 = up_conv_seq(512,256)
        self.conv5_ = conv_seq(512,256)

        self.attn4 = Attention_block(F_g=256,F_l=128,F_int=128)
        self.upconv4 = up_conv_seq(256,128)
        self.conv4_ = conv_seq(256,128)

        self.attn3 = Attention_block(F_g=128,F_l=64,F_int=64)
        self.upconv3 = up_conv_seq(128,64)
        self.conv3_ = conv_seq(128,64)

        self.attn2 = Attention_block(F_g=64,F_l=32,F_int=32)
        self.upconv2 = up_conv_seq(64,32)
        self.conv2_ = conv_seq_without_dropout(64,32)

        self.attn1 = Attention_block(F_g=32,F_l=16,F_int=16)
        self.upconv1 = up_conv_seq(32,16)
        self.conv1_ = conv_seq_without_dropout(32,16)

        self.conv0_ = nn.Conv3d(16, 1, 3, padding=1)


        self.maxpool = nn.MaxPool3d(2)
        # self.maxpool_ = nn.MaxPool3d((1,2,2))


        # self.upsample_ = nn.Upsample(size=(6,8,12), mode='trilinear')
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

        if (dsv == True):
            self.dsv4 = deep_sup(128, 1, 8)
            self.dsv3 = deep_sup(64, 1, 4)
            self.dsv2 = deep_sup(32, 1, 2)

            self.final = nn.Conv3d(4, 1, 1)


    def forward(self, image):

        x1 = self.conv1(image) # 96, 128, 192
        p1 = self.maxpool(x1) # 48, 64, 96

        x2 = self.conv2(p1)
        p2 = self.maxpool(x2) # 24, 32, 48

        x3 = self.conv3(p2)
        p3 = self.maxpool(x3) # 12, 16, 24

        x4 = self.conv4(p3)
        p4 = self.maxpool(x4) # 6, 8, 12

        x5 = self.conv5(p4)
        p5 = self.maxpool(x5) # 3, 4, 6

        x = self.conv6(p5) # 3, 4, 6

        x6 = self.upconv5(x)
        x6 = self.upsample(x6) # 6, 8, 12
        x5 = self.attn5(x,x5)
        x6 = torch.cat([x6, x5], dim=1)
        x6 = self.conv5_(x6)

        x7 = self.upconv4(x6)
        x7 = self.upsample(x7) # 12, 16, 24
        x4 = self.attn4(x6,x4)
        x7 = torch.cat([x7, x4], dim=1)
        x7 = self.conv4_(x7)
        # print(x7.size())

        x8 = self.upconv3(x7)
        x8 = self.upsample(x8) # 24, 32, 48
        x3 = self.attn3(x7,x3)
        x8 = torch.cat([x8, x3], dim=1)
        x8 = self.conv3_(x8)
        # print(x8.size())

        x9 = self.upconv2(x8)
        x9 = self.upsample(x9) # 48, 64, 96
        x2 = self.attn2(x8,x2)
        x9 = torch.cat([x9, x2], dim=1)
        x9 = self.conv2_(x9)
        # print(x9.size())

        x10 = self.upconv1(x9)
        x10 = self.upsample(x10) # 96, 128, 192
        x1 = self.attn1(x9,x1)
        x10 = torch.cat([x10, x1], dim=1)
        x10 = self.conv1_(x10)

        output = self.conv0_(x10)
        output = torch.sigmoid(output)

        if (self.dsv == True):
            aux4 = self.dsv4(x7)
            aux3 = self.dsv3(x8)
            aux2 = self.dsv2(x9)

            final = self.final(torch.cat([output, aux2, aux3, aux4], dim=1))
            final = torch.sigmoid(final)

            return final
        else:
            return output

if __name__=='__main__':
    net = UNet(1,1, dsv=True)
    # net = net.float()
    if torch.cuda.is_available():
        net.cuda()

    summary(net, input_size=(1, 96, 192, 192))

    batch_size = 2
    input = np.random.rand(batch_size, 1, 96, 192, 192)
    output = net(torch.from_numpy(input).float().cuda())
    # time.sleep(100)
    print(output.size())