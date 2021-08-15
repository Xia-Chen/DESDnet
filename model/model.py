import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_p(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, is_bias=True):
        super(Encoder_p, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1,
                                bias=is_bias),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1,
                                bias=is_bias),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1,
                                bias=is_bias),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1,
                                bias=is_bias),
            )

        self.moduleConv1 = Basic(n_channel * (t_length - 1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)

        return tensorConv4, tensorConv1, tensorConv2, tensorConv3


class Encoder_r(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(Encoder_r, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )

        def Fusion(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=1, stride=1),
                torch.nn.Tanh())

        self.moduleConv1 = Basic(n_channel * (t_length - 1), 64)
        self.fusion1 = Fusion(64 * 2, 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.fusion2 = Fusion(128 * 2, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(128, 256)
        self.fusion3 = Fusion(256 * 2, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)

    def forward(self, x, skip1, skip2, skip3):
        tensorConv1 = self.moduleConv1(x)

        cat1 = torch.cat((skip1, tensorConv1), dim=1)
        tensorFusion1 = self.fusion1(cat1)
        tensorPool1 = self.modulePool1(tensorFusion1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        cat2 = torch.cat((skip2, tensorConv2), dim=1)
        tensorFusion2 = self.fusion2(cat2)
        tensorPool2 = self.modulePool2(tensorFusion2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        cat3 = torch.cat((skip3, tensorConv3), dim=1)
        tensorFusion3 = self.fusion3(cat3)
        tensorPool3 = self.modulePool3(tensorFusion3)

        tensorConv4 = self.moduleConv4(tensorPool3)

        return tensorConv4, tensorConv1, tensorConv2, tensorConv3


class Decoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(Decoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        # self.moduleConv = Basic(1024, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(128, n_channel, 64)

    def forward(self, x, skip1, skip2, skip3):
        # tensorConv = self.moduleConv(x)      # for 1024
        tensorUpsample4 = self.moduleUpsample4(x)
        cat4 = torch.cat((skip3, tensorUpsample4), dim=1)

        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim=1)

        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim=1)

        output = self.moduleDeconv1(cat2)

        return output


#
class Fusion(torch.nn.Module):
    def __init__(self, feature_dim=512):
        super(Fusion, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=1, stride=1),
                torch.nn.Tanh())

        self.moduleConv = Basic(feature_dim * 2, feature_dim)

    def forward(self, x):
        output = self.moduleConv(x)
        return output


class Context(torch.nn.Module):
    def __init__(self, inputdim, middledim):
        super(Context, self).__init__()

        def Stream1(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=1, stride=1, dilation=1),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intInput, kernel_size=1, stride=1, dilation=1),
                torch.nn.ReLU())

        def Stream2(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=1, stride=1, dilation=1),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, dilation=1,
                                padding=1),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intInput, kernel_size=1, stride=1, dilation=1),
                torch.nn.ReLU())

        def Stream3(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=1, stride=1, dilation=1),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, dilation=3,
                                padding=3),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intInput, kernel_size=1, stride=1, dilation=1),
                torch.nn.ReLU())

        def Stream4(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=1, stride=1, dilation=1),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, dilation=5,
                                padding=5),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intInput, kernel_size=1, stride=1, dilation=1),
                torch.nn.ReLU())

        def Fusion(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=1, stride=1),
                torch.nn.Tanh())

        self.Conv1 = Stream1(inputdim, middledim)
        self.Conv2 = Stream2(inputdim, middledim)
        self.Conv3 = Stream3(inputdim, middledim)
        self.Conv4 = Stream4(inputdim, middledim)
        self.Fusion = Fusion(inputdim * 4, inputdim)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)
        x3 = self.Conv3(x)
        x4 = self.Conv4(x)
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        output = self.Fusion(x_cat)

        return output


class FC_net(torch.nn.Module):
    def __init__(self, input_dim, n_hidden_1, n_hidden_2, output_dim=1):
        super(FC_net, self).__init__()

        def Basic(input_dim, output_dim):
            return torch.nn.Sequential(
                torch.nn.Linear(in_features=input_dim, out_features=output_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.3))

        self.modulePool = torch.nn.MaxPool2d(kernel_size=32, stride=32)
        self.moduleFc1 = Basic(input_dim, n_hidden_1)
        self.moduleFc2 = Basic(n_hidden_1, n_hidden_2)
        self.moduleFc3 = torch.nn.Linear(n_hidden_2, output_dim)
        self.moduleOutput = torch.nn.Softmax(dim=1)

    def forward(self, x, is_training):
        x = self.modulePool(x)
        x = x.view(x.size(0), -1)
        if is_training:
            b_size, dim = x.size(0), x.size(1)
            mean = np.zeros(dim)
            var = np.eye(dim) * 0.0001
            pseudo_sample = np.random.multivariate_normal(mean, var, b_size)
            pseudo_sample = torch.tensor(pseudo_sample, device='cuda', dtype=torch.float32)
            appended_sample = torch.cat([x, pseudo_sample], dim=0)
        else:
            appended_sample = x
        x = self.moduleFc1(appended_sample)
        x = self.moduleFc2(x)
        logits = self.moduleFc3(x)
        output = self.moduleOutput(logits)
        return logits, output


class convAE(torch.nn.Module):
    def __init__(self, n_channel=3, t_length=5, feature_dim=512):
        super(convAE, self).__init__()

        self.encoder_p = Encoder_p(t_length, n_channel)  # for prediction
        self.encoder_r = Encoder_p(2, n_channel)  # for reconstruction
        self.encoder_e = Encoder_p(2, n_channel)  # for encoding
        self.decoder = Decoder(t_length, n_channel)
        self.fusion = Fusion(feature_dim)

    def forward(self, x1, x2, n_channel, is_training=True):
        fea_p, skip1, skip2, skip3 = self.encoder_p(x2[:, :4 * n_channel, ...])
        fea_r, _, _, _ = self.encoder_r(x2[:, 4 * n_channel:, ...])
        fea_s = torch.cat((fea_p, fea_r), dim=1)  # concate two feature map in different branches by channel
        fea = self.fusion(fea_s)
        output = self.decoder(fea, skip1, skip2, skip3)
        fea_r_r, _, _, _ = self.encoder_e(output)

        return output, fea_r_r, fea_r




