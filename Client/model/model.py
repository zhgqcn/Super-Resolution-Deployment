import torch
import torch.nn as nn
import numpy as np
import math


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class RecursiveBlock(nn.Module):
    def __init__(self, d):
        super(RecursiveBlock, self).__init__()

        self.block = nn.Sequential()
        for i in range(d):
            self.block.add_module("relu_" + str(i), nn.LeakyReLU(0.2, inplace=True))

            self.block.add_module("conv_" + str(i), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                                                              stride=1, padding=1, bias=True))

    def forward(self, x):
        output = self.block(x)
        return output


class FeatureEmbedding(nn.Module):
    def __init__(self, r, d):
        super(FeatureEmbedding, self).__init__()

        self.recursive_block = RecursiveBlock(d)
        self.num_recursion = r

    def forward(self, x):
        output = x.clone()

        # The weights are shared within the recursive block!
        for i in range(self.num_recursion):
            output = self.recursive_block(output) + x

        return output


class LapSrnMS(nn.Module):
    def __init__(self, r, d, scale):
        super(LapSrnMS, self).__init__()

        self.scale = scale
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, )

        self.transpose = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4,
                                            stride=2, padding=1, bias=True)

        self.relu_features = nn.LeakyReLU(0.2, inplace=True)

        self.scale_img = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=4,
                                            stride=2, padding=1, bias=False)

        self.predict = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)

        self.features = FeatureEmbedding(r, d)

    # 这里省略了权值初始化的代码
    def forward(self, x):
        features_x2 = self.conv_input(x)  # in:3   out:64
        features_x2 = self.features(features_x2)  # in:64  out:64
        features_x2 = self.transpose(self.relu_features(features_x2))  # in:64  out:64
        predict_x2 = self.predict(features_x2)  # in:64  out:3

        # rescaled_img_x2 = x.clone()                                     # 3 channels
        rescaled_img_x2 = self.scale_img(x)  # in:3  out:1

        out_2 = predict_x2 + rescaled_img_x2  # 3 channels

        features_x4 = self.conv_input(out_2)  # in:3   out:64
        features_x4 = self.features(features_x4)  # in:64  out:64
        features_x4 = self.transpose(self.relu_features(features_x4))  # in:64  out:64
        predict_x4 = self.predict(features_x4)  # in:64  out:3

        # rescaled_img_x4 = out_2.clone()                                 # 3 channels
        rescaled_img_x4 = self.scale_img(out_2)  # in:3  out:1

        out_4 = predict_x4 + rescaled_img_x4  # 3 channels

        return out_2, out_4


