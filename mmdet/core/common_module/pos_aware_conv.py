import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class PosAwareConv(nn.Module):

    def __init__(self,
                 in_channels,
                 pos_channels,
                 pos_kernel_size,
                 **conv_cfg):
        super().__init__()
        assert pos_kernel_size % 2 == 1
        self.in_channels = in_channels
        self.pos_channels = pos_channels
        self.pos_kernel_size = pos_kernel_size
        self.img_pad = int((pos_kernel_size - 1) / 2)

        offsets = np.arange(-self.img_pad, self.img_pad + 1)
        self.x_offsets = np.tile(offsets, pos_kernel_size)
        self.y_offsets = np.repeat(offsets, pos_kernel_size)

        out_channels = pos_channels * pos_kernel_size ** 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            **conv_cfg)

    def forward(self, feat):
        import pdb
        pdb.set_trace()
        img_pad = self.img_pad
        pos_xs = self.conv(feat).split(self.pos_channels, dim=1)

        pos_aware_xs = []
        for pos_x, x, y in zip(pos_xs, self.x_offsets, self.y_offsets):
            pad_shape = (img_pad-x, img_pad+x, img_pad-y, img_pad+y)
            pos_aware_xs.append(F.pad(
                pos_x, pad_shape, mode='constant', value=0))

        pos_aware_xs = torch.cat(pos_aware_xs, dim=1)
        pos_aware_xs = pos_aware_xs[..., img_pad:-img_pad, img_pad:-img_pad]
        return pos_aware_xs
