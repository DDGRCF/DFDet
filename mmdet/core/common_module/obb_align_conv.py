import torch
import torch.nn as nn

from mmdet.ops import DeformConv


class OBBAlignConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 corner=True,
                 center_offset=0.):
        super(OBBAlignConv, self).__init__()
        assert kernel_size > 1 and kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.corner = corner
        self.center_offset = center_offset
        pad = int(kernel_size // 2)

        self.pad = pad
        self.conv = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad)

        idx = torch.arange(-pad, pad+1).float()
        yy, xx = torch.meshgrid(idx, idx)
        self.register_buffer('xx', xx.view(-1))
        self.register_buffer('yy', yy.view(-1))

    def forward(self, x, offsets, stride=None):
        offsets = offsets.detach() # disable the grad
        if offsets.size(1) == 5:
            offsets = self.obb2offset(offsets, stride)
        x = self.conv(x, offsets)
        return x

    def obb2offset(self, obb, stride):
        assert stride is not None
        N, _, H, W = obb.size()

        obb = obb.permute(0, 2, 3, 1)
        Ctr
