import torch
import torch.nn as nn


class MultiScale(torch.nn.Module):

    def __init__(self, chn, scale=1.0):
        super(MultiScale, self).__init__()
        self.scale = torch.nn.Parameter(torch.full((1, chn, 1, 1), scale))

    def forward(self, x):
        return x * self.scale


class GMultiScale(torch.nn.Module):

    def __init__(self, chn, groups=1, scale=1.0):
        super(GMultiScale, self).__init__()
        assert chn % groups == 0
        self.groups = groups
        chn = int(chn // groups)
        self.scale = torch.nn.Parameter(torch.full((1, chn, 1, 1), scale))

    def forward(self, x):
        scale = self.scale.repeat(1, self.groups, 1, 1)
        return x * scale
