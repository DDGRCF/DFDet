import torch
from numpy import pi

from ..base_bbox_coder import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class TrigCircularCoder(BaseBBoxCoder):
    '''
    theta range (-pi/4, pi/4]
    '''
    def __init__(self,
                 target_means=(0., 0.),
                 target_stds=(1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, thetas):
        assert thetas.size(-1) == 1
        sin_targets, cos_targets = torch.sin(4 * thetas), torch.cos(4 * thetas)
        targets = torch.cat([sin_targets, cos_targets], dim=-1)

        means = targets.new_tensor(self.means)
        stds = targets.new_tensor(self.stds)
        targets = targets.sub_(means).div_(stds)
        return targets

    def decode(self, preds):
        assert preds.size(-1) % 2 == 0
        means = preds.new_tensor(self.means).repeat(preds.size(-1) // 2)
        stds = preds.new_tensor(self.stds).repeat(preds.size(-1) // 2)
        denorm_preds = preds * stds + means

        Sin = denorm_preds[..., 0::2]
        Cos = denorm_preds[..., 1::2]
        thetas = torch.atan2(Sin, Cos) / 4
        return thetas


@BBOX_CODERS.register_module()
class LinearCircularCoder(BaseBBoxCoder):

    def __init__(self,
                 target_means=(0., 0.),
                 target_stds=(1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, thetas):
        assert thetas.size(-1) == 1
        thetas = 4 * thetas # extend the range of thetas to (-180, 180)
        linear_sin_pos = 1 - torch.abs(thetas - pi/2) / (pi/2)
        linear_sin_neg = torch.abs(thetas + pi/2) / (pi/2) - 1
        linear_sin_targets = torch.where(thetas > 0, linear_sin_pos, linear_sin_neg)
        linear_cos_targets = 1 - torch.abs(thetas) / (pi/2)
        targets = torch.cat([linear_sin_targets, linear_cos_targets], dim=-1)

        means = targets.new_tensor(self.means)
        stds = targets.new_tensor(self.stds)
        targets = targets.sub_(means).div_(stds)
        return targets

    def decode(self, preds):
        assert preds.size(-1) % 2 == 0
        means = preds.new_tensor(self.means).repeat(preds.size(-1) // 2)
        stds = preds.new_tensor(self.stds).repeat(preds.size(-1) // 2)
        denorm_preds = preds * stds + means

        linear_sin = denorm_preds[..., 0::2]
        linear_cos = denorm_preds[..., 1::2]
        norm = torch.abs(linear_sin) + torch.abs(linear_cos)
        linear_sin = linear_sin / norm
        linear_cos = linear_cos / norm

        thetas1 = (pi / 2) * (linear_cos - 1)
        thetas2 = (pi / 2) * (1 - linear_cos)
        thetas = torch.where(linear_sin < 0, thetas1, thetas2)
        thetas = thetas / 4
        return thetas
