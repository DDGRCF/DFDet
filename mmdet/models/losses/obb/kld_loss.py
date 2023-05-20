import torch
import torch.nn as nn
import numpy as np
from ..utils import weighted_loss
from mmdet.models.builder import LOSSES

# @weighted_loss
# def compute_kld(pred, target, eps=1e-6, taf=1.0, angle_mode="rad"):
#     delta_x = pred[:, 0] - target[:, 0]
#     delta_y = pred[:, 1] - target[:, 1]
#     pre_angle_radian = np.pi * pred[:, 4] / 180.0
#     if angle_mode != "angle":
#         target_angle_radian = np.pi * target[:, 4] / 180.0
#     elif angle_mode == "rad":
#         target_angle_radian = target[:, 4]
#     else:
#         raise NotImplementedError

#     delta_angle_radian = pre_angle_radian - target_angle_radian 

#     kld =  0.5 * (
#                     4 * torch.pow( ( delta_x.mul(torch.cos(target_angle_radian)) + delta_y.mul(torch.sin(target_angle_radian)) ), 2) / torch.pow(target[:, 2], 2)
#                     + 4 * torch.pow( ( delta_y.mul(torch.cos(target_angle_radian)) - delta_x.mul(torch.sin(target_angle_radian)) ), 2) / torch.pow(target[:, 3], 2)
#                     )\
#             + 0.5 * (
#                     torch.pow(pred[:, 3], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
#                     + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
#                     + torch.pow(pred[:, 3], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
#                     + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
#                     )\
#             + 0.5 * (
#                     torch.log(torch.pow(target[:, 3], 2) / torch.pow(pred[:, 3], 2))
#                     + torch.log(torch.pow(target[:, 2], 2) / torch.pow(pred[:, 2], 2))
#                     )\
#             - 1.0

#     kld_loss = 1 - 1 / (taf + torch.log(kld + 1))

#     return kld_loss

def get_sigma(input, eps=1e-7):
    _, wh, theta = input.split([2, 2, 1], -1)
    wh = wh.clamp(min=eps)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    R = torch.cat((Cos, -Sin, Sin, Cos), -1).view(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    sigma = (R @ S.square() @ R.transpose(1, 2)).reshape(-1, 2, 2)
    return sigma

# def get_sigma(xywhr, eps=1e-7):
#     _shape = xywhr.shape
#     assert _shape[-1] == 5
#     xy = xywhr[..., :2]
#     wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
#     r = xywhr[..., 4]
#     cos_r = torch.cos(r)
#     sin_r = torch.sin(r)
#     R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
#     S = 0.5 * torch.diag_embed(wh)

#     sigma = R.bmm(S.square()).bmm(R.permute(0, 2, 1)).reshape(
#         _shape[:-1] + (2, 2))
#     return xy, sigma

@weighted_loss
def compute_gwd(pred, target, eps=1e-7, alpha=1.0, tau=1.0, norm=True):
    pred_xy = pred[..., :2]
    target_xy = target[..., :2]
    pred_sigma = get_sigma(pred, eps)
    target_sigma = get_sigma(target, eps)
    # m calculate
    xy_dist = (pred_xy - target_xy).square().sum(-1)
    whr_dist = pred_sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_dist = whr_dist + target_sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_tr = (pred_sigma @ target_sigma).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (pred_sigma.det() * target_sigma.det()).clamp(0).sqrt()
    whr_dist = whr_dist + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt()
    )
    dist = (xy_dist + alpha * alpha * whr_dist).clamp(0).sqrt()
    if norm:
        scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(eps)
        dist = dist / scale
    loss = 1 - 1 / (tau + torch.log1p(dist))
    return loss

# @weighted_loss
# def compute_gwd(pred, target, eps=1e-7, alpha=1.0, tau=1.0, norm=True):
#     xy_p, Sigma_p = get_sigma(pred)
#     xy_t, Sigma_t = get_sigma(target)
#     xy_distance = (xy_p - xy_t).square().sum(dim=-1)

#     whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
#     whr_distance = whr_distance + Sigma_t.diagonal(dim1=-2, dim2=-1).sum(
#         dim=-1)

#     _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
#     _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(0).sqrt()
#     whr_distance = whr_distance + (-2) * (
#         (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())

#     distance = (xy_distance + alpha * alpha * whr_distance).clamp(0).sqrt()

#     if norm:
#         scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(1e-7)
#         distance = distance / scale
    
#     loss = 1 - 1 / (tau + log1p(distance))

@weighted_loss
def compute_kld(pred, target, alpha=1.0, tau=1.0, sqrt=True, eps=1e-7):
    xy_p = pred[..., :2]
    xy_t = target[..., :2]
    Sigma_p = get_sigma(pred)
    Sigma_t = get_sigma(target)
    _shape = xy_p.shape
    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                            -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                            dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(
        dxy).view(-1)

    whr_distance = 0.5 * Sigma_p_inv.bmm(
        Sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = Sigma_p.det().log()
    Sigma_t_det_log = Sigma_t.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)
    if sqrt:
        distance = distance.clamp(0).sqrt()

    distance = distance.reshape(_shape[:-1])
    loss = 1 - 1 / (tau + torch.log1p(distance))

    return loss



@LOSSES.register_module()
class KLDLoss(nn.Module):
    def __init__(self, tau=1.0, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(KLDLoss, self).__init__()
        self.tau = tau
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred, 
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert pred.ndim == 2 and target.ndim == 2
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)

        loss = self.loss_weight * compute_kld(pred, 
                    target, weight, eps=self.eps, 
                    reduction=reduction, avg_factor=avg_factor, 
                    tau=self.tau, **kwargs)
        return loss
        

@LOSSES.register_module()
class GWDLoss(nn.Module):
    def __init__(self, tau=1.0, alpha=1.0, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(GWDLoss, self).__init__()
        self.alpha = alpha
        self.tau = tau
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred, 
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert pred.ndim == 2 and target.ndim == 2
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)

        loss = self.loss_weight * compute_gwd(pred, 
                    target, weight, eps=self.eps, 
                    reduction=reduction, avg_factor=avg_factor, 
                    alpha=self.alpha, tau=self.tau, **kwargs)
        return loss

    