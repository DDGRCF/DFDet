import torch

from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox.transforms import distance2bbox
from mmdet.core.bbox.transforms_obb import distance2obb

from ..base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class OBB2DistCoder(BaseBBoxCoder):

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert gt_bboxes.size(-1) == bboxes.size(-1) == 5
        encoded_bboxes = obb2dist(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self, bboxes, pred_bboxes, max_shape=None, mode="obb"):
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = dist2obb(
            bboxes, pred_bboxes, self.means, self.stds, mode=mode)
        return decoded_bboxes


def obb2dist(proposals,
             gt,
             means=(0., 0., 0., 0., 0.),
             stds=(1., 1., 1., 1., 1.)):
    proposals = proposals.float()
    gt = gt.float()
    gt_ctr_xy, gt_wh, gt_theta = gt.split([2, 2, 1], dim=-1)
    pr_ctr_xy, pr_wh, _ = proposals.split([2, 2, 1], dim=-1)
    Cos, Sin = torch.cos(gt_theta), torch.sin(gt_theta)
    Matrix = torch.stack((Cos, -Sin, Sin, Cos), dim=-1).reshape(-1, 2, 2)
    offset = pr_ctr_xy - gt_ctr_xy
    offset = torch.matmul(Matrix, offset[..., None])
    offset = offset.squeeze(-1)
    offset_x, offset_y = offset[..., 0], offset[..., 1]
    W, H = gt_wh[..., 0], gt_wh[..., 1]
    left = (W / 2 + offset_x) / pr_wh[..., 0]
    right = (W / 2 - offset_x) / pr_wh[..., 0]
    top = (H / 2 + offset_y) / pr_wh[..., 1]
    bottom = (H / 2 - offset_y) / pr_wh[..., 1]
    deltas = torch.stack((left, top, right, bottom, gt_theta.squeeze(-1)), -1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def dist2obb(proposals,
             deltas,
             means=(0., 0., 0., 0., 0.),
             stds=(1., 1., 1., 1., 1.),
             mode="obb"):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    pr_ctr_xy, pr_wh, _ = proposals.split([2, 2, 1], -1)
    denorm_deltas[..., :4] *= pr_wh.repeat(1, 2)
    if mode == "obb":
        bboxes = distance2obb(pr_ctr_xy, denorm_deltas)
    elif mode == "hbb":
        bboxes = distance2bbox(pr_ctr_xy, denorm_deltas)
    else:
        raise NotImplementedError
    return bboxes
