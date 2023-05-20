import torch

from .obb_base_sampler import OBBBaseSampler
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from .obb_sampling_result import OBBSamplingResult


@BBOX_SAMPLERS.register_module()
class OBBPseudoSampler(OBBBaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples"""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples"""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        """Directly returns the positive and negative indices  of samples

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = OBBSamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result