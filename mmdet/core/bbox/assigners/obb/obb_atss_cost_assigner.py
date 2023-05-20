import torch

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.iou_calculators import (BboxOverlaps2D, OBBOverlaps,
                                             build_iou_calculator,)
from ..assign_result import AssignResult
from ..base_assigner import BaseAssigner


def obb2hbb(bbox):
    ctr, wh, _ = bbox.split([2, 2, 1], dim=-1)
    lt = ctr - wh / 2
    br = ctr + wh / 2
    return torch.cat([lt, br], dim=-1)


@BBOX_ASSIGNERS.register_module()
class OBBATSSCostAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self, topk, iou_calculator=dict(type='OBBOverlaps'),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.hbb_iou_calculator = BboxOverlaps2D()
        self.ignore_iof_thr = ignore_iof_thr
        # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    @torch.no_grad()
    def assign(self,
               epoch,
               bboxes,
               num_level_bboxes,
               cls_scores,
               bbox_preds,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        bboxes = bboxes[:, :5]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps1 = self.hbb_iou_calculator(obb2hbb(bboxes), obb2hbb(gt_bboxes))
        overlaps1 = torch.maximum(overlaps1, overlaps1.new_tensor(0.))
        overlaps2 = self.iou_calculator(bbox_preds, gt_bboxes)
        overlaps2 = torch.maximum(overlaps2, overlaps2.new_tensor(0.))
        if epoch < 8:
            coefficient = (epoch - 1) * 0.07143
        elif epoch == 8:
            coefficient = 0.5
        elif epoch > 8 and epoch < 12:
            coefficient = 0.5 + 0.125 * (epoch - 8)
        else:
            coefficient = 1
        overlaps = overlaps1.pow(1 - coefficient) * overlaps2.pow(coefficient)

        # compute cls cost for bbox and gt
        # cls_cost = torch.sigmoid(cls_scores[:, gt_labels])

        # assert cls_cost.shape == overlaps.shape

        # with torch.no_grad():
        #     overlaps = cls_cost ** (1 - self.alpha) * overlaps ** self.alpha

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        # gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        # gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0 # gt_points = gt_bboxes[:, 0: 2]
        # gt_wh = gt_bboxes[:, 2: 4] # (num_gts, 2)
        # gt_theta = gt_bboxes[:, 4]
        gt_points, gt_wh, gt_theta = gt_bboxes.split([2, 2, 1], dim=1)

        # bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        # bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        # bboxes_cx, bboxes_cy = bboxes[:, 0], bboxes[:, 1]
        # bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)
        bboxes_points = bboxes[:, 0: 2]

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt() # (num_points, num_gts)


        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for _, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(
                self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0) # (num_stages * num_candi, num_gts)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :] # (num_stages * num_candi, num_gts)

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        # ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
        #     num_gt, num_bboxes).contiguous().view(-1) # (num_gt * num_bboxes)
        # ep_bboxes_cy = bboxes_cy.view(1, -1).expand( # (num_gt * num_bboxes)
        #     num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1) # (num_stages * num_candi * num_gt)
        ep_bboxes_cxy = bboxes_points.view(1, -1, 2).expand(num_gt, num_bboxes, 2).contiguous().view(-1, 2) # (num_gts * num_bboxes, 2)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        
        Cos, Sin = torch.cos(gt_theta), torch.sin(gt_theta)
        Matrix = torch.cat((Cos, -Sin, Sin, Cos), dim=-1).reshape(
            1, num_gt, 2, 2
        ) # (1, num_gt, 2, 2)
        offset = ep_bboxes_cxy[candidate_idxs].view(-1, num_gt, 2) - gt_points # (num_stages * num_candi, num_gt, 2)
        offset = torch.matmul(Matrix, offset[..., None]).squeeze(-1) # (num_candi, num_gts, 2)
        W, H = gt_wh[..., 0], gt_wh[..., 1]

        # if self.wh_thre > 0:
        #     wh_ratio_index = (torch.maximum(W / H, H / W) > self.wh_thre).nonzero(as_tuple=False).squeeze(1)
        #     wh_max_index = torch.argmax(gt_wh, dim=1)[wh_ratio_index]
        #     wh_min_index = torch.argmin(gt_wh, dim=1)[wh_ratio_index]
        #     gt_wh[wh_ratio_index, wh_max_index] = gt_wh[wh_ratio_index, wh_min_index] * self.wh_thre
        #     W, H = gt_wh[..., 0], gt_wh[..., 1]

        offset_x, offset_y = offset[..., 0], offset[..., 1]
        l_ = W / 2 + offset_x
        r_ = W / 2 - offset_x
        t_ = H / 2 + offset_y
        b_ = H / 2 - offset_y
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        # lt_ = gt_wh / 2 + offset # (num_candi, num_gts, 2)
        # rb_ = gt_wh / 2 - offset
        # is_in_gts = torch.cat((lt_, rb_), dim=-1).min(-1)[0] > 0.01 # (num_candi, num_gts)
        is_pos = is_pos & is_in_gts


        # l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        # t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        # r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        # b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        # is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        # is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1) # (num_bboxes * num_gts)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
