import torch

from mmdet.ops.nms_rotated import arb_batched_nms
from mmdet.core.bbox.transforms_obb import get_bbox_dim


def multiclass_arb_nms(multi_bboxes,
                       multi_scores,
                       score_thr,
                       nms_cfg,
                       max_num=-1,
                       score_factors=None,
                       bbox_type='hbb',
                       extra_dets=None
                       ):
    bbox_dim = get_bbox_dim(bbox_type)
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > bbox_dim:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, bbox_dim)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, bbox_dim)

    if extra_dets is not None:
        extra_dets = extra_dets[:, None].expand(*(-1, num_classes, *extra_dets.shape[1:]))
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]
    if extra_dets is not None:
        extra_dets = extra_dets[valid_mask]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, bbox_dim+1))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        if extra_dets is not None:
            extra_dets = multi_bboxes.new_zeros((0, extra_dets.shape[-1]), dtype=torch.long)
            return bboxes, labels, extra_dets
        else:
            return bboxes, labels, 

    dets, keep = arb_batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if extra_dets is not None:
        return dets, labels[keep], extra_dets[keep]
    else:
        return dets, labels[keep]
