from mmdet.models.builder import DETECTORS
from .obb_single_stage import OBBSingleStageDetector
from mmdet.core import arb2result_withmask, arb2result


@DETECTORS.register_module()
class FCOSOBB(OBBSingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOSOBB, self).__init__(backbone, neck, bbox_head, train_cfg,
                                      test_cfg, pretrained)

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale
        )
        bbox_type = getattr(self.bbox_head, 'bbox_type', 'hbb')
        with_polar_mask = getattr(self.bbox_head, 'with_polar_mask', False)
        if with_polar_mask:
            bbox_results = [
                arb2result_withmask(det_bboxes, det_masks, det_labels, self.bbox_head.num_classes, img_metas[0], bbox_type)
                for det_bboxes, det_labels, det_masks in bbox_list
            ]
        else:
            bbox_results = [
                arb2result(det_bboxes, det_labels, self.bbox_head.num_classes, bbox_type)
                for det_bboxes, det_labels in bbox_list
            ]
        return bbox_results[0]

    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch
