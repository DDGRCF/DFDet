from mmdet.models.builder import DETECTORS
from .obb_single_stage import OBBSingleStageDetector


@DETECTORS.register_module()
class ATSSOBB(OBBSingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)

    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch
