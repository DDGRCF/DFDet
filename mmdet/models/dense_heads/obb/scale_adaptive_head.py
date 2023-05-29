import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmdet.core import (
    build_anchor_generator,
    build_assigner,
    build_bbox_coder,
    build_sampler,
    force_fp32,
    images_to_levels,
    mintheta_obb,
    multi_apply,
    multiclass_arb_nms,
    obb_anchor_inside_flags,
    unmap,
)

from mmdet.models.builder import HEADS, build_loss

from .obb_fcos_head import OBBFCOSHead

INF = 1e8


class FeatureEnhanceModule(nn.Module):

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act'),
                 stack_num=1,
                 use_se=False):
        super().__init__()
        assert stack_num > 0, "stack num must greater than 0"

        self.use_se = use_se
        self.convs = nn.ModuleList()

        for i in range(stack_num + 2):
            if i == 0 or i == stack_num + 1:
                _kernel_size, _stride, _padding, _dilation, _groups = [
                    1, 1, 0, 1, 1
                ]
            else:
                _kernel_size, _stride, _padding, _dilation, _groups = kernel_size, stride, padding, dilation, groups

            if i == 0:
                in_chs, out_chs = in_channels, feat_channels
            elif i == stack_num + 1:
                in_chs, out_chs = feat_channels, out_channels
            else:
                in_chs, out_chs = feat_channels, feat_channels

            conv = ConvModule(
                in_channels=in_chs,
                out_channels=out_chs,
                kernel_size=_kernel_size,
                stride=_stride,
                padding=_padding,
                bias=bias,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=inplace,
                with_spectral_norm=with_spectral_norm,
                padding_mode=padding_mode,
                order=order,
                dilation=_dilation,
                groups=_groups)
            self.convs.append(conv)

        if self.use_se:
            self.layer_attention = SELayer(out_channels)

        if in_channels != out_channels:
            self.skip = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=inplace,
                with_spectral_norm=with_spectral_norm,
                padding_mode=padding_mode,
                order=order)
        else:
            self.skip = None

    def init_weights(self):
        if self.use_se:
            self.layer_attention.init_weights()

        for m in self.convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        if self.skip != None:
            for m in self.skip.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)

    def forward(self, x):
        _x = x.clone()
        if self.skip != None:
            _x = self.skip(_x)

        for conv in self.convs:
            x = conv(x)

        if self.use_se:
            x = self.layer_attention(x)

        return x + _x


class FeatureProjectionModule(nn.Module):

    def __init__(self,
                 feat_channels,
                 stacked_convs,
                 la_down_rate=32,
                 conv_cfg=None,
                 norm_cfg=None):
        super().__init__()
        self.in_channels = feat_channels * stacked_convs
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.la_down_rate = la_down_rate
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels // la_down_rate, stacked_convs, 1),
            nn.Sigmoid())

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=norm_cfg is None)

    def init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat):
        b, _, h, w = feat.shape
        avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)  # (b, in, 1, 1)

        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv.conv.weight.reshape(
                1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels,
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,
                                                    w)

        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)
        return feat


class SELayer(nn.Module):

    def __init__(self, channel, reduction=32):
        super().__init__()
        self.layer_attention = nn.Sequential(
            nn.Conv2d(channel, reduction, 1), nn.ReLU(inplace=True),
            nn.Conv2d(reduction, channel, 1), nn.Sigmoid())

    def init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, (1, 1))
        y = self.layer_attention(y)
        return x * y


@HEADS.register_module()
class ScaleAdaptiveHead(OBBFCOSHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_dcn=0,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 scale_theta=True,
                 dcn_on_first_conv=False,
                 reg_loss_wh_thre=5,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='CIoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_trig=dict(type='L1Loss', loss_weight=0.2),
                 bbox_coder=dict(
                     type='OBB2DistCoder',
                     target_means=(0., 0., 0., 0., 0.),
                     target_stds=(1., 1., 1., 1., 1.)),
                 anchor_generator=dict(
                     type='Theta0AnchorGenerator',
                     ratios=[1.0],
                     octave_base_scale=8,
                     scales_per_octave=1,
                     center_offset=0.5,
                     strides=[8, 16, 32, 64, 128]),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):

        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.scale_theta = scale_theta
        self.dcn_on_first_conv = dcn_on_first_conv
        self.num_dcn = num_dcn
        self.reg_loss_wh_thre = reg_loss_wh_thre

        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.anchor_center_offset = anchor_generator["center_offset"]
        self.num_anchors = self.anchor_generator.num_base_anchors[0]

        super().__init__(
            num_classes,
            in_channels,
            norm_on_bbox=norm_on_bbox,
            centerness_on_reg=centerness_on_reg,
            scale_theta=scale_theta,
            loss_centerness=loss_centerness,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)

        self.loss_trig = build_loss(loss_trig)
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type="OBBPseudoSampler")
            self.sampler = build_sampler(sampler_cfg, context=self)

    def _init_layers(self):
        self.inter_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if i < self.num_dcn:
                conv_cfg = dict(type='DCNv2', deform_groups=4)
            else:
                conv_cfg = self.conv_cfg

            chn = self.in_channels if i == 0 else self.feat_channels
            self.inter_convs.append(
                FeatureEnhanceModule(
                    chn,
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    use_se=True,
                ))

        self.cls_attn = FeatureProjectionModule(
            self.feat_channels,
            self.stacked_convs,
            64,
            conv_cfg,
            norm_cfg=self.norm_cfg)

        self.reg_attn = FeatureProjectionModule(
            self.feat_channels,
            self.stacked_convs,
            64,
            conv_cfg,
            norm_cfg=self.norm_cfg)

        self.conv_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)

        self.conv_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * self.reg_dim, 3, padding=1)

        self.conv_centerness = nn.Conv2d(
            self.feat_channels * self.num_anchors, 1, 3, padding=1)

        self.conv_theta = nn.Conv2d(
            self.feat_channels * self.num_anchors, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.scale_t = Scale(1.0)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.inter_convs:
            m.init_weights()
        bias_cls = bias_init_with_prob(0.01)

        self.cls_attn.init_weights()
        self.reg_attn.init_weights()

        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)
        normal_init(self.conv_centerness, std=0.01)
        normal_init(self.conv_theta, std=0.01)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        theta_preds = []
        ctr_preds = []
        for x, scale in zip(feats, self.scales):
            inter_feats = []
            for inter_conv in self.inter_convs:
                x = inter_conv(x)
                inter_feats.append(x)
            feat = torch.cat(inter_feats, 1)

            reg_feat = self.reg_attn(feat)
            cls_feat = self.cls_attn(feat)

            if self.centerness_on_reg:
                centerness = self.conv_centerness(reg_feat)
            else:
                centerness = self.conv_centerness(cls_feat)

            reg_pred = scale(self.conv_reg(reg_feat)).float().exp()
            theta_pred = self.scale_t(self.conv_theta(reg_feat)).float()

            cls_score = self.conv_cls(cls_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_pred)
            theta_preds.append(theta_pred)
            ctr_preds.append(centerness)

        return tuple(cls_scores), tuple(bbox_preds), tuple(theta_preds), tuple(
            ctr_preds)

    @force_fp32(
        apply_to=('cls_scores', 'reg_preds', 'theta_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             reg_preds,
             theta_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            reg_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): Centerss for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(reg_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=cls_scores[0].device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        labels, label_weights, bbox_targets, _ = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            gt_labels,
            img_metas,
            label_channels=label_channels)
        num_imgs = cls_scores[0].size(0)

        # flatten cls_scores, reg_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_reg_preds = [
            reg_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for reg_pred in reg_preds
        ]
        flatten_theta_preds = [
            theta_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for theta_pred in theta_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_reg_preds = torch.cat(flatten_reg_preds)
        flatten_theta_preds = torch.cat(flatten_theta_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_anchors = torch.cat(
            [torch.vstack(anchors) for anchors in zip(*anchor_list)])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero(
                        as_tuple=False).reshape(-1)

        num_pos = len(pos_inds)
        pos_centerness = flatten_centerness[pos_inds]

        # prepare for different loss
        # pos_theta_preds = flatten_theta_preds[pos_inds].clone()
        flatten_bbox_preds = torch.cat(
            [flatten_reg_preds, flatten_theta_preds], dim=1)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_theta_preds = flatten_theta_preds[pos_inds]
            pos_points = flatten_anchors[pos_inds]

            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds, mode="hbb")
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets, mode="hbb")

            # centerness weighted iou loss
            pos_decoded_target_w = (pos_decoded_target_preds[:, 2] -
                                    pos_decoded_target_preds[:, 0]).abs()
            pos_decoded_target_h = (pos_decoded_target_preds[:, 3] -
                                    pos_decoded_target_preds[:, 1]).abs()

            pos_decoded_target_ratio = torch.maximum(
                pos_decoded_target_w / pos_decoded_target_h,
                pos_decoded_target_h / pos_decoded_target_w)

            pos_decoded_target_scale = torch.where(
                pos_decoded_target_ratio < self.reg_loss_wh_thre,
                pos_decoded_target_ratio.new_tensor(1.),
                (0.02 * (pos_decoded_target_ratio - self.reg_loss_wh_thre)).exp())

            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets * pos_decoded_target_scale,
                avg_factor=pos_centerness_targets.sum())

            loss_trig = self.loss_trig(
                pos_theta_preds,
                pos_bbox_targets[..., [4]],
                weight=pos_decoded_target_scale)

            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_trig = pos_bbox_preds[..., 4].sum()
            loss_bbox = pos_bbox_preds[..., :4].sum()
            loss_centerness = pos_centerness.sum()

        loss_cls = self.loss_cls(
            flatten_cls_scores,
            flatten_labels,
            weight=label_weights,
            avg_factor=num_pos + num_imgs)

        loss_total = dict(
            loss_bbox=loss_bbox,
            loss_trig=loss_trig,
            loss_cls=loss_cls,
            loss_centerness=loss_centerness)

        return loss_total

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    gt_labels_list=None,
                    img_metas=None,
                    gt_bboxes_ignore=None,
                    unmap_outputs=True,
                    label_channels=1):
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        if gt_bboxes_ignore is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (_, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         _, _) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)

        bbox_targets_list = [
            bbox_targets.reshape(-1, 5) for bbox_targets in bbox_targets_list
        ]
        labels_list = [labels.reshape(-1) for labels in labels_list]
        label_weights_list = [
            label_weights.reshape(-1) for label_weights in label_weights_list
        ]
        bbox_weights_list = [
            bbox_weights.reshape(-1, 5) for bbox_weights in bbox_weights_list
        ]
        label_weights = torch.cat(label_weights_list)
        bbox_weights = torch.cat(bbox_weights_list)
        return labels_list, label_weights, bbox_targets_list, bbox_weights

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        inside_flags = obb_anchor_inside_flags(flat_anchors, valid_flags,
                                               img_meta["img_shape"][:2],
                                               self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        gt_bboxes = mintheta_obb(gt_bboxes)
        if gt_bboxes_ignore is not None:
            gt_bboxes_ignore = mintheta_obb(gt_bboxes)
        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if hasattr(self, 'bbox_coder'):
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    @force_fp32(
        apply_to=('cls_scores', 'reg_preds', 'trig_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   reg_preds,
                   trig_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=None):
        assert len(cls_scores) == len(reg_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            reg_pred_list = [
                reg_preds[i][img_id].detach() for i in range(num_levels)
            ]
            trig_pred_list = [
                trig_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list, reg_pred_list,
                                                 trig_pred_list,
                                                 centerness_pred_list,
                                                 mlvl_anchors, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           reg_preds,
                           theta_preds,
                           centernesses,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(reg_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, reg_pred, theta_pred, centerness, anchors in zip(
                cls_scores, reg_preds, theta_preds, centernesses,
                mlvl_anchors):
            assert cls_score.size()[-2:] == reg_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            theta_pred = theta_pred.permute(1, 2, 0).reshape(-1, 1)
            reg_pred = reg_pred.permute(1, 2, 0).reshape(-1, 4)
            # bbox_pred = torch.cat([reg_pred, theta_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                reg_pred = reg_pred[topk_inds, :]
                theta_pred = theta_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bbox_pred = torch.cat((reg_pred, theta_pred), dim=1)
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_arb_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness,
            bbox_type='obb')
        return det_bboxes, det_labels

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        # return torch.sqrt(torch.abs(centerness_targets))
        return torch.sqrt(centerness_targets)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        num_imgs = len(img_metas)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for _, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
