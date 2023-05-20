_base_ = './dadet_r50.py'

model = dict(
    pretrained=
    'https://download.openmmlab.com/pretrain/third_party/resnet101_caffe-3ad79236.pth',
    backbone=dict(depth=101))
