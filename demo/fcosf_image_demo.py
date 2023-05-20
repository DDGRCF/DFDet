import os
import cv2
import glob
import torch
import numpy as np
import random
from functools import partial
from tqdm import tqdm
from mmdet.core import obb2poly
from argparse import ArgumentParser
from BboxToolkit import list_named_colors
from mmdet.apis import inference_detector, init_detector


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--nproc', type=int, default=8, help='the processing to visualize images'
    )
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--save-dir', type=str, default=None, help='the dir to save images'
    )
    args = parser.parse_args()
    color_list = get_color_list()
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    CLASSES = model.CLASSES
    # test images
    img_set = tqdm(check_path(args.img))

    kwargs = dict(
        model=model,
        class_name=CLASSES,
        score_thr=args.score_thr,
        save_dir=args.save_dir,
        color_list=color_list,
        is_vis_bbox=False,
        is_vis_polys=True,
    )

    vis_func = partial(vis, **kwargs)
    for img in img_set:
        vis_func(img)

def vis(img_path, 
        model=None,
        class_name=None, 
        score_thr=0.3, 
        save_dir=None, 
        color_list=None,
        is_vis_bbox=False, 
        is_vis_polys=False): 
    result = inference_detector(model, img_path)
    img = cv2.imread(img_path)
    bboxes_result = result[0]
    mask_result = result[1]
    num_colors_per_classes = len(bboxes_result) // len(color_list)
    for cls_id, (cls_bboxes, cls_masks) in enumerate(zip(bboxes_result, mask_result)):
        # cls_name = class_name[cls_id]
        cls_conf = cls_bboxes[:, 5]
        mask_conf = cls_conf >= score_thr
        cls_bboxes = cls_bboxes[mask_conf]
        cls_masks = cls_masks[mask_conf]
        # color_id = int(cls_id * num_colors_per_classes + \
        #                 random.randint(0, num_colors_per_classes)) % len(color_list)
        color_id = cls_id % len(color_list)
        color = color_list[color_id]
        for i, cls_bbox in enumerate(cls_bboxes):
            cls_reg = cls_bbox[:5]
            cls_poly = obb2poly_(cls_reg).astype(np.int32).reshape(-1, 1, 2)
            im_mask = draw_mask(cls_masks[i], img.shape, color)
            img = merge_image_mask(img, im_mask) 
            if is_vis_bbox:
                cv2.polylines(img, [cls_poly], True, color, 1)
            if is_vis_polys:
                cls_mask = cls_masks[i].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [cls_mask], True, color, 1)
    if save_dir is None:
        save_prefix, save_suffix = os.path.splitext(img_path)
        save_path = save_prefix + '_vis' + save_suffix
    else:
        check_dir(save_dir)
        save_path = os.path.join(save_dir, os.path.basename(img_path))

    cv2.imwrite(save_path, img)
        
def draw_mask(polys, img_shape, color=None):          
    img_h, img_w = img_shape[:2]
    polys = polys.astype(np.int32).reshape(-1, 2)
    im_mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    im_mask = cv2.drawContours(im_mask, [polys], -1, color, -1) 
    return im_mask

def merge_image_mask(img, im_mask, alpha=1, beta=0.50, gamma=0.):
    img = cv2.addWeighted(img, alpha, im_mask, beta, gamma)
    return img
    
def obb2poly_(bbox):
    is_numpy = isinstance(bbox, np.ndarray)
    if is_numpy:
        bbox = torch.from_numpy(bbox)
    bbox = obb2poly(bbox)
    if is_numpy:
        bbox = bbox.numpy()
    return bbox

def check_path(path, suffix='.png'):
    if os.path.isfile(path):
        file_set = [path]
    elif os.path.isdir(path):
        file_set = glob.glob(os.path.join(path, "*" + suffix))
    else:
        raise TypeError
    return file_set

def check_dir(dir):
    if not os.path.isfile(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
    else:
        raise TypeError

def get_color_list():
    color_dict = list_named_colors()
    color_list = list(color_dict.values())
    return color_list

if __name__ == '__main__':
    main()
