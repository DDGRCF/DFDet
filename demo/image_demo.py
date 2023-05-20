import os
import glob
from argparse import ArgumentParser

import tqdm
import matplotlib as mpl
mpl.use('Agg')

from mmdet.apis import inference_detector, init_detector


def main():
    parser = ArgumentParser()
    parser.add_argument('img_source_path', help='Image source file')
    parser.add_argument('img_save_path', help="Image save file")
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    # show the results
    img_suffix = ["png", "jpg", "jpeg", "tif"]
    if os.path.isdir(args.img_source_path):
        img_set = []
        os.makedirs(args.img_save_path, exist_ok=True)
        for suffix in img_suffix:
            img_set.extend(list(glob.glob(os.path.join(args.img_source_path, "*." + suffix))))
        for img in tqdm.tqdm(img_set):
            result = inference_detector(model, img)
            save = os.path.join(args.img_save_path, os.path.basename(img))
            show_result_pyplot(model, img, save, result, score_thr=args.score_thr)
    else:
        result = inference_detector(model, args.img_source_path)
        show_result_pyplot(model, args.img_source_path, args.img_save_path, result, score_thr=args.score_thr)

def show_result_pyplot(model, img, save_path, result, score_thr=0.3, fig_size=(15, 10)):
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False, out_file=save_path)


if __name__ == '__main__':
    main()
