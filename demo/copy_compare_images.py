import shutil
from pathlib import Path
dir_names = ["ALL", "FSP", "TPL", "FSP_v1", "TPL_v1"]
Reference_dir = "/data/wjb/CVPR2022/vis/Selected/Compare1/ORIGIN"
Reference_dir = Path(Reference_dir)
re_img_file = list(Reference_dir.glob("*.png"))
re_img_file = [i.name for i in re_img_file]

for dir_name in dir_names:
    source_dir = "/data/wjb/CVPR2022/vis/{}"
    dist_dir = "/data/wjb/CVPR2022/vis/Selected/Compare1/{}"
    source_dir = source_dir.format(dir_name)
    dist_dir = dist_dir.format(dir_name)
    source_dir = Path(source_dir)
    dist_dir = Path(dist_dir)
    for img_file in source_dir.glob("*.png"): 
        img_file_name = img_file.name
        if img_file_name in re_img_file: 
            source_img_file = source_dir / img_file_name
            dist_img_file = dist_dir / img_file_name
            shutil.copyfile(source_img_file, dist_img_file)
            print("copy {} to {}".format(source_img_file, dist_img_file))

