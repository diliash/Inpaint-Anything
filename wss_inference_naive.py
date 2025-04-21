import os
from glob import glob

from tqdm import tqdm

data_path = "<path_to_exp>"

scenes = glob(f"{data_path}/*")
print(scenes)

# Merged
for scene in tqdm(scenes):
    scene_name = scene.split("/")[-1]
    # Launch remove_anything_masks_naive.py
    os.system(f"python remove_anything_masks_naive_predicted_seg.py\
              --input_img {scene}/scene.png\
                --seg_path {scene}/dichotomous_2_step.png\
                --dilate_kernel_size 10\
                --output_dir {scene}\
                --lama_config lama/configs/prediction/default.yaml\
                --lama_ckpt ./pretrained_models/big-lama\ ")
