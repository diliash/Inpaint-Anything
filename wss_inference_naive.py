import os
from glob import glob

from tqdm import tqdm

data_path = "/project/3dlg-hcvc/diorama/wss/scenes"

scenes = glob(f"{data_path}/*")
print(scenes)

# Merged
for scene in tqdm(scenes):
    scene_name = scene.split("/")[-1]
    # Launch remove_anything_masks_naive.py
    os.system(f"python remove_anything_masks_naive.py\
              --input_img {scene}/scene.png\
                --seg_path {scene}/seg.png\
                --dilate_kernel_size 10\
                --output_dir ./results/gt-rerun-vis-pcd/merged/{scene_name}\
                --sam_model_type 'vit_h'\
                --sam_ckpt sam_vit_h_4b8939.pth\
                --lama_config lama/configs/prediction/default.yaml\
                --lama_ckpt ./pretrained_models/big-lama\
                --method merged")
"""
# iterative-s-l
for scene in tqdm(scenes):
    scene_name = scene.split("/")[-1]
    # Launch remove_anything_masks_naive.py
    os.system(f"python remove_anything_masks_naive.py\
              --input_img {scene}/scene.png\
                --seg_path {scene}/seg.png\
                --dilate_kernel_size 0\
                --output_dir ./results/gt/iterative-s-l/{scene_name}\
                --sam_model_type 'vit_h'\
                --sam_ckpt sam_vit_h_4b8939.pth\
                --lama_config lama/configs/prediction/default.yaml\
                --lama_ckpt ./pretrained_models/big-lama\
                --method iterative-s-l")

# iterative-l-s
for scene in tqdm(scenes):
    scene_name = scene.split("/")[-1]
    # Launch remove_anything_masks_naive.py
    os.system(f"python remove_anything_masks_naive.py\
              --input_img {scene}/scene.png\
                --seg_path {scene}/seg.png\
                --dilate_kernel_size 0\
                --output_dir ./results/gt/iterative-l-s/{scene_name}\
                --sam_model_type 'vit_h'\
                --sam_ckpt sam_vit_h_4b8939.pth\
                --lama_config lama/configs/prediction/default.yaml\
                --lama_ckpt ./pretrained_models/big-lama\
                --method iterative-l-s")

# iterative-depth
for scene in tqdm(scenes):
    scene_name = scene.split("/")[-1]
    # Launch remove_anything_masks_naive.py
    os.system(f"python remove_anything_masks_naive.py\
              --input_img {scene}/scene.png\
                --seg_path {scene}/seg.png\
                --dilate_kernel_size 0\
                --output_dir ./results/gt/iterative-depth/{scene_name}\
                --sam_model_type 'vit_h'\
                --sam_ckpt sam_vit_h_4b8939.pth\
                --lama_config lama/configs/prediction/default.yaml\
                --lama_ckpt ./pretrained_models/big-lama\
                --method iterative-depth\
                --depth_path {scene}/depth.png")
"""