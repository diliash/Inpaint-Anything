import os
import sys
from glob import glob

import cv2
import numpy as np

sys.path.append("../..")
from diorama.utils.depth_util import back_project_depth_to_points

postfix = "marigold-lcm"
data_path = "/project/3dlg-hcvc/diorama/wss/scenes"
exp_path = "/local-scratch2/diliash/diorama/third_party/Inpaint-Anything/results/gt-rerun-vis-pcd/merged"

scene_dirs = glob(f"{exp_path}/*")
for scene_dir in scene_dirs:
    scene_id = scene_dir.split("/")[-1]

    depth = np.load(f"{scene_dir}/inpainted-scene-depth-{postfix}.npy")
    # depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_metric = depth * 1. + 1.

    image_path = f"{data_path}/{scene_id}/room.png"
    raw_image = cv2.imread(image_path)
    back_project_depth_to_points(depth_metric, image=cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB), save_path=os.path.join(exp_path, f"{scene_id}/inpainted_pcd_{postfix}.ply"))

    # Now the same for gt version

    depth = np.load(f"{scene_dir}/inpainted-scene-gt-depth-{postfix}.npy")
    # depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_metric = depth * 1. + 1.

    image_path = f"{data_path}/{scene_id}/scene.png"
    raw_image = cv2.imread(image_path)
    back_project_depth_to_points(depth_metric, image=cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB), save_path=os.path.join(exp_path, f"{scene_id}/inpainted_pcd_gt_{postfix}.ply"))
