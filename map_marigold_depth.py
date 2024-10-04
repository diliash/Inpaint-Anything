import os
import sys

import cv2
import numpy as np
import open3d as o3d

sys.path.append("../..")
from diorama.utils.depth_util import back_project_depth_to_points

os.makedirs("./marigold_mapped", exist_ok=True)

model_id = "scene00037"
data_path = "/project/3dlg-hcvc/diorama/wss/scenes"
depth_path = "./inpainted_merged_pred.npy"
image_path = f"{data_path}/{model_id}/room.png"

raw_image = cv2.imread(image_path)

depth = np.load(depth_path)
normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())
depth_metric = normalized_depth * 1. + 1.
back_project_depth_to_points(depth_metric, image=cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB), save_path=os.path.join("./marigold_mapped", f"inpainted-{model_id}.ply"))
