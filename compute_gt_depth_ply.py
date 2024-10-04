import os
import sys
from glob import glob

import cv2
import numpy as np
import open3d as o3d

sys.path.append("../..")
from diorama.utils.depth_util import back_project_depth_to_points

DEBUG = True

data_path = "/project/3dlg-hcvc/diorama/wss/scenes"
exp_path = "/local-scratch2/diliash/diorama/third_party/Inpaint-Anything/results/gt-rerun-vis-pcd/merged"

def debug_vis(depth, image):
    points = back_project_depth_to_points(depth, image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    colors = np.array(image).reshape(-1, 3) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

scene_dirs = glob(f"{exp_path}/*")
for scene_dir in scene_dirs:
    scene_id = scene_dir.split("/")[-1]

    image_path = f"{data_path}/{scene_id}/room.png"
    raw_image = cv2.imread(image_path)

    depth = cv2.imread(f"{data_path}/{scene_id}/depth.room.png", cv2.IMREAD_UNCHANGED)
    if DEBUG:
        debug_vis(depth, raw_image)

    depth = depth.astype(np.float32) / 1000.
    if DEBUG:
        debug_vis(depth, raw_image)

    depth = (depth - depth.min()) / (depth.max() - depth.min())
    if DEBUG:
        debug_vis(depth, raw_image)

    depth_metric = depth * 1. + 1.
    if DEBUG:
        debug_vis(depth_metric, raw_image)

    back_project_depth_to_points(depth, image=cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB), save_path=os.path.join(exp_path, f"{scene_id}/gt_room_depth.ply"))

    # Now the same for gt version

    image_path = f"{data_path}/{scene_id}/scene.png"
    raw_image = cv2.imread(image_path)

    depth = cv2.imread(f"{data_path}/{scene_id}/depth.png", cv2.IMREAD_UNCHANGED)
    if DEBUG:
        debug_vis(depth, raw_image)

    depth = depth.astype(np.float32) / 1000.
    if DEBUG:
        debug_vis(depth, raw_image)

    depth /= depth.max()
    if DEBUG:
        debug_vis(depth, raw_image)

    """depth_metric = depth * 1. + 1.
    if DEBUG:
        debug_vis(depth_metric, raw_image)"""

    back_project_depth_to_points(depth, image=cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB), save_path=os.path.join(exp_path, f"{scene_id}/gt_scene_depth.ply"))
