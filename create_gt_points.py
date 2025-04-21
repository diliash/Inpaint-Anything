import os
from glob import glob

import cv2
import numpy as np
import open3d as o3d
import pointops
import torch
from depth_util import back_project_depth_to_points
from tqdm import tqdm


def farthest_point_downsampling_idx_pointops(points, n_samples):
    points_tensor = torch.from_numpy(points).float().cuda()

    N = points.shape[0]
    offset = torch.tensor([N], device='cuda')

    new_offset = torch.tensor([n_samples], device='cuda')

    sampled_indices = pointops.farthest_point_sampling(points_tensor, offset, new_offset)

    sampled_indices_np = sampled_indices.cpu().numpy()

    return sampled_indices_np

gt_path = "path/to/gt/scenes"
export_path = "./gt_points"
os.makedirs(export_path, exist_ok=True)

scene_dirs = glob(f"{gt_path}/scene*")
for scene_dir in tqdm(scene_dirs):
    scene_id = scene_dir.split("/")[-1]

    depth_path = f"{scene_dir}/depth.room.png"
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000
    points = back_project_depth_to_points(depth, intrinsics="wss")
    points = points.reshape(-1, 3)

    # Downsample to 20k
    """pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.farthest_point_down_sample(20000)

    points = np.asarray(pcd.points)"""
    downsampled_idx = farthest_point_downsampling_idx_pointops(points, 20000)
    points = points[downsampled_idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(f"{export_path}/{scene_id}.ply", pcd)
    print(f"{export_path}/{scene_id}.ply")
    np.savez(f"{export_path}/{scene_id}.npz", points=points)