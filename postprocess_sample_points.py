import os
from glob import glob

import numpy as np
import open3d as o3d
import pointops
import torch
import trimesh
from tqdm import tqdm


def farthest_point_downsampling_idx_pointops(points, n_samples):
    points_tensor = torch.from_numpy(points).float().cuda()

    N = points.shape[0]
    offset = torch.tensor([N], device='cuda')

    new_offset = torch.tensor([n_samples], device='cuda')

    sampled_indices = pointops.farthest_point_sampling(points_tensor, offset, new_offset)

    sampled_indices_np = sampled_indices.cpu().numpy()

    return sampled_indices_np

def sample_points(arch_mesh, num_points=20000):
    points = trimesh.sample.sample_surface(arch_mesh, count=800000)[0]
    sampled_indices = farthest_point_downsampling_idx_pointops(points, num_points)
    points = points[sampled_indices]
    return points


paths = ["exp_path"]

for path in paths:
    scene_dirs = glob(f"{path}/*")
    for scene_dir in tqdm(scene_dirs):
        scene_id = scene_dir.split("/")[-1]
        if not os.path.exists(f"{scene_dir}/arch/arch.ply"):
            print(f"Skipping {scene_dir}")
            continue
        arch = trimesh.load(f"{scene_dir}/arch/arch.ply", process=False, force="mesh")
        points = sample_points(arch)
        np.savez(f"{scene_dir}/arch/arch_points.npz", points=points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(f"{scene_dir}/arch/arch_points.ply", pcd)