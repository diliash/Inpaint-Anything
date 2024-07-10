import os
from glob import glob

import cv2
import numpy as np
import open3d as o3d


def save_visualization(geometries, filename, width=512, height=512):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    vis.get_render_option().background_color = [1.0, 1.0, 1.0]  # White background
    vis.update_renderer()
    vis.poll_events()
    vis.update_geometry(geometries[0])
    
    image = vis.capture_screen_float_buffer(do_render=True)
    cv2.imwrite(filename, np.array(image)[:, :, ::-1] * 255)
    vis.destroy_window()

def segment_multiple_planes(pcd, distance_threshold=0.05, ransac_n=4, num_iterations=1000, min_points=100, max_planes=4):
    planes = []
    plane_colors = []
    remaining_points = pcd
    pcds = []
    plane_models = []
    
    while True and len(planes) < max_planes:
        plane_model, inliers = remaining_points.segment_plane(distance_threshold=distance_threshold,
                                                              ransac_n=ransac_n,
                                                              num_iterations=num_iterations)
        
        if len(inliers) < min_points:
            break
        
        inlier_cloud = remaining_points.select_by_index(inliers)
        pcds.append(inlier_cloud)
        plane_models.append(plane_model)
        print(f"Found a plane with {len(inliers)} inliers.")
        outlier_cloud = remaining_points.select_by_index(inliers, invert=True)
        
        plane_color = np.random.rand(3)
        inlier_cloud.paint_uniform_color(plane_color)
        
        planes.append(inlier_cloud)
        plane_colors.append(plane_color)
        
        remaining_points = outlier_cloud
    
    return planes, plane_colors, remaining_points, pcds, plane_models


if __name__ == "__main__":
    exp_path = "/local-scratch2/diliash/diorama/third_party/Inpaint-Anything/results/gt-rerun-vis-pcd/merged"
    pcd_type = "inpainted_pcd_marigold-lcm"

    failed_scenes = []
    scene_dirs = glob(f"{exp_path}/*")
    for scene_dir in scene_dirs:
        scene_id = scene_dir.split("/")[-1]
        if scene_id not in ['scene00113']:
            continue
        scene_pcd = o3d.io.read_point_cloud(f"{scene_dir}/{pcd_type}.ply")
        try:
            segmented_planes, plane_colors, scene_pcd_remain, pcds, plane_models = segment_multiple_planes(scene_pcd)
        except:
            print(f"Failed to segment planes for scene {scene_id}.")
            failed_scenes.append(scene_id)
            continue
        vis_geometries = segmented_planes + [scene_pcd_remain]
        save_visualization(vis_geometries, f"{scene_dir}/{pcd_type}_segmentation.png")
        os.makedirs(f"{scene_dir}/{pcd_type}_segmentation", exist_ok=True)
        for i, pcd in enumerate(pcds):
            o3d.io.write_point_cloud(f"{scene_dir}/{pcd_type}_segmentation/{i}.ply", pcd)
            np.save(f"{scene_dir}/{pcd_type}_segmentation/{i}.npy", plane_models[i])
    print(f"Failed to segment {len(failed_scenes)} scenes. Failed scenes: {failed_scenes}")