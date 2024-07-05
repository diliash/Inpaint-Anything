import copy

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def segment_multiple_planes(pcd, distance_threshold=0.05, ransac_n=4, num_iterations=1000, min_points=100, max_planes=4):
    planes = []
    plane_colors = []
    remaining_points = pcd
    
    while True and len(planes) < max_planes:
        plane_model, inliers = remaining_points.segment_plane(distance_threshold=distance_threshold,
                                                              ransac_n=ransac_n,
                                                              num_iterations=num_iterations)
        
        if len(inliers) < min_points:
            break
        
        inlier_cloud = remaining_points.select_by_index(inliers)
        print(f"Found a plane with {len(inliers)} inliers.")
        outlier_cloud = remaining_points.select_by_index(inliers, invert=True)
        
        plane_color = np.random.rand(3)
        inlier_cloud.paint_uniform_color(plane_color)
        
        planes.append(inlier_cloud)
        plane_colors.append(plane_color)
        
        remaining_points = outlier_cloud
    
    return planes, plane_colors, remaining_points

pcd_type = "inpainted_pcd_v2.ply"
exp_dir = "/local-scratch2/diliash/diorama/third_party/Inpaint-Anything/results/gt-rerun-vis-pcd/merged"
scene_id = "00037"

scene_pcd = o3d.io.read_point_cloud(f"{exp_dir}/scene{scene_id}/{pcd_type}")

o3d.visualization.draw_geometries([scene_pcd])

# Multiple plane segmentation
segmented_planes, plane_colors, scene_pcd_remain = segment_multiple_planes(copy.deepcopy(scene_pcd), max_planes=4)

# Visualize segmented planes
vis_geometries = segmented_planes + [scene_pcd_remain]
o3d.visualization.draw_geometries(vis_geometries)

"""# Clustering on remaining points
labels = np.array(scene_pcd.cluster_dbscan(eps=0.005, min_points=10, print_progress=True))
max_label = labels.max()
print(f"number of clusters: {max_label + 1}")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
scene_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Visualize clustering results
vis_geometries = [scene_pcd]
o3d.visualization.draw_geometries(vis_geometries)"""