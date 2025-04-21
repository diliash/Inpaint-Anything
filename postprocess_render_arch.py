import os
from glob import glob

import cv2
import numpy as np
import pyrender
import seaborn as sns
import trimesh
from pyrender import DirectionalLight, OffscreenRenderer, PerspectiveCamera
from tqdm import tqdm

fx = 784/(2*np.tan((np.pi/3)/2))
fy = 784/(2*np.tan((np.pi/3)/2))
cx = 504
cy = 392

os.environ["PYOPENGL_PLATFORM"] = "egl"

K = np.array([
    [fx, 0, cx, 0],
    [0, fy, cy, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

yfov = 2 * np.arctan(784/(2*fy))

views_dir = "views_dir"
segmentation_types = ["segmentation_name"]

exp_paths = ["exp_path"]

camera = PerspectiveCamera(yfov=yfov)
light = DirectionalLight(color=np.ones(3), intensity=3)
renderer = OffscreenRenderer(viewport_width=1008, viewport_height=784)

for exp_path in exp_paths:
    scene_dirs = glob(f"{exp_path}/*")
    for segmentation_type in segmentation_types:
        for scene_dir in tqdm(scene_dirs):
            scene_id = scene_dir.split("/")[-1]
            scene_dir = f"{scene_dir}/{segmentation_type}"
            # print(scene_dir)
            if not os.path.exists(scene_dir):
                print(f"Skipping {scene_dir}")
                continue
            if "segmentation_image.png" in os.listdir(scene_dir) and "segmentation_image_pcd.png" not in os.listdir(scene_dir):
                # Rename all segmentation_image.png, npy, _vis.png to segmentation_image_pcd
                os.rename(f"{scene_dir}/segmentation_image.png", f"{scene_dir}/segmentation_image_pcd.png")
                os.rename(f"{scene_dir}/segmentation_image.npy", f"{scene_dir}/segmentation_image_pcd.npy")
                os.rename(f"{scene_dir}/segmentation_image_vis.png", f"{scene_dir}/segmentation_image_pcd_vis.png")
            if not os.path.exists(f"{scene_dir}/arch/arch.ply"):
                print(f"Skipping {scene_dir}")
                continue
            arch = trimesh.load(f"{scene_dir}/arch/arch.ply", process=False, force="mesh")
            # arch.apply_transform(view_trans)
            arch_p = pyrender.Mesh.from_trimesh(arch)
            scene = pyrender.Scene()
            scene.add(arch_p)

            scene.add(camera)
            scene.add(light, pose=np.eye(4))
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES)
            cv2.imwrite(f"{scene_dir}/arch_vis.png", color)
            depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
            cv2.imwrite(f"{scene_dir}/arch_depth_vis.png", cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET))
            np.save(f"{scene_dir}/plane_depth.npy", depth)

            # Render segmentation image - load and color each plane separately, render flat, backgorund = 20
            plane_mesh_dict = {}
            for path in glob(f"{scene_dir}/arch/components/*.ply"):
                plane = trimesh.load(path, process=False, force="mesh")
                plane_id = int(path.split("/")[-1].split(".")[0].split("_")[-1])
                plane_mesh_dict[plane_id] = plane
            n_planes = len(plane_mesh_dict)
            segmentation_scene = pyrender.Scene()
            nodemap = {}
            for idx, plane in plane_mesh_dict.items():
                # color = [idx, idx, idx]
                # plane.visual.vertex_colors = color
                plane_p = pyrender.Mesh.from_trimesh(plane)
                plane_node = pyrender.Node(mesh=plane_p)
                segmentation_scene.add_node(plane_node)
                nodemap[plane_node] = idx
            segmentation_scene.add(camera)

            segmentation_color, _ = renderer.render(segmentation_scene, pyrender.RenderFlags.SEG, nodemap)
            segmentation_color = np.array(segmentation_color[:, :, 0])
            segmentation_color[segmentation_color == 255] = 20
            cv2.imwrite(f"{scene_dir}/segmentation_image.png", segmentation_color)
            np.save(f"{scene_dir}/segmentation_image.npy", segmentation_color)

            n_unique = len(np.unique(segmentation_color))
            colormap = sns.color_palette("husl", n_unique)
            segmentation_viz = np.zeros((segmentation_color.shape[0], segmentation_color.shape[1], 3), dtype=np.uint8)
            for i, unique_plane in enumerate(np.unique(segmentation_color)):
                segmentation_viz[segmentation_color == unique_plane] = np.array(colormap[i]) * 255
            cv2.imwrite(f"{scene_dir}/segmentation_image_vis.png", segmentation_viz)