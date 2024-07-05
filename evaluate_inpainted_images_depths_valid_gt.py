import argparse
import csv
import os
from glob import glob

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

EPSILON = 1e-8

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir", type=str, required=True,
    )
    parser.add_argument(
        "--gt_path", type=str, required=True,
    )
    parser.add_argument("--selected_scenes", type=str, default=None)  # /project/3dlg-hcvc/diorama/wss/wss_scenes_selected.txt
    parser.add_argument("--v2", action="store_true")

    args = parser.parse_args()

    if args.selected_scenes:
        with open(args.selected_scenes, "r") as f:
            selected_scenes = f.read().splitlines()

    subexp_dirs = [path for path in glob(f"{args.exp_dir}/*") if os.path.isdir(path)]

    # Now do same with applicable metrics for depth (also normalize the depths first)
    rows = []
    rows.append(["subexp_dir", "image_name", "rmse", "absrel"])
    for subexp_dir in subexp_dirs:
        print(subexp_dir)
        inpainted_depths = glob(f"{subexp_dir}/*/inpainted-scene-gt-depth.npy")
        for inpainted_depth in inpainted_depths:
            scene_id = os.path.dirname(inpainted_depth).split("/")[-1]
            if args.selected_scenes and scene_id not in selected_scenes:
                continue
            gt_depth = cv2.imread(f"{args.gt_path}/{scene_id}/depth.room.png", cv2.IMREAD_UNCHANGED) / 1000
            inpainted_depth = np.load(inpainted_depth)

            mask = cv2.imread(f"{subexp_dir}/{scene_id}/merged_mask.png", cv2.IMREAD_UNCHANGED)
            gt_depth[mask != 0] = 0
            inpainted_depth[mask != 0] = 0

            valid_mask = ((gt_depth > 0) & (inpainted_depth > 0))
            gt_depth_masked = gt_depth
            gt_depth_masked[~valid_mask] = 0
            inpainted_depth_masked = inpainted_depth
            inpainted_depth_masked[~valid_mask] = 0
            gt_depth_norm = (gt_depth_masked - gt_depth_masked.min()) / (gt_depth_masked.max() - gt_depth_masked.min())
            inpainted_depth_norm = (inpainted_depth_masked - inpainted_depth_masked.min()) / (inpainted_depth_masked.max() - inpainted_depth_masked.min())

            gt_depth_valid = gt_depth_norm[valid_mask]
            inpainted_depth_valid = 1 - inpainted_depth_norm[valid_mask]

            depth_color = np.zeros(inpainted_depth.shape + (3,))
            depth_color[valid_mask] = np.squeeze(cv2.applyColorMap((inpainted_depth_valid * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
            cv2.imwrite(f"{subexp_dir}/{scene_id}/inpainted-scene-depth-gt-valid-color.png", depth_color)

            rmse = np.sqrt(mean_squared_error(gt_depth_valid, inpainted_depth_valid))
            abs_diff = np.abs(inpainted_depth_valid - gt_depth_valid)
            absrel = np.mean(abs_diff / gt_depth_valid)
            rows.append([subexp_dir, scene_id, rmse, absrel])
    with open(f"{args.exp_dir}/results_depth_gt.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Calculate and write averages per subexp_dir
    rows = []
    header = ["subexp_dir", "rmse", "absrel"]
    with open(f"{args.exp_dir}/results_depth_gt.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        data = []
        for recorded_row in reader:
            row = [recorded_row[0], recorded_row[2], recorded_row[3]]
            data.append(row)
        data = np.array(data)
        subexp_dirs = np.unique(data[:, 0])
        for subexp_dir in subexp_dirs:
            subexp_rows = data[data[:, 0] == subexp_dir]
            subexp_rows = subexp_rows[:, 1:].astype(float)
            avg_row = np.mean(subexp_rows, axis=0)
            rows.append([subexp_dir, *avg_row])
    rows = [header] + rows
    with open(f"{args.exp_dir}/results_depth_gt_avg.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# Now for v2 if exists
if args.v2:
    rows = []
    rows.append(["subexp_dir", "image_name", "rmse", "absrel"])
    for subexp_dir in subexp_dirs:
        print(subexp_dir)
        inpainted_depths = glob(f"{subexp_dir}/*/inpainted-scene-gt-depth-v2.npy")
        for inpainted_depth in inpainted_depths:
            scene_id = os.path.dirname(inpainted_depth).split("/")[-1]
            if args.selected_scenes and scene_id not in selected_scenes:
                continue
            gt_depth = cv2.imread(f"{args.gt_path}/{scene_id}/depth.room.png", cv2.IMREAD_UNCHANGED) / 1000
            inpainted_depth = np.load(inpainted_depth)

            mask = cv2.imread(f"{subexp_dir}/{scene_id}/merged_mask.png", cv2.IMREAD_UNCHANGED)
            gt_depth[mask != 0] = 0
            inpainted_depth[mask != 0] = 0

            valid_mask = ((gt_depth > 0) & (inpainted_depth > 0))
            gt_depth_masked = gt_depth
            gt_depth_masked[~valid_mask] = 0
            inpainted_depth_masked = inpainted_depth
            inpainted_depth_masked[~valid_mask] = 0
            gt_depth_norm = (gt_depth_masked - gt_depth_masked.min()) / (gt_depth_masked.max() - gt_depth_masked.min())
            inpainted_depth_norm = (inpainted_depth_masked - inpainted_depth_masked.min()) / (inpainted_depth_masked.max() - inpainted_depth_masked.min())

            gt_depth_valid = gt_depth_norm[valid_mask]
            inpainted_depth_valid = 1 - inpainted_depth_norm[valid_mask]

            depth_color = np.zeros(inpainted_depth.shape + (3,))
            depth_color[valid_mask] = np.squeeze(cv2.applyColorMap((inpainted_depth_valid * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
            cv2.imwrite(f"{subexp_dir}/{scene_id}/inpainted-scene-depth-gt-valid-color-v2.png", depth_color)

            rmse = np.sqrt(mean_squared_error(gt_depth_valid, inpainted_depth_valid))
            abs_diff = np.abs(inpainted_depth_valid - gt_depth_valid)
            absrel = np.mean(abs_diff / gt_depth_valid)
            rows.append([subexp_dir, scene_id, rmse, absrel])
    with open(f"{args.exp_dir}/results_depth_gt-v2.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Calculate and write averages per subexp_dir
    rows = []
    header = ["subexp_dir", "rmse", "absrel"]
    with open(f"{args.exp_dir}/results_depth_gt-v2.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        data = []
        for recorded_row in reader:
            row = [recorded_row[0], recorded_row[2], recorded_row[3]]
            data.append(row)
        data = np.array(data)
        subexp_dirs = np.unique(data[:, 0])
        for subexp_dir in subexp_dirs:
            subexp_rows = data[data[:, 0] == subexp_dir]
            subexp_rows = subexp_rows[:, 1:].astype(float)
            avg_row = np.mean(subexp_rows, axis=0)
            rows.append([subexp_dir, *avg_row])
    rows = [header] + rows
    with open(f"{args.exp_dir}/results_depth_gt_avg-v2.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
