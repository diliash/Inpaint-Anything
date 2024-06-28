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
    parser.add_argument("--v2", action="store_true")

    args = parser.parse_args()

    lpips_metric = lpips.LPIPS(net='alex')

    subexp_dirs = glob(f"{args.exp_dir}/*")[:2]
    rows = []
    rows.append(["subexp_dir", "image_name", "psnr", "ssim", "lpips"])
    for subexp_dir in subexp_dirs:
        print(subexp_dir)
        inpainted_images = glob(f"{subexp_dir}/*/inpainted_merged.png")
        for inpainted_image in inpainted_images:
            scene_id = os.path.dirname(inpainted_image).split("/")[-1]
            gt_image = cv2.imread(f"{args.gt_path}/{scene_id}/room.png")
            inpainted_image = cv2.imread(inpainted_image)
            gt_image = cv2.resize(gt_image, (inpainted_image.shape[1], inpainted_image.shape[0]))
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)

            psnr = cv2.PSNR(gt_image, inpainted_image)
            ssim_score = ssim(gt_image, inpainted_image, multichannel=True, win_size=3)

            lpips_score = lpips_metric(torch.tensor(gt_image).permute(2, 0, 1).unsqueeze(0).float() / 255,
                           torch.tensor(inpainted_image).permute(2, 0, 1).unsqueeze(0).float() / 255)
            lpips_score = lpips_score.item()

            rows.append([subexp_dir, scene_id, psnr, ssim_score, lpips_score])
    with open(f"{args.exp_dir}/results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Calculate and write averages per subexp_dir
    rows = []
    header = ["subexp_dir", "psnr", "ssim", "lpips"]
    with open(f"{args.exp_dir}/results.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        data = []
        for recorded_row in reader:
            row = [recorded_row[0], recorded_row[2], recorded_row[3], recorded_row[4]]
            data.append(row)
        data = np.array(data)
        subexp_dirs = np.unique(data[:, 0])
        for subexp_dir in subexp_dirs:
            subexp_rows = data[data[:, 0] == subexp_dir]
            subexp_rows = subexp_rows[:, 1:].astype(float)  # Exclude the subexp_dir column
            avg_row = np.mean(subexp_rows, axis=0)
            rows.append([subexp_dir, *avg_row])
    # Prepend header
    rows = [header] + rows
    with open(f"{args.exp_dir}/results_avg.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Now do same with applicable metrics for depth (also normalize the depths first)
    rows = []
    rows.append(["subexp_dir", "image_name", "rmse", "absrel"])
    for subexp_dir in subexp_dirs:
        print(subexp_dir)
        inpainted_depths = glob(f"{subexp_dir}/*/inpainted-scene-depth.npy")
        for inpainted_depth in inpainted_depths:
            scene_id = os.path.dirname(inpainted_depth).split("/")[-1]
            gt_depth = cv2.imread(f"{args.gt_path}/{scene_id}/depth.room.png", cv2.IMREAD_UNCHANGED) / 1000
            gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())
            inpainted_depth = np.load(inpainted_depth)

            valid_mask = ((gt_depth > 0) & (inpainted_depth > 0))
            gt_depth_valid = gt_depth[valid_mask]
            inpainted_depth_valid = inpainted_depth[valid_mask]
            inpainted_depth_valid = 1 - (inpainted_depth_valid - inpainted_depth_valid.min()) / (inpainted_depth_valid.max() - inpainted_depth_valid.min())

            inpainted_depth_valid_save = np.zeros_like(inpainted_depth)
            inpainted_depth_valid_save[valid_mask] = (inpainted_depth_valid * 1000).astype(np.uint16)
            depth_color = np.zeros(inpainted_depth.shape + (3,))
            depth_color[valid_mask] = np.squeeze(cv2.applyColorMap((inpainted_depth_valid * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
            cv2.imwrite(f"{subexp_dir}/{scene_id}/inpainted-scene-depth-valid-color.png", depth_color)

            rmse = np.sqrt(mean_squared_error(gt_depth_valid, inpainted_depth_valid))

            abs_diff = np.abs(inpainted_depth_valid - gt_depth_valid)
            absrel = np.mean(abs_diff / gt_depth_valid)
            rows.append([subexp_dir, scene_id, rmse, absrel])
    with open(f"{args.exp_dir}/results_depth.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Calculate and write averages per subexp_dir
    rows = []
    header = ["subexp_dir", "rmse", "absrel"]
    with open(f"{args.exp_dir}/results_depth.csv", "r") as f:
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
    with open(f"{args.exp_dir}/results_depth_avg.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# Now for v2 if exists
if args.v2:
    rows = []
    rows.append(["subexp_dir", "image_name", "rmse", "absrel"])
    for subexp_dir in subexp_dirs:
        print(subexp_dir)
        inpainted_depths = glob(f"{subexp_dir}/*/inpainted-scene-depth-v2.npy")
        for inpainted_depth in inpainted_depths:
            scene_id = os.path.dirname(inpainted_depth).split("/")[-1]
            gt_depth = cv2.imread(f"{args.gt_path}/{scene_id}/depth.room.png", cv2.IMREAD_UNCHANGED) / 1000
            gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())
            inpainted_depth = np.load(inpainted_depth)

            valid_mask = ((gt_depth > 0) & (inpainted_depth > 0))
            gt_depth_valid = gt_depth[valid_mask]
            inpainted_depth_valid = inpainted_depth[valid_mask]
            inpainted_depth_valid = 1 - (inpainted_depth_valid - inpainted_depth_valid.min()) / (inpainted_depth_valid.max() - inpainted_depth_valid.min())

            inpainted_depth_valid_save = np.zeros_like(inpainted_depth)
            inpainted_depth_valid_save[valid_mask] = (inpainted_depth_valid * 1000).astype(np.uint16)
            depth_color = np.zeros(inpainted_depth.shape + (3,))
            depth_color[valid_mask] = np.squeeze(cv2.applyColorMap((inpainted_depth_valid * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
            cv2.imwrite(f"{subexp_dir}/{scene_id}/inpainted-scene-depth-valid-color-v2.png", depth_color)

            rmse = np.sqrt(mean_squared_error(gt_depth_valid, inpainted_depth_valid))

            abs_diff = np.abs(inpainted_depth_valid - gt_depth_valid)
            absrel = np.mean(abs_diff / gt_depth_valid)
            rows.append([subexp_dir, scene_id, rmse, absrel])
    with open(f"{args.exp_dir}/results_depth-v2.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Calculate and write averages per subexp_dir
    rows = []
    header = ["subexp_dir", "rmse", "absrel"]
    with open(f"{args.exp_dir}/results_depth-v2.csv", "r") as f:
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
    with open(f"{args.exp_dir}/results_depth_avg-v2.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
