import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from lama_inpaint import inpaint_img_with_lama

# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from utils import dilate_mask, load_img_to_array, save_array_to_img


def visualize_mask(mask, title="Mask Visualization"):
    if mask.ndim == 2:
        plt.figure(figsize=(10, 8))
        plt.imshow(mask, cmap='gray')
        plt.title(title)
        plt.colorbar(label='Pixel Value')
        plt.axis('on')
        plt.show()
        plt.close()


def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return np.array([[x_min, y_min], [x_max, y_max]])

def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--seg_path", type=str, required=True,
        help="Path to the segmentation mask",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int,
        help="The size of the kernel for dilation. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_img_to_array(args.input_img)

    mask = load_img_to_array(args.seg_path)
    masks = {}
    for insatnce in np.unique(mask):
        if insatnce > 1:
            masks[insatnce] = ((mask == insatnce).astype(np.uint8) * 255)
    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        masks = {idx: dilate_mask(mask, args.dilate_kernel_size) for idx, mask in masks.items()}
    os.makedirs(args.output_dir, exist_ok=True)
    # merge all masks
    merged_mask = np.zeros_like(mask)
    for idx, mask in masks.items():
        merged_mask += mask
    # Save a version with merged mask removed
    masked_img = img.copy()
    masked_img[merged_mask != 0] = 0
    import cv2
    cv2.imwrite(f"{args.output_dir}/inpainted_merged_masked.png", masked_img)
    cv2.imwrite(f"{args.output_dir}/merged_mask.png", merged_mask)
    img_inpainted_p = f"{args.output_dir}/inpainted_merged.png"
    img_inpainted = inpaint_img_with_lama(
        img, merged_mask, args.lama_config, args.lama_ckpt, device=device)
    save_array_to_img(img_inpainted, img_inpainted_p)