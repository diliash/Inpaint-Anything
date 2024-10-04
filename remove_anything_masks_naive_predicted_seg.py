import argparse
import json
import os
import sys
from pathlib import Path

import cv2
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
        "--seg_anno", type=str,
        help="Path to the segmentation annotation",
    )
    parser.add_argument(
        "--depth_path", type=str,
        help="Path to the depth map",
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
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
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
    parser.add_argument(
        "--method", type=str, default="separate",
        choices=["separate", "merged", "iterative-s-l", "iterative-l-s", "iterative-depth"],
        help="The method to use for inpainting. Default: separate",
    )
    parser.add_argument(
        "--save_intermediate", action="store_true",
        help="Save intermediate results for iterative methods.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])

    print(args.seg_path)

    if args.method == "iterative-depth" and args.depth_path is None:
        raise ValueError("Depth path is required for iterative-depth method")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_img_to_array(args.input_img)
    # Also read RGBA
    img_a = cv2.imread(args.input_img, cv2.IMREAD_UNCHANGED)

    mask = load_img_to_array(args.seg_path)
    masks = {0: mask}
    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        """depth = load_img_to_array(args.depth_path)
        sorted_masks = sorted(masks.items(), key=lambda x: np.max(depth[x[1] == 255]), reverse=True)"""
        masks = {idx: dilate_mask(mask, args.dilate_kernel_size) for idx, mask in masks.items()}
    os.makedirs(args.output_dir, exist_ok=True)
    if args.method == "separate":
        for idx, mask in masks.items():
            img_inpainted_p = f"{args.output_dir}/inpainted_with_{idx}.png"
            img_inpainted = inpaint_img_with_lama(
                img, mask, args.lama_config, args.lama_ckpt, device=device)
            save_array_to_img(img_inpainted, img_inpainted_p)
    elif args.method == "merged":
        merged_mask = np.zeros_like(mask, dtype=np.uint8)
        for idx, mask in masks.items():
            merged_mask += mask

        img_with_alpha = cv2.imread(args.input_img, cv2.IMREAD_UNCHANGED)
        if img_with_alpha.shape[2] == 3:
            img_with_alpha = cv2.cvtColor(img_with_alpha, cv2.COLOR_BGR2BGRA)

        masked_img = img_with_alpha.copy()
        masked_img[merged_mask != 0] = [0, 0, 0, 255]

        cv2.imwrite(f"{args.output_dir}/inpainted_merged_masked.png", masked_img)
        cv2.imwrite(f"{args.output_dir}/merged_mask.png", merged_mask)

        img_inpainted_p = f"{args.output_dir}/inpainted_merged.png"
        img_inpainted = inpaint_img_with_lama(
            img_with_alpha[:,:,:3], merged_mask, args.lama_config, args.lama_ckpt, device=device)

        img_inpainted_bgra = cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2BGRA)
        img_inpainted_bgra[:,:,3] = img_with_alpha[:,:,3]

        cv2.imwrite(img_inpainted_p, img_inpainted_bgra)
    elif args.method == "iterative-s-l":
        # Sort masks from smallest to largest area
        # Inpaint iteratively starting from the smallest mask
        # Keep track of bboxes of all inpainted masks
        # If the current mask overlaps with any of the previous masks, merge them and inpaint
        sorted_masks = sorted(masks.items(), key=lambda x: np.sum(x[1]))
        img_inpainted = img
        for idx, mask in sorted_masks:
            img_inpainted = inpaint_img_with_lama(
                img_inpainted, mask, args.lama_config, args.lama_ckpt, device=device)
        img_inpainted_p = f"{args.output_dir}/inpainted_iterative.png"
        save_array_to_img(img_inpainted, img_inpainted_p)
    elif args.method == "iterative-l-s":
        # Sort masks from largest to smallest area
        # Inpaint iteratively starting from the largest mask
        sorted_masks = sorted(masks.items(), key=lambda x: np.sum(x[1]), reverse=True)
        img_inpainted = img
        for idx, mask in sorted_masks:
            img_inpainted = inpaint_img_with_lama(
                img_inpainted, mask, args.lama_config, args.lama_ckpt, device=device)
        img_inpainted_p = f"{args.output_dir}/inpainted_iterative.png"
        save_array_to_img(img_inpainted, img_inpainted_p)
    elif args.method == "iterative-depth":
        # Sort masks by the largest depth value covered by each mask
        # Inpaint iteratively starting from the mask with the smallest depth value
        depth = load_img_to_array(args.depth_path)
        sorted_masks = sorted(masks.items(), key=lambda x: np.max(depth[x[1] == 255]))
        img_inpainted = img
        for idx, mask in sorted_masks:
            img_inpainted = inpaint_img_with_lama(
                img_inpainted, mask, args.lama_config, args.lama_ckpt, device=device)
        img_inpainted_p = f"{args.output_dir}/inpainted_iterative.png"
        save_array_to_img(img_inpainted, img_inpainted_p)
    else:
        raise ValueError(f"Invalid method: {args.method}")
