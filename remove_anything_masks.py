import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from lama_inpaint import inpaint_img_with_lama

matplotlib.use('TkAgg')
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


def refine_masks(masks):
    # Iteratively merge masks with overlapping bboxes

    bboxes = {idx: get_bbox_from_mask(mask) for idx, mask in masks.items()}
    merged_masks = {}
    merged_bboxes = {}
    for i, (idx, bbox) in enumerate(bboxes.items()):
        if i == 0:
            merged_masks[i] = masks[idx]
            merged_bboxes[i] = bbox
        else:
            merged = False
            for j in range(i):
                if np.any(np.logical_and(bbox[0] >= merged_bboxes[j][0], bbox[1] <= merged_bboxes[j][1])):
                    merged_masks[j] += masks[idx]
                    merged_bboxes[j] = np.array([
                        np.minimum(merged_bboxes[j][0], bbox[0]),
                        np.maximum(merged_bboxes[j][1], bbox[1])
                    ])
                    merged = True
                    break
            if not merged:
                merged_masks[i] = masks[idx]
                merged_bboxes[i] = bbox
    for i, mask in merged_masks.items():
        visualize_mask(mask)


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
        "--depth_path", type=str, required=True,
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
    """Example usage:
    python remove_anything_gt.py \
        --input_img /project/3dlg-hcvc/diorama/wss/scenes/scene00000/scene.png \
        --seg_path /project/3dlg-hcvc/diorama/wss/scenes/scene00000/seg.png \
        --dilate_kernel_size 0 \
        --output_dir ./results/gt/separate/scene00000 \
        --sam_model_type "vit_h" \
        --sam_ckpt sam_vit_h_4b8939.pth \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt ./pretrained_models/big-lama
        --method iterative-depth
        --depth_path /project/3dlg-hcvc/diorama/wss/scenes/scene00000/depth.png
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])

    if args.method == "iterative-depth" and args.depth_path is None:
        raise ValueError("Depth path is required for iterative-depth method")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_img_to_array(args.input_img)

    mask = load_img_to_array(args.seg_path)
    masks = {}
    for insatnce in np.unique(mask):
        if insatnce != 1:
            masks[insatnce] = ((mask == insatnce).astype(np.uint8) * 255)

    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        masks = {idx: dilate_mask(mask, args.dilate_kernel_size) for idx, mask in masks.items()}
    os.makedirs(args.output_dir, exist_ok=True)
    if args.method == "separate":
        for idx, mask in masks.items():
            img_inpainted_p = f"{args.output_dir}/inpainted_with_{idx}.png"
            img_inpainted = inpaint_img_with_lama(
                img, mask, args.lama_config, args.lama_ckpt, device=device)
            save_array_to_img(img_inpainted, img_inpainted_p)
    elif args.method == "merged":
        # merge all masks
        merged_mask = np.zeros_like(mask)
        for idx, mask in masks.items():
            merged_mask += mask
        img_inpainted_p = f"{args.output_dir}/inpainted_merged.png"
        img_inpainted = inpaint_img_with_lama(
            img, merged_mask, args.lama_config, args.lama_ckpt, device=device)
        save_array_to_img(img_inpainted, img_inpainted_p)
    elif args.method == "iterative-s-l":
        # Sort masks from smallest to largest area
        # Inpaint iteratively starting from the smallest mask
        # Keep track of bboxes of all inpainted masks
        # If the current mask overlaps with any of the previous masks, merge them and inpaint
        refine_masks(masks)
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
