"""Standalone DINOv3 bbox visualization test.

Usage:
    uv run test_dino_bbox.py                          # uses train/train_0001.jpg
    uv run test_dino_bbox.py --image train/train_1940.jpg
    uv run test_dino_bbox.py --image train/train_0001.jpg --threshold 0.6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from transformers import AutoImageProcessor, AutoModel


# ── Config ────────────────────────���───────────���─────────────────────────────
DINO_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_IMAGE = "train/train_0001.jpg"
DEFAULT_THRESHOLD = 0.6   # top 60% attention mass → foreground
MIN_BLOB_SIZE = 50        # pixels
CROP_PADDING = 20         # pixels
OUTPUT_PATH = "test_dino_bbox_result.png"


# ── Core functions ─────────────────────────────��─────────────────────────────

def load_model():
    print(f"Loading DINOv3 from {DINO_MODEL} ...")
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
    model = AutoModel.from_pretrained(
        DINO_MODEL,
        torch_dtype=torch.bfloat16,    # bfloat16: same exponent range as float32, no softmax NaN
        device_map="auto",
        attn_implementation="eager",   # required for output_attentions=True
    )
    model.eval()
    print(f"  device: {next(model.parameters()).device}")
    print(f"  num_register_tokens: {getattr(model.config, 'num_register_tokens', 0)}")
    print(f"  patch_size: {model.config.patch_size}")
    return model, processor


@torch.inference_mode()
def get_attention_map(model, processor, image: Image.Image):
    """Return raw CLS→patch attention map (num_h, num_w) and image size."""
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    outputs = model(**inputs, output_attentions=True)

    # Debug: print attention tensor info
    attn_last = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
    print(f"  attention shape: {attn_last.shape}")
    print(f"  attention[-1] min/max/mean: "
          f"{attn_last.min().item():.5f} / {attn_last.max().item():.5f} / {attn_last.mean().item():.5f}")

    num_register = getattr(model.config, "num_register_tokens", 0)
    patch_size = model.config.patch_size
    _, _, h, w = inputs["pixel_values"].shape
    num_h, num_w = h // patch_size, w // patch_size

    cls_attn = attn_last[:, 0, 1 + num_register:].mean(dim=0)  # (num_patches,)
    attn_map = cls_attn.reshape(num_h, num_w).float().cpu().numpy()

    print(f"  attn_map shape: {attn_map.shape}  "
          f"min={attn_map.min():.5f} max={attn_map.max():.5f} mean={attn_map.mean():.5f}")
    return attn_map, (image.width, image.height)


def upsample(attn_map: np.ndarray, w: int, h: int, mode: str = "nearest") -> np.ndarray:
    t = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0).float()
    kw = {"align_corners": False} if mode == "bilinear" else {}
    up = F.interpolate(t, size=(h, w), mode=mode, **kw)
    return up.squeeze().numpy()


def mass_threshold(attn_map: np.ndarray, threshold: float) -> np.ndarray:
    """Official DINO mass-based threshold: keep top `threshold` fraction of attention mass."""
    flat = attn_map.flatten()
    sorted_vals = np.sort(flat)
    sorted_norm = sorted_vals / (sorted_vals.sum() + 1e-8)
    cumsum = np.cumsum(sorted_norm)
    above = sorted_vals[cumsum > (1 - threshold)]
    cutoff = above[0] if len(above) > 0 else flat.min()
    binary = (attn_map >= cutoff).astype(np.uint8)
    print(f"  threshold cutoff={cutoff:.5f}  foreground pixels={binary.sum()} / {binary.size} "
          f"({100*binary.sum()/binary.size:.1f}%)")
    return binary


def find_blobs(binary: np.ndarray, min_blob_size: int, orig_w: int, orig_h: int):
    """Return list of (x1,y1,x2,y2) bboxes for significant blobs."""
    labeled, n = ndimage.label(binary)
    print(f"  total blobs before size filter: {n}")
    blobs = []
    for lid in range(1, n + 1):
        mask = labeled == lid
        if mask.sum() >= min_blob_size:
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            y1, y2 = int(rows[0]), int(rows[-1])
            x1, x2 = int(cols[0]), int(cols[-1])
            blobs.append((x1, y1, x2, y2))
    print(f"  blobs after size filter (>={min_blob_size}px): {len(blobs)}")
    return blobs


# ── Visualization ───────────────────────────────────────────────────���─────────

def visualize(image: Image.Image, attn_map: np.ndarray, blobs: list, threshold: float, output_path: str):
    orig_np = np.array(image)
    orig_w, orig_h = image.width, image.height

    # Smooth bilinear heatmap for display
    attn_full_vis = upsample(attn_map, orig_w, orig_h, mode="bilinear")
    attn_norm = (attn_full_vis - attn_full_vis.min()) / (attn_full_vis.max() - attn_full_vis.min() + 1e-8)

    # Binary mask (nearest) for display
    attn_full_nearest = upsample(attn_map, orig_w, orig_h, mode="nearest")
    binary = mass_threshold(attn_full_nearest, threshold)

    # Overlay
    cmap = plt.colormaps["jet"]
    heat_rgba = cmap(attn_norm)
    heat_rgb = (heat_rgba[:, :, :3] * 255).astype(np.uint8)
    overlay = (orig_np.astype(float) * 0.55 + heat_rgb.astype(float) * 0.45).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#1a1a2e")
    fig.suptitle(f"DINOv3 Attention Bbox Test  (threshold={threshold}, blobs={len(blobs)})",
                 color="white", fontsize=11)

    titles = ["Original + Bboxes", "Attention Heatmap", "Attention Overlay", "Binary Mask"]
    for ax, title in zip(axes, titles):
        ax.set_facecolor("#0d0d1a")
        ax.set_title(title, color="white", fontsize=9)
        ax.axis("off")

    # Panel 1: original + blobs
    axes[0].imshow(orig_np)
    for (x1, y1, x2, y2) in blobs:
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="#ff4444", facecolor="none",
        )
        axes[0].add_patch(rect)
    if not blobs:
        axes[0].text(0.5, 0.5, "NO BLOBS DETECTED", transform=axes[0].transAxes,
                     color="red", ha="center", va="center", fontsize=12)

    # Panel 2: heatmap
    axes[1].imshow(attn_norm, cmap="jet", interpolation="bilinear")

    # Panel 3: overlay
    axes[2].imshow(overlay)

    # Panel 4: binary mask
    axes[3].imshow(binary, cmap="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ── Main ──────────────────────────��───────────────────────────────���──────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-blob-size", type=int, default=MIN_BLOB_SIZE)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== DINOv3 Bbox Test ===")
    print(f"Image    : {image_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Min blob : {args.min_blob_size} px")

    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.width} × {image.height}")

    model, processor = load_model()

    print("\n[1] Extracting attention map ...")
    attn_map, (orig_w, orig_h) = get_attention_map(model, processor, image)

    print("\n[2] Upsampling & thresholding ...")
    attn_full_nearest = upsample(attn_map, orig_w, orig_h, mode="nearest")
    binary = mass_threshold(attn_full_nearest, args.threshold)

    print("\n[3] Finding blobs ...")
    blobs = find_blobs(binary, args.min_blob_size, orig_w, orig_h)
    for i, (x1, y1, x2, y2) in enumerate(blobs):
        print(f"  blob {i+1}: ({x1},{y1}) → ({x2},{y2})  size={x2-x1}×{y2-y1}")

    print("\n[4] Saving visualization ...")
    visualize(image, attn_map, blobs, args.threshold, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
