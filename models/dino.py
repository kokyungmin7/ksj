from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from transformers import AutoImageProcessor, AutoModel


def load_dino(cfg: SimpleNamespace) -> tuple:
    """Load DINOv3 model and image processor."""
    processor = AutoImageProcessor.from_pretrained(cfg.model.dino_name)
    model = AutoModel.from_pretrained(
        cfg.model.dino_name,
        torch_dtype=torch.bfloat16,    # bfloat16: same exponent range as float32 → no NaN in softmax
        device_map="auto",
        attn_implementation="eager",   # sdpa does not support output_attentions=True
    )
    model.eval()
    return model, processor


@torch.inference_mode()
def _extract_patch_features(model, processor, image: Image.Image) -> torch.Tensor:
    """Return patch features of shape (num_patches, hidden_size)."""
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    num_register = getattr(model.config, "num_register_tokens", 0)
    patch_features = outputs.last_hidden_state[0, 1 + num_register :]  # (P, D)
    return patch_features


@torch.inference_mode()
def _get_raw_attention_map(
    model, processor, image: Image.Image
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return mean [CLS->patch] attention as (num_h, num_w) numpy array
    and (img_w, img_h) for upsampling reference."""
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    outputs = model(**inputs, output_attentions=True)

    attn = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
    num_register = getattr(model.config, "num_register_tokens", 0)
    patch_size = model.config.patch_size
    _, _, h, w = inputs["pixel_values"].shape
    num_h, num_w = h // patch_size, w // patch_size

    cls_attn = attn[:, 0, 1 + num_register :].mean(dim=0)  # (num_patches,)
    attn_map = cls_attn.reshape(num_h, num_w).float().cpu().numpy()
    return attn_map, (image.width, image.height)


def _upsample_attn(
    attn_map: np.ndarray, target_w: int, target_h: int, mode: str = "bilinear"
) -> np.ndarray:
    """Upsample attention map to (target_h, target_w).

    Use mode='bilinear' for smooth heatmap visualization.
    Use mode='nearest' for binary mask generation (preserves patch boundaries).
    """
    t = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0).float()
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    up = F.interpolate(t, size=(target_h, target_w), mode=mode, **kwargs)
    return up.squeeze().numpy()


def _mass_threshold(attn_map: np.ndarray, threshold: float) -> np.ndarray:
    """Binary mask keeping the top `threshold` fraction of attention mass.

    Official Facebook DINO (visualize_attention.py) approach:
      1. Sort attention values ascending.
      2. Normalize to a probability distribution (sum = 1).
      3. Compute cumulative sum; pixels where cumsum > (1 - threshold) form the
         smallest set that carries `threshold` fraction of total attention mass.

    This is image-agnostic — the result always covers a consistent *proportion*
    of foreground regardless of the absolute attention scale.
    """
    flat = attn_map.flatten()
    sorted_vals = np.sort(flat)                            # ascending
    sorted_norm = sorted_vals / (sorted_vals.sum() + 1e-8)
    cumsum = np.cumsum(sorted_norm)
    above = sorted_vals[cumsum > (1 - threshold)]
    cutoff = above[0] if len(above) > 0 else flat.min()
    return (attn_map >= cutoff).astype(np.uint8)


def get_attention_crop(
    model,
    processor,
    image: Image.Image,
    cfg: SimpleNamespace,
) -> dict:
    """Extract the ROI crop from an image using DINOv3 attention.

    Returns:
        {
            "crop": PIL.Image,              # cropped ROI (or original if too small)
            "bbox": (x1, y1, x2, y2),      # bbox in original pixel coords
            "attn_map": np.ndarray,         # raw attention map (num_h × num_w)
            "attn_map_full": np.ndarray,    # upsampled to original image size (H × W)
            "used_full": bool,              # True if crop fell back to full image
        }
    """
    attn_map, (orig_w, orig_h) = _get_raw_attention_map(model, processor, image)
    # bilinear: smooth heatmap for visualization
    attn_full = _upsample_attn(attn_map, orig_w, orig_h, mode="bilinear")
    # nearest: sharp patch boundaries for binary mask
    attn_full_nearest = _upsample_attn(attn_map, orig_w, orig_h, mode="nearest")

    # Mass-based threshold (official DINO approach)
    binary = _mass_threshold(attn_full_nearest, cfg.dino.attention_threshold)

    # Find bounding box via ndimage
    labeled, _ = ndimage.label(binary)
    slices = ndimage.find_objects(labeled)

    # Filter blobs by min_blob_size to exclude noise
    min_blob_size = getattr(cfg.dino, "min_blob_size", 50)
    significant: list[tuple] = []  # (slice_row, slice_col) for significant blobs
    blobs_bbox: list[tuple[int, int, int, int]] = []  # (x1, y1, x2, y2) per blob

    for label_id, s in enumerate(slices, start=1):
        if (labeled == label_id).sum() >= min_blob_size:
            significant.append(s)
            by1, by2 = s[0].start, s[0].stop
            bx1, bx2 = s[1].start, s[1].stop
            blobs_bbox.append((bx1, by1, bx2, by2))

    if significant:
        rows = [s[0] for s in significant]
        cols = [s[1] for s in significant]
        y1 = min(r.start for r in rows)
        y2 = max(r.stop for r in rows)
        x1 = min(c.start for c in cols)
        x2 = max(c.stop for c in cols)
    else:
        y1, y2, x1, x2 = 0, orig_h, 0, orig_w

    # Add padding and clip to image bounds
    pad = cfg.dino.crop_padding
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(orig_w, x2 + pad)
    y2 = min(orig_h, y2 + pad)

    crop_w = x2 - x1
    crop_h = y2 - y1
    min_size = cfg.dino.crop_min_size
    used_full = crop_w < min_size or crop_h < min_size

    if used_full:
        crop = image.copy()
        x1, y1, x2, y2 = 0, 0, orig_w, orig_h
        blobs_bbox = []
    else:
        crop = image.crop((x1, y1, x2, y2))

    return {
        "crop": crop,
        "bbox": (x1, y1, x2, y2),
        "blobs_bbox": blobs_bbox,   # individual bbox per significant blob
        "attn_map": attn_map,
        "attn_map_full": attn_full,
        "used_full": used_full,
    }


def _count_blobs(binary_map: np.ndarray, min_size: int) -> int:
    """Count connected components in a binary map, ignoring small blobs."""
    labeled, num_features = ndimage.label(binary_map.astype(bool))
    count = 0
    for label_id in range(1, num_features + 1):
        if (labeled == label_id).sum() >= min_size:
            count += 1
    return count


def _attention_blob_count(
    model, processor, image: Image.Image, cfg: SimpleNamespace
) -> int:
    # Upsample to pixel-space so min_blob_size is always in pixel units (not patches)
    attn_map, (orig_w, orig_h) = _get_raw_attention_map(model, processor, image)
    attn_full = _upsample_attn(attn_map, orig_w, orig_h, mode="nearest")
    binary = _mass_threshold(attn_full, cfg.dino.attention_threshold)
    return _count_blobs(binary, cfg.dino.min_blob_size)


@torch.inference_mode()
def _reference_similarity_count(
    model,
    processor,
    image: Image.Image,
    reference_images: list[Image.Image],
    cfg: SimpleNamespace,
) -> int:
    """Count objects by comparing query patch features to mean reference features."""
    ref_feats = []
    for ref_img in reference_images:
        patch_feat = _extract_patch_features(model, processor, ref_img)
        ref_feats.append(patch_feat.mean(dim=0))
    ref_mean = F.normalize(torch.stack(ref_feats).mean(dim=0).unsqueeze(0), dim=-1)

    query_feats = _extract_patch_features(model, processor, image)
    query_norm = F.normalize(query_feats, dim=-1)

    sim = (query_norm @ ref_mean.T).squeeze(-1).float().cpu()

    patch_size = model.config.patch_size
    inputs = processor(images=image, return_tensors="pt")
    _, _, h, w = inputs["pixel_values"].shape
    sim_map = sim.reshape(h // patch_size, w // patch_size).numpy()

    binary = sim_map >= cfg.dino.similarity_threshold
    count = _count_blobs(binary, cfg.dino.min_region_size)

    if count == 0:
        count = _attention_blob_count(model, processor, image, cfg)

    return count


def dino_count(
    model,
    processor,
    image_path: str,
    cfg: SimpleNamespace,
    reference_paths: list[str] | None = None,
) -> int:
    """Count objects in the image using DINOv3 attention or reference similarity."""
    image = Image.open(image_path).convert("RGB")

    if reference_paths:
        ref_images = [Image.open(p).convert("RGB") for p in reference_paths]
        return _reference_similarity_count(model, processor, image, ref_images, cfg)

    return _attention_blob_count(model, processor, image, cfg)


def pick_answer_by_count(count: int, row: dict) -> str | None:
    """Select the choice whose text contains the number closest to count.

    Returns None if no choice contains a number (signals Qwen fallback).
    """
    choices = {"a": row["a"], "b": row["b"], "c": row["c"], "d": row["d"]}
    best_key = None
    best_diff = float("inf")

    for key, text in choices.items():
        for n in re.findall(r"\d+", str(text)):
            diff = abs(int(n) - count)
            if diff < best_diff:
                best_diff = diff
                best_key = key

    return best_key


def load_reference_paths(reference_dir: str) -> list[str]:
    """Load all image paths from a reference gallery directory."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [
        str(p)
        for p in sorted(Path(reference_dir).iterdir())
        if p.suffix.lower() in exts
    ]
