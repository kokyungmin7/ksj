"""SAM-based instance counting within a GroundingDINO bbox.

Pipeline:
  1. Crop image to the GroundingDINO union bbox.
  2. Generate an NxN grid of points over the crop.
  3. Run SAM inference (in chunks to manage VRAM).
  4. Per point: select the mask with the highest IoU score.
  5. Filter masks by area ratio (remove noise / background masks).
  6. IoU-based NMS to deduplicate overlapping masks.
  7. Return the count of remaining unique masks.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor


def load_sam(cfg: SimpleNamespace) -> tuple:
    """Load SAM model and processor."""
    model_name = getattr(cfg.sam, "model_name", "facebook/sam-vit-base")
    print(f"  model: {model_name}")
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name, device_map="auto")
    model.eval()
    return model, processor


def _nms_masks(masks: list[np.ndarray], iou_threshold: float) -> list[int]:
    """IoU-based NMS on binary masks. Returns indices of kept masks."""
    if not masks:
        return []
    n = len(masks)
    areas = [m.sum() for m in masks]
    keep = []
    suppressed = [False] * n

    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(i)
        for j in range(i + 1, n):
            if suppressed[j]:
                continue
            inter = (masks[i] & masks[j]).sum()
            union = areas[i] + areas[j] - inter
            if union > 0 and inter / union > iou_threshold:
                suppressed[j] = True

    return keep


@torch.inference_mode()
def count_instances_sam(
    sam_model,
    sam_processor,
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    cfg: SimpleNamespace,
) -> int:
    """Count individual object instances within a bbox using SAM.

    Args:
        image:  Original PIL image.
        bbox:   (x1, y1, x2, y2) GroundingDINO union bbox in pixel coords.

    Returns:
        Number of detected instances (0 if none pass filters).
    """
    grid_size = getattr(cfg.sam, "grid_size", 8)
    min_area = getattr(cfg.sam, "min_mask_area", 0.02)
    max_area = getattr(cfg.sam, "max_mask_area", 0.90)
    iou_thr = getattr(cfg.sam, "iou_threshold", 0.5)
    chunk_size = getattr(cfg.sam, "chunk_size", 16)

    x1, y1, x2, y2 = bbox
    crop = image.crop((x1, y1, x2, y2))
    cw, ch = crop.width, crop.height
    crop_area = cw * ch

    # ── Build grid of (x, y) points in crop pixel coords ──────────────────
    grid_points: list[list[float]] = []
    for row in range(grid_size):
        for col in range(grid_size):
            px = (col + 0.5) * cw / grid_size
            py = (row + 0.5) * ch / grid_size
            grid_points.append([px, py])

    # ── SAM inference in chunks ────────────────────────────────────────────
    candidate_masks: list[np.ndarray] = []

    for start in range(0, len(grid_points), chunk_size):
        chunk = grid_points[start : start + chunk_size]

        inputs = sam_processor(
            images=[crop] * len(chunk),
            input_points=[[[p]] for p in chunk],   # (B, 1, 1, 2)
            return_tensors="pt",
        ).to(sam_model.device)

        outputs = sam_model(**inputs)

        # pred_masks: (B, 1, 3, H, W)  iou_scores: (B, 1, 3)
        pred_masks = outputs.pred_masks          # (B, 1, 3, H, W)
        iou_scores = outputs.iou_scores          # (B, 1, 3)

        # Post-process to original crop size
        masks_np = sam_processor.post_process_masks(
            pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )  # list of length B, each (1, 3, H, W) bool tensors

        for b_idx in range(len(chunk)):
            # masks_np[b_idx]: (1, 3, H, W)
            b_masks = masks_np[b_idx][0]           # (3, H, W)
            b_scores = iou_scores[b_idx, 0]        # (3,)

            best = int(b_scores.argmax())
            mask = b_masks[best].numpy().astype(bool)  # (H, W)

            area_ratio = mask.sum() / crop_area
            if min_area <= area_ratio <= max_area:
                candidate_masks.append(mask)

    if not candidate_masks:
        return 0

    # ── IoU NMS ───────────────────────────────────────────────────────────
    keep_idx = _nms_masks(candidate_masks, iou_thr)
    return len(keep_idx)
