"""GroundingDINO-based object detection for crop and counting.

Replaces models/dino.py. Accepts both image and text prompt, enabling
class-specific detection (e.g., "styrofoam box", "PET bottle").
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def load_grounding_dino(cfg: SimpleNamespace) -> tuple:
    """Load GroundingDINO model and processor."""
    model_name = getattr(cfg.dino, "model_name", "IDEA-Research/grounding-dino-tiny")
    print(f"  model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_name,
        device_map="auto",
    )
    model.eval()
    return model, processor


@torch.inference_mode()
def _detect(
    model,
    processor,
    image: Image.Image,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> list[tuple[int, int, int, int]]:
    """Run GroundingDINO and return (x1,y1,x2,y2) boxes in pixel coords."""
    if not text_prompt.strip():
        return []

    prompt = text_prompt.strip().rstrip(".") + "."

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs)

    # Parameter name changed across transformers versions:
    #   older: threshold=  newer: box_threshold=
    import inspect
    sig = inspect.signature(processor.post_process_grounded_object_detection)
    thr_kwarg = "box_threshold" if "box_threshold" in sig.parameters else "threshold"

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        **{thr_kwarg: box_threshold},
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],  # (H, W)
    )

    boxes = results[0]["boxes"].cpu().numpy()  # (N, 4) xyxy
    return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in boxes]


def get_grounding_crop(
    model,
    processor,
    image: Image.Image,
    text_prompt: str,
    cfg: SimpleNamespace,
) -> dict:
    """Detect objects by text prompt and return ROI crop from the union bbox.

    Returns:
        {
            "crop": PIL.Image,                          # ROI crop or full image fallback
            "bbox": (x1, y1, x2, y2),                  # union bbox in pixel coords
            "blobs_bbox": list[(x1, y1, x2, y2)],      # individual detection boxes
            "used_full": bool,                          # True if fell back to full image
            "attn_map": None,                           # compat placeholder
            "attn_map_full": None,                      # compat placeholder
        }
    """
    box_thr = getattr(cfg.dino, "box_threshold", 0.25)
    text_thr = getattr(cfg.dino, "text_threshold", 0.20)
    pad = getattr(cfg.dino, "crop_padding", 20)
    min_size = getattr(cfg.dino, "crop_min_size", 64)
    orig_w, orig_h = image.width, image.height

    boxes = _detect(model, processor, image, text_prompt, box_thr, text_thr)

    if boxes:
        ux1 = max(0, min(b[0] for b in boxes) - pad)
        uy1 = max(0, min(b[1] for b in boxes) - pad)
        ux2 = min(orig_w, max(b[2] for b in boxes) + pad)
        uy2 = min(orig_h, max(b[3] for b in boxes) + pad)

        used_full = (ux2 - ux1) < min_size or (uy2 - uy1) < min_size
        if used_full:
            crop = image.copy()
            union_bbox = (0, 0, orig_w, orig_h)
            boxes = []
        else:
            crop = image.crop((ux1, uy1, ux2, uy2))
            union_bbox = (ux1, uy1, ux2, uy2)
    else:
        crop = image.copy()
        union_bbox = (0, 0, orig_w, orig_h)
        used_full = True

    return {
        "crop": crop,
        "bbox": union_bbox,
        "blobs_bbox": boxes,
        "used_full": used_full,
        "attn_map": None,
        "attn_map_full": None,
    }



# ── Korean noun extraction & translation ──────────────────────────────────────

def extract_object_noun(question: str, row: dict | None = None) -> str:
    """Extract the object noun from a Korean question and translate to English.

    Delegates to utils.ko2en which has comprehensive vocabulary + parsing.
    """
    from utils.ko2en import extract_grounding_prompt, extract_grounding_prompt_from_row

    if row is not None:
        return extract_grounding_prompt_from_row(row)
    return extract_grounding_prompt(question)


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
