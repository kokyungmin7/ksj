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

# Longest-match first (sorted in extract_object_noun)
_KO_EN: dict[str, str] = {
    "발포 스티로폼 상자": "styrofoam box",
    "발포 스티로폼": "styrofoam",
    "스티로폼 상자": "styrofoam box",
    "스티로폼": "styrofoam",
    "페트병": "PET bottle",
    "음료 페트": "beverage PET bottle",
    "페트": "plastic bottle",
    "음료캔": "beverage can",
    "철캔": "steel can",
    "알루미늄캔": "aluminum can",
    "캔": "can",
    "골판지": "cardboard",
    "종이류": "paper",
    "종이팩": "paper pack",
    "종이": "paper",
    "유리병": "glass bottle",
    "유리": "glass",
    "플라스틱병": "plastic bottle",
    "플라스틱": "plastic",
    "비닐봉지": "plastic bag",
    "비닐": "plastic bag",
    "재활용품": "recyclable item",
    "쓰레기": "trash",
    "병": "bottle",
}

_COUNTING_SPLIT = re.compile(
    r"몇\s*개|개수|몇\s*가지|몇\s*종류|몇\s*번|몇\s*마리|몇"
)
_PARTICLES = re.compile(r"[은는이가을를의에서으로로이고이며]+$")


def extract_object_noun(question: str) -> str:
    """Extract the object noun phrase from a Korean question and translate to English.

    Examples:
        "발포 스티로폼 상자는 몇 개입니까?" → "styrofoam box"
        "페트병의 개수는?"              → "PET bottle"
        "사진 속 캔은 몇 개인가요?"     → "can"
    """
    # Split at the first counting keyword
    parts = _COUNTING_SPLIT.split(question, maxsplit=1)
    text = parts[0].strip()

    # Remove trailing Korean particles
    text = _PARTICLES.sub("", text).strip()

    # Remove common leading phrases
    for prefix in ["사진에 보이는 재활용품 중", "사진에 보이는", "사진 속", "사진에서"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Translate: try longest match first
    for ko, en in sorted(_KO_EN.items(), key=lambda x: -len(x[0])):
        if ko in text:
            return en

    # Fallback: return Korean (GroundingDINO's BERT tokenizer has partial Korean support)
    return text.strip()


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
