"""Two-stage counting: GroundingDINO group crop → adaptive SAHI sliced counting.

Stage 1: 전체 이미지에서 GroundingDINO로 묶음(group) 탐지 → union bbox crop
Stage 2: crop 크기에 맞춰 slice_size 자동 결정 → SAHI 슬라이스 추론 → NMS → count

Usage:
    # 기본 (group prompt + count prompt 분리)
    uv run test_sahi_count.py --image train/train_1940.jpg \
        --group-prompt "recyclable items" --count-prompt "styrofoam box"

    # group/count 프롬프트 동일하게
    uv run test_sahi_count.py --image train/train_1940.jpg --prompt "bottle"

    # crop padding, NMS threshold 조정
    uv run test_sahi_count.py --image train/train_1940.jpg --prompt "can" \
        --crop-pad 30 --nms-iou 0.4

Output:
    test_sahi_count_result.png — 4-panel visualization
"""
from __future__ import annotations

import argparse
import gc
import inspect
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    BitsAndBytesConfig,
)

GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_IMAGE = "train/train_0001.jpg"
DEFAULT_PROMPT = "recyclable item"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
CROP_PAD = 20
NMS_IOU_THRESHOLD = 0.5
OUTPUT_PATH = "test_sahi_count_result.png"

ADAPTIVE_MIN_SLICE = 256
ADAPTIVE_MAX_SLICE = 640
ADAPTIVE_NO_SLICE_THRESHOLD = 400


# ── Utilities ─────────────────────────────────────────────────────────────────

def _device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── GroundingDINO ─────────────────────────────────────────────────────────────

def load_gdino(quant: str = "8"):
    print(f"Loading GroundingDINO ({GDINO_MODEL})  quant={quant} ...")
    processor = AutoProcessor.from_pretrained(GDINO_MODEL)

    kwargs: dict = {"device_map": "auto"}
    if quant == "8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_MODEL, **kwargs
    )
    model.eval()
    print(f"  device: {_device_of(model)}")
    return model, processor


@torch.inference_mode()
def gdino_detect(model, processor, image, prompt, box_thr, text_thr):
    """단일 이미지에 대한 GroundingDINO 추론. (x1,y1,x2,y2,score,label) 리스트 반환."""
    p = prompt.strip().rstrip(".") + "."
    dev = _device_of(model)
    inputs = processor(images=image, text=p, return_tensors="pt").to(dev)
    outputs = model(**inputs)

    sig = inspect.signature(processor.post_process_grounded_object_detection)
    thr_kwarg = "box_threshold" if "box_threshold" in sig.parameters else "threshold"
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        **{thr_kwarg: box_thr},
        text_threshold=text_thr,
        target_sizes=[image.size[::-1]],
    )
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    labels = results[0]["labels"]
    return [
        (float(x1), float(y1), float(x2), float(y2), float(s), str(lb))
        for (x1, y1, x2, y2), s, lb in zip(boxes, scores, labels)
    ]


# ── Stage 1: Group Detection + Crop ──────────────────────────────────────────

def group_crop(
    model, processor, image: Image.Image,
    group_prompt: str, box_thr: float, text_thr: float, pad: int,
) -> tuple[Image.Image, list, tuple[int, int, int, int]]:
    """전체 이미지에서 묶음 탐지 → union bbox crop.

    Returns:
        (crop_image, group_detections, (ux1, uy1, ux2, uy2))
    """
    print(f"  group prompt: '{group_prompt}'")
    dets = gdino_detect(model, processor, image, group_prompt, box_thr, text_thr)
    print(f"  {len(dets)} group detection(s)")

    if not dets:
        print("  No group detections → using full image as crop")
        bbox = (0, 0, image.width, image.height)
        return image.copy(), dets, bbox

    ux1 = max(0, int(min(d[0] for d in dets)) - pad)
    uy1 = max(0, int(min(d[1] for d in dets)) - pad)
    ux2 = min(image.width, int(max(d[2] for d in dets)) + pad)
    uy2 = min(image.height, int(max(d[3] for d in dets)) + pad)
    crop = image.crop((ux1, uy1, ux2, uy2))
    print(f"  union crop: ({ux1},{uy1})->({ux2},{uy2})  "
          f"size={ux2-ux1}x{uy2-uy1}")
    return crop, dets, (ux1, uy1, ux2, uy2)


# ── Stage 2: Adaptive SAHI ───────────────────────────────────────────────────

def compute_adaptive_slice(
    crop_w: int, crop_h: int, n_group_dets: int,
) -> tuple[int, float]:
    """crop 크기와 1차 탐지 수를 기반으로 slice_size와 overlap 자동 결정.

    로직:
    - crop이 작으면 (<=400px) 슬라이싱 불필요 → 전체 한 번 추론
    - 1차 탐지 수가 많으면 밀집도 높음 → 더 작은 슬라이스
    - 긴 축 기준으로 축당 슬라이스 수 결정

    Returns:
        (slice_size, overlap_ratio)
    """
    long_side = max(crop_w, crop_h)

    if long_side <= ADAPTIVE_NO_SLICE_THRESHOLD:
        return long_side, 0.0

    if n_group_dets >= 5:
        divisor = 300
    elif n_group_dets >= 2:
        divisor = 450
    else:
        divisor = ADAPTIVE_MAX_SLICE

    target_n = max(2, math.ceil(long_side / divisor))
    slice_size = math.ceil(long_side / target_n)
    slice_size = max(ADAPTIVE_MIN_SLICE, min(slice_size, ADAPTIVE_MAX_SLICE))

    overlap = min(0.35, 0.15 + 0.05 * target_n)

    return slice_size, round(overlap, 2)


def generate_slices(
    img_w: int, img_h: int,
    slice_size: int, overlap_ratio: float,
) -> list[tuple[int, int, int, int]]:
    """겹치는 슬라이스 좌표 (x1, y1, x2, y2) 목록 생성."""
    if overlap_ratio <= 0 or slice_size >= max(img_w, img_h):
        return [(0, 0, img_w, img_h)]

    step = max(1, int(slice_size * (1 - overlap_ratio)))
    slices = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            x2 = min(x + slice_size, img_w)
            y2 = min(y + slice_size, img_h)
            x1 = max(0, x2 - slice_size)
            y1 = max(0, y2 - slice_size)
            slices.append((x1, y1, x2, y2))
    return list(dict.fromkeys(slices))


def sahi_count(
    model, processor, crop_img: Image.Image, count_prompt: str,
    box_thr: float, text_thr: float,
    n_group_dets: int, nms_iou: float,
) -> tuple[list, list, list[tuple[int, int, int, int]], int, float]:
    """crop 이미지에 adaptive SAHI 적용 → NMS → 카운트.

    Returns:
        (raw_dets, final_dets, slices, slice_size, overlap)
    """
    crop_w, crop_h = crop_img.size
    slice_size, overlap = compute_adaptive_slice(crop_w, crop_h, n_group_dets)
    slices = generate_slices(crop_w, crop_h, slice_size, overlap)

    print(f"  crop: {crop_w}x{crop_h}  group_dets={n_group_dets}")
    print(f"  → auto slice_size={slice_size}  overlap={overlap}  "
          f"slices={len(slices)}")

    all_dets = []
    for i, (sx1, sy1, sx2, sy2) in enumerate(slices):
        patch = crop_img.crop((sx1, sy1, sx2, sy2))
        dets = gdino_detect(model, processor, patch, count_prompt, box_thr, text_thr)
        mapped = [
            (x1 + sx1, y1 + sy1, x2 + sx1, y2 + sy1, s, lb)
            for x1, y1, x2, y2, s, lb in dets
        ]
        if mapped:
            print(f"    slice[{i}] ({sx1},{sy1})-({sx2},{sy2}): "
                  f"{len(mapped)} det(s)")
        all_dets.extend(mapped)

    if len(slices) > 1:
        full_dets = gdino_detect(
            model, processor, crop_img, count_prompt, box_thr, text_thr,
        )
        if full_dets:
            print(f"    crop-full: {len(full_dets)} det(s)")
        all_dets.extend(full_dets)

    print(f"  total before NMS: {len(all_dets)}")
    final_dets = apply_nms(all_dets, nms_iou)
    print(f"  after NMS: {len(final_dets)}")

    return all_dets, final_dets, slices, slice_size, overlap


def apply_nms(
    detections: list[tuple[float, float, float, float, float, str]],
    iou_threshold: float,
) -> list[tuple[float, float, float, float, float, str]]:
    """torchvision NMS로 중복 bbox 제거."""
    if not detections:
        return []

    boxes = torch.tensor([(x1, y1, x2, y2) for x1, y1, x2, y2, _, _ in detections])
    scores = torch.tensor([s for _, _, _, _, s, _ in detections])

    keep_idx = nms(boxes, scores, iou_threshold).numpy()
    return [detections[i] for i in keep_idx]


# ── Visualization ─────────────────────────────────────────────────────────────

COLORS = [
    "#ff4444", "#44ff44", "#4488ff", "#ffaa00", "#ff44ff",
    "#00ffff", "#ff8800", "#88ff00", "#8844ff", "#ff0088",
]


def visualize_pipeline(
    image: Image.Image,
    group_dets: list,
    crop_bbox: tuple[int, int, int, int],
    crop_img: Image.Image,
    slices: list[tuple[int, int, int, int]],
    raw_dets: list, final_dets: list,
    group_prompt: str, count_prompt: str,
    slice_size: int, overlap: float,
    output: str,
):
    orig_np = np.array(image)
    crop_np = np.array(crop_img)
    fig, axes = plt.subplots(1, 4, figsize=(28, 7), facecolor="#1a1a2e")
    fig.suptitle(
        f'Stage1: "{group_prompt}" → crop {crop_img.width}x{crop_img.height}  │  '
        f'Stage2: "{count_prompt}" slice={slice_size} ov={overlap}  │  '
        f'Count = {len(final_dets)}',
        color="white", fontsize=11, fontweight="bold",
    )
    for ax in axes:
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")

    ux1, uy1, ux2, uy2 = crop_bbox

    # Panel 1: 원본 + group bbox
    axes[0].imshow(orig_np)
    axes[0].set_title(
        f"Stage 1: Group ({len(group_dets)} det)", color="white", fontsize=9,
    )
    for x1, y1, x2, y2, score, label in group_dets:
        axes[0].add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor="#ff8800", facecolor="none",
        ))
        axes[0].text(x1, y1 - 4, f"{score:.2f}", color="#ff8800", fontsize=7)
    axes[0].add_patch(mpatches.Rectangle(
        (ux1, uy1), ux2 - ux1, uy2 - uy1,
        linewidth=2.5, edgecolor="#44ff44", facecolor="#44ff44",
        alpha=0.1, linestyle="--",
    ))
    axes[0].add_patch(mpatches.Rectangle(
        (ux1, uy1), ux2 - ux1, uy2 - uy1,
        linewidth=2.5, edgecolor="#44ff44", facecolor="none", linestyle="--",
    ))

    # Panel 2: crop + SAHI 슬라이스 그리드
    axes[1].imshow(crop_np)
    axes[1].set_title(
        f"Adaptive Slices ({len(slices)})", color="white", fontsize=9,
    )
    for i, (sx1, sy1, sx2, sy2) in enumerate(slices):
        c = COLORS[i % len(COLORS)]
        axes[1].add_patch(mpatches.Rectangle(
            (sx1, sy1), sx2 - sx1, sy2 - sy1,
            linewidth=1.5, edgecolor=c, facecolor=c, alpha=0.08,
        ))
        axes[1].add_patch(mpatches.Rectangle(
            (sx1, sy1), sx2 - sx1, sy2 - sy1,
            linewidth=1.5, edgecolor=c, facecolor="none",
        ))
        cx = (sx1 + sx2) / 2
        cy = (sy1 + sy2) / 2
        axes[1].text(
            cx, cy, str(i), color=c, fontsize=8,
            ha="center", va="center", fontweight="bold",
        )

    # Panel 3: raw detections (NMS 전)
    axes[2].imshow(crop_np)
    axes[2].set_title(
        f"Raw Detections ({len(raw_dets)})", color="white", fontsize=9,
    )
    for x1, y1, x2, y2, score, label in raw_dets:
        axes[2].add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1, edgecolor="#ff4444", facecolor="none", alpha=0.5,
        ))

    # Panel 4: NMS 후 최종 = count
    axes[3].imshow(crop_np)
    axes[3].set_title(
        f"Count = {len(final_dets)}",
        color="#44ff44", fontsize=11, fontweight="bold",
    )
    for i, (x1, y1, x2, y2, score, label) in enumerate(final_dets):
        c = COLORS[i % len(COLORS)]
        axes[3].add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=c, facecolor=c, alpha=0.12,
        ))
        axes[3].add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=c, facecolor="none",
        ))
        axes[3].text(
            x1, y1 - 4, f"#{i+1} {score:.2f}",
            color=c, fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.7),
        )

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {output}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage counting: GDino group crop → adaptive SAHI",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=None,
                        help="group/count 동일 프롬프트 (편의용)")
    parser.add_argument("--group-prompt", default=None,
                        help="Stage 1 묶음 탐지 프롬프트")
    parser.add_argument("--count-prompt", default=None,
                        help="Stage 2 개별 카운팅 프롬프트")
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    parser.add_argument("--crop-pad", type=int, default=CROP_PAD,
                        help="union crop padding (px)")
    parser.add_argument("--nms-iou", type=float, default=NMS_IOU_THRESHOLD,
                        help="NMS IoU threshold (낮을수록 공격적 제거)")
    parser.add_argument(
        "--quant", choices=("none", "4", "8"), default="8",
        help="GroundingDINO quantization",
    )
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    group_prompt = args.group_prompt or args.prompt or DEFAULT_PROMPT
    count_prompt = args.count_prompt or args.prompt or DEFAULT_PROMPT

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Two-Stage Counting: GDino Group Crop → Adaptive SAHI ===")
    print(f"Image        : {image_path}")
    print(f"Group prompt : {group_prompt}")
    print(f"Count prompt : {count_prompt}")
    print(f"Crop pad     : {args.crop_pad}px")
    print(f"NMS IoU      : {args.nms_iou}")

    image = Image.open(image_path).convert("RGB")
    print(f"Size         : {image.width} x {image.height}")

    model, processor = load_gdino(args.quant)

    # ── Stage 1: Group detection + crop ──
    print("\n[Stage 1] Group detection ...")
    crop_img, group_dets, crop_bbox = group_crop(
        model, processor, image,
        group_prompt, args.box_threshold, args.text_threshold, args.crop_pad,
    )
    for i, (x1, y1, x2, y2, score, label) in enumerate(group_dets):
        print(f"  [{i+1}] ({x1:.0f},{y1:.0f})->({x2:.0f},{y2:.0f})  "
              f"score={score:.3f}  label={label}")

    # ── Stage 2: Adaptive SAHI counting on crop ──
    print("\n[Stage 2] Adaptive SAHI counting on crop ...")
    raw_dets, final_dets, slices, slice_size, overlap = sahi_count(
        model, processor, crop_img, count_prompt,
        args.box_threshold, args.text_threshold,
        n_group_dets=len(group_dets),
        nms_iou=args.nms_iou,
    )

    count = len(final_dets)
    print(f"\n  ★ Final Count: {count}")
    for i, (x1, y1, x2, y2, score, label) in enumerate(final_dets):
        print(f"  [{i+1}] ({x1:.0f},{y1:.0f})->({x2:.0f},{y2:.0f})  "
              f"score={score:.3f}  label={label}")

    # ── Visualization ──
    print("\n[3] Saving visualization ...")
    visualize_pipeline(
        image, group_dets, crop_bbox,
        crop_img, slices, raw_dets, final_dets,
        group_prompt, count_prompt,
        slice_size, overlap, args.output,
    )

    del model
    _free_vram()
    print("\nDone.")


if __name__ == "__main__":
    main()
