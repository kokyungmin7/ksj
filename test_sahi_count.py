"""Standalone test: GroundingDINO + SAHI sliced inference for object counting.

SAHI (Slicing Aided Hyper Inference) 적용:
  이미지를 겹치는 패치로 슬라이싱 → 각 패치에서 GroundingDINO 추론
  → 좌표를 원본으로 매핑 → NMS로 중복 제거 → 최종 detection 수 = count

Usage:
    # 기본 (512x512 슬라이스, 0.2 overlap)
    uv run test_sahi_count.py --image train/train_1940.jpg --prompt "styrofoam box"

    # 슬라이스 크기/overlap 조정
    uv run test_sahi_count.py --image train/train_1940.jpg --prompt "bottle" \
        --slice-size 640 --overlap-ratio 0.25

    # full image 추론도 병합 (기본 활성화)
    uv run test_sahi_count.py --image train/train_1940.jpg --prompt "can" \
        --no-full-image

    # NMS IoU threshold 조정
    uv run test_sahi_count.py --image train/train_1940.jpg --prompt "box" \
        --nms-iou 0.4

Output:
    test_sahi_count_result.png — 3-panel visualization
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
SLICE_SIZE = 512
OVERLAP_RATIO = 0.2
NMS_IOU_THRESHOLD = 0.5
OUTPUT_PATH = "test_sahi_count_result.png"


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


# ── SAHI: Sliced Inference ────────────────────────────────────────────────────

def generate_slices(
    img_w: int, img_h: int,
    slice_size: int, overlap_ratio: float,
) -> list[tuple[int, int, int, int]]:
    """겹치는 슬라이스 좌표 (x1, y1, x2, y2) 목록 생성.

    이미지가 slice_size보다 작으면 전체를 하나의 슬라이스로 반환.
    """
    step = max(1, int(slice_size * (1 - overlap_ratio)))
    slices = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            x2 = min(x + slice_size, img_w)
            y2 = min(y + slice_size, img_h)
            x1 = max(0, x2 - slice_size)
            y1 = max(0, y2 - slice_size)
            slices.append((x1, y1, x2, y2))
    # 중복 슬라이스 제거 (이미지가 작을 때)
    return list(dict.fromkeys(slices))


def sliced_inference(
    model, processor, image: Image.Image, prompt: str,
    slice_size: int, overlap_ratio: float,
    box_thr: float, text_thr: float,
    include_full: bool = True,
) -> list[tuple[float, float, float, float, float, str]]:
    """SAHI 방식 슬라이스 추론 → 원본 좌표계 detection 목록."""
    img_w, img_h = image.size
    slices = generate_slices(img_w, img_h, slice_size, overlap_ratio)
    print(f"  Image: {img_w}x{img_h} → {len(slices)} slice(s) "
          f"(size={slice_size}, overlap={overlap_ratio})")

    all_dets: list[tuple[float, float, float, float, float, str]] = []

    for i, (sx1, sy1, sx2, sy2) in enumerate(slices):
        crop = image.crop((sx1, sy1, sx2, sy2))
        dets = gdino_detect(model, processor, crop, prompt, box_thr, text_thr)
        mapped = [
            (x1 + sx1, y1 + sy1, x2 + sx1, y2 + sy1, s, lb)
            for x1, y1, x2, y2, s, lb in dets
        ]
        if mapped:
            print(f"    slice[{i}] ({sx1},{sy1})-({sx2},{sy2}): {len(mapped)} det(s)")
        all_dets.extend(mapped)

    if include_full:
        full_dets = gdino_detect(model, processor, image, prompt, box_thr, text_thr)
        if full_dets:
            print(f"    full-image: {len(full_dets)} det(s)")
        all_dets.extend(full_dets)

    print(f"  Total before NMS: {len(all_dets)}")
    return all_dets


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


def visualize_sahi(
    image: Image.Image,
    slices: list[tuple[int, int, int, int]],
    raw_dets: list[tuple[float, float, float, float, float, str]],
    final_dets: list[tuple[float, float, float, float, float, str]],
    prompt: str, output: str,
):
    orig_np = np.array(image)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor="#1a1a2e")
    fig.suptitle(
        f'GroundingDINO + SAHI  │  prompt="{prompt}"  │  '
        f'raw={len(raw_dets)} → NMS → final={len(final_dets)}',
        color="white", fontsize=12, fontweight="bold",
    )
    for ax in axes:
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")

    # Panel 1: 슬라이스 그리드
    axes[0].imshow(orig_np)
    axes[0].set_title(f"SAHI Slices ({len(slices)})", color="white", fontsize=10)
    for i, (sx1, sy1, sx2, sy2) in enumerate(slices):
        c = COLORS[i % len(COLORS)]
        axes[0].add_patch(mpatches.Rectangle(
            (sx1, sy1), sx2 - sx1, sy2 - sy1,
            linewidth=1.5, edgecolor=c, facecolor=c, alpha=0.08,
        ))
        axes[0].add_patch(mpatches.Rectangle(
            (sx1, sy1), sx2 - sx1, sy2 - sy1,
            linewidth=1.5, edgecolor=c, facecolor="none",
        ))

    # Panel 2: NMS 전 raw detections
    axes[1].imshow(orig_np)
    axes[1].set_title(f"Raw Detections ({len(raw_dets)})", color="white", fontsize=10)
    for x1, y1, x2, y2, score, label in raw_dets:
        axes[1].add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1, edgecolor="#ff4444", facecolor="none", alpha=0.5,
        ))

    # Panel 3: NMS 후 최종 detections = count
    axes[2].imshow(orig_np)
    axes[2].set_title(
        f"After NMS — Count = {len(final_dets)}",
        color="#44ff44", fontsize=11, fontweight="bold",
    )
    for i, (x1, y1, x2, y2, score, label) in enumerate(final_dets):
        c = COLORS[i % len(COLORS)]
        axes[2].add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=c, facecolor=c, alpha=0.12,
        ))
        axes[2].add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=c, facecolor="none",
        ))
        axes[2].text(
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
        description="GroundingDINO + SAHI sliced inference counting",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    parser.add_argument("--slice-size", type=int, default=SLICE_SIZE,
                        help="슬라이스 크기 (px)")
    parser.add_argument("--overlap-ratio", type=float, default=OVERLAP_RATIO,
                        help="슬라이스 간 overlap 비율 (0~1)")
    parser.add_argument("--nms-iou", type=float, default=NMS_IOU_THRESHOLD,
                        help="NMS IoU threshold (낮을수록 공격적 제거)")
    parser.add_argument("--no-full-image", action="store_true",
                        help="full image 추론 비활성화 (슬라이스만)")
    parser.add_argument(
        "--quant", choices=("none", "4", "8"), default="8",
        help="GroundingDINO quantization (8bit default)",
    )
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print("\n=== GroundingDINO + SAHI Sliced Counting ===")
    print(f"Image     : {image_path}")
    print(f"Prompt    : {args.prompt}")
    print(f"Slice     : {args.slice_size}px, overlap={args.overlap_ratio}")
    print(f"NMS IoU   : {args.nms_iou}")
    print(f"Full-image: {'Yes' if not args.no_full_image else 'No'}")

    image = Image.open(image_path).convert("RGB")
    print(f"Size      : {image.width} x {image.height}")

    # ── Step 1: Load model ──
    model, processor = load_gdino(args.quant)

    # ── Step 2: SAHI sliced inference ──
    print("\n[1] SAHI Sliced Inference ...")
    slices = generate_slices(
        image.width, image.height, args.slice_size, args.overlap_ratio,
    )
    raw_dets = sliced_inference(
        model, processor, image, args.prompt,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap_ratio,
        box_thr=args.box_threshold,
        text_thr=args.text_threshold,
        include_full=not args.no_full_image,
    )

    # ── Step 3: NMS ──
    print("\n[2] NMS merging ...")
    final_dets = apply_nms(raw_dets, args.nms_iou)
    print(f"  {len(raw_dets)} → {len(final_dets)} detections")
    for i, (x1, y1, x2, y2, score, label) in enumerate(final_dets):
        print(f"  [{i+1}] ({x1:.0f},{y1:.0f})->({x2:.0f},{y2:.0f})  "
              f"score={score:.3f}  label={label}")

    count = len(final_dets)
    print(f"\n  ★ Final Count: {count}")

    # ── Step 4: Visualization ──
    print("\n[3] Saving visualization ...")
    visualize_sahi(image, slices, raw_dets, final_dets, args.prompt, args.output)

    del model
    _free_vram()
    print("\nDone.")


if __name__ == "__main__":
    main()
