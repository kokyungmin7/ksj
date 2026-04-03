"""Standalone test: GroundingDINO zero-shot detection + union crop visualization.

Usage:
    uv run test_dino_bbox.py --image train/train_1940.jpg --prompt "styrofoam box"
    uv run test_dino_bbox.py --image train/train_0001.jpg --prompt "plastic bottle" \\
        --box-threshold 0.3 --text-threshold 0.25

Output:
    test_dino_bbox_result.png — 원본+검출, union crop 2패널
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    BitsAndBytesConfig,
)

GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_IMAGE = "train/train_0001.jpg"
DEFAULT_PROMPT = "recyclable item"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
CROP_PAD = 20
OUTPUT_PATH = "test_dino_bbox_result.png"


def _device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


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
    p = prompt.strip().rstrip(".") + "."
    print(f"  prompt: '{p}'")
    dev = _device_of(model)
    inputs = processor(images=image, text=p, return_tensors="pt").to(dev)
    outputs = model(**inputs)

    sig = inspect.signature(processor.post_process_grounded_object_detection)
    thr_kwarg = "box_threshold" if "box_threshold" in sig.parameters else "threshold"
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        **{thr_kwarg: box_thr},
        text_threshold=text_thr,
        target_sizes=[image.size[::-1]],
    )
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    labels = results[0]["labels"]
    detections = [
        (int(x1), int(y1), int(x2), int(y2), float(s), str(lb))
        for (x1, y1, x2, y2), s, lb in zip(boxes, scores, labels)
    ]
    print(f"  {len(detections)} detection(s)")
    return detections


def union_crop_box(
    image: Image.Image,
    detections: list[tuple],
    pad: int,
) -> tuple[int, int, int, int]:
    """모든 검출 bbox의 union + 패딩 (파이프라인과 동일)."""
    if not detections:
        return 0, 0, image.width, image.height
    ux1 = max(0, min(d[0] for d in detections) - pad)
    uy1 = max(0, min(d[1] for d in detections) - pad)
    ux2 = min(image.width, max(d[2] for d in detections) + pad)
    uy2 = min(image.height, max(d[3] for d in detections) + pad)
    return ux1, uy1, ux2, uy2


def visualize(
    image: Image.Image,
    crop_img: Image.Image,
    detections: list[tuple],
    crop_origin: tuple[int, int, int, int],
    prompt: str,
    output: str,
) -> None:
    orig_np = np.array(image)
    crop_np = np.array(crop_img)
    ox, oy, _, _ = crop_origin

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#1a1a2e")
    fig.suptitle(
        f'prompt="{prompt}"  detections={len(detections)}',
        color="white",
        fontsize=11,
        fontweight="bold",
    )
    for ax in axes:
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")

    # Panel 1: original + boxes
    axes[0].imshow(orig_np)
    axes[0].set_title("GroundingDINO (full image)", color="white", fontsize=10)
    for x1, y1, x2, y2, score, _ in detections:
        axes[0].add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="#ff4444",
                facecolor="none",
            )
        )
        axes[0].text(x1, y1 - 4, f"{score:.2f}", color="#ff4444", fontsize=8)
    if not detections:
        axes[0].text(
            0.5,
            0.5,
            "NO DETECTIONS",
            transform=axes[0].transAxes,
            color="red",
            ha="center",
            va="center",
            fontsize=12,
        )

    # Panel 2: union crop + boxes in crop coordinates
    axes[1].imshow(crop_np)
    axes[1].set_title("Union crop (+padding)", color="white", fontsize=10)
    for x1, y1, x2, y2, score, _ in detections:
        cx1, cy1 = x1 - ox, y1 - oy
        cx2, cy2 = x2 - ox, y2 - oy
        if cx2 <= 0 or cy2 <= 0 or cx1 >= crop_np.shape[1] or cy1 >= crop_np.shape[0]:
            continue
        axes[1].add_patch(
            patches.Rectangle(
                (cx1, cy1),
                cx2 - cx1,
                cy2 - cy1,
                linewidth=2,
                edgecolor="#44ff88",
                facecolor="none",
            )
        )
        axes[1].text(cx1, cy1 - 4, f"{score:.2f}", color="#44ff88", fontsize=8)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {output}")


def main():
    parser = argparse.ArgumentParser(
        description="GroundingDINO zero-shot detection + union crop viz",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    parser.add_argument(
        "--crop-pad",
        type=int,
        default=CROP_PAD,
        help="Union crop 시 bbox 주변 패딩(px)",
    )
    parser.add_argument(
        "--quant",
        choices=("none", "4", "8"),
        default="8",
        help="GroundingDINO quantization",
    )
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print("\n=== GroundingDINO detection ===")
    print(f"Image  : {image_path}")
    print(f"Prompt : {args.prompt}")

    image = Image.open(image_path).convert("RGB")
    print(f"Size   : {image.width} x {image.height}")

    model, proc = load_gdino(args.quant)
    print("\n[detect]")
    detections = gdino_detect(
        model,
        proc,
        image,
        args.prompt,
        args.box_threshold,
        args.text_threshold,
    )
    for i, (x1, y1, x2, y2, score, label) in enumerate(detections):
        print(
            f"  [{i + 1}] ({x1},{y1})->({x2},{y2})  "
            f"score={score:.3f}  label={label}"
        )

    ux1, uy1, ux2, uy2 = union_crop_box(image, detections, args.crop_pad)
    crop_img = image.crop((ux1, uy1, ux2, uy2))
    print(f"  union bbox: ({ux1},{uy1})->({ux2},{uy2})  pad={args.crop_pad}")

    print("\n[viz]")
    visualize(
        image,
        crop_img,
        detections,
        (ux1, uy1, ux2, uy2),
        args.prompt,
        args.output,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
