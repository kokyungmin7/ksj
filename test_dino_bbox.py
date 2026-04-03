"""Standalone GroundingDINO bbox visualization test.

Usage:
    uv run test_dino_bbox.py --image train/train_1940.jpg --prompt "styrofoam box"
    uv run test_dino_bbox.py --image train/train_0001.jpg --prompt "PET bottle"
    uv run test_dino_bbox.py --image train/train_0001.jpg --prompt "can"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_NAME = "IDEA-Research/grounding-dino-tiny"
DEFAULT_IMAGE = "train/train_0001.jpg"
DEFAULT_PROMPT = "recyclable item"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
OUTPUT_PATH = "test_dino_bbox_result.png"


def load_model():
    print(f"Loading GroundingDINO ({MODEL_NAME}) ...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_NAME, device_map="auto"
    )
    model.eval()
    print(f"  device: {next(model.parameters()).device}")
    return model, processor


@torch.inference_mode()
def detect(model, processor, image: Image.Image, prompt: str, box_thr: float, text_thr: float):
    # GroundingDINO expects prompt ending with "."
    p = prompt.strip().rstrip(".") + "."
    print(f"  prompt fed to model: '{p}'")

    inputs = processor(images=image, text=p, return_tensors="pt").to(model.device)
    print(f"  pixel_values: {inputs['pixel_values'].shape}  "
          f"min={inputs['pixel_values'].min():.3f}  max={inputs['pixel_values'].max():.3f}")

    outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_thr,
        text_threshold=text_thr,
        target_sizes=[image.size[::-1]],
    )

    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    labels = results[0]["labels"]

    detections = [(int(x1), int(y1), int(x2), int(y2), float(s), str(l))
                  for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels)]
    return detections


def visualize(image: Image.Image, detections: list, prompt: str, output_path: str):
    import numpy as np
    orig_np = np.array(image)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#1a1a2e")
    fig.suptitle(f'GroundingDINO  prompt="{prompt}"  detections={len(detections)}',
                 color="white", fontsize=10)

    for ax in axes:
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")

    # Panel 1: full image + all bboxes
    axes[0].imshow(orig_np)
    axes[0].set_title(f"Detections (box_thr={BOX_THRESHOLD})", color="white", fontsize=9)
    for x1, y1, x2, y2, score, label in detections:
        axes[0].add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="#ff4444", facecolor="none",
        ))
        axes[0].text(x1, y1 - 4, f"{label} {score:.2f}",
                     color="#ff4444", fontsize=7, va="bottom")
    if not detections:
        axes[0].text(0.5, 0.5, "NO DETECTIONS", transform=axes[0].transAxes,
                     color="red", ha="center", va="center", fontsize=13)

    # Panel 2: union crop
    if detections:
        pad = 20
        ux1 = max(0, min(d[0] for d in detections) - pad)
        uy1 = max(0, min(d[1] for d in detections) - pad)
        ux2 = min(image.width, max(d[2] for d in detections) + pad)
        uy2 = min(image.height, max(d[3] for d in detections) + pad)
        crop = image.crop((ux1, uy1, ux2, uy2))
        axes[1].imshow(crop)
        axes[1].set_title(f"Union Crop ({ux2-ux1}x{uy2-uy1})", color="white", fontsize=9)
    else:
        axes[1].imshow(orig_np)
        axes[1].set_title("Crop (fallback: full image)", color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="English text prompt, e.g. 'styrofoam box'")
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== GroundingDINO Bbox Test ===")
    print(f"Image  : {image_path}")
    print(f"Prompt : {args.prompt}")
    print(f"box_thr: {args.box_threshold}  text_thr: {args.text_threshold}")

    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.width} x {image.height}")

    model, processor = load_model()

    print("\n[1] Running detection ...")
    detections = detect(model, processor, image, args.prompt, args.box_threshold, args.text_threshold)
    print(f"  {len(detections)} detections:")
    for i, (x1, y1, x2, y2, score, label) in enumerate(detections):
        print(f"  [{i+1}] ({x1},{y1})->({x2},{y2})  score={score:.3f}  label={label}")

    print("\n[2] Saving visualization ...")
    visualize(image, detections, args.prompt, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
