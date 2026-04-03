"""Standalone Grounded SAM test: GroundingDINO bbox → SAM instance count.

Usage:
    uv run test_dino_bbox.py --image train/train_1940.jpg --prompt "styrofoam box"
    uv run test_dino_bbox.py --image train/train_0001.jpg --prompt "PET bottle"
    uv run test_dino_bbox.py --image train/train_0001.jpg --prompt "can"

Output:
    test_dino_bbox_result.png  — 3-panel: original+bboxes | union crop | SAM masks

VRAM (L4 24GB 등): GroundingDINO만 bitsandbytes 양자화 + 추론 후 SAM 로드(순차).
    SAM은 attention에 4D 텐서가 들어가 bnb Linear와 호환되지 않아 항상 FP16.
    --quant 8   # GDINO 8bit (기본)
    --quant 4   # GDINO NF4
    --quant none  # GDINO도 FP16
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import BitsAndBytesConfig, SamModel, SamProcessor

GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM_MODEL = "facebook/sam-vit-base"
DEFAULT_IMAGE = "train/train_0001.jpg"
DEFAULT_PROMPT = "recyclable item"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
OUTPUT_PATH = "test_dino_bbox_result.png"

# Minimal cfg namespace for models/sam.py
SAM_CFG = SimpleNamespace(
    grid_size=8,
    min_mask_area=0.02,
    max_mask_area=0.90,
    iou_threshold=0.5,
    chunk_size=16,
)


# ── Model loading ─────────────────────────────────────────────────────────────

def _bnb_config(quant: str) -> BitsAndBytesConfig | None:
    if quant == "none":
        return None
    if quant == "8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant == "4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"quant must be none|4|8, got {quant!r}")


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _batch_to_model_dtype(batch, model: torch.nn.Module) -> object:
    """SamProcessor 등이 float32를 주면 FP16 모델과 conv dtype이 어긋난다."""
    dev = _model_device(model)
    dt = next(model.parameters()).dtype
    batch = batch.to(dev)
    for k in list(batch.keys()):
        v = batch[k]
        if torch.is_tensor(v) and v.is_floating_point():
            batch[k] = v.to(dtype=dt)
    return batch


def load_gdino(quant: str = "8"):
    print(f"Loading GroundingDINO ({GDINO_MODEL})  quant={quant} ...")
    processor = AutoProcessor.from_pretrained(GDINO_MODEL)
    qcfg = _bnb_config(quant)
    kwargs: dict = {"device_map": "auto"}
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_MODEL, **kwargs
    )
    model.eval()
    print(f"  device: {_model_device(model)}")
    return model, processor


def load_sam_model():
    # bitsandbytes 8bit/4bit Linear는 2D/3D 입력만 지원. SAM ViT qkv는 4D → RuntimeError.
    print(f"Loading SAM ({SAM_MODEL})  dtype=fp16 (no bnb) ...")
    processor = SamProcessor.from_pretrained(SAM_MODEL)
    model = SamModel.from_pretrained(
        SAM_MODEL, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()
    print(f"  device: {_model_device(model)}")
    return model, processor


def free_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── GroundingDINO detection ───────────────────────────────────────────────────

@torch.inference_mode()
def gdino_detect(model, processor, image, prompt, box_thr, text_thr):
    p = prompt.strip().rstrip(".") + "."
    print(f"  prompt: '{p}'")
    dev = _model_device(model)
    inputs = processor(images=image, text=p, return_tensors="pt").to(dev)
    outputs = model(**inputs)

    import inspect
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
    detections = [(int(x1), int(y1), int(x2), int(y2), float(s), str(l))
                  for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels)]
    print(f"  {len(detections)} detection(s)")
    return detections


# ── SAM instance counting ─────────────────────────────────────────────────────

def _nms_masks(masks, iou_thr):
    if not masks:
        return []
    areas = [m.sum() for m in masks]
    keep, suppressed = [], [False] * len(masks)
    for i in range(len(masks)):
        if suppressed[i]:
            continue
        keep.append(i)
        for j in range(i + 1, len(masks)):
            if suppressed[j]:
                continue
            inter = (masks[i] & masks[j]).sum()
            union = areas[i] + areas[j] - inter
            if union > 0 and inter / union > iou_thr:
                suppressed[j] = True
    return keep


@torch.inference_mode()
def sam_count(sam_model, sam_processor, image, bbox, cfg=SAM_CFG):
    x1, y1, x2, y2 = bbox
    crop = image.crop((x1, y1, x2, y2))
    cw, ch = crop.width, crop.height
    crop_area = cw * ch
    print(f"  SAM crop size: {cw}x{ch}")

    grid_points = [
        [(col + 0.5) * cw / cfg.grid_size, (row + 0.5) * ch / cfg.grid_size]
        for row in range(cfg.grid_size)
        for col in range(cfg.grid_size)
    ]

    candidate_masks = []
    for start in range(0, len(grid_points), cfg.chunk_size):
        chunk = grid_points[start: start + cfg.chunk_size]
        inputs = _batch_to_model_dtype(
            sam_processor(
                images=[crop] * len(chunk),
                input_points=[[[p]] for p in chunk],
                return_tensors="pt",
            ),
            sam_model,
        )

        outputs = sam_model(**inputs)
        masks_np = sam_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        for b in range(len(chunk)):
            b_masks = masks_np[b][0]           # (3, H, W)
            b_scores = outputs.iou_scores[b, 0] # (3,)
            best = int(b_scores.argmax())
            mask = b_masks[best].numpy().astype(bool)
            ratio = mask.sum() / crop_area
            if cfg.min_mask_area <= ratio <= cfg.max_mask_area:
                candidate_masks.append(mask)

    print(f"  {len(candidate_masks)} candidates before NMS")
    keep_idx = _nms_masks(candidate_masks, cfg.iou_threshold)
    print(f"  {len(keep_idx)} instances after NMS")
    kept_masks = [candidate_masks[i] for i in keep_idx]
    return len(keep_idx), kept_masks, crop


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize(image, detections, union_bbox, sam_count_val, kept_masks, crop_img, prompt, output):
    orig_np = np.array(image)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor="#1a1a2e")
    fig.suptitle(
        f'Grounded SAM  prompt="{prompt}"  '
        f'GDino={len(detections)} bbox  SAM={sam_count_val} instances',
        color="white", fontsize=10,
    )
    for ax in axes:
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")

    # Panel 1: original + GroundingDINO bboxes
    axes[0].imshow(orig_np)
    axes[0].set_title("GroundingDINO Detections", color="white", fontsize=9)
    for x1, y1, x2, y2, score, label in detections:
        axes[0].add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="#ff4444", facecolor="none",
        ))
        axes[0].text(x1, y1 - 4, f"{score:.2f}", color="#ff4444", fontsize=7)
    if not detections:
        axes[0].text(0.5, 0.5, "NO DETECTIONS", transform=axes[0].transAxes,
                     color="red", ha="center", va="center", fontsize=12)

    # Panel 2: union crop
    axes[1].imshow(np.array(crop_img))
    axes[1].set_title(f"Union Crop (SAM input)", color="white", fontsize=9)

    # Panel 3: SAM masks overlay on crop
    crop_np = np.array(crop_img)
    overlay = crop_np.copy()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(kept_masks), 1)))
    for i, mask in enumerate(kept_masks):
        color = (np.array(colors[i][:3]) * 255).astype(np.uint8)
        overlay[mask] = (overlay[mask] * 0.4 + color * 0.6).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title(f"SAM Masks ({sam_count_val} instances)", color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {output}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument(
        "--quant",
        choices=("none", "4", "8"),
        default="8",
        help="GroundingDINO only: 8=8bit, 4=NF4, none=FP16 (SAM은 항상 FP16)",
    )
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    SAM_CFG.grid_size = args.grid_size

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Grounded SAM Test ===")
    print(f"Image : {image_path}")
    print(f"Prompt: {args.prompt}")

    image = Image.open(image_path).convert("RGB")
    print(f"Size  : {image.width} x {image.height}")

    gdino_model, gdino_proc = load_gdino(args.quant)

    print("\n[1] GroundingDINO detection ...")
    detections = gdino_detect(gdino_model, gdino_proc, image,
                               args.prompt, args.box_threshold, args.text_threshold)
    for i, (x1, y1, x2, y2, score, label) in enumerate(detections):
        print(f"  [{i+1}] ({x1},{y1})->({x2},{y2})  score={score:.3f}  label={label}")

    del gdino_model
    free_cuda_memory()

    if detections:
        pad = 20
        ux1 = max(0, min(d[0] for d in detections) - pad)
        uy1 = max(0, min(d[1] for d in detections) - pad)
        ux2 = min(image.width,  max(d[2] for d in detections) + pad)
        uy2 = min(image.height, max(d[3] for d in detections) + pad)
        union_bbox = (ux1, uy1, ux2, uy2)
        print(f"  union bbox: {union_bbox}")

        print("\n[2] SAM instance counting ...")
        sam_model, sam_proc = load_sam_model()
        count, kept_masks, crop_img = sam_count(sam_model, sam_proc, image, union_bbox)
        print(f"  Final count: {count}")
    else:
        union_bbox = (0, 0, image.width, image.height)
        count, kept_masks, crop_img = 0, [], image

    print("\n[3] Saving visualization ...")
    visualize(image, detections, union_bbox, count, kept_masks, crop_img,
              args.prompt, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
