"""Standalone test: GroundingDINO bbox → DINOv2 attention blob counting.

Usage:
    # Attention blob counting (기본)
    uv run test_dino_bbox.py --image train/train_1940.jpg --prompt "styrofoam box"

    # Attention threshold / min blob size 조정
    uv run test_dino_bbox.py --image train/train_1940.jpg --prompt "styrofoam box" \
        --attn-threshold 0.6 --min-blob-size 30

    # 임베딩 갤러리 생성 (향후 reference similarity 확장용)
    uv run test_dino_bbox.py --build-gallery reference/styrofoam/ \
        --gallery-out embeddings/styrofoam.pt

Output:
    test_dino_bbox_result.png — 4-panel visualization
"""
from __future__ import annotations

import argparse
import gc
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
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    BitsAndBytesConfig,
)

GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DINO_MODEL = "facebook/dinov2-small"
DEFAULT_IMAGE = "train/train_0001.jpg"
DEFAULT_PROMPT = "recyclable item"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
ATTN_THRESHOLD = 0.5
MIN_BLOB_SIZE = 50
OUTPUT_PATH = "test_dino_bbox_result.png"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


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
    p = prompt.strip().rstrip(".") + "."
    print(f"  prompt: '{p}'")
    dev = _device_of(model)
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
    detections = [
        (int(x1), int(y1), int(x2), int(y2), float(s), str(lb))
        for (x1, y1, x2, y2), s, lb in zip(boxes, scores, labels)
    ]
    print(f"  {len(detections)} detection(s)")
    return detections


# ── DINOv2 ────────────────────────────────────────────────────────────────────

def load_dinov2(model_name: str = DINO_MODEL):
    print(f"Loading DINOv2 ({model_name}) ...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print(f"  device: {_device_of(model)}")
    return model, processor


# ── Attention Blob Counting ───────────────────────────────────────────────────

@torch.inference_mode()
def get_attention_map(
    model, processor, image: Image.Image,
) -> tuple[np.ndarray, int, int]:
    """CLS→patch attention map (num_h, num_w) + crop pixel dims."""
    inputs = processor(images=image, return_tensors="pt").to(_device_of(model))
    outputs = model(**inputs, output_attentions=True)

    attn = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
    num_register = getattr(model.config, "num_register_tokens", 0)
    patch_size = model.config.patch_size
    _, _, h, w = inputs["pixel_values"].shape
    num_h, num_w = h // patch_size, w // patch_size

    cls_attn = attn[:, 0, 1 + num_register:].mean(dim=0)  # (num_patches,)
    attn_map = cls_attn.reshape(num_h, num_w).float().cpu().numpy()
    return attn_map, image.width, image.height


def upsample_map(
    arr: np.ndarray, target_w: int, target_h: int, mode: str = "bilinear",
) -> np.ndarray:
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    up = F.interpolate(t, size=(target_h, target_w), mode=mode, **kwargs)
    return up.squeeze().numpy()


def mass_threshold(attn_map: np.ndarray, threshold: float) -> np.ndarray:
    """Top-`threshold` fraction of attention mass → binary mask.

    Facebook DINO official approach: sort ascending → cumsum → keep pixels
    where cumsum > (1 - threshold).  Image-agnostic adaptive threshold.
    """
    flat = attn_map.flatten()
    sorted_vals = np.sort(flat)
    sorted_norm = sorted_vals / (sorted_vals.sum() + 1e-8)
    cumsum = np.cumsum(sorted_norm)
    above = sorted_vals[cumsum > (1 - threshold)]
    cutoff = above[0] if len(above) > 0 else flat.min()
    return (attn_map >= cutoff).astype(np.uint8)


def count_blobs(binary_map: np.ndarray, min_size: int) -> tuple[int, np.ndarray]:
    """Count connected components ≥ min_size. Returns (count, labeled_map)."""
    labeled, num_features = ndimage.label(binary_map.astype(bool))
    count = 0
    for label_id in range(1, num_features + 1):
        if (labeled == label_id).sum() >= min_size:
            count += 1
        else:
            labeled[labeled == label_id] = 0
    return count, labeled


def attention_blob_count(
    model, processor, image: Image.Image,
    attn_threshold: float = ATTN_THRESHOLD,
    min_blob_size: int = MIN_BLOB_SIZE,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Attention map → upsample → mass threshold → blob count.

    Returns:
        (count, attn_map_raw, attn_heatmap, binary_full)
        attn_map_raw:  (num_h, num_w) raw patch-level attention
        attn_heatmap:  (crop_h, crop_w) bilinear upsampled (for visualization)
        binary_full:   (crop_h, crop_w) pixel-level binary mask
    """
    attn_map, crop_w, crop_h = get_attention_map(model, processor, image)
    attn_heatmap = upsample_map(attn_map, crop_w, crop_h, mode="bilinear")
    binary_full = mass_threshold(
        upsample_map(attn_map, crop_w, crop_h, mode="nearest"),
        attn_threshold,
    )
    count, _ = count_blobs(binary_full, min_blob_size)
    return count, attn_map, attn_heatmap, binary_full


# ── Reference Embedding (optional, 향후 확장용) ──────────────────────────────

@torch.inference_mode()
def extract_patch_features(model, processor, image: Image.Image) -> torch.Tensor:
    """(num_patches, hidden_dim) patch features."""
    inputs = processor(images=image, return_tensors="pt").to(_device_of(model))
    outputs = model(**inputs)
    num_register = getattr(model.config, "num_register_tokens", 0)
    return outputs.last_hidden_state[0, 1 + num_register:]


def build_reference_embedding(
    model, processor, images: list[Image.Image],
) -> torch.Tensor:
    """여러 레퍼런스 이미지 → 단일 L2-정규화 prototype (1, D)."""
    feats = []
    for img in images:
        pf = extract_patch_features(model, processor, img)
        feats.append(pf.mean(dim=0))
    prototype = torch.stack(feats).mean(dim=0, keepdim=True)
    return F.normalize(prototype, dim=-1)


def load_reference_prototype(
    path: str, model=None, processor=None,
) -> torch.Tensor:
    """레퍼런스 로드: 이미지 / 디렉토리 / .pt → (1, D) prototype (CPU)."""
    p = Path(path)

    if p.suffix == ".pt":
        proto = torch.load(p, map_location="cpu", weights_only=True)
        if proto.ndim == 1:
            proto = proto.unsqueeze(0)
        print(f"  Loaded embedding: {p}  shape={tuple(proto.shape)}")
        return F.normalize(proto.float(), dim=-1)

    if p.is_dir():
        imgs = [
            Image.open(f).convert("RGB")
            for f in sorted(p.iterdir()) if f.suffix.lower() in IMAGE_EXTS
        ]
        if not imgs:
            raise FileNotFoundError(f"No images in {p}")
        print(f"  {len(imgs)} reference image(s) from {p}")
    elif p.suffix.lower() in IMAGE_EXTS:
        imgs = [Image.open(p).convert("RGB")]
        print(f"  1 reference image: {p}")
    else:
        raise ValueError(f"Unsupported reference: {p}")

    assert model is not None and processor is not None, \
        "model/processor required when reference is image(s)"
    return build_reference_embedding(model, processor, imgs).cpu()


def build_gallery(image_dir: str, output_path: str, model_name: str = DINO_MODEL):
    """레퍼런스 이미지 디렉토리 → .pt 임베딩 파일 생성."""
    model, processor = load_dinov2(model_name)
    proto = load_reference_prototype(image_dir, model, processor)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(proto.squeeze(0), output_path)
    print(f"  Saved embedding: {output_path}  shape={tuple(proto.shape)}")


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize(
    image, detections, crop_img,
    attn_heatmap, binary_full, count,
    prompt, attn_threshold, output,
):
    orig_np = np.array(image)
    crop_np = np.array(crop_img)
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor="#1a1a2e")
    fig.suptitle(
        f'prompt="{prompt}"  '
        f'GDino={len(detections)} bbox  '
        f'attn_thr={attn_threshold}  count={count}',
        color="white", fontsize=11, fontweight="bold",
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
        axes[0].text(
            0.5, 0.5, "NO DETECTIONS", transform=axes[0].transAxes,
            color="red", ha="center", va="center", fontsize=12,
        )

    # Panel 2: union crop
    axes[1].imshow(crop_np)
    axes[1].set_title("Union Crop", color="white", fontsize=9)

    # Panel 3: attention heatmap overlaid on crop
    axes[2].imshow(crop_np)
    axes[2].imshow(attn_heatmap, cmap="jet", alpha=0.55)
    axes[2].set_title("DINOv2 Attention Heatmap", color="white", fontsize=9)

    # Panel 4: binary mask with labeled blobs
    _, labeled_map = count_blobs(binary_full, MIN_BLOB_SIZE)
    cmap_tab = plt.cm.tab10
    overlay = crop_np.copy().astype(np.float32) / 255.0 * 0.3
    n_labels = labeled_map.max()
    for lbl in range(1, n_labels + 1):
        mask = labeled_map == lbl
        if not mask.any():
            continue
        color = np.array(cmap_tab(lbl % 10)[:3])
        overlay[mask] = overlay[mask] + color * 0.7
    overlay = np.clip(overlay, 0, 1)
    axes[3].imshow(overlay)
    axes[3].set_title(f"Attention Blobs (count={count})", color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {output}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GroundingDINO + DINOv2 attention blob counting",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    parser.add_argument("--attn-threshold", type=float, default=ATTN_THRESHOLD,
                        help="Mass threshold (0-1): fraction of attention to keep")
    parser.add_argument("--min-blob-size", type=int, default=MIN_BLOB_SIZE,
                        help="Min blob area in pixels (after upsample)")
    parser.add_argument(
        "--quant", choices=("none", "4", "8"), default="8",
        help="GroundingDINO quantization (8bit default)",
    )
    parser.add_argument("--dino-model", default=DINO_MODEL,
                        help="DINOv2 model name or path")
    parser.add_argument("--output", default=OUTPUT_PATH)

    parser.add_argument(
        "--reference",
        help="(optional) 레퍼런스: 이미지 / 디렉토리 / .pt 임베딩",
    )
    parser.add_argument("--build-gallery",
                        help="레퍼런스 이미지 디렉토리 → .pt 임베딩 생성 후 종료")
    parser.add_argument("--gallery-out", default="embedding.pt",
                        help="--build-gallery 출력 경로")
    args = parser.parse_args()

    # ── Gallery build mode ──
    if args.build_gallery:
        build_gallery(args.build_gallery, args.gallery_out, args.dino_model)
        return

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print("\n=== GroundingDINO + DINOv2 Attention Blob Counting ===")
    print(f"Image     : {image_path}")
    print(f"Prompt    : {args.prompt}")
    print(f"Threshold : attn={args.attn_threshold}  blob_min={args.min_blob_size}px")

    image = Image.open(image_path).convert("RGB")
    print(f"Size      : {image.width} x {image.height}")

    # ── Step 1: GroundingDINO detection ──
    gdino_model, gdino_proc = load_gdino(args.quant)
    print("\n[1] GroundingDINO detection ...")
    detections = gdino_detect(
        gdino_model, gdino_proc, image,
        args.prompt, args.box_threshold, args.text_threshold,
    )
    for i, (x1, y1, x2, y2, score, label) in enumerate(detections):
        print(f"  [{i+1}] ({x1},{y1})->({x2},{y2})  score={score:.3f}  label={label}")

    del gdino_model
    _free_vram()

    # ── Compute union crop ──
    if detections:
        pad = 20
        ux1 = max(0, min(d[0] for d in detections) - pad)
        uy1 = max(0, min(d[1] for d in detections) - pad)
        ux2 = min(image.width, max(d[2] for d in detections) + pad)
        uy2 = min(image.height, max(d[3] for d in detections) + pad)
        crop_img = image.crop((ux1, uy1, ux2, uy2))
        print(f"  union bbox: ({ux1},{uy1})->({ux2},{uy2})")
    else:
        crop_img = image
        print("  No detections → using full image")

    # ── Step 2: DINOv2 attention blob counting ──
    print("\n[2] DINOv2 attention blob counting ...")
    dino_model, dino_proc = load_dinov2(args.dino_model)

    count, attn_raw, attn_heatmap, binary_full = attention_blob_count(
        dino_model, dino_proc, crop_img,
        attn_threshold=args.attn_threshold,
        min_blob_size=args.min_blob_size,
    )
    print(f"  attn_map shape: {attn_raw.shape}")
    print(f"  attn range: [{attn_raw.min():.4f}, {attn_raw.max():.4f}]")
    print(f"  binary pixels: {binary_full.sum()} / {binary_full.size}")
    print(f"  Final count: {count}")

    # ── Step 3: Visualization ──
    print("\n[3] Saving visualization ...")
    visualize(
        image, detections, crop_img,
        attn_heatmap, binary_full, count,
        args.prompt, args.attn_threshold, args.output,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
