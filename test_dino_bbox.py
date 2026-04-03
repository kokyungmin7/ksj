"""Standalone test: GroundingDINO bbox → DINOv2 reference similarity counting.

Usage:
    # 레퍼런스 이미지로 카운팅
    uv run test_dino_bbox.py --image train/train_1940.jpg --prompt "styrofoam box" \
        --reference reference/ex_styrofoam_box.png

    # 사전 계산된 임베딩 갤러리 (.pt) 사용
    uv run test_dino_bbox.py --image train/train_1940.jpg --prompt "styrofoam box" \
        --reference embeddings/styrofoam.pt

    # 임베딩 갤러리 생성 (레퍼런스 이미지 → .pt 파일)
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
SIM_THRESHOLD = 0.5
MIN_BLOB_SIZE = 3
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
    """여러 레퍼런스 이미지 → 단일 L2-정규화 prototype (1, D).

    향후 임베딩 갤러리에서는 이 함수 대신 .pt 파일을 직접 로드.
    """
    feats = []
    for img in images:
        pf = extract_patch_features(model, processor, img)
        feats.append(pf.mean(dim=0))
    prototype = torch.stack(feats).mean(dim=0, keepdim=True)
    return F.normalize(prototype, dim=-1)


def load_reference_prototype(
    path: str, model=None, processor=None,
) -> torch.Tensor:
    """레퍼런스 로드: 이미지 / 이미지 디렉토리 / .pt 임베딩 → (1, D) prototype.

    Returns:
        L2-normalized prototype tensor on CPU.
    """
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


# ── Reference Similarity Counting ─────────────────────────────────────────────

@torch.inference_mode()
def reference_similarity_count(
    model,
    processor,
    image: Image.Image,
    prototype: torch.Tensor,
    sim_threshold: float = SIM_THRESHOLD,
    min_blob_size: int = MIN_BLOB_SIZE,
) -> tuple[int, np.ndarray, np.ndarray]:
    """코사인 유사도 맵 → binary → blob count.

    Args:
        prototype: (1, D) L2-normalized reference embedding.

    Returns:
        (count, sim_map, binary_map)
        sim_map:    (num_h, num_w) float in [-1, 1]
        binary_map: (num_h, num_w) uint8
    """
    dev = _device_of(model)
    proto = prototype.to(dev)

    query_feats = extract_patch_features(model, processor, image)
    query_norm = F.normalize(query_feats, dim=-1)
    sim = (query_norm @ proto.T).squeeze(-1).float().cpu()

    patch_size = model.config.patch_size
    inputs = processor(images=image, return_tensors="pt")
    _, _, h, w = inputs["pixel_values"].shape
    num_h, num_w = h // patch_size, w // patch_size
    sim_map = sim.reshape(num_h, num_w).numpy()

    binary = sim_map >= sim_threshold
    labeled, num_features = ndimage.label(binary)
    count = 0
    for label_id in range(1, num_features + 1):
        if (labeled == label_id).sum() >= min_blob_size:
            count += 1

    return count, sim_map, binary.astype(np.uint8)


# ── Embedding Gallery Builder ─────────────────────────────────────────────────

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
    sim_map, binary_map, count,
    prompt, reference_name, sim_threshold, output,
):
    orig_np = np.array(image)
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor="#1a1a2e")
    fig.suptitle(
        f'prompt="{prompt}"  ref={reference_name}  '
        f'GDino={len(detections)} bbox  count={count}',
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
    axes[1].imshow(np.array(crop_img))
    axes[1].set_title("Union Crop (DINOv2 input)", color="white", fontsize=9)

    # Panel 3: similarity heatmap with threshold line on colorbar
    im = axes[2].imshow(sim_map, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    axes[2].set_title("Cosine Similarity Map", color="white", fontsize=9)
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white", labelsize=7)
    cbar.ax.axhline(y=sim_threshold, color="cyan", linewidth=1.5, linestyle="--")

    # Panel 4: binary + labeled blobs (blob별 다른 색상)
    labeled_map, _ = ndimage.label(binary_map)
    cmap_tab = plt.cm.tab10
    rgb = np.full((*binary_map.shape, 3), 0.12, dtype=np.float32)
    for lbl in range(1, labeled_map.max() + 1):
        mask = labeled_map == lbl
        if mask.sum() < MIN_BLOB_SIZE:
            continue
        color = cmap_tab(lbl % 10)[:3]
        rgb[mask] = color
    axes[3].imshow(rgb, interpolation="nearest")
    axes[3].set_title(f"Binary Blobs (count={count})", color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {output}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GroundingDINO + DINOv2 reference similarity counting",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--reference",
        help="레퍼런스: 이미지 경로 / 디렉토리 / .pt 임베딩 파일",
    )
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    parser.add_argument("--sim-threshold", type=float, default=SIM_THRESHOLD)
    parser.add_argument("--min-blob-size", type=int, default=MIN_BLOB_SIZE)
    parser.add_argument(
        "--quant", choices=("none", "4", "8"), default="8",
        help="GroundingDINO quantization (8bit default)",
    )
    parser.add_argument("--dino-model", default=DINO_MODEL,
                        help="DINOv2 model name or path")
    parser.add_argument("--output", default=OUTPUT_PATH)

    parser.add_argument("--build-gallery",
                        help="레퍼런스 이미지 디렉토리 → .pt 임베딩 생성 후 종료")
    parser.add_argument("--gallery-out", default="embedding.pt",
                        help="--build-gallery 출력 경로")
    args = parser.parse_args()

    # ── Gallery build mode ──
    if args.build_gallery:
        build_gallery(args.build_gallery, args.gallery_out, args.dino_model)
        return

    if not args.reference:
        parser.error("--reference is required (image, directory, or .pt file)")

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print("\n=== GroundingDINO + DINOv2 Reference Counting ===")
    print(f"Image     : {image_path}")
    print(f"Prompt    : {args.prompt}")
    print(f"Reference : {args.reference}")
    print(f"Threshold : sim={args.sim_threshold}  blob_min={args.min_blob_size}")

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

    # ── Step 2: DINOv2 reference similarity counting ──
    print("\n[2] DINOv2 reference similarity counting ...")
    dino_model, dino_proc = load_dinov2(args.dino_model)

    prototype = load_reference_prototype(
        args.reference, dino_model, dino_proc,
    ).to(_device_of(dino_model))

    count, sim_map, binary_map = reference_similarity_count(
        dino_model, dino_proc, crop_img, prototype,
        sim_threshold=args.sim_threshold,
        min_blob_size=args.min_blob_size,
    )
    print(f"  sim_map shape: {sim_map.shape}")
    print(f"  sim range: [{sim_map.min():.3f}, {sim_map.max():.3f}]")
    print(f"  Final count: {count}")

    # ── Step 3: Visualization ──
    print("\n[3] Saving visualization ...")
    ref_name = Path(args.reference).stem
    visualize(
        image, detections, crop_img,
        sim_map, binary_map, count,
        args.prompt, ref_name, args.sim_threshold, args.output,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
