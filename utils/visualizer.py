from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.font_manager as fm

# Candidate font names (by registered name in matplotlib font manager)
_KO_FONT_NAMES = [
    "NanumGothic",
    "NanumBarunGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "UnDotum",
    "Malgun Gothic",
    "AppleGothic",
]

# Candidate filename substrings to search for in system font paths
_KO_FONT_FILE_KEYWORDS = [
    "NanumGothic",
    "NanumBarunGothic",
    "NotoSansCJK",
    "NotoSansKR",
    "UnDotum",
    "malgun",
    "AppleGothic",
]


def _find_korean_font() -> str | None:
    """Return a Korean-capable font name, or None if none found.

    Strategy (in order):
    0. Check project-local assets/fonts/ directory first (portable, no root needed).
    1. Match by registered font name in matplotlib's font manager.
    2. Search system font file paths for known Korean font filenames,
       register the found file, then return its family name.
    3. Rebuild the font cache once and retry strategy 1.
    """
    # Strategy 0: project-local font directory (assets/fonts/ next to project root)
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _local_font_dir = os.path.join(_project_root, "assets", "fonts")
    if os.path.isdir(_local_font_dir):
        for fname in sorted(os.listdir(_local_font_dir)):
            if fname.lower().endswith((".ttf", ".otf")):
                fpath = os.path.join(_local_font_dir, fname)
                try:
                    prop = fm.FontProperties(fname=fpath)
                    family = prop.get_name()
                    fm.fontManager.addfont(fpath)
                    return family
                except Exception:
                    continue

    # Strategy 1: name lookup
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _KO_FONT_NAMES:
        if name in available:
            return name

    # Strategy 2: file-path search (handles fonts installed but not yet cached)
    search_paths: list[str] = []
    for ext in ("ttf", "otf"):
        search_paths += fm.findSystemFonts(fontpaths=None, fontext=ext)
    # also check user font dir explicitly (may not be in system paths on some distros)
    user_font_dir = os.path.expanduser("~/.local/share/fonts")
    if os.path.isdir(user_font_dir):
        for ext in ("ttf", "otf"):
            search_paths += fm.findSystemFonts(fontpaths=[user_font_dir], fontext=ext)
    for path in search_paths:
        basename = os.path.basename(path)
        if any(kw.lower() in basename.lower() for kw in _KO_FONT_FILE_KEYWORDS):
            try:
                prop = fm.FontProperties(fname=path)
                family = prop.get_name()
                fm.fontManager.addfont(path)
                return family
            except Exception:
                continue

    # Strategy 3: rebuild cache and retry name lookup
    try:
        fm._load_fontmanager(try_read_cache=False)  # type: ignore[attr-defined]
    except Exception:
        pass
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _KO_FONT_NAMES:
        if name in available:
            return name

    return None


_ko_font = _find_korean_font()
if _ko_font:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [_ko_font, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
else:
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


def save_visualization(
    original_image: Image.Image,
    trace: dict,
    row: dict,
    pred: str,
    output_path: str,
) -> None:
    """Save a 4-panel visualization of the prediction process.

    Panels (left to right):
        1. Original image with attention bbox overlay
        2. DINOv3 attention heatmap (jet colormap)
        3. Attention overlay on original image (semi-transparent)
        4. DINOv3 attention crop

    Below the panels:
        - Question text
        - Choices with correct answer marked
        - Prediction result and routing info
    """
    orig_np = np.array(original_image)
    attn_full = trace.get("attn_map_full")   # None for GroundingDINO / disabled
    crop_img = trace.get("crop")
    bbox = trace.get("bbox")
    blobs_bbox = trace.get("blobs_bbox") or []
    route = trace["route"]
    dino_cnt = trace["dino_count"]
    text_prompt = trace.get("text_prompt")
    gt = str(row.get("answer", "?")).strip().lower()
    is_correct = pred == gt

    # Layout mode:
    #   "attn"       → DINOv3 mode: 4 panels (original+bbox, heatmap, overlay, crop)
    #   "grounding"  → GroundingDINO mode: 2 panels (original+bbox, crop)
    #   "disabled"   → no detector: 1 panel (original only)
    if attn_full is not None:
        mode = "attn"
        num_panels = 4
    elif crop_img is not None:
        mode = "grounding"
        num_panels = 2
    else:
        mode = "disabled"
        num_panels = 1

    fig = plt.figure(figsize=(5 * num_panels + 2, 10), facecolor="#1a1a2e")

    gs = fig.add_gridspec(
        2, num_panels,
        height_ratios=[3, 1],
        hspace=0.05, wspace=0.05,
        left=0.02, right=0.98, top=0.95, bottom=0.02,
    )

    ax_text = fig.add_subplot(gs[1, :])

    # ── Shared: draw bboxes helper ───────────────────────────────────────────
    def _draw_bboxes(ax, title: str):
        ax.imshow(orig_np)
        draw_bboxes = blobs_bbox if blobs_bbox else ([bbox] if bbox else [])
        for bx1, by1, bx2, by2 in draw_bboxes:
            ax.add_patch(patches.Rectangle(
                (bx1, by1), bx2 - bx1, by2 - by1,
                linewidth=2, edgecolor="#ff4444", facecolor="none",
            ))
        n = len(draw_bboxes)
        label = f"{title} ({n}개)" if n > 1 else title
        ax.set_title(label, color="white", fontsize=9, pad=4)
        ax.axis("off")

    if mode == "attn":
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])

        _draw_bboxes(ax1, "원본 이미지 + ROI bbox")

        attn_norm_vis = (attn_full - attn_full.min()) / (attn_full.max() - attn_full.min() + 1e-8)
        ax2.imshow(attn_norm_vis, cmap="jet", interpolation="bilinear")
        ax2.set_title("Attention Heatmap", color="white", fontsize=9, pad=4)
        ax2.axis("off")

        cmap = plt.colormaps["jet"]
        heat_rgba = cmap(attn_norm_vis)
        heat_rgb = (heat_rgba[:, :, :3] * 255).astype(np.uint8)
        overlay = (orig_np.astype(float) * 0.55 + heat_rgb.astype(float) * 0.45).clip(0, 255).astype(np.uint8)
        ax3.imshow(overlay)
        ax3.set_title("Attention Overlay", color="white", fontsize=9, pad=4)
        ax3.axis("off")

        ax4.imshow(np.array(crop_img))
        crop_label = "Crop (원본 사용)" if trace.get("used_full") else "ROI Crop"
        ax4.set_title(crop_label, color="white", fontsize=9, pad=4)
        ax4.axis("off")

    elif mode == "grounding":
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        prompt_label = f'"{text_prompt}"' if text_prompt else ""
        _draw_bboxes(ax1, f"GroundingDINO {prompt_label}")

        ax2.imshow(np.array(crop_img))
        crop_label = "Crop (원본 사용)" if trace.get("used_full") else "ROI Crop"
        ax2.set_title(crop_label, color="white", fontsize=9, pad=4)
        ax2.axis("off")

    else:
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(orig_np)
        ax1.set_title("원본 이미지 (detector 비활성화)", color="white", fontsize=9, pad=4)
        ax1.axis("off")

    # ── Text panel ───────────────────────────────────────────────────────────
    ax_text.set_facecolor("#0d0d1a")
    ax_text.axis("off")

    q_text = str(row.get("question", ""))
    choices = {k: str(row.get(k, "")) for k in "abcd"}

    choice_parts = []
    for k, v in choices.items():
        if k == gt and k == pred:
            marker = " ✓"
        elif k == gt:
            marker = " ←정답"
        elif k == pred:
            marker = " ←예측"
        else:
            marker = ""
        choice_parts.append(f"{k}) {v}{marker}")

    choices_str = "    ".join(choice_parts)

    if route == "grounding_count":
        route_str = f"grounding_count({dino_cnt}개 감지, prompt={text_prompt}) → {pred}"
    elif route == "qwen_with_crop":
        route_str = f"qwen_with_crop(prompt={text_prompt}) → {pred}"
    else:
        route_str = f"qwen_only → {pred}"
    result_str = f"예측: {pred}  |  정답: {gt}  |  {'✓ 정답' if is_correct else '✗ 오답'}"

    full_text = (
        f"Q: {q_text}\n"
        f"{choices_str}\n"
        f"경로: {route_str}    {result_str}"
    )

    text_color = "#44ff88" if is_correct else "#ff6666"
    ax_text.text(
        0.01, 0.95, full_text,
        transform=ax_text.transAxes,
        fontsize=8.5,
        color=text_color,
        verticalalignment="top",
        wrap=True,
        fontfamily="sans-serif",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
