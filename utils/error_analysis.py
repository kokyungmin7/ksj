"""Epoch별 오답 분석 및 학습 종료 시 종합 요약 시각화.

출력 구조:
    analysis/
    ├── epoch_1/
    │   ├── wrong_predictions.json
    │   └── wrong_grid.png
    ├── epoch_2/
    │   └── ...
    └── summary/
        ├── training_curves.png
        ├── confusion_matrix.png
        └── error_analysis.png
"""

from __future__ import annotations

import json
import os
import textwrap
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm

# ── 한국어 폰트 설정 ────────────────────────────────────────────────────────

_KO_FONTS = [
    "NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR",
    "Noto Sans KR", "UnDotum", "Malgun Gothic", "AppleGothic",
]
_KO_FILE_KEYWORDS = [
    "NanumGothic", "NanumBarunGothic", "NotoSansCJK",
    "NotoSansKR", "UnDotum", "malgun", "AppleGothic",
]


def _setup_korean_font() -> None:
    """Configure matplotlib rcParams with a Korean-capable font."""
    # 1. Project-local assets/fonts/
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_dir = os.path.join(root, "assets", "fonts")
    if os.path.isdir(local_dir):
        for fn in sorted(os.listdir(local_dir)):
            if fn.lower().endswith((".ttf", ".otf")):
                fp = os.path.join(local_dir, fn)
                try:
                    prop = fm.FontProperties(fname=fp)
                    fm.fontManager.addfont(fp)
                    _apply_font(prop.get_name())
                    return
                except Exception:
                    continue

    # 2. System font name lookup
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _KO_FONTS:
        if name in available:
            _apply_font(name)
            return

    # 3. System font file path search
    sys_paths = fm.findSystemFonts(fontpaths=None, fontext="ttf")
    sys_paths += fm.findSystemFonts(fontpaths=None, fontext="otf")
    user_dir = os.path.expanduser("~/.local/share/fonts")
    if os.path.isdir(user_dir):
        sys_paths += fm.findSystemFonts(fontpaths=[user_dir], fontext="ttf")
    for path in sys_paths:
        if any(kw.lower() in os.path.basename(path).lower() for kw in _KO_FILE_KEYWORDS):
            try:
                prop = fm.FontProperties(fname=path)
                fm.fontManager.addfont(path)
                _apply_font(prop.get_name())
                return
            except Exception:
                continue

    warnings.filterwarnings("ignore", message="Glyph .* missing from font")


def _apply_font(name: str) -> None:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


_setup_korean_font()

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix


# ── Public API ───────────────────────────────────────────────────────────────

def save_epoch_errors(
    all_results: list[dict],
    epoch: int,
    analysis_dir: str,
    image_root: str,
    dpi: int = 150,
    max_grid: int = 20,
) -> None:
    """Save wrong predictions JSON and grid visualization for one epoch."""
    epoch_dir = os.path.join(analysis_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    wrong_list = [r for r in all_results if not r["correct"]]

    with open(os.path.join(epoch_dir, "wrong_predictions.json"), "w", encoding="utf-8") as f:
        json.dump({
            "epoch": epoch,
            "total": len(all_results),
            "wrong_count": len(wrong_list),
            "wrong": wrong_list,
        }, f, ensure_ascii=False, indent=2)

    if wrong_list:
        _save_wrong_grid(
            wrong_list[:max_grid], epoch, len(wrong_list),
            os.path.join(epoch_dir, "wrong_grid.png"), image_root, dpi,
        )


def save_training_summary(
    epoch_records: list[dict],
    analysis_dir: str,
    dpi: int = 150,
) -> None:
    """Generate end-of-training summary: curves, confusion matrix, error patterns."""
    if not epoch_records:
        return

    summary_dir = os.path.join(analysis_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    _save_training_curves(epoch_records, summary_dir, dpi)

    last = epoch_records[-1]
    _save_confusion_matrix(last["all_results"], last["epoch"], summary_dir, dpi)
    _save_error_patterns(epoch_records, summary_dir, dpi)


# ── Wrong Grid ───────────────────────────────────────────────────────────────

def _save_wrong_grid(
    wrong_list: list[dict],
    epoch: int,
    total_wrong: int,
    output_path: str,
    image_root: str,
    dpi: int,
) -> None:
    n = len(wrong_list)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    cell_w, img_h, txt_h = 3.8, 3.0, 1.3
    fig = plt.figure(
        figsize=(cell_w * n_cols + 0.6, (img_h + txt_h) * n_rows + 1.2),
        facecolor="#1a1a2e",
    )

    shown = f" (상위 {n}건)" if total_wrong > n else ""
    fig.suptitle(
        f"Epoch {epoch} — 오답 {total_wrong}건{shown}",
        color="white", fontsize=13, fontweight="bold",
    )

    row_ratios: list[float] = []
    for _ in range(n_rows):
        row_ratios.extend([img_h, txt_h])

    gs = fig.add_gridspec(
        n_rows * 2, n_cols,
        height_ratios=row_ratios,
        hspace=0.08, wspace=0.12,
        left=0.02, right=0.98, top=0.93, bottom=0.01,
    )

    for idx, item in enumerate(wrong_list):
        lr, col = divmod(idx, n_cols)

        ax_img = fig.add_subplot(gs[lr * 2, col])
        _draw_thumbnail(ax_img, item, image_root)

        ax_txt = fig.add_subplot(gs[lr * 2 + 1, col])
        _draw_caption(ax_txt, item)

    for idx in range(n, n_rows * n_cols):
        lr, col = divmod(idx, n_cols)
        for row_off in (0, 1):
            fig.add_subplot(gs[lr * 2 + row_off, col]).set_visible(False)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_thumbnail(ax, item: dict, image_root: str) -> None:
    img_path = os.path.join(image_root, str(item["path"]))
    try:
        img = Image.open(img_path).convert("RGB")
        ax.imshow(np.array(img))
    except Exception:
        ax.text(
            0.5, 0.5, "이미지\n로드 실패",
            ha="center", va="center", color="gray",
            transform=ax.transAxes, fontsize=10,
        )
    ax.axis("off")


def _draw_caption(ax, item: dict) -> None:
    ax.set_facecolor("#0d0d1a")
    ax.axis("off")

    gt, pred = item["ground_truth"], item["predicted"]
    q = textwrap.shorten(str(item["question"]), width=42, placeholder="…")
    gt_txt = textwrap.shorten(str(item.get(gt, "")), width=20, placeholder="…")
    pred_txt = textwrap.shorten(str(item.get(pred, "")), width=20, placeholder="…")

    caption = (
        f"Q: {q}\n"
        f"정답: {gt}) {gt_txt}\n"
        f"예측: {pred}) {pred_txt}"
    )
    ax.text(
        0.04, 0.92, caption,
        transform=ax.transAxes,
        fontsize=7, color="#ff6666",
        va="top", ha="left", linespacing=1.5,
    )


# ── Training Curves ──────────────────────────────────────────────────────────

def _save_training_curves(
    epoch_records: list[dict],
    summary_dir: str,
    dpi: int,
) -> None:
    epochs = [r["epoch"] for r in epoch_records]
    accs = [r["accuracy"] for r in epoch_records]
    f1s = [r["f1_macro"] for r in epoch_records]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#0d0d1a")

    c_acc, c_f1 = "#44ff88", "#4488ff"
    ax.plot(epochs, accs, "o-", color=c_acc, label="Accuracy", lw=2, ms=8)
    ax.plot(epochs, f1s, "s-", color=c_f1, label="F1 Macro", lw=2, ms=8)

    for e, a, f in zip(epochs, accs, f1s):
        ax.annotate(f"{a:.3f}", (e, a), textcoords="offset points",
                    xytext=(0, 10), color=c_acc, fontsize=8, ha="center")
        ax.annotate(f"{f:.3f}", (e, f), textcoords="offset points",
                    xytext=(0, -14), color=c_f1, fontsize=8, ha="center")

    ax.set_xlabel("Epoch", color="white", fontsize=12)
    ax.set_ylabel("Score", color="white", fontsize=12)
    ax.set_title("학습 진행 곡선 (Accuracy & F1 Macro)", color="white", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.tick_params(colors="white")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.15)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    plt.tight_layout()
    plt.savefig(
        os.path.join(summary_dir, "training_curves.png"),
        dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


# ── Confusion Matrix ─────────────────────────────────────────────────────────

def _save_confusion_matrix(
    all_results: list[dict],
    epoch: int,
    summary_dir: str,
    dpi: int,
) -> None:
    labels = ["a", "b", "c", "d"]
    gts = [r["ground_truth"] for r in all_results]
    preds = [r["predicted"] for r in all_results]

    cm = confusion_matrix(gts, preds, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5.5), facecolor="#1a1a2e")
    ax.set_facecolor("#0d0d1a")

    im = ax.imshow(cm, cmap="YlOrRd", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white")

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, color="white", fontsize=13)
    ax.set_yticklabels(labels, color="white", fontsize=13)
    ax.set_xlabel("예측", color="white", fontsize=12)
    ax.set_ylabel("정답", color="white", fontsize=12)
    ax.set_title(f"혼동 행렬 (Epoch {epoch})", color="white", fontsize=14)
    ax.tick_params(colors="white")

    thresh = cm.max() * 0.5
    for i in range(4):
        for j in range(4):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=15, fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(summary_dir, "confusion_matrix.png"),
        dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


# ── Error Pattern Analysis ───────────────────────────────────────────────────

def _save_error_patterns(
    epoch_records: list[dict],
    summary_dir: str,
    dpi: int,
) -> None:
    last = epoch_records[-1]
    wrong = [r for r in last["all_results"] if not r["correct"]]
    if not wrong:
        return

    pairs = Counter(f"{r['ground_truth']}→{r['predicted']}" for r in wrong)
    sorted_pairs = pairs.most_common()
    pair_labels = [p[0] for p in sorted_pairs]
    pair_counts = [p[1] for p in sorted_pairs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#1a1a2e")

    # ── Left: confusion pair bars ──
    ax1.set_facecolor("#0d0d1a")
    colors = plt.cm.Reds(np.linspace(0.35, 0.85, len(pair_labels)))
    bars = ax1.barh(range(len(pair_labels)), pair_counts, color=colors)
    ax1.set_yticks(range(len(pair_labels)))
    ax1.set_yticklabels(
        [f"정답 {l.split('→')[0]} → 예측 {l.split('→')[1]}" for l in pair_labels],
        color="white", fontsize=9,
    )
    ax1.set_xlabel("오답 횟수", color="white", fontsize=11)
    ax1.set_title(
        f"오답 패턴 (Epoch {last['epoch']})", color="white", fontsize=13,
    )
    ax1.tick_params(colors="white")
    ax1.invert_yaxis()
    for spine in ax1.spines.values():
        spine.set_color("#333333")
    for bar, cnt in zip(bars, pair_counts):
        ax1.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(cnt), color="white", va="center", fontsize=10,
        )

    # ── Right: epoch-wise wrong count ──
    ax2.set_facecolor("#0d0d1a")
    ep_x = [r["epoch"] for r in epoch_records]
    wrong_n = [r["total"] - r["correct"] for r in epoch_records]
    total_n = [r["total"] for r in epoch_records]

    ax2.bar(ep_x, total_n, color="#334466", label="전체 샘플", alpha=0.7)
    ax2.bar(ep_x, wrong_n, color="#ff6666", label="오답 수", alpha=0.9)

    for e, w, t in zip(ep_x, wrong_n, total_n):
        pct = w / t * 100 if t > 0 else 0
        ax2.text(
            e, w + max(total_n) * 0.02,
            f"{w}건 ({pct:.1f}%)",
            ha="center", va="bottom", color="#ff6666", fontsize=9,
        )

    ax2.set_xlabel("Epoch", color="white", fontsize=11)
    ax2.set_ylabel("건수", color="white", fontsize=11)
    ax2.set_title("Epoch별 오답 수 추이", color="white", fontsize=13)
    ax2.set_xticks(ep_x)
    ax2.legend(fontsize=10)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("#333333")

    plt.tight_layout()
    plt.savefig(
        os.path.join(summary_dir, "error_analysis.png"),
        dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
