from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace

from PIL import Image
from tqdm import tqdm

from data.dataset import load_and_split
from models.qwen import load_finetuned_qwen, load_base_qwen
from models.grounding_dino import load_grounding_dino
from pipeline.predictor import Predictor
from utils.visualizer import save_visualization


def run_evaluation(
    cfg: SimpleNamespace,
    checkpoint_dir: str | None = None,
    reference_dir: str | None = None,
) -> dict:
    """Evaluate on the validation split and save results + visualizations.

    Args:
        checkpoint_dir: Path to fine-tuned checkpoint. If None, uses base model.
        reference_dir: Optional reference gallery for DINOv3 counting.

    Returns:
        Dict with accuracy, per_class accuracy, and per-sample results.
    """
    # Load models
    if checkpoint_dir:
        print(f"Loading fine-tuned model from {checkpoint_dir} ...")
        qwen_model, qwen_processor = load_finetuned_qwen(checkpoint_dir, cfg)
    else:
        print("Loading base model (zero-shot evaluation) ...")
        qwen_model, qwen_processor = load_base_qwen(cfg)

    dino_enabled = getattr(cfg.dino, "enabled", True)
    if dino_enabled:
        print("Loading GroundingDINO ...")
        dino_model, dino_processor = load_grounding_dino(cfg)
    else:
        print("GroundingDINO disabled — skipping load.")
        dino_model, dino_processor = None, None

    predictor = Predictor(
        qwen_model, qwen_processor,
        dino_model, dino_processor,
        cfg,
        reference_dir=reference_dir,
    )

    _, val_df = load_and_split(cfg)
    print(f"Evaluating on {len(val_df)} validation samples ...")

    viz_enabled = getattr(cfg, "visualization", None) and cfg.visualization.enabled
    if viz_enabled:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = os.path.join(cfg.visualization.output_dir, run_ts)
    else:
        viz_dir = None
    save_every = cfg.visualization.save_every if viz_enabled else 1
    max_viz = cfg.visualization.max_samples if viz_enabled else None

    predictions, ground_truths, samples = [], [], []
    viz_count = 0

    for idx, (_, row) in enumerate(tqdm(val_df.iterrows(), total=len(val_df), desc="Evaluating")):
        row_dict = row.to_dict()
        pred, trace = predictor.predict_with_trace(row_dict)
        gt = str(row_dict["answer"]).strip().lower()

        predictions.append(pred)
        ground_truths.append(gt)
        samples.append({
            "id": row_dict["id"],
            "question": row_dict["question"],
            "ground_truth": gt,
            "predicted": pred,
            "correct": pred == gt,
            "route": trace["route"],
            "bbox": list(trace["bbox"]) if trace["bbox"] is not None else [],
            "dino_count": trace["dino_count"],
        })

        # Visualization
        should_viz = (
            viz_enabled
            and (idx % save_every == 0)
            and (max_viz is None or viz_count < max_viz)
        )
        if should_viz:
            image_path = os.path.join(cfg.data.image_root, str(row_dict["path"]))
            original_image = Image.open(image_path).convert("RGB")
            sample_id = str(row_dict["id"]).replace("/", "_")
            out_path = os.path.join(viz_dir, f"{sample_id}.png")
            save_visualization(original_image, trace, row_dict, pred, out_path)
            viz_count += 1

    # Metrics
    accuracy = sum(p == g for p, g in zip(predictions, ground_truths)) / len(predictions)

    per_class: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for p, g in zip(predictions, ground_truths):
        per_class[g]["total"] += 1
        if p == g:
            per_class[g]["correct"] += 1

    per_class_acc = {
        cls: d["correct"] / d["total"]
        for cls, d in sorted(per_class.items())
    }

    # Route breakdown
    route_counts: dict[str, int] = defaultdict(int)
    for s in samples:
        route_counts[s["route"] or "unknown"] += 1

    print(f"\nValidation Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    for cls, acc in per_class_acc.items():
        d = per_class[cls]
        print(f"  Class {cls}: {d['correct']}/{d['total']} = {acc:.3f}")
    print(f"\nRouting breakdown:")
    for route, cnt in sorted(route_counts.items()):
        print(f"  {route}: {cnt} samples ({cnt / len(samples) * 100:.1f}%)")

    if viz_enabled:
        print(f"\nVisualizations saved to {viz_dir}/ ({viz_count} files)")

    results = {
        "accuracy": accuracy,
        "per_class": per_class_acc,
        "route_breakdown": dict(route_counts),
        "num_samples": len(val_df),
        "checkpoint": checkpoint_dir,
        "samples": samples,
    }

    os.makedirs(cfg.evaluation.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.evaluation.output_dir, "val_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}")

    return results
