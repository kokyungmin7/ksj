import argparse
import yaml
from types import SimpleNamespace


def load_config(path: str) -> SimpleNamespace:
    """Recursively convert a YAML dict to SimpleNamespace for dot-access."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def _to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
        return obj

    return _to_ns(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Korean Recycling VQA — Qwen3-VL + DINOv3 pipeline"
    )
    parser.add_argument(
        "mode",
        choices=["train", "evaluate"],
        help="train: QLoRA fine-tuning | evaluate: val set accuracy",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="[evaluate] Path to fine-tuned checkpoint dir. "
             "Omit for zero-shot evaluation with the base model.",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        dest="reference_dir",
        help="[evaluate] Path to reference gallery directory for DINOv3 counting.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "train":
        from training.trainer import run_training
        run_training(cfg)

    elif args.mode == "evaluate":
        from evaluation.evaluator import run_evaluation
        run_evaluation(cfg, checkpoint_dir=args.checkpoint, reference_dir=args.reference_dir)


if __name__ == "__main__":
    main()
