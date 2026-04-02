import argparse
import os
import yaml
from types import SimpleNamespace


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. override values take precedence."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str) -> SimpleNamespace:
    """Load YAML config and deep-merge configs/local.yaml if it exists.

    local.yaml only needs to contain the keys you want to override.
    All other values are inherited from the base config.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Auto-detect local.yaml next to the given config file
    local_path = os.path.join(os.path.dirname(path), "local.yaml")
    if os.path.exists(local_path):
        with open(local_path, encoding="utf-8") as f:
            local_raw = yaml.safe_load(f) or {}
        raw = _deep_merge(raw, local_raw)
        print(f"[config] Loaded local overrides from {local_path}")

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
