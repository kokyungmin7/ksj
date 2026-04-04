#!/usr/bin/env python3
"""test.csv + test/ 이미지로 추론 후 sample_submission.csv 형식(id,answer)으로 저장.

평가 파이프라인(`Predictor`)과 동일하게 GroundingDINO(설정상 켜져 있을 때) + Qwen을 사용합니다.

예시 (GPU 추론 서버):
  uv run infer_submission.py \\
    --config configs/default.yaml \\
    --checkpoint /home/kokyungmin/ksj/outputs/qwen35-lora/run_20260403_175736/checkpoint-1270 \\
    --test-csv test.csv \\
    --image-root /home/kokyungmin/ksj \\
    --output submission.csv
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import pandas as pd
from tqdm import tqdm

from main import load_config
from models.grounding_dino import load_grounding_dino
from models.qwen import load_finetuned_qwen
from pipeline.predictor import Predictor


def _row_to_dict(row: pd.Series) -> dict:
    d = row.to_dict()
    text_keys = ("a", "b", "c", "d", "question", "path", "id")
    for k, v in list(d.items()):
        if pd.isna(v):
            d[k] = ""
        elif k in text_keys:
            d[k] = str(v)
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="test 세트 제출용 CSV 생성")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 설정")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="LoRA 체크포인트 디렉터리 (예: .../checkpoint-1270)",
    )
    parser.add_argument("--test-csv", default="test.csv", help="테스트 메타 CSV 경로")
    parser.add_argument(
        "--image-root",
        default=".",
        help="test.csv의 path 컬럼 기준 상대 경로 루트 (보통 프로젝트 루트)",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="출력 파일 (컬럼: id,answer — sample_submission.csv와 동일)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="배치 크기 (미지정 시 config evaluation.batch_size)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.data.image_root = os.path.abspath(args.image_root)
    if args.batch_size is not None:
        ev = getattr(cfg, "evaluation", None)
        if ev is None:
            cfg.evaluation = SimpleNamespace(batch_size=args.batch_size)
        else:
            ev.batch_size = args.batch_size

    required_cols = {"path", "question", "a", "b", "c", "d"}
    df = pd.read_csv(args.test_csv)
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"test.csv에 필요한 컬럼이 없습니다: {sorted(missing)}")

    if "id" not in df.columns:
        df = df.copy()
        df["id"] = df["path"].map(lambda p: os.path.basename(str(p)))

    print(f"Loading fine-tuned Qwen from {args.checkpoint} ...")
    qwen_model, qwen_processor = load_finetuned_qwen(args.checkpoint, cfg)

    dino_enabled = getattr(cfg.dino, "enabled", True)
    if dino_enabled:
        print("Loading GroundingDINO ...")
        dino_model, dino_processor = load_grounding_dino(cfg)
    else:
        print("GroundingDINO disabled (config).")
        dino_model, dino_processor = None, None

    predictor = Predictor(
        qwen_model,
        qwen_processor,
        dino_model,
        dino_processor,
        cfg,
    )

    batch_size = getattr(cfg.evaluation, "batch_size", 8)
    all_rows = [_row_to_dict(df.iloc[i]) for i in range(len(df))]
    ids = [str(all_rows[i]["id"]) for i in range(len(all_rows))]
    preds: list[str] = []

    n_batch = (len(all_rows) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(all_rows), batch_size), desc="Inference", total=n_batch):
        batch = all_rows[start : start + batch_size]
        batch_out = predictor.predict_batch_with_trace(batch)
        for pred, _trace in batch_out:
            preds.append(str(pred).strip().lower()[:1] if pred else "a")

    out_df = pd.DataFrame({"id": ids, "answer": preds})
    out_path = os.path.abspath(args.output)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
