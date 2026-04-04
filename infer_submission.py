#!/usr/bin/env python3
"""test.csv + 이미지로 Qwen만 배치 추론 후 sample_submission.csv 형식(id,answer)으로 저장.

GroundingDINO는 로드하지 않습니다. `dino.enabled` 설정과 무관하게 원본 이미지 1장 + 4지선다
프롬프트로 `qwen_batch_predict`만 사용합니다 (`evaluate`에서 DINO 끈 경우와 동일한 추론 경로).

예시 (GPU 추론 서버):
  uv run infer_submission.py \\
    --config configs/default.yaml \\
    --checkpoint /path/to/checkpoint-1270 \\
    --test-csv /path/to/test.csv \\
    --image-root /path/to/data_root \\
    --output submission.csv
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import pandas as pd
from tqdm import tqdm

from main import load_config
from models.qwen import load_finetuned_qwen, qwen_batch_predict


def _row_to_dict(row: pd.Series) -> dict:
    d = row.to_dict()
    text_keys = ("a", "b", "c", "d", "question", "path", "id")
    for k, v in list(d.items()):
        if pd.isna(v):
            d[k] = ""
        elif k in text_keys:
            d[k] = str(v)
    return d


def _eval_batch_size(cfg) -> int:
    ev = getattr(cfg, "evaluation", None)
    raw = getattr(ev, "batch_size", 8) if ev is not None else 8
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 8


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen 전용 — test 제출용 CSV 생성 (GroundingDINO 미사용)",
    )
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
        help="test.csv의 path 컬럼 기준 상대 경로 루트",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="출력 (컬럼: id,answer)",
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
            cfg.evaluation = SimpleNamespace(
                batch_size=int(args.batch_size),
                max_new_tokens=5,
            )
        else:
            ev.batch_size = int(args.batch_size)

    batch_size = _eval_batch_size(cfg)
    if getattr(cfg.dino, "enabled", False):
        print(
            "[infer_submission] 참고: config의 dino.enabled가 true여도 "
            "이 스크립트는 Qwen 단일 경로만 사용합니다.",
        )

    required_cols = {"path", "question", "a", "b", "c", "d"}
    df = pd.read_csv(args.test_csv)
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"test.csv에 필요한 컬럼이 없습니다: {sorted(missing)}")

    if "id" not in df.columns:
        df = df.copy()
        df["id"] = df["path"].map(lambda p: os.path.basename(str(p)))

    print(f"Loading fine-tuned Qwen from {args.checkpoint} (Qwen-only, batch_size={batch_size}) ...")
    model, processor = load_finetuned_qwen(args.checkpoint, cfg)

    all_rows = [_row_to_dict(row) for _, row in df.iterrows()]
    ids = [str(r["id"]) for r in all_rows]
    preds: list[str] = []

    n_rows = len(all_rows)
    image_root = cfg.data.image_root
    for start in tqdm(
        range(0, n_rows, batch_size),
        desc="Qwen inference",
        total=(n_rows + batch_size - 1) // batch_size if n_rows else 0,
    ):
        batch = all_rows[start : start + batch_size]
        letters = qwen_batch_predict(model, processor, batch, image_root, cfg)
        for letter in letters:
            c = str(letter).strip().lower()[:1] if letter else "a"
            preds.append(c if c in "abcd" else "a")

    out_df = pd.DataFrame({"id": ids, "answer": preds})
    out_path = os.path.abspath(args.output)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
