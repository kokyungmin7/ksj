#!/usr/bin/env python3
"""
Qwen3.5(AutoModelForImageTextToText)에 프롬프트를 넣어 생성 텍스트를 확인하는 최소 테스트.

사용 예 (GPU 추론 환경):
  uv run test_qwen_chat.py --prompt "1+1은?"
  uv run test_qwen_chat.py --prompt "이 이미지에 뭐가 보이나요?" --image path/to.jpg
  uv run test_qwen_chat.py --config configs/default.yaml --checkpoint outputs/qwen35-lora/run_xxx

--config 없으면 Qwen/Qwen3.5-9B 등 기본값을 사용합니다.
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import torch
from PIL import Image
from main import load_config
from models.qwen import load_base_qwen, load_finetuned_qwen


def _default_cfg() -> SimpleNamespace:
    """configs/default.yaml 이 없을 때 쓰는 기본 설정."""
    return SimpleNamespace(
        model=SimpleNamespace(
            qwen_name="Qwen/Qwen3.5-9B",
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
            device="cuda",
            enable_thinking=False,
            min_pixels=200704,
            max_pixels=1003520,
        ),
        training=SimpleNamespace(max_seq_length=2048),
        evaluation=SimpleNamespace(max_new_tokens=256),
    )


def _resolve_cfg(config_path: str | None) -> SimpleNamespace:
    if config_path and os.path.isfile(config_path):
        return load_config(config_path)
    return _default_cfg()


def _build_messages(
    prompt: str,
    image_path: str | None,
    system: str | None,
) -> list[dict]:
    user_parts: list[dict] = []
    if image_path:
        user_parts.append({"type": "image", "image": os.path.abspath(image_path)})
    user_parts.append({"type": "text", "text": prompt})

    messages: list[dict] = []
    if system:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system}]},
        )
    messages.append({"role": "user", "content": user_parts})
    return messages


@torch.inference_mode()
def run_generate(
    model,
    processor,
    messages: list[dict],
    cfg: SimpleNamespace,
    max_new_tokens: int,
) -> str:
    enable_thinking = getattr(cfg.model, "enable_thinking", False)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    images = []
    for msg in messages:
        if not isinstance(msg.get("content"), list):
            continue
        for part in msg["content"]:
            if part.get("type") == "image":
                p = str(part["image"]).replace("file://", "")
                images.append(Image.open(p).convert("RGB"))

    max_len = getattr(
        getattr(cfg, "training", SimpleNamespace(max_seq_length=2048)),
        "max_seq_length",
        2048,
    )
    batch = processor(
        text=[text],
        images=images if images else None,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    batch = {k: v.to(model.device) for k, v in batch.items()}

    output_ids = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    prompt_len = batch["input_ids"].shape[1]
    new_ids = output_ids[:, prompt_len:]
    return processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 프롬프트 생성 테스트")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML 설정 경로 (없으면 내장 기본값)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="LoRA 등 미세조정 체크포인트 디렉터리 (없으면 베이스 모델만 로드)",
    )
    parser.add_argument(
        "--prompt",
        default="간단히 자기소개 해 주세요.",
        help="사용자 프롬프트",
    )
    parser.add_argument("--image", default=None, help="선택: 이미지 파일 경로")
    parser.add_argument("--system", default=None, help="선택: 시스템 메시지")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="생성 max_new_tokens (VQA용 config의 5보다 크게 두는 것을 권장)",
    )
    args = parser.parse_args()

    cfg = _resolve_cfg(args.config)
    if args.checkpoint:
        model, processor = load_finetuned_qwen(args.checkpoint, cfg)
    else:
        model, processor = load_base_qwen(cfg)

    messages = _build_messages(args.prompt, args.image, args.system)
    out = run_generate(model, processor, messages, cfg, args.max_new_tokens)
    print(out.strip())


if __name__ == "__main__":
    main()
