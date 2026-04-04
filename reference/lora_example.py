# 회로 분석을 위한 신속한 설계

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)


"""
이 단계에서는 시각-언어 학습을 위한 프롬프트, 목표 및 이미지를 준비하는 데 필요한 경량 유틸리티를 정의합니다. 단일 파이프라인을 사용하여 TASK변수(구성 요소, YAML, JSON 또는 도식 재구성)를 전환함으로써 여러 작업을 수행할 수 있습니다.

기본 안전 제한이 적용되어 목표 길이와 이미지 크기를 제어함으로써 학습을 안정적이고 메모리 효율적으로 유지할 수 있습니다.
"""
TASK = "components"  # "components" | "yaml" | "json" | "schematic"
MAX_TARGET_CHARS = 5000  # safety cap for long targets like schematic/json
MAX_IMAGE_SIDE = 1024  # bigger side
MAX_IMAGE_PIXELS = 1024 * 1024  # safety cap (1.0 MP). raise to 1.5MP if stable


# 프롬프트 생성


def build_prompt(example):
    # Use dataset fields to give better context (name/type are helpful)
    name = example.get("name") or "Unknown project"
    ftype = example.get("type") or "unknown format"

    if TASK == "components":
        return (
            f"Project: {name}\nFormat: {ftype}\n"
            "From the schematic image, extract all component labels and identifiers exactly as shown "
            "(part numbers, values, footprints, net labels like +5V/GND).\n"
            "Output only a comma-separated list. Do not generalize or add extra text."
        )

    if TASK == "yaml":
        return (
            f"Project: {name}\nFormat: {ftype}\n"
            "From the schematic image, produce YAML metadata for the design.\n"
            "Return valid YAML only. No markdown, no explanations."
        )

    if TASK == "json":
        return (
            f"Project: {name}\nFormat: {ftype}\n"
            "From the schematic image, produce a JSON representation of the schematic structure.\n"
            "Return valid JSON only. No markdown, no explanations."
        )

    if TASK == "schematic":
        return (
            f"Project: {name}\nFormat: {ftype}\n"
            "From the schematic image, reconstruct the raw KiCad schematic content.\n"
            "Return only the schematic text. No markdown, no explanations."
        )

    raise ValueError("Unknown TASK")


def build_target(example):
    if TASK == "components":
        comps = example.get("components_used") or []
        return ", ".join(comps)

    if TASK == "yaml":
        return (example.get("yaml") or "").strip()

    if TASK == "json":
        return (example.get("json") or "").strip()

    if TASK == "schematic":
        return (example.get("schematic") or "").strip()

    raise ValueError("Unknown TASK")


"""
이 clamp_text()함수는 대상에 엄격한 문자 제한을 적용합니다. 이를 통해 크기가 지나치게 큰 JSON, YAML 또는 스키매틱 파일로 인해 학습 중에 메모리 문제가 발생하는 것을 방지합니다.
"""


def clamp_text(s: str, max_chars: int = MAX_TARGET_CHARS) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[:max_chars].rstrip()


"""
이 _resize_pil()함수는 처리 전에 개략도 이미지를 정규화하고 크기를 조정합니다. 최대 변의 길이와 최대 총 픽셀 수를 모두 제한하여 시각적 세부 정보를 유지하면서 예측 가능한 GPU 메모리 사용량을 보장합니다.
"""


def _resize_pil(
    pil: Image.Image, max_side: int = MAX_IMAGE_SIDE, max_pixels: int = MAX_IMAGE_PIXELS
) -> Image.Image:
    pil = pil.convert("RGB")
    w, h = pil.size

    # Scale down if max side too large
    scale_side = min(1.0, max_side / float(max(w, h)))

    # Scale down if too many pixels (area cap)
    scale_area = (max_pixels / float(w * h)) ** 0.5 if (w * h) > max_pixels else 1.0

    scale = min(scale_side, scale_area)

    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        pil = pil.resize((nw, nh), resample=Image.BICUBIC)

    return pil


def to_messages(example):
    prompt = build_prompt(example)
    target = clamp_text(build_target(example))

    example["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target}],
        },
    ]
    return example


# Start small (increase later)
train_ds = (
    ds_clean.shuffle(seed=42).select(range(min(800, len(ds_clean)))).map(to_messages)
)
train_ds = train_ds.cast_column("image", HFImage(decode=True))


# 학습

import torch


def run_inference(model_, example, max_new_tokens=256):
    prompt = build_prompt(example)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": _resize_pil(example["image"])},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model_.device)

    with torch.inference_mode():
        out = model_.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    gen = out[0][inputs["input_ids"].shape[1] :]
    return processor.decode(gen, skip_special_tokens=True)


baseline_ex = train_ds.shuffle(seed=120).select(range(1))[0]

# 훈련을 위한 비전-언어 데이터 수집기 ​​구축
"""
이 섹션에서는 멀티모달 채팅 스타일 배치 데이터를 학습에 사용할 수 있도록 준비하는 사용자 지정 데이터 수집기를 정의합니다. 이 수집기는 텍스트와 이미지를 함께 인코딩하여 각 예제를 모델에서 바로 사용할 수 있는 텐서로 변환하며, 손실은 음성 비서의 응답에 대해서만 계산되도록 합니다.

콜레이터는 채팅 텍스트의 두 가지 버전을 생성합니다. 하나는 입력 인코딩을 위한 전체 버전(프롬프트 및 대상)이고, 다른 하나는 프롬프트 토큰 길이를 계산하기 위한 프롬프트만 포함된 버전입니다. 이러한 길이를 사용하여 레이블에서 모든 프롬프트 및 패딩 토큰이 마스킹되므로 어시스턴트 출력만 손실에 영향을 미칩니다. 이미지 크기는 일관되게 조정되며, 메모리 제어를 위해 고정된 최대 시퀀스 길이가 적용됩니다.
"""

from typing import Any, Dict, List

MAX_LEN = 1500


def collate_fn(batch: List[Dict[str, Any]]):
    # 1) Build full chat text (includes assistant answer)
    full_texts = [
        processor.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        for ex in batch
    ]

    # 2) Build prompt-only text (up to user turn; generation prompt on)
    prompt_texts = [
        processor.apply_chat_template(
            ex["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in batch
    ]

    # 3) Images
    images = [_resize_pil(ex["image"]) for ex in batch]

    # 4) Tokenize full inputs ONCE (text + images)
    enc = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )

    input_ids = enc["input_ids"]
    pad_id = processor.tokenizer.pad_token_id

    # 5) Compute prompt lengths with TEXT-ONLY tokenization (much cheaper than text+images)
    prompt_ids = processor.tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        add_special_tokens=False,  # chat template already includes special tokens
    )["input_ids"]

    # Count non-pad tokens in prompt
    prompt_lens = (prompt_ids != pad_id).sum(dim=1)

    # 6) Labels: copy + mask prompt tokens + mask padding
    labels = input_ids.clone()
    bs, seqlen = labels.shape

    for i in range(bs):
        pl = int(prompt_lens[i].item())
        pl = min(pl, seqlen)
        labels[i, :pl] = -100

    # Mask padding positions too
    labels[labels == pad_id] = -100

    # If your processor produces pixel_values / image_grid_thw, keep them
    enc["labels"] = labels
    return enc


from peft import LoraConfig, TaskType

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)


from trl import SFTConfig

args = SFTConfig(
    output_dir=f"qwen3vl-open-schematics-{TASK}-lora",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    learning_rate=1e-4,
    warmup_steps=10,
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=True,
    fp16=False,
    lr_scheduler_type="cosine",
    logging_steps=10,
    report_to="none",
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collate_fn,
    peft_config=lora,
)


trainer.train()
