from __future__ import annotations

import torch
from PIL import Image
from types import SimpleNamespace
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import BitsAndBytesConfig
from peft import PeftModel

from utils.prompt import build_messages, build_messages_with_crop, extract_answer


def _get_enable_thinking(cfg: SimpleNamespace) -> bool:
    return getattr(cfg.model, "enable_thinking", False)


def _extract_images(messages: list[dict]) -> list[Image.Image]:
    """Extract PIL images from chat messages containing image paths."""
    images = []
    for msg in messages:
        if not isinstance(msg.get("content"), list):
            continue
        for part in msg["content"]:
            if part.get("type") == "image":
                path = str(part["image"]).replace("file://", "")
                images.append(Image.open(path).convert("RGB"))
    return images


def _load_processor(name: str, cfg: SimpleNamespace) -> AutoProcessor:
    """Load AutoProcessor with optional min/max pixel budget from config."""
    kwargs: dict = {}
    if hasattr(cfg.model, "min_pixels"):
        kwargs["min_pixels"] = cfg.model.min_pixels
    if hasattr(cfg.model, "max_pixels"):
        kwargs["max_pixels"] = cfg.model.max_pixels
    return AutoProcessor.from_pretrained(name, **kwargs)


def load_qwen(cfg: SimpleNamespace) -> tuple:
    """Load Qwen3.5-9B with 4-bit NF4 quantization for QLoRA training."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model.qwen_name,
            quantization_config=bnb_config,
            attn_implementation=cfg.model.attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model.qwen_name,
            quantization_config=bnb_config,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.config.use_cache = False
    processor = _load_processor(cfg.model.qwen_name, cfg)
    return model, processor


def load_finetuned_qwen(checkpoint_dir: str, cfg: SimpleNamespace) -> tuple:
    """Load a fine-tuned Qwen3.5 checkpoint with LoRA weights merged."""
    try:
        base = AutoModelForImageTextToText.from_pretrained(
            cfg.model.qwen_name,
            attn_implementation=cfg.model.attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except Exception:
        base = AutoModelForImageTextToText.from_pretrained(
            cfg.model.qwen_name,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model = model.merge_and_unload()
    model.eval()

    try:
        processor = _load_processor(checkpoint_dir, cfg)
    except Exception:
        processor = _load_processor(cfg.model.qwen_name, cfg)

    return model, processor


def load_base_qwen(cfg: SimpleNamespace) -> tuple:
    """Load Qwen3.5 base model in bfloat16 for zero-shot evaluation."""
    device = getattr(cfg.model, "device", "cuda")
    dtype = torch.bfloat16 if cfg.model.torch_dtype == "bfloat16" else torch.float16
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model.qwen_name,
            attn_implementation=cfg.model.attn_implementation,
            torch_dtype=dtype,
            device_map=device,
        )
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model.qwen_name,
            attn_implementation="eager",
            torch_dtype=dtype,
            device_map=device,
        )

    model.eval()
    processor = _load_processor(cfg.model.qwen_name, cfg)
    return model, processor


@torch.inference_mode()
def qwen_predict(
    model,
    processor,
    row: dict,
    image_root: str,
    cfg: SimpleNamespace,
) -> str:
    """Run inference on a single row and return the predicted answer letter."""
    messages = build_messages(row, image_root)
    images = _extract_images(messages)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=_get_enable_thinking(cfg),
    )
    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=cfg.evaluation.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return extract_answer(text)


@torch.inference_mode()
def qwen_predict_with_crop(
    model,
    processor,
    row: dict,
    image_root: str,
    crop_path: str,
    cfg: SimpleNamespace,
) -> str:
    """Run inference using both original image and GroundingDINO crop."""
    messages = build_messages_with_crop(row, image_root, crop_path)
    images = _extract_images(messages)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=_get_enable_thinking(cfg),
    )
    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=cfg.evaluation.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return extract_answer(text)
