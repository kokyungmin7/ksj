"""VLM-based ReID verification module using Qwen3-VL-8B-Instruct.

Compares two person crops and determines if they are the same individual.
Designed for GCP L4 (24GB) GPU with bfloat16 precision.

HITL integration: set hitl_threshold to automatically queue low-confidence
predictions for human review via HITLCollector.

LoRA integration: set lora_adapter_path to load a PEFT adapter on top of
the base model (e.g., from VLMLoRATrainer output).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

SYSTEM_PROMPT = (
    "You are an expert in person re-identification. "
    "You carefully analyze visual appearance features such as clothing color, "
    "clothing style, accessories, body shape, and other distinguishing characteristics "
    "to determine if two person images show the same individual."
)

USER_PROMPT = (
    "The two images below are person crops from surveillance cameras.\n"
    "Analyze their appearance carefully.\n"
    "Answer in exactly this format:\n"
    "SAME_PERSON: <YES or NO>\n"
    "CONFIDENCE: <0.0 to 1.0>\n"
    "REASONING: <one sentence>"
)


@dataclass
class VerificationResult:
    """Result from VLM-based person verification.

    Attributes:
        is_same: Whether the two crops are the same person.
        confidence: Confidence score in [0.0, 1.0]. 0.5 on parse failure.
        reasoning: One-sentence explanation from the model.
        raw_output: Full model output for debugging.
    """

    is_same: bool
    confidence: float
    reasoning: str
    raw_output: str


def _parse_output(raw: str) -> VerificationResult:
    """Parse structured VLM output into a VerificationResult.

    Args:
        raw: Raw text output from the model.

    Returns:
        Parsed VerificationResult. Falls back to is_same=False, confidence=0.5
        if parsing fails.
    """
    same_match = re.search(r"SAME_PERSON:\s*(YES|NO)", raw, re.IGNORECASE)
    conf_match = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", raw)
    reason_match = re.search(r"REASONING:\s*(.+)", raw)

    if same_match is None:
        return VerificationResult(
            is_same=False,
            confidence=0.5,
            reasoning=raw.strip(),
            raw_output=raw,
        )

    is_same = same_match.group(1).upper() == "YES"
    confidence = float(conf_match.group(1)) if conf_match else 0.5
    confidence = float(np.clip(confidence, 0.0, 1.0))
    reasoning = reason_match.group(1).strip() if reason_match else raw.strip()

    return VerificationResult(
        is_same=is_same,
        confidence=confidence,
        reasoning=reasoning,
        raw_output=raw,
    )


class VLMVerifier:
    """VLM-based person re-identification verifier.

    Uses Qwen3-VL-8B-Instruct to compare two person crops and determine
    if they depict the same individual.

    Args:
        device: Target device string (e.g., "cuda", "cuda:0").
        model_id: HuggingFace model identifier.
        dtype: Model precision dtype.
        hitl_threshold: If set, predictions with confidence below this value
            are automatically logged to the HITL review queue.
        hitl_data_dir: Root directory for HITL data storage.
        lora_adapter_path: Path to a PEFT LoRA adapter directory. If set,
            the adapter is loaded on top of the base model.
    """

    def __init__(
        self,
        device: str,
        model_id: str,
        dtype: torch.dtype,
        hitl_threshold: float | None = None,
        hitl_data_dir: str = "data/hitl",
        lora_adapter_path: str | None = None,
    ) -> None:
        self.device = device
        self.model_id = model_id
        self.dtype = dtype

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

        if lora_adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
            self.model.eval()

        self._hitl_threshold = hitl_threshold
        if hitl_threshold is not None:
            from src.models.hitl_collector import HITLCollector

            self._hitl: HITLCollector | None = HITLCollector(hitl_data_dir)
        else:
            self._hitl = None

    def verify(
        self,
        bgr_a: np.ndarray,
        bgr_b: np.ndarray,
        max_new_tokens: int = 256,
    ) -> VerificationResult:
        """Verify whether two BGR person crops are the same individual.

        Args:
            bgr_a: First person crop in BGR format (H, W, 3).
            bgr_b: Second person crop in BGR format (H, W, 3).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            VerificationResult with is_same, confidence, reasoning, raw_output.
        """
        pil_a = Image.fromarray(bgr_a[:, :, ::-1])
        pil_b = Image.fromarray(bgr_b[:, :, ::-1])

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_a},
                    {"type": "image", "image": pil_b},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        generated_ids = output_ids[:, input_len:]
        raw = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        result = _parse_output(raw)

        if self._hitl is not None and result.confidence < self._hitl_threshold:  # type: ignore[operator]
            self._hitl.log(bgr_a, bgr_b, result)

        return result

    def verify_batch(
        self,
        bgr_pairs: list[tuple[np.ndarray, np.ndarray]],
        max_new_tokens: int = 256,
    ) -> list[VerificationResult]:
        """Verify multiple BGR person crop pairs in a single forward pass.

        Processes all pairs together as a padded batch, which is significantly
        faster than calling verify() in a loop on GPU.

        Args:
            bgr_pairs: List of (bgr_a, bgr_b) tuples, each crop in BGR format.
            max_new_tokens: Maximum tokens to generate per item.

        Returns:
            List of VerificationResult, one per input pair (same order).
        """
        if not bgr_pairs:
            return []

        # Tokenize each pair individually (same path as verify()) then left-pad into a batch.
        # Using tokenize=True per item avoids PIL-serialization issues with tokenize=False.
        per_item_inputs: list[dict] = []
        for bgr_a, bgr_b in bgr_pairs:
            pil_a = Image.fromarray(bgr_a[:, :, ::-1])
            pil_b = Image.fromarray(bgr_b[:, :, ::-1])
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_a},
                        {"type": "image", "image": pil_b},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ]
            item = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            per_item_inputs.append(item)

        # Left-pad input_ids and attention_mask so generation starts at the same position.
        max_len = max(x["input_ids"].shape[1] for x in per_item_inputs)
        pad_id = self.processor.tokenizer.pad_token_id or 0

        padded_ids = torch.full(
            (len(per_item_inputs), max_len), pad_id, dtype=torch.long
        )
        padded_mask = torch.zeros(len(per_item_inputs), max_len, dtype=torch.long)

        for i, item in enumerate(per_item_inputs):
            seq_len = item["input_ids"].shape[1]
            padded_ids[i, max_len - seq_len:] = item["input_ids"][0]
            padded_mask[i, max_len - seq_len:] = item["attention_mask"][0]

        inputs: dict = {"input_ids": padded_ids.to(self.device),
                        "attention_mask": padded_mask.to(self.device)}

        # pixel_values and image_grid_thw must be concatenated along the batch dim.
        if "pixel_values" in per_item_inputs[0]:
            inputs["pixel_values"] = torch.cat(
                [x["pixel_values"] for x in per_item_inputs], dim=0
            ).to(self.device)
        if "image_grid_thw" in per_item_inputs[0]:
            inputs["image_grid_thw"] = torch.cat(
                [x["image_grid_thw"] for x in per_item_inputs], dim=0
            ).to(self.device)

        input_len = max_len

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Decode only the generated portion for each item
        generated_ids = output_ids[:, input_len:]
        raws = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        results = []
        for i, (raw, (bgr_a, bgr_b)) in enumerate(zip(raws, bgr_pairs)):
            result = _parse_output(raw)
            if self._hitl is not None and result.confidence < self._hitl_threshold:  # type: ignore[operator]
                self._hitl.log(bgr_a, bgr_b, result)
            results.append(result)

        return results


def load_vlm_verifier(
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    hitl_threshold: float | None = None,
    hitl_data_dir: str = "data/hitl",
    lora_adapter_path: str | None = None,
) -> VLMVerifier:
    """Load a VLMVerifier with the specified configuration.

    Args:
        model_id: HuggingFace model identifier for the VLM.
        device: Target device. None auto-detects CUDA.
        dtype: Model precision. Defaults to bfloat16.
        hitl_threshold: Queue predictions below this confidence for human review.
        hitl_data_dir: Root directory for HITL data storage.
        lora_adapter_path: Path to PEFT LoRA adapter. None uses base model.

    Returns:
        Loaded VLMVerifier ready for inference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return VLMVerifier(
        device=device,
        model_id=model_id,
        dtype=dtype,
        hitl_threshold=hitl_threshold,
        hitl_data_dir=hitl_data_dir,
        lora_adapter_path=lora_adapter_path,
    )
