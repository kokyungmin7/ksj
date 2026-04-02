import os
import re

SYSTEM_PROMPT = (
    "당신은 재활용품 분류 전문가입니다. "
    "주어진 이미지와 질문을 보고 정확한 답을 선택하세요."
)

_CHOICE_TEXT = (
    "질문: {question}\n\n"
    "선택지:\n"
    "a) {a}\n"
    "b) {b}\n"
    "c) {c}\n"
    "d) {d}\n\n"
    "올바른 답의 알파벳(a, b, c, d 중 하나)만 답하세요."
)


def build_messages(row: dict, image_root: str) -> list[dict]:
    """Build Qwen3-VL chat messages for inference (no assistant turn)."""
    image_path = os.path.abspath(os.path.join(image_root, row["path"]))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": _CHOICE_TEXT.format(**row)},
            ],
        },
    ]


def build_messages_with_crop(row: dict, image_root: str, crop_path: str) -> list[dict]:
    """Build Qwen3-VL messages with original image + DINOv3 attention crop."""
    image_path = os.path.abspath(os.path.join(image_root, row["path"]))
    crop_abs = os.path.abspath(crop_path)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "image", "image": f"file://{crop_abs}"},
                {
                    "type": "text",
                    "text": (
                        "(두 번째 이미지는 첫 번째 이미지에서 "
                        "재활용품이 집중된 영역을 확대한 것입니다.)\n\n"
                        + _CHOICE_TEXT.format(**row)
                    ),
                },
            ],
        },
    ]


def build_training_messages(row: dict, image_root: str) -> list[dict]:
    """Build Qwen3-VL chat messages for SFT training (includes assistant turn)."""
    messages = build_messages(row, image_root)
    messages.append({"role": "assistant", "content": str(row["answer"]).strip().lower()})
    return messages


def extract_answer(text: str) -> str:
    """Extract the first a/b/c/d token from generated text. Falls back to 'a'."""
    text = text.strip().lower()
    match = re.search(r"\b([abcd])\b", text)
    if match:
        return match.group(1)
    if text and text[0] in "abcd":
        return text[0]
    return "a"
