from __future__ import annotations

import random

import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from types import SimpleNamespace

from utils.prompt import build_training_messages


def load_and_split(cfg: SimpleNamespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv and split into train/val with stratification.

    If cfg.data.subset is set, only that many samples are used total
    (train/val ratio is preserved via stratified sampling).
    """
    df = pd.read_csv(cfg.data.csv_path)

    if cfg.data.subset:
        df, _ = train_test_split(
            df,
            train_size=int(cfg.data.subset),
            stratify=df["answer"],
            random_state=cfg.data.seed,
        )

    train_df, val_df = train_test_split(
        df,
        test_size=cfg.data.val_split,
        stratify=df["answer"],
        random_state=cfg.data.seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def shuffle_choices(row: dict) -> dict:
    """Shuffle a/b/c/d choices and remap the answer label accordingly.

    Prevents position bias by randomizing choice order each time.
    """
    keys = ["a", "b", "c", "d"]
    original_answer = str(row["answer"]).strip().lower()

    shuffled = keys.copy()
    random.shuffle(shuffled)

    new_row = row.copy()
    for new_key, old_key in zip(keys, shuffled):
        new_row[new_key] = row[old_key]

    new_row["answer"] = keys[shuffled.index(original_answer)]
    return new_row


class KoreanRecyclingVQADataset(Dataset):
    """Returns {"messages": [...]} dicts for use with VQADataCollator."""

    def __init__(
        self, df: pd.DataFrame, image_root: str, shuffle: bool = False
    ) -> None:
        self.df = df
        self.image_root = image_root
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx].to_dict()
        if self.shuffle:
            row = shuffle_choices(row)
        return {"messages": build_training_messages(row, self.image_root)}


class VQADataCollator:
    """Tokenizes a batch of {"messages": [...]} items for Qwen3.5 SFT.

    - Applies the chat template to produce input_ids.
    - Masks all tokens before the assistant turn with -100 so the loss
      is computed only on the single answer token.
    """

    def __init__(self, processor, max_seq_length: int = 1024, enable_thinking: bool = False) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.enable_thinking = enable_thinking
        self._assistant_token_ids: list[int] | None = None

    def _get_assistant_token_ids(self) -> list[int]:
        if self._assistant_token_ids is None:
            self._assistant_token_ids = self.processor.tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
        return self._assistant_token_ids

    def __call__(self, features: list[dict]) -> dict:
        texts = [
            self.processor.apply_chat_template(
                f["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=self.enable_thinking,
            )
            for f in features
        ]

        images = []
        for f in features:
            for msg in f["messages"]:
                if not isinstance(msg["content"], list):
                    continue
                for part in msg["content"]:
                    if part.get("type") == "image":
                        from PIL import Image
                        path = str(part["image"]).replace("file://", "")
                        images.append(Image.open(path).convert("RGB"))

        batch = self.processor(
            text=texts,
            images=images if images else None,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        assistant_ids = self._get_assistant_token_ids()
        for i, seq in enumerate(batch["input_ids"].tolist()):
            pos = _find_last_subseq(seq, assistant_ids)
            if pos != -1:
                labels[i, : pos + len(assistant_ids)] = -100

        batch["labels"] = labels
        return batch


def _find_last_subseq(sequence: list[int], subseq: list[int]) -> int:
    """Return the start index of the last occurrence of subseq in sequence, or -1."""
    n, m = len(sequence), len(subseq)
    for i in range(n - m, -1, -1):
        if sequence[i : i + m] == subseq:
            return i
    return -1
