from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

from data.dataset import KoreanRecyclingVQADataset, VQADataCollator, load_and_split
from models.qwen import load_qwen
from utils.prompt import build_messages, extract_answer


class F1EvalCallback(TrainerCallback):
    """Epoch 종료 시 val subset에서 macro F1 / accuracy를 계산해 TensorBoard에 기록.

    배치 추론을 사용하여 단건 추론 대비 속도를 크게 개선.
    """

    def __init__(self, val_df, processor, cfg: SimpleNamespace) -> None:
        self.val_df = val_df
        self.processor = processor
        self.cfg = cfg
        self.batch_size = getattr(cfg.training, "f1_eval_batch_size", 4)
        self.writer = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=args.logging_dir)

    @torch.inference_mode()
    def _batch_predict(self, model, rows: list[dict]) -> list[str]:
        """Run batched generation on a list of row dicts."""
        all_messages = [
            build_messages(r, self.cfg.data.image_root) for r in rows
        ]

        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True,
            )
            for m in all_messages
        ]

        images = []
        for m_list in all_messages:
            for msg in m_list:
                if not isinstance(msg["content"], list):
                    continue
                for part in msg["content"]:
                    if part.get("type") == "image":
                        path = str(part["image"]).replace("file://", "")
                        images.append(Image.open(path).convert("RGB"))

        self.processor.tokenizer.padding_side = "left"
        batch = self.processor(
            text=texts,
            images=images if images else None,
            padding=True,
            truncation=True,
            max_length=self.cfg.training.max_seq_length,
            return_tensors="pt",
        )
        batch = {k: v.to(model.device) for k, v in batch.items()}

        output_ids = model.generate(
            **batch,
            max_new_tokens=self.cfg.evaluation.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        prompt_len = batch["input_ids"].shape[1]
        new_ids = output_ids[:, prompt_len:]
        decoded = self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        self.processor.tokenizer.padding_side = "right"

        return [extract_answer(t) for t in decoded]

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        max_samples = getattr(self.cfg.training, "f1_eval_samples", 200)
        df = self.val_df.sample(min(max_samples, len(self.val_df)), random_state=42)

        prev_use_cache = model.config.use_cache
        model.config.use_cache = True
        model.eval()

        preds, gts = [], []
        rows = [row.to_dict() for _, row in df.iterrows()]

        for i in range(0, len(rows), self.batch_size):
            batch_rows = rows[i : i + self.batch_size]
            batch_preds = self._batch_predict(model, batch_rows)
            preds.extend(batch_preds)
            gts.extend(
                str(r["answer"]).strip().lower() for r in batch_rows
            )

        model.config.use_cache = prev_use_cache
        model.train()

        labels = ["a", "b", "c", "d"]
        f1 = f1_score(gts, preds, labels=labels, average="macro", zero_division=0)
        acc = sum(p == g for p, g in zip(preds, gts)) / len(gts)
        epoch = int(state.epoch)

        if self.writer:
            self.writer.add_scalar("eval/f1_macro", f1, epoch)
            self.writer.add_scalar("eval/accuracy", acc, epoch)

        print(
            f"\n[Epoch {epoch}] F1 macro: {f1:.4f}  |  Accuracy: {acc:.4f}"
            f"  (n={len(gts)})"
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self.writer:
            self.writer.close()


def run_training(cfg: SimpleNamespace) -> None:
    """Fine-tune Qwen3-VL with QLoRA on the training split."""
    model, processor = load_qwen(cfg)

    # Enable grad for frozen base model (required for gradient checkpointing + PEFT)
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(cfg.lora.target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_df, val_df = load_and_split(cfg)
    print(f"Train: {len(train_df)} samples | Val: {len(val_df)} samples")

    do_shuffle = getattr(cfg.training, "shuffle_choices", True)
    train_dataset = KoreanRecyclingVQADataset(train_df, cfg.data.image_root, shuffle=do_shuffle)
    val_dataset = KoreanRecyclingVQADataset(val_df, cfg.data.image_root, shuffle=False)
    collator = VQADataCollator(processor, cfg.training.max_seq_length)

    training_args = SFTConfig(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=cfg.training.bf16,
        fp16=False,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=cfg.training.remove_unused_columns,
        dataloader_num_workers=0,
        report_to="tensorboard",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=cfg.training.max_seq_length,
    )

    patience = getattr(cfg.training, "early_stopping_patience", 2)
    f1_callback = F1EvalCallback(val_df, processor, cfg)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[
            f1_callback,
            EarlyStoppingCallback(early_stopping_patience=patience),
        ],
    )

    trainer.train()
    trainer.save_model(cfg.training.output_dir)
    processor.save_pretrained(cfg.training.output_dir)
    print(f"Saved to {cfg.training.output_dir}")
