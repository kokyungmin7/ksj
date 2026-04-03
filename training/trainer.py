from __future__ import annotations

import os
from datetime import datetime
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
    """Epoch 종료 시 val subset에서 macro F1 / accuracy를 계산하고,
    오답을 상세 기록하여 시각화까지 수행하는 콜백.

    학습 종료 시 전체 epoch에 걸친 종합 요약(학습 곡선, 혼동 행렬,
    오답 패턴 분석)을 자동 생성한다.
    """

    def __init__(
        self, val_df, processor, cfg: SimpleNamespace, run_dir: str,
    ) -> None:
        self.val_df = val_df
        self.processor = processor
        self.cfg = cfg
        self.batch_size = getattr(cfg.training, "f1_eval_batch_size", 4)
        self.run_dir = run_dir
        self.analysis_dir = os.path.join(run_dir, "analysis")
        self.writer = None
        self.epoch_records: list[dict] = []

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
        enable_thinking = getattr(self.cfg.model, "enable_thinking", False)
        all_messages = [
            build_messages(r, self.cfg.data.image_root) for r in rows
        ]

        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
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

        preds, gts, all_results = [], [], []
        rows = [row.to_dict() for _, row in df.iterrows()]

        for i in range(0, len(rows), self.batch_size):
            batch_rows = rows[i : i + self.batch_size]
            batch_preds = self._batch_predict(model, batch_rows)
            preds.extend(batch_preds)
            for r, p in zip(batch_rows, batch_preds):
                gt = str(r["answer"]).strip().lower()
                gts.append(gt)
                all_results.append({
                    "id": str(r.get("id", "")),
                    "question": str(r.get("question", "")),
                    "a": str(r.get("a", "")),
                    "b": str(r.get("b", "")),
                    "c": str(r.get("c", "")),
                    "d": str(r.get("d", "")),
                    "path": str(r.get("path", "")),
                    "ground_truth": gt,
                    "predicted": p,
                    "correct": p == gt,
                })

        model.config.use_cache = prev_use_cache
        model.train()

        labels = ["a", "b", "c", "d"]
        f1 = f1_score(gts, preds, labels=labels, average="macro", zero_division=0)
        acc = sum(p == g for p, g in zip(preds, gts)) / len(gts)
        correct = sum(p == g for p, g in zip(preds, gts))
        epoch = int(state.epoch)

        if self.writer:
            self.writer.add_scalar("eval/f1_macro", f1, epoch)
            self.writer.add_scalar("eval/accuracy", acc, epoch)

        wrong_n = len(gts) - correct
        print(
            f"\n[Epoch {epoch}] F1 macro: {f1:.4f}  |  Accuracy: {acc:.4f}"
            f"  (n={len(gts)}, 오답 {wrong_n}건)"
        )

        from utils.error_analysis import save_epoch_errors
        save_epoch_errors(
            all_results, epoch, self.analysis_dir, self.cfg.data.image_root,
        )

        self.epoch_records.append({
            "epoch": epoch,
            "accuracy": acc,
            "f1_macro": f1,
            "total": len(gts),
            "correct": correct,
            "all_results": all_results,
        })

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self.writer:
            self.writer.close()

        if self.epoch_records:
            from utils.error_analysis import save_training_summary
            save_training_summary(self.epoch_records, self.analysis_dir)
            print(f"[analysis] 오답 분석 저장 완료: {self.analysis_dir}")


def run_training(cfg: SimpleNamespace) -> None:
    """Fine-tune Qwen3.5 with QLoRA on the training split."""

    # Timestamped run directory — 이전 학습 결과를 덮어쓰지 않음
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.training.output_dir, f"run_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[run] 출력 디렉토리: {run_dir}")

    model, processor = load_qwen(cfg)
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
    enable_thinking = getattr(cfg.model, "enable_thinking", False)
    train_dataset = KoreanRecyclingVQADataset(train_df, cfg.data.image_root, shuffle=do_shuffle)
    val_dataset = KoreanRecyclingVQADataset(val_df, cfg.data.image_root, shuffle=False)
    collator = VQADataCollator(processor, cfg.training.max_seq_length, enable_thinking=enable_thinking)

    training_args = SFTConfig(
        output_dir=run_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_steps=getattr(cfg.training, "warmup_steps", 10),
        weight_decay=cfg.training.weight_decay,
        logging_steps=cfg.training.logging_steps,
        save_strategy="epoch",
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
    )

    patience = getattr(cfg.training, "early_stopping_patience", 2)
    f1_callback = F1EvalCallback(val_df, processor, cfg, run_dir)

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
    trainer.save_model(run_dir)
    processor.save_pretrained(run_dir)
    print(f"[run] 학습 완료. 모델 저장: {run_dir}")
