from __future__ import annotations

from types import SimpleNamespace

import torch
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig

from data.dataset import KoreanRecyclingVQADataset, VQADataCollator, load_and_split
from models.qwen import load_qwen


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

    train_dataset = KoreanRecyclingVQADataset(train_df, cfg.data.image_root)
    val_dataset = KoreanRecyclingVQADataset(val_df, cfg.data.image_root)
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
        eval_strategy="steps",
        eval_steps=cfg.training.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=cfg.training.bf16,
        fp16=False,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=cfg.training.remove_unused_columns,
        dataloader_num_workers=0,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=cfg.training.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(cfg.training.output_dir)
    processor.save_pretrained(cfg.training.output_dir)
    print(f"Saved to {cfg.training.output_dir}")
