import os


# Set BEFORE any torch import
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    TrainerCallback,
)
import numpy as np
import torch
from google.cloud import storage
from dataclasses import dataclass
from typing import Any, Dict, List
import gc
from accelerate import Accelerator


class MemoryMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 5 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(
                f"Step {state.global_step}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB"
            )


@dataclass
class AudioDataCollator:
    """Custom data collator for audio token sequences with dynamic padding."""

    tokenizer: Any
    pad_token_id: int = 128263

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature.get("labels", input_ids)
            padding_length = max_length - len(input_ids)

            padded_input_ids = input_ids + [self.pad_token_id] * padding_length
            batch_input_ids.append(padded_input_ids)

            padded_labels = labels + [-100] * padding_length
            batch_labels.append(padded_labels)

            attention_mask = [1] * len(input_ids) + [0] * padding_length
            batch_attention_mask.append(attention_mask)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }


def main():
    # Initialize Accelerator first
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "no",
        log_with="tensorboard",
        project_dir=os.environ.get("AIP_CHECKPOINT_DIR", "/tmp/checkpoints"),
    )

    # Print device info
    accelerator.print(f"Using device: {accelerator.device}")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Process index: {accelerator.process_index}")

    # Clear memory at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        if accelerator.is_main_process:
            print(
                f"Starting GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )

    model_dir = os.environ.get("AIP_MODEL_DIR", "/tmp/model")
    checkpoint_dir = os.environ.get("AIP_CHECKPOINT_DIR", "/tmp/checkpoints")

    dsn = "cubbk/audio_swedish_2_dataset_cleaned"
    model_name = "canopylabs/orpheus-tts-0.1-pretrained"

    # Optimized settings
    epochs = 3
    batch_size = 24
    pad_token = 128263
    save_steps = 1000
    learning_rate = 5.0e-5

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_supported else torch.float32
    if not bf16_supported and accelerator.is_main_process:
        print("bfloat16 not supported on this device; using float32.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Load dataset only on main process to avoid duplication
    with accelerator.main_process_first():
        raw_ds = load_dataset(dsn, split="train", data_dir="8sidor_tokenized")
        raw_ds = raw_ds.select(range(1000))

        # Filter long sequences
        raw_ds = raw_ds.filter(lambda x: len(x["input_ids"]) <= 1000)
        if accelerator.is_main_process:
            print(f"Dataset: {len(raw_ds)} samples")

        split = raw_ds.train_test_split(test_size=0.05, seed=42)
        train_ds, eval_ds = split["train"], split["test"]

    data_collator = AudioDataCollator(tokenizer=tokenizer, pad_token_id=pad_token)

    training_args = TrainingArguments(
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=10,
        bf16=bf16_supported,
        output_dir=checkpoint_dir,
        save_steps=save_steps,
        remove_unused_columns=False,
        learning_rate=learning_rate,
        save_total_limit=1,  # Reduced from 2 to save memory
        logging_dir=f"{checkpoint_dir}/logs",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        dataloader_pin_memory=False,  # Disable pin memory
        dataloader_num_workers=0,  # Disable multiprocessing
        logging_first_step=True,
        logging_nan_inf_filter=False,  # Reduce logging overhead
        ddp_find_unused_parameters=False,  # Let Accelerate handle DDP
        save_on_each_node=False,  # Save only on main process
    )

    # Use custom trainer with memory management
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=[MemoryMonitorCallback()],
    )

    accelerator.print("Starting training...")
    trainer.train()

    # Save only on main process
    if accelerator.is_main_process:
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Training completed. Model saved to {model_dir}")

        trainer.push_to_hub(
            repo_id="cubbk/orpheus-swedish-2",
            commit_message="Fine-tuned Orpheus TTS on Swedish audio",
            blocking=True,
        )


if __name__ == "__main__":
    main()
