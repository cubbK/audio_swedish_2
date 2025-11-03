import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import torch
from google.cloud import storage


def main():
    # Environment variables for Vertex AI
    model_dir = os.environ.get("AIP_MODEL_DIR", "/tmp/model")
    checkpoint_dir = os.environ.get("AIP_CHECKPOINT_DIR", "/tmp/checkpoints")

    # Training configuration
    dsn = "cubbk/audio_swedish_2_dataset_cleaned"
    model_name = "canopylabs/orpheus-tts-0.1-pretrained"

    # Training Args
    epochs = 1
    batch_size = 1
    pad_token = 128263
    save_steps = 5000
    learning_rate = 5.0e-5

    # Ensure bf16 only when supported
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_supported else torch.float32
    if not bf16_supported:
        print("bfloat16 not supported on this device; using float32.")

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )

    # Load dataset
    raw_ds = load_dataset(dsn, split="train", data_dir="8sidor_tokenized")
    raw_ds = raw_ds.select(range(50))  # Take only first 50 items

    print(f"Dataset loaded: {raw_ds}")

    split = raw_ds.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    # Training arguments for Vertex AI
    training_args = TrainingArguments(
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=10,
        bf16=bf16_supported,
        output_dir=checkpoint_dir,
        save_steps=save_steps,
        remove_unused_columns=True,
        learning_rate=learning_rate,
        save_total_limit=2,  # Limit checkpoints to save storage
        logging_dir=f"{checkpoint_dir}/logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # Start training
    trainer.train()

    # Save final model
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"Training completed. Model saved to {model_dir}")


if __name__ == "__main__":
    main()
