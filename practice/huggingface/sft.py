#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) script using Hugging Face TRL.
- Base model: SmolLM3-3B-Base (loaded from local model folder)
- Dataset: Smoltalk2 SFT (loaded from local data folder)
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# Paths relative to project root (parent of practice/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/SmolLM2-360M"
print(DEFAULT_MODEL_PATH)
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "smoltalk2"
print(DEFAULT_DATA_PATH)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "sft_output"
print(DEFAULT_OUTPUT_PATH)


def resolve_model_path(model_path: Path) -> Path:
    """Resolve model path, trying common locations."""
    candidates = [
        model_path,
        model_path / "SmolLM3-3B-Base",
        PROJECT_ROOT / "models" / "SmolLM3-3B-Base",
        PROJECT_ROOT / "model",
    ]
    for p in candidates:
        if p.exists() and (p / "config.json").exists():
            return p
    return model_path  # Return as-is, let from_pretrained fail with clear error


def load_model_and_tokenizer(
    model_path: Path,
    use_4bit: bool = True,
    device_map: str = "auto",
):
    """Load model and tokenizer from local path (no download)."""
    model_path = resolve_model_path(Path(model_path))
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Place SmolLM3-3B-Base in 'model' or 'models/SmolLM3-3B-Base'."
        )

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_smoltalk_dataset(
    data_path: Path,
    config: str = "Mid",
    split: str = "smoltalk_smollm3_everyday_conversations_no_think",
    subset: str | None = None,
    max_samples: int | None = None,
):
    """
    Load Smoltalk2 dataset from local path or HuggingFace Hub.
    SFT config has: messages (list of {content, role}), source, chat_template_kwargs.
    """
    data_path = Path(data_path) / config

    print("data_path is ", data_path)

    # Try local load first: data/smoltalk2 with SFT parquet files
    sft_dir = data_path
    parquet_files = list(sft_dir.glob("*.parquet")) if sft_dir.exists() else []

    print("parquet_files is ", parquet_files)

    if parquet_files:
        data_files = {"train": [str(f) for f in parquet_files[:1]]}  # Limit for memory

        print("data_files is ", data_files)

        dataset = load_dataset("parquet", data_files=data_files, split="train")

    if subset:
        # Filter by source if subset name provided
        if "source" in dataset.column_names:
            dataset = dataset.filter(lambda x: subset in x.get("source", ""))
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


def format_messages_to_text(example, tokenizer):
    """Format messages using the model's chat template."""
    messages = example["messages"]
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        # Fallback: simple format for base models without chat template
        parts = []
        for m in messages:
            role, content = m.get("role", "user"), m.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        text = "\n".join(parts)
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(
        description="SFT with SmolLM3-3B-Base and Smoltalk2"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to model folder (SmolLM3-3B-Base)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to data folder (smoltalk2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Max training samples (for testing)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="smoltalk_smollm3_everyday_conversations_no_think",
        help="Dataset split name (SFT config)",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Max sequence length"
    )
    parser.add_argument("--num-epochs", type=int, default=1, help="Training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Per-device batch size"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--no-4bit", action="store_true", help="Disable 4-bit quantization"
    )
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        use_4bit=not args.no_4bit,
    )
    print(f"Model loaded from {args.model_path}")

    print("Loading dataset...")
    dataset = load_smoltalk_dataset(
        args.data_path,
        config="Mid",
        split=args.split,
        max_samples=args.max_samples,
    )
    print(f"Dataset size: {len(dataset)}")

    # Format messages to text for SFTTrainer (num_proc=1 to avoid tokenizer pickling)
    dataset = dataset.map(
        lambda ex: format_messages_to_text(ex, tokenizer),
        remove_columns=[c for c in dataset.column_names if c != "text"],
        desc="Formatting",
        num_proc=1,
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        fp16=False,
        bf16=True,  # Avoid "not implemented for BFloat16" AMP error
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=50,
        report_to="none",
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        packing=False,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
