import os
import json
from pathlib import Path

# MPS caps GPU memory conservatively (~20GB) and fragments under long-sequence training,
# triggering spurious OOMs even when system RAM is mostly free. Lifting the soft cap lets
# training use the real available unified memory. (Must be set before torch is imported.)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from expand_notes import STYLE_INSTRUCTION

BASE_MODEL = os.getenv("BASE_MODEL", "google/flan-t5-base")
TRAIN_FILE = os.getenv("TRAIN_FILE", "training_data.jsonl")
OUTPUT_DIR = os.getenv("FINETUNED_DIR", "finetuned_model")


def load_pairs(path: str):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def main(epochs: int = 4, batch_size: int = 2, lr: float = 3e-4, max_in: int = 448, max_out: int = 448):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
    )

    if not Path(TRAIN_FILE).exists():
        raise SystemExit(f"{TRAIN_FILE} not found. Run: python prepare_dataset.py")
    pairs = load_pairs(TRAIN_FILE)
    if not pairs:
        raise SystemExit("No training pairs found.")
    print(f"Training on {len(pairs)} pairs with base model '{BASE_MODEL}'.")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    def preprocess(batch):
        inputs = [f"{STYLE_INSTRUCTION}\n\nExplanation:\n{x}\n\nNotes:" for x in batch["input"]]
        model_inputs = tok(inputs, max_length=max_in, truncation=True)
        labels = tok(text_target=batch["output"], max_length=max_out, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = Dataset.from_list(pairs).map(preprocess, batched=True, remove_columns=["input", "output"])

    # Plain config: the MPS watermark env var (top of file) lifts the artificial memory cap,
    # and the 448-token limit bounds the long NLP sequences. Gradient checkpointing /
    # accumulation are deliberately avoided — on this MPS stack they yielded a model that
    # never learned (loss stuck at ln|vocab| ~= 10.4).
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned condenser to '{OUTPUT_DIR}/'. expand_notes will use it automatically.")


if __name__ == "__main__":
    main()
