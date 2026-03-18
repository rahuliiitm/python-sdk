"""
Fine-tune all-MiniLM-L6-v2 for 7-class attack classification.

Trains a sequence classifier on the curated LaunchPromptly attack corpus.
Outputs a PyTorch model ready for ONNX export.

Usage:
    pip install -r scripts/requirements-training.txt
    python scripts/collect_training_data.py
    python scripts/train_attack_classifier.py [--epochs 10] [--output ./output/attack-classifier-v1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 256


def load_data(path: str) -> tuple[list[str], list[int], dict, dict]:
    with open(path) as f:
        data = json.load(f)

    label2id = data["label2id"]
    id2label = {int(k): v for k, v in data["id2label"].items()}

    texts = [ex["text"] for ex in data["examples"]]
    labels = [label2id[ex["label"]] for ex in data["examples"]]

    return texts, labels, label2id, id2label


def main():
    parser = argparse.ArgumentParser(description="Train attack classifier")
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parent / "training_data.json"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "output" / "attack-classifier-v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    texts, labels, label2id, id2label = load_data(args.data)
    print(f"  {len(texts)} examples, {len(label2id)} classes")

    # Stratified split
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.15, stratify=labels, random_state=args.seed,
    )
    print(f"  Train: {len(train_texts)}, Eval: {len(eval_texts)}")

    # Load tokenizer and model
    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(
        tokenize, batched=True
    )
    eval_ds = Dataset.from_dict({"text": eval_texts, "label": eval_labels}).map(
        tokenize, batched=True
    )

    # Metrics
    def compute_metrics(pred):
        preds = pred.predictions.argmax(-1)
        labs = pred.label_ids
        return {
            "accuracy": accuracy_score(labs, preds),
            "f1_macro": f1_score(labs, preds, average="macro"),
            "f1_weighted": f1_score(labs, preds, average="weighted"),
        }

    # Training args
    output_dir = Path(args.output)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        logging_steps=10,
        fp16=False,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nTraining...")
    trainer.train()

    # Save best model
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Final evaluation
    print("\nFinal evaluation on held-out set:")
    results = trainer.evaluate()
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Detailed classification report
    preds = trainer.predict(eval_ds)
    pred_labels = preds.predictions.argmax(-1)
    print("\nClassification Report:")
    print(
        classification_report(
            eval_labels,
            pred_labels,
            target_names=[id2label[i] for i in range(len(id2label))],
            digits=3,
        )
    )

    print(f"\nBest model saved to: {best_dir}")
    print("Next step: python scripts/export_onnx.py")


if __name__ == "__main__":
    main()
