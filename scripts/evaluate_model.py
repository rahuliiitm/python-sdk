"""
Evaluate the trained attack classifier on the full dataset.

Prints per-class metrics, confusion matrix, and inference speed.

Usage:
    python scripts/evaluate_model.py [--model-dir ./output/attack-classifier-v1/onnx]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Evaluate attack classifier")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent / "output" / "attack-classifier-v1" / "onnx"
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parent / "training_data.json"),
    )
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model for eval")
    args = parser.parse_args()

    # Load data
    with open(args.data) as f:
        data = json.load(f)

    id2label = {int(k): v for k, v in data["id2label"].items()}
    label2id = data["label2id"]
    texts = [ex["text"] for ex in data["examples"]]
    true_labels = [label2id[ex["label"]] for ex in data["examples"]]

    model_dir = Path(args.model_dir)

    if args.onnx:
        # ONNX inference
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_path = model_dir / "onnx" / "model.onnx"
        if not onnx_path.exists():
            onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"No ONNX model found in {model_dir}")

        tok_path = model_dir / "tokenizer.json"
        if not tok_path.exists():
            # Try parent dir
            tok_path = model_dir.parent / "tokenizer.json"

        print(f"Loading ONNX model from {onnx_path}...")
        session = ort.InferenceSession(str(onnx_path))
        tokenizer = Tokenizer.from_file(str(tok_path))
        tokenizer.enable_truncation(max_length=256)
        tokenizer.enable_padding(length=256)

        def predict_batch(batch_texts):
            encodings = tokenizer.encode_batch(batch_texts)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            )
            feeds = {"input_ids": input_ids, "attention_mask": attention_mask}

            # Add token_type_ids if model expects it
            input_names = [i.name for i in session.get_inputs()]
            if "token_type_ids" in input_names:
                feeds["token_type_ids"] = np.zeros_like(input_ids)

            outputs = session.run(None, feeds)
            logits = outputs[0]
            return logits.argmax(axis=-1).tolist()

    else:
        # PyTorch inference
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        best_dir = model_dir.parent / "best" if "onnx" in str(model_dir) else model_dir
        if not (best_dir / "config.json").exists():
            best_dir = model_dir

        print(f"Loading PyTorch model from {best_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(str(best_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(best_dir))
        model.eval()

        def predict_batch(batch_texts):
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.logits.argmax(dim=-1).tolist()

    # Run predictions in batches
    print(f"Evaluating on {len(texts)} examples...")
    pred_labels = []
    batch_size = 32
    latencies = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        start = time.perf_counter()
        preds = predict_batch(batch)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed / len(batch))
        pred_labels.extend(preds)

    # Metrics
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted")

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  F1 (macro):  {f1_macro:.4f}")
    print(f"  F1 (weight): {f1_weighted:.4f}")

    # Targets
    print(f"\n  Targets:")
    print(f"  {'Accuracy >= 0.90':30s}: {'PASS' if accuracy >= 0.90 else 'FAIL'}")
    print(f"  {'F1 macro >= 0.80':30s}: {'PASS' if f1_macro >= 0.80 else 'FAIL'}")

    # Per-class report
    target_names = [id2label[i] for i in range(len(id2label))]
    print(f"\nPer-class Report:")
    print(classification_report(true_labels, pred_labels, target_names=target_names, digits=3))

    # Safe class precision
    from sklearn.metrics import precision_score
    safe_mask = np.array(pred_labels) == label2id["safe"]
    true_arr = np.array(true_labels)
    if safe_mask.any():
        safe_precision = (true_arr[safe_mask] == label2id["safe"]).sum() / safe_mask.sum()
        print(f"  Safe precision: {safe_precision:.4f} (target >= 0.95: {'PASS' if safe_precision >= 0.95 else 'FAIL'})")

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print(f"\nConfusion Matrix:")
    header = "".join(f"{name[:6]:>8s}" for name in target_names)
    print(f"{'':>20s}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>8d}" for v in row)
        print(f"  {target_names[i]:>18s}{row_str}")

    # Latency
    avg_ms = np.mean(latencies) * 1000
    p95_ms = np.percentile(latencies, 95) * 1000
    print(f"\nInference Latency:")
    print(f"  Average: {avg_ms:.2f} ms/example")
    print(f"  P95:     {p95_ms:.2f} ms/example")
    print(f"  Target < 10ms: {'PASS' if avg_ms < 10 else 'FAIL'}")


if __name__ == "__main__":
    main()
