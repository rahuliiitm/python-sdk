"""
Collect and format training data for the attack classifier.

Reads attack-patterns.json and safe-prompts.json, outputs a single
training_data.json file with labeled examples ready for fine-tuning.

Usage:
    python scripts/collect_training_data.py [--output training_data.json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "launchpromptly" / "ml" / "data"

LABEL2ID = {
    "safe": 0,
    "injection": 1,
    "jailbreak": 2,
    "data_extraction": 3,
    "manipulation": 4,
    "role_escape": 5,
    "social_engineering": 6,
}


def load_attack_patterns() -> list[dict]:
    path = DATA_DIR / "attack-patterns.json"
    with open(path) as f:
        raw = json.load(f)

    examples = []
    for category, patterns in raw["categories"].items():
        label = category  # category keys match our label schema
        if label not in LABEL2ID:
            print(f"  WARNING: unknown category '{label}', skipping")
            continue
        for text in patterns:
            examples.append({"text": text, "label": label})
    return examples


def load_safe_prompts() -> list[dict]:
    path = DATA_DIR / "safe-prompts.json"
    with open(path) as f:
        raw = json.load(f)

    examples = []
    for _category, prompts in raw["prompts"].items():
        for text in prompts:
            examples.append({"text": text, "label": "safe"})
    return examples


def main():
    parser = argparse.ArgumentParser(description="Collect training data")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "training_data.json"),
        help="Output path for training data JSON",
    )
    args = parser.parse_args()

    print("Loading attack patterns...")
    attacks = load_attack_patterns()
    print(f"  Loaded {len(attacks)} attack examples")

    print("Loading safe prompts...")
    safe = load_safe_prompts()
    print(f"  Loaded {len(safe)} safe examples")

    all_examples = attacks + safe

    # Print label distribution
    counts = Counter(ex["label"] for ex in all_examples)
    print("\nLabel distribution:")
    for label in sorted(counts, key=lambda l: LABEL2ID.get(l, 99)):
        print(f"  {label:20s}: {counts[label]:4d}")
    print(f"  {'TOTAL':20s}: {len(all_examples):4d}")

    output = {
        "version": "1.0.0",
        "label2id": LABEL2ID,
        "id2label": {v: k for k, v in LABEL2ID.items()},
        "examples": all_examples,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(all_examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
