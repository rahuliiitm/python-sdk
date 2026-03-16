"""
CLI for pre-downloading ML models.

Usage::

    python -m launchpromptly.ml.cli download-models
    python -m launchpromptly.ml.cli download-models --models toxicity,injection
    python -m launchpromptly.ml.cli download-models --cache-dir /app/.models
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .model_cache import (
    MODEL_NAME_MAP,
    ensure_model,
    get_registered_models,
)


def _resolve_model_id(specifier: str) -> str:
    """Resolve a model specifier (friendly name or full HF ID) to a model ID."""
    if specifier in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[specifier]
    registered = get_registered_models()
    if specifier in registered:
        return specifier
    raise ValueError(
        f'Unknown model: "{specifier}". '
        f"Available names: {', '.join(MODEL_NAME_MAP.keys())}. "
        f"Available IDs: {', '.join(registered)}"
    )


def _dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    if not path.exists():
        return 0
    total = 0
    for entry in path.iterdir():
        if entry.is_dir():
            total += _dir_size(entry)
        else:
            total += entry.stat().st_size
    return total


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="launchpromptly",
        description="LaunchPromptly ML model downloader",
    )
    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser(
        "download-models",
        help="Pre-download ML models for offline use",
    )
    dl.add_argument(
        "--models",
        type=str,
        default=None,
        help=f"Comma-separated model names (default: toxicity,injection). Names: {', '.join(MODEL_NAME_MAP.keys())}",
    )
    dl.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to store models (default: ~/.launchpromptly/models)",
    )

    args = parser.parse_args(argv)

    if args.command != "download-models":
        parser.print_help()
        sys.exit(1)

    # Resolve model IDs
    if args.models:
        names = [s.strip() for s in args.models.split(",")]
        model_ids = [_resolve_model_id(n) for n in names]
    else:
        model_ids = [
            MODEL_NAME_MAP["toxicity"],
            MODEL_NAME_MAP["injection"],
        ]

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    print(f"Downloading {len(model_ids)} model(s)...")
    if cache_dir:
        print(f"Cache directory: {cache_dir}")

    total_size = 0
    for model_id in model_ids:
        friendly = next(
            (k for k, v in MODEL_NAME_MAP.items() if v == model_id),
            model_id,
        )
        print(f"  {friendly} ({model_id})... ", end="", flush=True)

        try:
            quantized = not model_id.startswith("protectai/")
            model_dir = ensure_model(
                model_id, quantized=quantized, cache_dir=cache_dir
            )
            size = _dir_size(model_dir)
            total_size += size
            print(f"OK ({_format_bytes(size)})")
        except Exception as exc:
            print("FAILED")
            print(f"    {exc}")
            sys.exit(1)

    print(f"\nTotal: {_format_bytes(total_size)}")
    print("All models downloaded.")


if __name__ == "__main__":
    main()
