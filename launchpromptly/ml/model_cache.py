"""
Model cache -- downloads and caches ONNX model files from HuggingFace Hub.

Models are stored in ~/.launchpromptly/models/<model-id>/
and reused across sessions.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

DEFAULT_CACHE_DIR = Path.home() / ".launchpromptly" / "models"

_MODEL_REGISTRY: dict[str, dict] = {
    "meta-llama/Prompt-Guard-86M": {
        "onnx_file": "onnx/model.onnx",
        "quantized_file": "onnx/model_quantized.onnx",
        "files": [
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ],
    },
    "Xenova/bert-base-NER": {
        "onnx_file": "onnx/model.onnx",
        "quantized_file": "onnx/model_quantized.onnx",
        "files": [
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        ],
    },
    "Xenova/toxic-bert": {
        "onnx_file": "onnx/model.onnx",
        "quantized_file": "onnx/model_quantized.onnx",
        "files": [
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        ],
    },
    "unitary/toxic-bert": {
        "onnx_file": "onnx/model.onnx",
        "quantized_file": "onnx/model_quantized.onnx",
        "files": [
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        ],
    },
    "vectara/hallucination_evaluation_model": {
        "onnx_file": "onnx/model.onnx",
        "quantized_file": "onnx/model_quantized.onnx",
        "files": [
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        ],
    },
}


def ensure_model(
    model_id: str,
    *,
    quantized: bool = True,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Ensure model files are downloaded and cached locally.

    Returns the local directory path containing model.onnx and config files.
    Downloads on first call; subsequent calls return the cached path immediately.
    """
    base_dir = cache_dir or DEFAULT_CACHE_DIR
    model_dir = base_dir / model_id.replace("/", "--")

    entry = _MODEL_REGISTRY.get(model_id)
    if entry is None:
        supported = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {model_id}. Supported models: {supported}"
        )

    onnx_remote = (
        entry["quantized_file"]
        if quantized and entry.get("quantized_file")
        else entry["onnx_file"]
    )
    local_onnx = model_dir / "model.onnx"

    # Fast path: already cached
    if local_onnx.exists() and (model_dir / "config.json").exists():
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)

    repo = entry.get("repo", model_id)

    # Download ONNX model file
    if not local_onnx.exists():
        _download_hf_file(repo, onnx_remote, local_onnx)

    # Download supporting files
    for file_path in entry["files"]:
        local_path = model_dir / Path(file_path).name
        if not local_path.exists():
            _download_hf_file(repo, file_path, local_path)

    return model_dir


def _download_hf_file(repo: str, file_path: str, local_path: Path) -> None:
    """Download a single file from HuggingFace Hub."""
    url = f"https://huggingface.co/{repo}/resolve/main/{file_path}"

    headers = {}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    request = Request(url, headers=headers)
    try:
        with urlopen(request) as response:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(response.read())
    except Exception as exc:
        hint = ""
        if "401" in str(exc):
            hint = " This model may require authentication. Set the HF_TOKEN environment variable."
        elif "404" in str(exc):
            hint = f" File not found. The ONNX weights may not be published for {repo}."
        raise RuntimeError(
            f"Failed to download {file_path} from {repo}: {exc}.{hint}"
        ) from exc


def get_cache_dir() -> Path:
    """Get the default cache directory."""
    return DEFAULT_CACHE_DIR


def remove_model(model_id: str, cache_dir: Optional[Path] = None) -> None:
    """Remove a cached model."""
    import shutil

    model_dir = (cache_dir or DEFAULT_CACHE_DIR) / model_id.replace("/", "--")
    if model_dir.exists():
        shutil.rmtree(model_dir)


def list_cached_models(cache_dir: Optional[Path] = None) -> list[str]:
    """List all cached model IDs."""
    base_dir = cache_dir or DEFAULT_CACHE_DIR
    if not base_dir.exists():
        return []
    return [
        name.replace("--", "/")
        for name in os.listdir(base_dir)
        if (base_dir / name / "model.onnx").exists()
    ]
