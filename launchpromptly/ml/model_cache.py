"""
Model cache -- downloads and caches ONNX model files from HuggingFace Hub.

Models are stored in ~/.launchpromptly/models/<model-id>/
and reused across sessions.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

DEFAULT_CACHE_DIR = Path.home() / ".launchpromptly" / "models"

_MIN_ONNX_FILE_SIZE = 1024  # 1KB -- any valid model is much larger
_MAX_DOWNLOAD_RETRIES = 3
_BASE_RETRY_DELAY_S = 1.0


def validate_onnx_file(file_path: Path) -> bool:
    """Check that a cached ONNX file looks valid.

    Verifies minimum file size and ONNX protobuf magic byte (0x08 = ir_version field).
    """
    try:
        if file_path.stat().st_size < _MIN_ONNX_FILE_SIZE:
            return False
        with open(file_path, "rb") as f:
            header = f.read(4)
        return len(header) >= 1 and header[0] == 0x08
    except OSError:
        return False

_MODEL_REGISTRY: dict[str, dict] = {
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
    "protectai/deberta-v3-base-prompt-injection-v2": {
        "onnx_file": "onnx/model.onnx",
        "files": [
            "onnx/tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ],
    },
    "protectai/deberta-v3-small-prompt-injection-v2": {
        "onnx_file": "onnx/model.onnx",
        "files": [
            "onnx/tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ],
    },
    "Xenova/all-MiniLM-L6-v2": {
        "onnx_file": "onnx/model.onnx",
        "quantized_file": "onnx/model_quantized.onnx",
        "files": [
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        ],
    },
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {
        "onnx_file": "onnx/model.onnx",
        "quantized_file": "onnx/model_quantized.onnx",
        "files": [
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        ],
    },
    "launchpromptly/attack-classifier-v1": {
        "onnx_file": "onnx/model.onnx",
        "quantized_file": "onnx/model_quantized.onnx",
        "files": [
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        ],
    },
}

MODEL_NAME_MAP: dict[str, str] = {
    "toxicity": "Xenova/toxic-bert",
    "injection": "protectai/deberta-v3-base-prompt-injection-v2",
    "injection-small": "protectai/deberta-v3-small-prompt-injection-v2",
    "ner": "Xenova/bert-base-NER",
    "embedding": "Xenova/all-MiniLM-L6-v2",
    "nli": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "attack-classifier": "launchpromptly/attack-classifier-v1",
}


def get_registered_models() -> list[str]:
    """Get list of all registered model IDs."""
    return list(_MODEL_REGISTRY.keys())


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

    # Fast path: already cached and valid
    if (
        local_onnx.exists()
        and (model_dir / "config.json").exists()
        and validate_onnx_file(local_onnx)
    ):
        return model_dir

    # Remove corrupted ONNX file so it gets re-downloaded
    if local_onnx.exists() and not validate_onnx_file(local_onnx):
        local_onnx.unlink()

    model_dir.mkdir(parents=True, exist_ok=True)

    repo = entry.get("repo", model_id)

    # Download ONNX model file
    if not local_onnx.exists():
        _download_hf_file(repo, onnx_remote, local_onnx)

    # Validate downloaded file
    if not validate_onnx_file(local_onnx):
        local_onnx.unlink()
        raise RuntimeError(
            f"Downloaded ONNX file for {model_id} failed integrity check. "
            "The file may be corrupted or incomplete."
        )

    # Download supporting files
    for file_path in entry["files"]:
        local_path = model_dir / Path(file_path).name
        if not local_path.exists():
            _download_hf_file(repo, file_path, local_path)

    return model_dir


def _download_hf_file(repo: str, file_path: str, local_path: Path) -> None:
    """Download a single file from HuggingFace Hub.

    Retries up to 3 times with exponential backoff on server/network errors.
    Uses atomic write (temp file + rename) to prevent partial downloads.
    """
    url = f"https://huggingface.co/{repo}/resolve/main/{file_path}"

    headers = {}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    request = Request(url, headers=headers)

    for attempt in range(_MAX_DOWNLOAD_RETRIES):
        try:
            with urlopen(request) as response:
                data = response.read()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: temp file then rename
            tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
            tmp_path.write_bytes(data)
            tmp_path.rename(local_path)
            return
        except Exception as exc:
            exc_str = str(exc)
            # Client errors (401, 404) won't change on retry
            if "401" in exc_str:
                raise RuntimeError(
                    f"Failed to download {file_path} from {repo}: {exc}."
                    " This model may require authentication. Set the HF_TOKEN environment variable."
                ) from exc
            if "404" in exc_str:
                raise RuntimeError(
                    f"Failed to download {file_path} from {repo}: {exc}."
                    f" File not found. The ONNX weights may not be published for {repo}."
                ) from exc
            if attempt < _MAX_DOWNLOAD_RETRIES - 1:
                delay = _BASE_RETRY_DELAY_S * (2 ** attempt)
                time.sleep(delay)
                continue
            raise RuntimeError(
                f"Failed to download {file_path} from {repo} after {_MAX_DOWNLOAD_RETRIES} attempts: {exc}."
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
