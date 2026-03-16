"""Tests for model cache validation, retry, and atomic writes."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from launchpromptly.ml.model_cache import (
    validate_onnx_file,
    ensure_model,
    remove_model,
    list_cached_models,
)


# ── validate_onnx_file ───────────────────────────────────────────────────────


class TestValidateOnnxFile:
    def test_missing_file(self, tmp_path: Path) -> None:
        assert validate_onnx_file(tmp_path / "missing.onnx") is False

    def test_file_too_small(self, tmp_path: Path) -> None:
        f = tmp_path / "tiny.onnx"
        f.write_bytes(bytes([0x08]) * 512)
        assert validate_onnx_file(f) is False

    def test_wrong_magic_byte(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.onnx"
        buf = bytearray(2048)
        buf[0] = 0xFF
        f.write_bytes(bytes(buf))
        assert validate_onnx_file(f) is False

    def test_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "good.onnx"
        buf = bytearray(2048)
        buf[0] = 0x08
        f.write_bytes(bytes(buf))
        assert validate_onnx_file(f) is True

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.onnx"
        f.write_bytes(b"")
        assert validate_onnx_file(f) is False


# ── ensure_model ──────────────────────────────────────────────────────────────


class TestEnsureModel:
    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            ensure_model("unknown/model")


# ── remove_model ──────────────────────────────────────────────────────────────


class TestRemoveModel:
    def test_removes_existing_model(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "org--model"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_text("data")
        remove_model("org/model", cache_dir=tmp_path)
        assert not model_dir.exists()

    def test_no_error_for_missing_model(self, tmp_path: Path) -> None:
        remove_model("missing/model", cache_dir=tmp_path)  # should not raise


# ── list_cached_models ────────────────────────────────────────────────────────


class TestListCachedModels:
    def test_empty_cache(self, tmp_path: Path) -> None:
        assert list_cached_models(cache_dir=tmp_path) == []

    def test_lists_models_with_onnx(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "org--model"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_text("data")
        assert list_cached_models(cache_dir=tmp_path) == ["org/model"]

    def test_skips_dirs_without_onnx(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "org--incomplete"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        assert list_cached_models(cache_dir=tmp_path) == []

    def test_nonexistent_cache_dir(self, tmp_path: Path) -> None:
        assert list_cached_models(cache_dir=tmp_path / "nope") == []
