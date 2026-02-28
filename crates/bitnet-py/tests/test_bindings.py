"""
Smoke tests for the bitnet_py Python package surface.

These tests validate that the compiled extension module exposes the expected
classes, functions, exceptions, and constants **without** loading a real model
(which would require GGUF files and significant memory).

Run with:
    maturin develop -p bitnet-py --features cpu && pytest crates/bitnet-py/tests/test_bindings.py -v
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _try_import():
    """Attempt to import the compiled extension; skip the entire module if unavailable."""
    try:
        import bitnet_py  # noqa: F401

        return bitnet_py
    except ImportError:
        pytest.skip(
            "bitnet_py native module not built — run 'maturin develop' first",
            allow_module_level=True,
        )


bitnet = _try_import()


# ── Module Surface ───────────────────────────────────────────────────


class TestModuleSurface:
    """Verify that all expected symbols are importable."""

    @pytest.mark.parametrize(
        "name",
        [
            "BitNetModel",
            "InferenceEngine",
            "Tokenizer",
            "BitNetConfig",
            "GenerationConfig",
            "ModelLoader",
            "ModelInfo",
            "StreamingGenerator",
        ],
    )
    def test_class_exists(self, name: str):
        assert hasattr(bitnet, name), f"Missing class: {name}"
        assert callable(getattr(bitnet, name))

    @pytest.mark.parametrize(
        "name",
        [
            "load_model",
            "list_available_models",
            "get_device_info",
            "set_num_threads",
            "batch_generate",
            "get_model_info",
            "is_cuda_available",
            "is_metal_available",
            "get_cuda_device_count",
        ],
    )
    def test_function_exists(self, name: str):
        assert hasattr(bitnet, name), f"Missing function: {name}"
        assert callable(getattr(bitnet, name))

    def test_version_string(self):
        assert isinstance(bitnet.__version__, str)
        assert len(bitnet.__version__) > 0


# ── Exceptions ───────────────────────────────────────────────────────


class TestExceptions:
    """Verify the exception hierarchy."""

    def test_base_error_is_exception(self):
        assert issubclass(bitnet.BitNetBaseError, Exception)

    @pytest.mark.parametrize(
        "name",
        [
            "ModelError",
            "QuantizationError",
            "InferenceError",
            "KernelError",
            "ConfigError",
            "ValidationError",
        ],
    )
    def test_error_subclass(self, name: str):
        exc_cls = getattr(bitnet, name)
        assert issubclass(exc_cls, bitnet.BitNetBaseError)

    def test_catch_base_catches_subclass(self):
        with pytest.raises(bitnet.BitNetBaseError):
            raise bitnet.ModelError("test")


# ── GenerationConfig ─────────────────────────────────────────────────


class TestGenerationConfig:
    def test_default_construction(self):
        cfg = bitnet.GenerationConfig()
        assert repr(cfg).startswith("GenerationConfig(")

    def test_custom_params(self):
        cfg = bitnet.GenerationConfig(
            max_tokens=50, temperature=0.5, top_p=0.8, top_k=10
        )
        assert repr(cfg)  # no crash


# ── BitNetConfig ─────────────────────────────────────────────────────


class TestBitNetConfig:
    def test_default_construction(self):
        cfg = bitnet.BitNetConfig()
        assert repr(cfg).startswith("BitNetConfig(")


# ── ModelLoader ──────────────────────────────────────────────────────


class TestModelLoader:
    def test_create_cpu(self):
        loader = bitnet.ModelLoader(device="cpu")
        assert loader.device == "cpu"

    def test_available_formats(self):
        loader = bitnet.ModelLoader()
        fmts = loader.available_formats()
        assert "GGUF" in fmts

    def test_repr(self):
        loader = bitnet.ModelLoader()
        assert "ModelLoader" in repr(loader)


# ── list_available_models ────────────────────────────────────────────


class TestListModels:
    def test_empty_dir(self, tmp_path):
        result = bitnet.list_available_models(str(tmp_path))
        assert result == []


# ── Device helpers ───────────────────────────────────────────────────


class TestDeviceHelpers:
    def test_is_cuda_returns_bool(self):
        assert isinstance(bitnet.is_cuda_available(), bool)

    def test_is_metal_returns_bool(self):
        assert isinstance(bitnet.is_metal_available(), bool)

    def test_cuda_device_count_int(self):
        assert isinstance(bitnet.get_cuda_device_count(), int)

    def test_get_device_info_dict(self):
        info = bitnet.get_device_info()
        assert "cpu" in info
        assert "gpu" in info


# ── set_num_threads ──────────────────────────────────────────────────


class TestSetThreads:
    def test_set_threads_no_crash(self):
        bitnet.set_num_threads(2)  # should not raise
