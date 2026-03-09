"""Pytest fixtures for mechinterp-lab tests."""

import os

import pytest
import torch


def _skip_model_tests() -> bool:
    """Skip tests that require model download (e.g. in CI without cache)."""
    return os.environ.get("MECHINTERP_SKIP_MODEL_TESTS", "").lower() in ("1", "true", "yes")


@pytest.fixture
def small_model():
    """Load a small model for testing. Uses gpt2 which is ~124M params."""
    if _skip_model_tests():
        pytest.skip("MECHINTERP_SKIP_MODEL_TESTS is set; skipping model download")
    from mechinterp_lab.models import load_model

    return load_model("gpt2", device="cpu")


@pytest.fixture
def mock_model():
    """Create a minimal mock for unit tests that don't need a real model."""
    import unittest.mock as mock

    model = mock.MagicMock()
    model.to_tokens = mock.MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
    model.to_single_token = mock.MagicMock(return_value=42)
    model.hook_dict = {
        "blocks.0.attn.hook_z": None,
        "blocks.0.hook_resid_pre": None,
    }
    return model
