"""
Pytest configuration and fixtures for BitNet Python baseline tests.
"""
import pytest
import torch
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import tempfile
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "gpu"))
sys.path.insert(0, str(project_root / "utils"))

# Set deterministic behavior
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        "numerical_tolerance": {
            "rtol": 1e-4,
            "atol": 1e-5,
        },
        "performance_tolerance": {
            "max_regression_percent": 5.0,
        },
        "model_shapes": [
            (2560, 2560),   # Square matrix
            (3840, 2560),   # Rectangular
            (13824, 2560),  # Large input
            (2560, 6912),   # Large output
        ],
        "batch_sizes": [1, 4, 8],
        "sequence_lengths": [64, 128, 512],
    }

@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for test models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def mock_model_config():
    """Mock model configuration for testing."""
    return {
        "dim": 2560,
        "n_layers": 30,
        "n_heads": 20,
        "n_kv_heads": 5,
        "vocab_size": 128256,
        "ffn_dim": 6912,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "use_kernel": False,
    }

@pytest.fixture
def sample_weights():
    """Generate sample weight tensors for testing."""
    def _generate_weights(shape: Tuple[int, int], dtype=torch.float32):
        # Generate weights with known distribution for reproducible tests
        weights = torch.randn(shape, dtype=dtype)
        # Normalize to [-1, 1] range for BitNet
        weights = torch.sign(weights)
        return weights
    return _generate_weights

@pytest.fixture
def cuda_available():
    """Check if CUDA is available for GPU tests."""
    return torch.cuda.is_available()

@pytest.fixture
def device(cuda_available):
    """Get appropriate device for testing."""
    return torch.device("cuda" if cuda_available else "cpu")

class TestDataGenerator:
    """Generate test data with known properties for validation."""

    @staticmethod
    def generate_quantization_test_data(shape: Tuple[int, int], seed: int = 42) -> torch.Tensor:
        """Generate test data specifically for quantization testing."""
        torch.manual_seed(seed)
        # Create data with specific patterns to test quantization accuracy
        data = torch.randn(shape)
        # Add some extreme values to test edge cases
        data[0, 0] = 1000.0  # Large positive
        data[0, 1] = -1000.0  # Large negative
        data[1, 0] = 1e-8  # Very small positive
        data[1, 1] = -1e-8  # Very small negative
        data[2, 0] = 0.0  # Exact zero
        return data

    @staticmethod
    def generate_inference_test_data(batch_size: int, seq_len: int, vocab_size: int, seed: int = 42) -> Dict[str, torch.Tensor]:
        """Generate test data for inference testing."""
        torch.manual_seed(seed)
        return {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "position_ids": torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1),
        }

@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "quantization: marks tests related to quantization")
    config.addinivalue_line("markers", "inference: marks tests related to inference")
    config.addinivalue_line("markers", "conversion: marks tests related to model conversion")

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test file names
        if "quantization" in item.nodeid:
            item.add_marker(pytest.mark.quantization)
        if "inference" in item.nodeid:
            item.add_marker(pytest.mark.inference)
        if "conversion" in item.nodeid:
            item.add_marker(pytest.mark.conversion)
        if "gpu" in item.nodeid or "cuda" in item.nodeid:
            item.add_marker(pytest.mark.gpu)
        # Mark slow tests
        if any(keyword in item.nodeid for keyword in ["benchmark", "performance", "large"]):
            item.add_marker(pytest.mark.slow)
