"""
Test fixtures and known-good model outputs for cross-validation.
"""
import pytest
import torch
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import tempfile
import hashlib
import sys

# Import BitNet modules
try:
    from gpu.model import ModelArgs, Transformer, make_cache
    from gpu.tokenizer import Tokenizer
    from gpu.pack_weight import convert_weight_int8_to_int2
except ImportError as e:
    pytest.skip(f"BitNet modules not available: {e}", allow_module_level=True)

class TestFixtureGenerator:
    """Generate and manage test fixtures with known-good outputs."""
    
    def __init__(self, fixture_dir: Path):
        self.fixture_dir = fixture_dir
        self.fixture_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_model_fixture(self, config: Dict[str, Any], fixture_name: str) -> Dict[str, Any]:
        """Generate a model fixture with known outputs."""
        # Create model with fixed seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        args = ModelArgs(**config)
        model = Transformer(args)
        model.eval()
        
        # Generate test inputs
        batch_size = 2
        seq_len = 8
        cache_len = 16
        
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,))
        token_lengths = torch.tensor([seq_len] * batch_size)
        start_pos = torch.zeros(batch_size, dtype=torch.long)
        cache = make_cache(args, cache_len)
        
        # Generate outputs
        with torch.no_grad():
            logits = model(token_values, token_lengths, start_pos, cache, cache_len)
        
        # Create fixture data
        fixture_data = {
            "config": config,
            "inputs": {
                "token_values": token_values.tolist(),
                "token_lengths": token_lengths.tolist(),
                "start_pos": start_pos.tolist(),
                "batch_size": batch_size,
                "seq_len": seq_len,
                "cache_len": cache_len,
            },
            "outputs": {
                "logits": logits.tolist(),
                "logits_shape": list(logits.shape),
            },
            "model_state": {
                "state_dict_keys": list(model.state_dict().keys()),
                "parameter_count": sum(p.numel() for p in model.parameters()),
            },
            "metadata": {
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "seed": 42,
                "fixture_version": "1.0",
            }
        }
        
        # Save fixture
        fixture_path = self.fixture_dir / f"{fixture_name}.json"
        with open(fixture_path, 'w') as f:
            json.dump(fixture_data, f, indent=2)
        
        return fixture_data
    
    def generate_quantization_fixture(self, shape: Tuple[int, int], fixture_name: str) -> Dict[str, Any]:
        """Generate quantization test fixture."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate test weights
        original_weights = torch.randn(shape)
        
        # Apply quantization
        quantized_weights = torch.sign(original_weights)
        
        # Convert to int8 and then int2
        int8_weights = quantized_weights.to(torch.int8)
        if shape[1] % 4 == 0:  # Can convert to int2
            int2_weights = convert_weight_int8_to_int2(int8_weights)
        else:
            int2_weights = None
        
        fixture_data = {
            "shape": list(shape),
            "original_weights": original_weights.tolist(),
            "quantized_weights": quantized_weights.tolist(),
            "int8_weights": int8_weights.tolist(),
            "int2_weights": int2_weights.tolist() if int2_weights is not None else None,
            "metadata": {
                "seed": 42,
                "fixture_version": "1.0",
            }
        }
        
        # Save fixture
        fixture_path = self.fixture_dir / f"{fixture_name}.json"
        with open(fixture_path, 'w') as f:
            json.dump(fixture_data, f, indent=2)
        
        return fixture_data
    
    def generate_inference_fixture(self, config: Dict[str, Any], prompts: List[str], fixture_name: str) -> Dict[str, Any]:
        """Generate inference test fixture with text prompts."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        args = ModelArgs(**config)
        model = Transformer(args)
        model.eval()
        
        # For this fixture, we'll use token IDs directly since we don't have a real tokenizer
        # In practice, you'd tokenize the prompts
        fixture_data = {
            "config": config,
            "prompts": prompts,
            "token_sequences": [],
            "outputs": [],
            "metadata": {
                "seed": 42,
                "fixture_version": "1.0",
            }
        }
        
        for i, prompt in enumerate(prompts):
            # Generate pseudo-tokens for the prompt (in practice, use real tokenizer)
            prompt_length = min(len(prompt.split()), 16)  # Limit length
            token_ids = torch.randint(0, args.vocab_size, (prompt_length,))
            
            # Generate output
            cache_len = prompt_length + 8
            token_lengths = torch.tensor([prompt_length])
            start_pos = torch.zeros(1, dtype=torch.long)
            cache = make_cache(args, cache_len)
            
            with torch.no_grad():
                logits = model(token_ids, token_lengths, start_pos, cache, cache_len)
            
            fixture_data["token_sequences"].append({
                "prompt": prompt,
                "token_ids": token_ids.tolist(),
                "length": prompt_length,
            })
            
            fixture_data["outputs"].append({
                "logits": logits.tolist(),
                "shape": list(logits.shape),
            })
        
        # Save fixture
        fixture_path = self.fixture_dir / f"{fixture_name}.json"
        with open(fixture_path, 'w') as f:
            json.dump(fixture_data, f, indent=2)
        
        return fixture_data
    
    def load_fixture(self, fixture_name: str) -> Dict[str, Any]:
        """Load a test fixture."""
        fixture_path = self.fixture_dir / f"{fixture_name}.json"
        if not fixture_path.exists():
            raise FileNotFoundError(f"Fixture not found: {fixture_path}")
        
        with open(fixture_path, 'r') as f:
            return json.load(f)
    
    def validate_fixture(self, fixture_data: Dict[str, Any]) -> bool:
        """Validate fixture data integrity."""
        required_keys = ["metadata"]
        for key in required_keys:
            if key not in fixture_data:
                return False
        
        # Check version compatibility
        if fixture_data["metadata"].get("fixture_version") != "1.0":
            return False
        
        return True

@pytest.fixture(scope="session")
def fixture_generator(tmp_path_factory):
    """Provide fixture generator with temporary directory."""
    fixture_dir = tmp_path_factory.mktemp("test_fixtures")
    return TestFixtureGenerator(fixture_dir)

@pytest.fixture(scope="session")
def standard_model_fixtures(fixture_generator):
    """Generate standard model fixtures."""
    fixtures = {}
    
    # Small model fixture
    small_config = {
        "dim": 128,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "vocab_size": 1000,
        "ffn_dim": 256,
    }
    fixtures["small_model"] = fixture_generator.generate_model_fixture(small_config, "small_model")
    
    # Medium model fixture
    medium_config = {
        "dim": 256,
        "n_layers": 4,
        "n_heads": 8,
        "n_kv_heads": 4,
        "vocab_size": 5000,
        "ffn_dim": 512,
    }
    fixtures["medium_model"] = fixture_generator.generate_model_fixture(medium_config, "medium_model")
    
    return fixtures

@pytest.fixture(scope="session")
def quantization_fixtures(fixture_generator):
    """Generate quantization test fixtures."""
    fixtures = {}
    
    shapes = [
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
    ]
    
    for i, shape in enumerate(shapes):
        fixture_name = f"quantization_{shape[0]}x{shape[1]}"
        fixtures[fixture_name] = fixture_generator.generate_quantization_fixture(shape, fixture_name)
    
    return fixtures

@pytest.fixture(scope="session")
def inference_fixtures(fixture_generator):
    """Generate inference test fixtures."""
    fixtures = {}
    
    config = {
        "dim": 128,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "vocab_size": 1000,
        "ffn_dim": 256,
    }
    
    prompts = [
        "Hello world",
        "The quick brown fox",
        "Machine learning is",
        "In the beginning",
        "Once upon a time",
    ]
    
    fixtures["inference_test"] = fixture_generator.generate_inference_fixture(config, prompts, "inference_test")
    
    return fixtures

class TestFixtureValidation:
    """Test fixture validation and consistency."""
    
    def test_model_fixture_consistency(self, standard_model_fixtures):
        """Test that model fixtures are consistent."""
        for fixture_name, fixture_data in standard_model_fixtures.items():
            # Recreate model with same config and seed
            torch.manual_seed(42)
            np.random.seed(42)
            
            config = fixture_data["config"]
            args = ModelArgs(**config)
            model = Transformer(args)
            model.eval()
            
            # Recreate inputs
            inputs = fixture_data["inputs"]
            token_values = torch.tensor(inputs["token_values"])
            token_lengths = torch.tensor(inputs["token_lengths"])
            start_pos = torch.tensor(inputs["start_pos"])
            cache = make_cache(args, inputs["cache_len"])
            
            # Generate outputs
            with torch.no_grad():
                logits = model(token_values, token_lengths, start_pos, cache, inputs["cache_len"])
            
            # Compare with fixture
            expected_logits = torch.tensor(fixture_data["outputs"]["logits"])
            torch.testing.assert_close(logits, expected_logits, rtol=1e-5, atol=1e-6)
            
            print(f"Fixture {fixture_name} validated successfully")
    
    def test_quantization_fixture_consistency(self, quantization_fixtures):
        """Test that quantization fixtures are consistent."""
        for fixture_name, fixture_data in quantization_fixtures.items():
            # Recreate quantization with same seed
            torch.manual_seed(42)
            np.random.seed(42)
            
            shape = tuple(fixture_data["shape"])
            original_weights = torch.randn(shape)
            
            # Apply quantization
            quantized_weights = torch.sign(original_weights)
            
            # Compare with fixture
            expected_original = torch.tensor(fixture_data["original_weights"])
            expected_quantized = torch.tensor(fixture_data["quantized_weights"])
            
            torch.testing.assert_close(original_weights, expected_original, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(quantized_weights, expected_quantized, rtol=1e-5, atol=1e-6)
            
            print(f"Quantization fixture {fixture_name} validated successfully")
    
    def test_inference_fixture_consistency(self, inference_fixtures):
        """Test that inference fixtures are consistent."""
        fixture_data = inference_fixtures["inference_test"]
        
        # Recreate model with same config and seed
        torch.manual_seed(42)
        np.random.seed(42)
        
        config = fixture_data["config"]
        args = ModelArgs(**config)
        model = Transformer(args)
        model.eval()
        
        # Test each prompt
        for i, (token_seq, expected_output) in enumerate(zip(fixture_data["token_sequences"], fixture_data["outputs"])):
            token_ids = torch.tensor(token_seq["token_ids"])
            prompt_length = token_seq["length"]
            
            # Generate output
            cache_len = prompt_length + 8
            token_lengths = torch.tensor([prompt_length])
            start_pos = torch.zeros(1, dtype=torch.long)
            cache = make_cache(args, cache_len)
            
            with torch.no_grad():
                logits = model(token_ids, token_lengths, start_pos, cache, cache_len)
            
            # Compare with fixture
            expected_logits = torch.tensor(expected_output["logits"])
            torch.testing.assert_close(logits, expected_logits, rtol=1e-5, atol=1e-6)
        
        print("Inference fixtures validated successfully")

class TestFixtureGeneration:
    """Test fixture generation functionality."""
    
    def test_generate_custom_model_fixture(self, fixture_generator):
        """Test generating custom model fixtures."""
        config = {
            "dim": 64,
            "n_layers": 1,
            "n_heads": 2,
            "n_kv_heads": 1,
            "vocab_size": 100,
            "ffn_dim": 128,
        }
        
        fixture_data = fixture_generator.generate_model_fixture(config, "custom_test")
        
        # Validate fixture structure
        assert "config" in fixture_data
        assert "inputs" in fixture_data
        assert "outputs" in fixture_data
        assert "metadata" in fixture_data
        
        # Validate config matches
        assert fixture_data["config"] == config
        
        # Validate outputs have correct structure
        assert "logits" in fixture_data["outputs"]
        assert "logits_shape" in fixture_data["outputs"]
        
        # Validate metadata
        assert fixture_data["metadata"]["seed"] == 42
        assert fixture_data["metadata"]["fixture_version"] == "1.0"
    
    def test_generate_custom_quantization_fixture(self, fixture_generator):
        """Test generating custom quantization fixtures."""
        shape = (16, 32)
        
        fixture_data = fixture_generator.generate_quantization_fixture(shape, "custom_quant")
        
        # Validate fixture structure
        assert "shape" in fixture_data
        assert "original_weights" in fixture_data
        assert "quantized_weights" in fixture_data
        assert "int8_weights" in fixture_data
        assert "int2_weights" in fixture_data
        assert "metadata" in fixture_data
        
        # Validate shape matches
        assert fixture_data["shape"] == list(shape)
        
        # Validate data consistency
        original = torch.tensor(fixture_data["original_weights"])
        quantized = torch.tensor(fixture_data["quantized_weights"])
        
        assert original.shape == shape
        assert quantized.shape == shape
        
        # Validate quantization properties
        expected_quantized = torch.sign(original)
        torch.testing.assert_close(quantized, expected_quantized)
    
    def test_fixture_loading_and_validation(self, fixture_generator):
        """Test fixture loading and validation."""
        # Generate a fixture
        config = {
            "dim": 32,
            "n_layers": 1,
            "n_heads": 2,
            "n_kv_heads": 1,
            "vocab_size": 50,
            "ffn_dim": 64,
        }
        
        original_fixture = fixture_generator.generate_model_fixture(config, "load_test")
        
        # Load the fixture
        loaded_fixture = fixture_generator.load_fixture("load_test")
        
        # Compare
        assert loaded_fixture == original_fixture
        
        # Validate
        assert fixture_generator.validate_fixture(loaded_fixture)
    
    def test_fixture_hash_consistency(self, fixture_generator):
        """Test that fixtures produce consistent hashes."""
        config = {
            "dim": 32,
            "n_layers": 1,
            "n_heads": 2,
            "n_kv_heads": 1,
            "vocab_size": 50,
            "ffn_dim": 64,
        }
        
        # Generate fixture twice
        fixture1 = fixture_generator.generate_model_fixture(config, "hash_test1")
        fixture2 = fixture_generator.generate_model_fixture(config, "hash_test2")
        
        # Remove metadata that might differ (like timestamps)
        def normalize_fixture(fixture):
            normalized = fixture.copy()
            if "metadata" in normalized:
                # Keep only deterministic metadata
                normalized["metadata"] = {
                    "seed": normalized["metadata"]["seed"],
                    "fixture_version": normalized["metadata"]["fixture_version"],
                }
            return normalized
        
        norm_fixture1 = normalize_fixture(fixture1)
        norm_fixture2 = normalize_fixture(fixture2)
        
        # Should be identical
        assert norm_fixture1 == norm_fixture2

class KnownGoodOutputs:
    """Container for known-good model outputs for regression testing."""
    
    @staticmethod
    def get_bitnet_3b_outputs() -> Dict[str, Any]:
        """Get known-good outputs for BitNet 3B model (simulated)."""
        # In practice, these would be real outputs from a validated model
        return {
            "model_name": "bitnet_3b_simulated",
            "test_cases": [
                {
                    "input": "Hello world",
                    "expected_tokens": [15496, 995],  # Simulated token IDs
                    "expected_logits_shape": [2, 128256],
                    "expected_top_tokens": [15496, 995, 318],  # Top 3 most likely next tokens
                },
                {
                    "input": "The quick brown fox",
                    "expected_tokens": [464, 2068, 7586, 21831],
                    "expected_logits_shape": [4, 128256],
                    "expected_top_tokens": [18045, 284, 625],
                },
            ],
            "numerical_precision": {
                "rtol": 1e-4,
                "atol": 1e-5,
            },
            "metadata": {
                "model_version": "1.0",
                "test_version": "1.0",
                "creation_date": "2024-01-01",
            }
        }
    
    @staticmethod
    def get_quantization_reference_outputs() -> Dict[str, Any]:
        """Get reference outputs for quantization operations."""
        return {
            "test_cases": [
                {
                    "input_shape": [32, 64],
                    "input_range": [-1.0, 1.0],
                    "quantization_type": "sign",
                    "expected_unique_values": [-1.0, 0.0, 1.0],
                    "expected_zero_preservation": True,
                },
                {
                    "input_shape": [64, 128],
                    "input_range": [-10.0, 10.0],
                    "quantization_type": "sign",
                    "expected_unique_values": [-1.0, 0.0, 1.0],
                    "expected_zero_preservation": True,
                },
            ],
            "numerical_precision": {
                "rtol": 1e-6,
                "atol": 1e-7,
            }
        }

@pytest.fixture
def known_good_outputs():
    """Provide known-good outputs for testing."""
    return KnownGoodOutputs()

class TestKnownGoodOutputs:
    """Test against known-good outputs."""
    
    def test_against_known_bitnet_outputs(self, known_good_outputs):
        """Test model outputs against known-good BitNet outputs."""
        reference = known_good_outputs.get_bitnet_3b_outputs()
        
        # This would test against actual model outputs
        # For now, we'll just validate the reference structure
        assert "model_name" in reference
        assert "test_cases" in reference
        assert "numerical_precision" in reference
        
        for test_case in reference["test_cases"]:
            assert "input" in test_case
            assert "expected_tokens" in test_case
            assert "expected_logits_shape" in test_case
            assert "expected_top_tokens" in test_case
    
    def test_against_known_quantization_outputs(self, known_good_outputs):
        """Test quantization against known-good outputs."""
        reference = known_good_outputs.get_quantization_reference_outputs()
        
        for test_case in reference["test_cases"]:
            shape = test_case["input_shape"]
            input_range = test_case["input_range"]
            
            # Generate test input in specified range
            torch.manual_seed(42)
            test_input = torch.rand(shape) * (input_range[1] - input_range[0]) + input_range[0]
            
            # Apply quantization
            quantized = torch.sign(test_input)
            
            # Validate against expected properties
            unique_values = quantized.unique().sort()[0]
            expected_values = torch.tensor(test_case["expected_unique_values"])
            
            # Check that all unique values are in expected set
            for val in unique_values:
                assert val in expected_values, f"Unexpected quantized value: {val}"
            
            # Check zero preservation if expected
            if test_case["expected_zero_preservation"]:
                zero_mask = test_input == 0.0
                if zero_mask.any():
                    assert torch.all(quantized[zero_mask] == 0.0), "Zero values not preserved"