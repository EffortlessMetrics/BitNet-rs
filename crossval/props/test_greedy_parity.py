"""
Property-based tests for greedy decoding parity across systems.
Uses Hypothesis to generate adversarial prompts and verify approximate agreement.
"""
import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pytest
from hypothesis import given, settings, HealthCheck, assume, note
import hypothesis.strategies as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crossval.props.run_model import BitNetRunner, LlamaCppRunner, HFRuntimeRunner
from crossval.props.metrics import (
    basic_text_metrics, 
    combined_similarity_score,
    extract_json,
    validate_json_schema,
    relative_metrics
)
from crossval.props.strategies import prompt_strategy


# Environment configuration
BITNET_BIN = os.environ.get("BITNET_BIN", "target/release/bitnet")
BITNET_GGUF = os.environ.get("MODEL_PATH")  # Required
BITNET_TOKENIZER = os.environ.get("TOKENIZER")  # Required
LLAMA_BIN = os.environ.get("LLAMA_BIN")  # Optional
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", BITNET_GGUF)  # Default to same model
HF_MODEL_ID = os.environ.get("HF_MODEL_ID")  # Optional

# Test parameters
MAX_TOKENS = int(os.environ.get("PROP_MAX_NEW_TOKENS", "128"))
NUM_EXAMPLES = int(os.environ.get("PROP_EXAMPLES", "30"))
TIMEOUT = int(os.environ.get("PROP_TIMEOUT", "180"))

# Thresholds (tune based on your model and requirements)
LEV_MAX = int(os.environ.get("PROP_LEV_MAX", "60"))  # Max edit distance
PREF_MIN = int(os.environ.get("PROP_PREFIX_MIN", "10"))  # Min prefix match
F1_MIN = float(os.environ.get("PROP_BIGRAM_F1_MIN", "0.55"))  # Min bigram F1
COMBINED_MIN = float(os.environ.get("PROP_COMBINED_MIN", "0.65"))  # Min combined score

# Artifact saving
SAVE_ARTIFACTS = os.environ.get("PROP_SAVE_ARTIFACTS", "1") == "1"
ARTIFACTS_DIR = Path(os.environ.get("PROP_ARTIFACTS_DIR", "test-artifacts"))


def save_artifact(test_name: str, data: Dict):
    """Save test artifact for debugging and reproduction."""
    if not SAVE_ARTIFACTS:
        return
    
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = ARTIFACTS_DIR / f"{test_name}_{timestamp}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filename


def get_runners() -> Dict[str, any]:
    """Initialize available model runners."""
    runners = {}
    
    # BitNet is required
    if not BITNET_GGUF:
        pytest.skip("MODEL_PATH environment variable not set")
    
    runners["bitnet"] = BitNetRunner(
        BITNET_BIN, 
        BITNET_GGUF, 
        BITNET_TOKENIZER, 
        threads=1
    )
    
    # Optional runners
    if LLAMA_BIN and LLAMA_MODEL:
        runners["llama.cpp"] = LlamaCppRunner(LLAMA_BIN, LLAMA_MODEL, threads=1)
    
    if HF_MODEL_ID:
        runners["hf"] = HFRuntimeRunner(HF_MODEL_ID, device="cpu")
    
    return runners


@pytest.fixture(scope="session")
def runners():
    """Pytest fixture for model runners."""
    return get_runners()


class TestGreedyDeterminism:
    """Test that BitNet produces deterministic outputs in greedy mode."""
    
    @settings(
        max_examples=min(NUM_EXAMPLES, 10),  # Fewer examples for determinism test
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    )
    @given(
        prompt=prompt_strategy(),
        seed=st.integers(min_value=1, max_value=2**31-1)
    )
    def test_bitnet_deterministic(self, prompt, seed, runners):
        """Verify BitNet produces identical outputs for same seed."""
        runner = runners["bitnet"]
        
        # Skip empty prompts for this test
        assume(prompt.strip())
        
        # Run twice with same seed
        result1 = runner.run(prompt, MAX_TOKENS, seed=seed, greedy=True, timeout=TIMEOUT)
        result2 = runner.run(prompt, MAX_TOKENS, seed=seed, greedy=True, timeout=TIMEOUT)
        
        # Should be identical
        if result1.text != result2.text:
            artifact = save_artifact("determinism_failure", {
                "prompt": prompt,
                "seed": seed,
                "run1": result1.text,
                "run2": result2.text,
                "meta1": result1.meta,
                "meta2": result2.meta,
            })
            
            pytest.fail(
                f"BitNet not deterministic for seed={seed}\n"
                f"Prompt: {prompt!r}\n"
                f"Run 1: {result1.text!r}\n"
                f"Run 2: {result2.text!r}\n"
                f"Artifact: {artifact}"
            )


class TestGreedyParity:
    """Test approximate parity between BitNet and reference implementations."""
    
    @settings(
        max_examples=NUM_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    )
    @given(
        prompt=prompt_strategy(),
        seed=st.integers(min_value=1, max_value=2**31-1)
    )
    def test_cross_system_parity(self, prompt, seed, runners):
        """Verify BitNet output approximately matches reference system."""
        
        # Skip if no reference system available
        ref_name = next((n for n in ["llama.cpp", "hf"] if n in runners), None)
        if ref_name is None:
            pytest.skip("No reference system available (set LLAMA_BIN or HF_MODEL_ID)")
        
        # Skip empty prompts
        assume(prompt.strip())
        
        # Run both systems
        note(f"Testing with prompt: {prompt!r}")
        note(f"Seed: {seed}")
        
        bitnet_result = runners["bitnet"].run(
            prompt, MAX_TOKENS, seed=seed, greedy=True, timeout=TIMEOUT
        )
        ref_result = runners[ref_name].run(
            prompt, MAX_TOKENS, seed=seed, greedy=True, timeout=TIMEOUT
        )
        
        # Compute metrics
        metrics = basic_text_metrics(bitnet_result.text, ref_result.text)
        combined = combined_similarity_score(metrics)
        
        # Add relative metrics
        ref_len = max(1, len(ref_result.text.split()))
        rel_metrics = relative_metrics(metrics, ref_len)
        metrics.update(rel_metrics)
        
        # Log for debugging
        note(f"BitNet: {bitnet_result.text!r}")
        note(f"{ref_name}: {ref_result.text!r}")
        note(f"Metrics: {metrics}")
        note(f"Combined score: {combined:.3f}")
        
        # Check thresholds
        failures = []
        
        if metrics["prefix_match"] < PREF_MIN:
            failures.append(
                f"Prefix too short: {metrics['prefix_match']} < {PREF_MIN}"
            )
        
        if metrics["bigram_f1"] < F1_MIN:
            failures.append(
                f"Bigram F1 too low: {metrics['bigram_f1']:.3f} < {F1_MIN}"
            )
        
        # Use relative threshold for longer outputs
        REL_LEV_MAX = float(os.environ.get("PROP_REL_LEV_MAX", "0.55"))
        if metrics["levenshtein"] > LEV_MAX and metrics.get("levenshtein_rel", 1.0) > REL_LEV_MAX:
            failures.append(
                f"Edit distance too large: abs={metrics['levenshtein']} > {LEV_MAX}, "
                f"rel={metrics.get('levenshtein_rel', 1.0):.2f} > {REL_LEV_MAX}"
            )
        
        if combined < COMBINED_MIN:
            failures.append(
                f"Combined score too low: {combined:.3f} < {COMBINED_MIN}"
            )
        
        # JSON validation for JSON tasks
        json_keywords = ["Respond ONLY with JSON", "Return a valid JSON", "Output JSON"]
        if any(kw in prompt for kw in json_keywords):
            bitnet_json = extract_json(bitnet_result.text)
            ref_json = extract_json(ref_result.text)
            
            if bitnet_json is None:
                failures.append(f"BitNet did not produce valid JSON. Output: {bitnet_result.text[:200]}")
            
            if ref_json is None:
                failures.append(f"{ref_name} did not produce valid JSON. Output: {ref_result.text[:200]}")
            
            # Check schema if both produced JSON
            if bitnet_json is not None and ref_json is not None:
                # Check for specific schemas based on prompt
                if '"title" and "items"' in prompt:
                    if not validate_json_schema(bitnet_json, ["title", "items"]):
                        failures.append(f"BitNet JSON missing required keys: {list(bitnet_json.keys())}")
                elif '"answer"' in prompt and '"reason"' in prompt:
                    if not validate_json_schema(bitnet_json, ["answer", "reason"]):
                        failures.append(f"BitNet JSON missing required keys: {list(bitnet_json.keys())}")
                elif '"lang"' in prompt and '"summary"' in prompt:
                    if not validate_json_schema(bitnet_json, ["lang", "summary"]):
                        failures.append(f"BitNet JSON missing required keys: {list(bitnet_json.keys())}")
        
        # Save artifact if failing
        if failures:
            artifact = save_artifact("parity_failure", {
                "prompt": prompt,
                "seed": seed,
                "bitnet": bitnet_result.text,
                ref_name: ref_result.text,
                "metrics": metrics,
                "combined_score": combined,
                "failures": failures,
                "bitnet_meta": bitnet_result.meta,
                f"{ref_name}_meta": ref_result.meta,
            })
            
            pytest.fail(
                f"Parity check failed:\n"
                f"{chr(10).join(failures)}\n"
                f"Prompt: {prompt!r}\n"
                f"BitNet: {bitnet_result.text!r}\n"
                f"{ref_name}: {ref_result.text!r}\n"
                f"Artifact: {artifact}"
            )


class TestEdgeCases:
    """Test handling of edge cases and adversarial inputs."""
    
    def test_empty_prompt(self, runners):
        """Test empty prompt handling."""
        runner = runners["bitnet"]
        result = runner.run("", MAX_TOKENS, seed=42, greedy=True, timeout=TIMEOUT)
        
        # Should handle gracefully (either generate something or return empty)
        assert result is not None
        assert isinstance(result.text, str)
    
    def test_unicode_normalization(self, runners):
        """Test Unicode handling and normalization."""
        runner = runners["bitnet"]
        
        # Various Unicode challenges
        prompts = [
            "café naïve résumé",  # Diacritics
            "🔥💯✨ Test",  # Emoji
            "ﬁle ﬀort",  # Ligatures
            "​Zero​Width​",  # Zero-width spaces
        ]
        
        for prompt in prompts:
            result = runner.run(prompt, 32, seed=42, greedy=True, timeout=TIMEOUT)
            
            # Should handle without crashing
            assert result is not None
            assert isinstance(result.text, str)
    
    def test_very_long_prompt(self, runners):
        """Test handling of very long prompts."""
        runner = runners["bitnet"]
        
        # Create a long but valid prompt
        long_prompt = "Please summarize the following: " + ("word " * 500)
        
        result = runner.run(long_prompt, 32, seed=42, greedy=True, timeout=TIMEOUT)
        
        # Should handle without timeout or crash
        assert result is not None
        assert isinstance(result.text, str)


class TestPerformanceMetrics:
    """Test that performance metrics are properly reported."""
    
    def test_timing_metrics(self, runners):
        """Verify timing metrics are present and reasonable."""
        runner = runners["bitnet"]
        
        result = runner.run(
            "Hello, world!", 
            MAX_TOKENS, 
            seed=42, 
            greedy=True, 
            timeout=TIMEOUT
        )
        
        # Check timing metrics exist
        assert "timing_ms" in result.meta
        timing = result.meta["timing_ms"]
        
        # Should have key phases
        expected_phases = ["tokenize", "prefill", "decode", "total"]
        for phase in expected_phases:
            assert phase in timing, f"Missing timing for {phase}"
            assert timing[phase] >= 0, f"Negative timing for {phase}"
        
        # Total should be sum of phases (approximately)
        sum_phases = timing["tokenize"] + timing["prefill"] + timing["decode"]
        assert abs(timing["total"] - sum_phases) < 100, "Total timing doesn't match sum"
    
    def test_throughput_metrics(self, runners):
        """Verify throughput metrics are calculated correctly."""
        runner = runners["bitnet"]
        
        result = runner.run(
            "Generate some text", 
            MAX_TOKENS, 
            seed=42, 
            greedy=True, 
            timeout=TIMEOUT
        )
        
        # Check throughput metrics
        if "throughput_tps" in result.meta:
            tps = result.meta["throughput_tps"]
            
            # Should have decode throughput at minimum
            assert "decode" in tps
            assert tps["decode"] > 0, "Decode throughput should be positive"
            
            # Sanity check: should be reasonable (1-10000 tokens/sec for CPU)
            assert 0.1 < tps["decode"] < 100000, f"Unrealistic decode TPS: {tps['decode']}"


# Regression test with fixed prompts
class TestRegression:
    """Regression tests with fixed prompts for stability tracking."""
    
    REGRESSION_PROMPTS = [
        "What is 2+2?",
        "Complete this: The quick brown",
        "def fibonacci(n):",
        '{"name": "test", "value":',
        "Translate to French: Hello",
        "List three colors:",
    ]
    
    @pytest.mark.parametrize("prompt", REGRESSION_PROMPTS)
    def test_regression_prompt(self, prompt, runners):
        """Test specific prompts for regression tracking."""
        runner = runners["bitnet"]
        
        # Fixed seed for reproducibility
        result = runner.run(prompt, 64, seed=12345, greedy=True, timeout=TIMEOUT)
        
        # Save for manual inspection
        artifact = save_artifact("regression", {
            "prompt": prompt,
            "output": result.text,
            "meta": result.meta,
        })
        
        # Basic sanity checks
        assert result.text, f"Empty output for prompt: {prompt}"
        assert len(result.text) > 0, f"No generation for prompt: {prompt}"
        
        # Could add specific expected outputs here once stable


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])