"""
Performance benchmarks for BitNet Python baseline.
"""
import pytest
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
import sys

# Import BitNet modules
try:
    from gpu.model import ModelArgs, Transformer, make_cache, BitLinear, BitLinearKernel
    from gpu.generate import FastGen, GenArgs
    from gpu.pack_weight import convert_weight_int8_to_int2
    import gpu.test as bitnet_test
except ImportError as e:
    pytest.skip(f"BitNet modules not available: {e}", allow_module_level=True)

class PerformanceTracker:
    """Track and store performance metrics."""

    def __init__(self):
        self.metrics = {}

    def record_metric(self, name: str, value: float, unit: str = "seconds", metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        self.metrics[name] = {
            "value": value,
            "unit": unit,
            "metadata": metadata or {}
        }

    def get_metric(self, name: str) -> Dict[str, Any]:
        """Get a recorded metric."""
        return self.metrics.get(name)

    def save_to_file(self, filepath: Path):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def load_from_file(self, filepath: Path):
        """Load metrics from JSON file."""
        if filepath.exists():
            with open(filepath, 'r') as f:
                self.metrics = json.load(f)

@pytest.fixture
def performance_tracker():
    """Provide performance tracker."""
    return PerformanceTracker()

class TestModelLoadingPerformance:
    """Test model loading performance."""

    @pytest.mark.slow
    def test_model_initialization_time(self, performance_tracker):
        """Test model initialization time."""
        model_configs = [
            (256, 2, 4, 1000, "small"),
            (512, 6, 8, 5000, "medium"),
            (1024, 12, 16, 10000, "large"),
        ]

        for dim, n_layers, n_heads, vocab_size, size_name in model_configs:
            args = ModelArgs(dim=dim, n_layers=n_layers, n_heads=n_heads, vocab_size=vocab_size)

            # Time model initialization
            start_time = time.time()
            model = Transformer(args)
            end_time = time.time()

            init_time = end_time - start_time
            performance_tracker.record_metric(
                f"model_init_time_{size_name}",
                init_time,
                "seconds",
                {"dim": dim, "n_layers": n_layers, "n_heads": n_heads, "vocab_size": vocab_size}
            )

            # Model initialization should be fast (less than 1 second)
            assert init_time < 1.0, f"Model initialization too slow: {init_time:.4f}s"

            print(f"Model {size_name} initialization: {init_time:.4f}s")

    @pytest.mark.slow
    def test_cache_creation_time(self, performance_tracker, device):
        """Test cache creation time."""
        args = ModelArgs(dim=512, n_layers=6, n_heads=8)

        cache_sizes = [128, 512, 1024, 2048]

        for cache_size in cache_sizes:
            # Time cache creation
            start_time = time.time()
            cache = make_cache(args, cache_size, device=device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()

            creation_time = end_time - start_time
            performance_tracker.record_metric(
                f"cache_creation_time_{cache_size}",
                creation_time,
                "seconds",
                {"cache_size": cache_size, "device": str(device)}
            )

            # Cache creation should be fast
            assert creation_time < 0.1, f"Cache creation too slow: {creation_time:.4f}s"

            print(f"Cache size {cache_size} creation: {creation_time:.4f}s")

class TestInferencePerformance:
    """Test inference performance benchmarks."""

    @pytest.mark.slow
    def test_forward_pass_throughput(self, performance_tracker, device):
        """Test forward pass throughput."""
        args = ModelArgs(dim=512, n_layers=6, n_heads=8, vocab_size=10000)
        model = Transformer(args).to(device)
        model.eval()

        batch_sizes = [1, 2, 4, 8]
        seq_lengths = [64, 128, 256, 512]

        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                cache_len = seq_len + 64

                # Create inputs
                token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
                token_lengths = torch.tensor([seq_len] * batch_size, device=device)
                start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
                cache = make_cache(args, cache_len, device=device)

                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(token_values, token_lengths, start_pos, cache, cache_len)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                # Time forward passes
                num_runs = 20
                start_time = time.time()

                with torch.no_grad():
                    for _ in range(num_runs):
                        _ = model(token_values, token_lengths, start_pos, cache, cache_len)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.time()
                avg_time = (end_time - start_time) / num_runs

                # Calculate throughput metrics
                tokens_per_second = (batch_size * seq_len) / avg_time

                performance_tracker.record_metric(
                    f"forward_throughput_b{batch_size}_s{seq_len}",
                    tokens_per_second,
                    "tokens/second",
                    {
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "avg_time": avg_time,
                        "device": str(device)
                    }
                )

                print(f"Batch {batch_size}, Seq {seq_len}: {tokens_per_second:.1f} tokens/s ({avg_time:.4f}s)")

    @pytest.mark.slow
    def test_single_token_generation_latency(self, performance_tracker, device):
        """Test single token generation latency."""
        args = ModelArgs(dim=512, n_layers=6, n_heads=8, vocab_size=10000)
        model = Transformer(args).to(device)
        model.eval()

        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            seq_len = 1  # Single token generation
            cache_len = 128

            # Create inputs
            token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
            token_lengths = torch.tensor([seq_len] * batch_size, device=device)
            start_pos = torch.tensor([64] * batch_size, dtype=torch.long, device=device)  # Simulate decode phase
            cache = make_cache(args, cache_len, device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(token_values, token_lengths, start_pos, cache, cache_len)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            # Time single token generation
            num_runs = 100
            start_time = time.time()

            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(token_values, token_lengths, start_pos, cache, cache_len)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()
            avg_latency = (end_time - start_time) / num_runs

            performance_tracker.record_metric(
                f"single_token_latency_b{batch_size}",
                avg_latency,
                "seconds",
                {
                    "batch_size": batch_size,
                    "device": str(device)
                }
            )

            print(f"Single token latency (batch {batch_size}): {avg_latency*1000:.2f}ms")

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_memory_bandwidth_utilization(self, performance_tracker, device):
        """Test memory bandwidth utilization."""
        if device.type != 'cuda':
            pytest.skip("Memory bandwidth testing only relevant for CUDA")

        args = ModelArgs(dim=1024, n_layers=12, n_heads=16, vocab_size=20000)
        model = Transformer(args).to(device)
        model.eval()

        batch_size = 8
        seq_len = 256
        cache_len = 512

        # Create inputs
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
        token_lengths = torch.tensor([seq_len] * batch_size, device=device)
        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        cache = make_cache(args, cache_len, device=device)

        # Clear memory stats
        torch.cuda.reset_peak_memory_stats()

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(token_values, token_lengths, start_pos, cache, cache_len)

        torch.cuda.synchronize()

        # Measure memory usage and timing
        start_memory = torch.cuda.memory_allocated()
        start_time = time.time()

        num_runs = 10
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(token_values, token_lengths, start_pos, cache, cache_len)

        torch.cuda.synchronize()
        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated()

        avg_time = (end_time - start_time) / num_runs
        memory_used = peak_memory - start_memory

        performance_tracker.record_metric(
            "memory_bandwidth_test",
            avg_time,
            "seconds",
            {
                "memory_used_mb": memory_used / 1024**2,
                "peak_memory_mb": peak_memory / 1024**2,
                "batch_size": batch_size,
                "seq_len": seq_len
            }
        )

        print(f"Memory bandwidth test: {avg_time:.4f}s, Memory used: {memory_used/1024**2:.1f}MB")

class TestQuantizationPerformance:
    """Test quantization performance benchmarks."""

    @pytest.mark.slow
    def test_bitlinear_vs_linear_performance(self, performance_tracker, device):
        """Compare BitLinear vs standard Linear performance."""
        shapes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
        batch_size = 32

        for in_features, out_features in shapes:
            # Create layers
            linear = torch.nn.Linear(in_features, out_features).to(device)
            bitlinear = BitLinear(in_features, out_features).to(device)

            # Create input
            input_tensor = torch.randn(batch_size, in_features, device=device)

            # Time standard linear
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = linear(input_tensor)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                start_time = time.time()
                for _ in range(100):
                    _ = linear(input_tensor)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                linear_time = (time.time() - start_time) / 100

            # Time BitLinear
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = bitlinear(input_tensor)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                start_time = time.time()
                for _ in range(100):
                    _ = bitlinear(input_tensor)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                bitlinear_time = (time.time() - start_time) / 100

            speedup = linear_time / bitlinear_time if bitlinear_time > 0 else float('inf')

            performance_tracker.record_metric(
                f"bitlinear_speedup_{in_features}x{out_features}",
                speedup,
                "ratio",
                {
                    "linear_time": linear_time,
                    "bitlinear_time": bitlinear_time,
                    "shape": [in_features, out_features],
                    "device": str(device)
                }
            )

            print(f"Shape {in_features}x{out_features}: Linear {linear_time*1000:.2f}ms, "
                  f"BitLinear {bitlinear_time*1000:.2f}ms, Speedup: {speedup:.2f}x")

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_bitlinear_kernel_performance(self, performance_tracker):
        """Test BitLinearKernel performance."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device('cuda')
        shapes = [(256, 256), (512, 512), (1024, 1024)]
        batch_size = 32

        for in_features, out_features in shapes:
            # Create kernel
            kernel = BitLinearKernel(in_features, out_features).to(device)

            # Create input
            input_tensor = torch.randn(batch_size, in_features, device=device)

            # Time kernel
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = kernel(input_tensor)

                torch.cuda.synchronize()

                start_time = time.time()
                for _ in range(100):
                    _ = kernel(input_tensor)

                torch.cuda.synchronize()
                kernel_time = (time.time() - start_time) / 100

            # Calculate throughput
            ops_per_second = (batch_size * in_features * out_features) / kernel_time

            performance_tracker.record_metric(
                f"bitlinear_kernel_throughput_{in_features}x{out_features}",
                ops_per_second,
                "ops/second",
                {
                    "kernel_time": kernel_time,
                    "shape": [in_features, out_features],
                    "batch_size": batch_size
                }
            )

            print(f"BitLinear kernel {in_features}x{out_features}: {kernel_time*1000:.2f}ms, "
                  f"{ops_per_second/1e9:.2f} GOps/s")

    @pytest.mark.slow
    def test_weight_conversion_performance(self, performance_tracker):
        """Test weight conversion performance."""
        shapes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]

        for M, K in shapes:
            # Create test weight
            weight_int8 = torch.randint(-1, 2, (M, K), dtype=torch.int8)

            # Time conversion
            start_time = time.time()
            for _ in range(10):
                _ = convert_weight_int8_to_int2(weight_int8)
            conversion_time = (time.time() - start_time) / 10

            # Calculate throughput
            elements_per_second = (M * K) / conversion_time

            performance_tracker.record_metric(
                f"weight_conversion_throughput_{M}x{K}",
                elements_per_second,
                "elements/second",
                {
                    "conversion_time": conversion_time,
                    "shape": [M, K]
                }
            )

            print(f"Weight conversion {M}x{K}: {conversion_time*1000:.2f}ms, "
                  f"{elements_per_second/1e6:.1f} M elements/s")

class TestEndToEndPerformance:
    """Test end-to-end performance scenarios."""

    @pytest.mark.slow
    def test_text_generation_throughput(self, performance_tracker, device):
        """Test text generation throughput."""
        args = ModelArgs(dim=512, n_layers=6, n_heads=8, vocab_size=10000)
        model = Transformer(args).to(device)
        model.eval()

        batch_sizes = [1, 4, 8]
        prompt_lengths = [64, 128, 256]
        generation_lengths = [32, 64, 128]

        for batch_size in batch_sizes:
            for prompt_len in prompt_lengths:
                for gen_len in generation_lengths:
                    cache_len = prompt_len + gen_len + 64

                    # Create prompt
                    prompt_tokens = torch.randint(0, args.vocab_size, (batch_size * prompt_len,), device=device)

                    # Simulate text generation
                    start_time = time.time()

                    with torch.no_grad():
                        # Prefill phase
                        token_lengths = torch.tensor([prompt_len] * batch_size, device=device)
                        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
                        cache = make_cache(args, cache_len, device=device)

                        logits = model(prompt_tokens, token_lengths, start_pos, cache, cache_len)
                        next_token = torch.argmax(logits[-batch_size:], dim=-1)

                        # Decode phase
                        for step in range(gen_len):
                            decode_lengths = torch.tensor([1] * batch_size, device=device)
                            decode_start_pos = torch.tensor([prompt_len + step] * batch_size, dtype=torch.long, device=device)

                            logits = model(next_token, decode_lengths, decode_start_pos, cache, cache_len)
                            next_token = torch.argmax(logits, dim=-1)

                    if device.type == 'cuda':
                        torch.cuda.synchronize()

                    total_time = time.time() - start_time
                    tokens_generated = batch_size * gen_len
                    tokens_per_second = tokens_generated / total_time

                    performance_tracker.record_metric(
                        f"generation_throughput_b{batch_size}_p{prompt_len}_g{gen_len}",
                        tokens_per_second,
                        "tokens/second",
                        {
                            "batch_size": batch_size,
                            "prompt_len": prompt_len,
                            "gen_len": gen_len,
                            "total_time": total_time,
                            "device": str(device)
                        }
                    )

                    print(f"Generation B{batch_size} P{prompt_len} G{gen_len}: "
                          f"{tokens_per_second:.1f} tokens/s ({total_time:.2f}s)")

    @pytest.mark.slow
    def test_concurrent_inference_performance(self, performance_tracker, device):
        """Test concurrent inference performance."""
        args = ModelArgs(dim=256, n_layers=4, n_heads=8, vocab_size=5000)
        model = Transformer(args).to(device)
        model.eval()

        # Simulate multiple concurrent requests
        num_requests = 8
        seq_len = 64
        cache_len = 128

        # Create multiple request inputs
        requests = []
        caches = []
        for i in range(num_requests):
            token_values = torch.randint(0, args.vocab_size, (1 * seq_len,), device=device)
            token_lengths = torch.tensor([seq_len], device=device)
            start_pos = torch.zeros(1, dtype=torch.long, device=device)
            cache = make_cache(args, cache_len, device=device)

            requests.append((token_values, token_lengths, start_pos))
            caches.append(cache)

        # Time sequential processing
        start_time = time.time()
        with torch.no_grad():
            for i, (token_values, token_lengths, start_pos) in enumerate(requests):
                _ = model(token_values, token_lengths, start_pos, caches[i], cache_len)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        sequential_time = time.time() - start_time

        # Time batch processing (simulate concurrent)
        batch_tokens = torch.cat([req[0] for req in requests])
        batch_lengths = torch.cat([req[1] for req in requests])
        batch_start_pos = torch.cat([req[2] for req in requests])
        batch_cache = make_cache(args, cache_len * num_requests, device=device)

        start_time = time.time()
        with torch.no_grad():
            _ = model(batch_tokens, batch_lengths, batch_start_pos, batch_cache, cache_len * num_requests)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        batch_time = time.time() - start_time

        efficiency = sequential_time / batch_time if batch_time > 0 else float('inf')

        performance_tracker.record_metric(
            f"concurrent_inference_efficiency_{num_requests}",
            efficiency,
            "ratio",
            {
                "num_requests": num_requests,
                "sequential_time": sequential_time,
                "batch_time": batch_time,
                "device": str(device)
            }
        )

        print(f"Concurrent inference ({num_requests} requests): "
              f"Sequential {sequential_time:.3f}s, Batch {batch_time:.3f}s, "
              f"Efficiency: {efficiency:.2f}x")

@pytest.mark.slow
class TestPerformanceRegression:
    """Test for performance regressions."""

    def test_performance_regression_detection(self, performance_tracker, test_config):
        """Test performance regression detection."""
        # This would compare against baseline performance metrics
        # For now, we'll create a simple example

        baseline_file = Path("tests/baseline_performance.json")
        current_metrics = performance_tracker.metrics

        if baseline_file.exists():
            baseline_tracker = PerformanceTracker()
            baseline_tracker.load_from_file(baseline_file)
            baseline_metrics = baseline_tracker.metrics

            regression_threshold = test_config["performance_tolerance"]["max_regression_percent"] / 100

            for metric_name in current_metrics:
                if metric_name in baseline_metrics:
                    current_value = current_metrics[metric_name]["value"]
                    baseline_value = baseline_metrics[metric_name]["value"]

                    # For throughput metrics (higher is better)
                    if "throughput" in metric_name or "speedup" in metric_name:
                        regression = (baseline_value - current_value) / baseline_value
                    else:
                        # For latency metrics (lower is better)
                        regression = (current_value - baseline_value) / baseline_value

                    if regression > regression_threshold:
                        pytest.fail(f"Performance regression detected in {metric_name}: "
                                  f"{regression*100:.1f}% (threshold: {regression_threshold*100:.1f}%)")

                    print(f"{metric_name}: {regression*100:+.1f}% vs baseline")
        else:
            print("No baseline performance file found, creating new baseline")
            performance_tracker.save_to_file(baseline_file)

def pytest_runtest_teardown(item):
    """Clean up after each test."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
