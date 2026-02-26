#!/usr/bin/env python3
"""
Performance comparison between BitNet-rs and BitNet.cpp
"""

import subprocess
import time
import json
import os
import statistics
import argparse
from pathlib import Path

# Default values (can be overridden via command line)
DEFAULT_MODEL_PATH = os.environ.get("BITNET_GGUF", "models/bitnet/ggml-model-i2_s.gguf")
DEFAULT_CPP_DIR = os.environ.get("BITNET_CPP_DIR", os.path.expanduser("~/.cache/bitnet_cpp"))
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_ITERATIONS = 3

def run_cpp_benchmark(args):
    """Run the C++ implementation benchmark"""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["GGML_NUM_THREADS"] = "1"

    cpp_bin = f"{args.cpp_dir}/build/bin/llama-cli"
    if not Path(cpp_bin).exists():
        print(f"C++ binary not found at {cpp_bin}")
        print(f"Run 'cargo xtask fetch-cpp' to download and build the C++ implementation")
        return None

    if not Path(args.model).exists():
        print(f"Model not found at {args.model}")
        print(f"Set BITNET_GGUF environment variable or use --model argument")
        return None

    times = []
    for i in range(args.iterations):
        start = time.time()
        try:
            result = subprocess.run(
                [cpp_bin,
                 "-m", args.model,
                 "-p", args.prompt,
                 "-n", str(args.tokens),
                 "-t", "1",
                 "--no-display-prompt"],
                capture_output=True,
                text=True,
                timeout=args.timeout,
                env=env
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  C++ iteration {i+1}: {elapsed:.3f}s")
        except Exception as e:
            print(f"  C++ error: {e}")
            return None

    return {
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "iterations": len(times)
    }

def run_rust_benchmark(args):
    """Run the Rust implementation benchmark"""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"

    if not Path(args.model).exists():
        print(f"Model not found at {args.model}")
        print(f"Set BITNET_GGUF environment variable or use --model argument")
        return None

    # Detect if we're running with GPU features
    features = "gpu" if args.gpu else "cpu"

    times = []
    sample_output = None
    response_correctness = "unknown"

    for i in range(args.iterations):
        start = time.time()
        try:
            result = subprocess.run(
                ["cargo", "run", "--release", "-p", "bitnet-cli",
                 "--no-default-features", "--features", features, "--",
                 "run",
                 "--model", args.model,
                 "--prompt", args.prompt,
                 "--max-new-tokens", str(args.tokens),
                 "--deterministic", "--greedy", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=args.timeout,
                env=env,
                cwd=args.cwd
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Rust iteration {i+1}: {elapsed:.3f}s")

            # Capture output from first successful iteration
            if sample_output is None and result.returncode == 0:
                try:
                    output_json = json.loads(result.stdout)
                    if isinstance(output_json, list) and len(output_json) > 0:
                        sample_output = output_json[0].get("generated_text", "").strip()
                    elif isinstance(output_json, dict):
                        sample_output = output_json.get("generated_text", "").strip()
                    else:
                        sample_output = result.stdout.strip()
                except json.JSONDecodeError:
                    sample_output = result.stdout.strip()

                # Check correctness for known prompts
                if sample_output:
                    prompt_lower = args.prompt.lower()
                    output_lower = sample_output.lower()

                    if "capital of france" in prompt_lower and "paris" in output_lower:
                        response_correctness = "correct"
                    elif "2 + 2" in prompt_lower and "4" in output_lower:
                        response_correctness = "correct"
                    elif "president" in prompt_lower and "washington" in output_lower:
                        response_correctness = "correct"
                    elif "sky" in prompt_lower and "blue" in output_lower:
                        response_correctness = "correct"
                    elif sample_output and not sample_output.lower().startswith("error"):
                        response_correctness = "generated"  # At least something was generated
                    else:
                        response_correctness = "incorrect"

        except Exception as e:
            print(f"  Rust error: {e}")
            return None

    return {
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "iterations": len(times),
        "actual_text_output": sample_output,
        "response_correctness": response_correctness
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Performance comparison between BitNet-rs and BitNet.cpp")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_PATH,
                        help=f"Path to GGUF model file (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--cpp-dir", default=DEFAULT_CPP_DIR,
                        help=f"Path to C++ implementation directory (default: {DEFAULT_CPP_DIR})")
    parser.add_argument("--prompt", "-p", default=DEFAULT_PROMPT,
                        help=f"Prompt to use for inference (default: '{DEFAULT_PROMPT}')")
    parser.add_argument("--tokens", "-n", type=int, default=32,
                        help="Number of tokens to generate (default: 32)")
    parser.add_argument("--iterations", "-i", type=int, default=DEFAULT_ITERATIONS,
                        help=f"Number of benchmark iterations (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--timeout", "-t", type=int, default=60,
                        help="Timeout per iteration in seconds (default: 60)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU features for Rust implementation")
    parser.add_argument("--cwd", default=os.getcwd(),
                        help="Working directory for Rust build (default: current directory)")
    parser.add_argument("--skip-cpp", action="store_true",
                        help="Skip C++ benchmark (Rust only)")
    parser.add_argument("--skip-rust", action="store_true",
                        help="Skip Rust benchmark (C++ only)")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 60)
    print("BitNet-rs vs BitNet.cpp Performance Comparison")
    print("=" * 60)
    print(f"Model: {Path(args.model).name}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Tokens to generate: {args.tokens}")
    print(f"Threads: 1 (for deterministic comparison)")
    print(f"Iterations: {args.iterations}")
    print(f"GPU mode: {'Enabled' if args.gpu else 'Disabled'}")
    print()

    cpp_results = None
    rust_results = None

    if not args.skip_cpp:
        print("Running C++ implementation benchmark...")
        cpp_results = run_cpp_benchmark(args)

    if not args.skip_rust:
        print("\nRunning Rust implementation benchmark...")
        rust_results = run_rust_benchmark(args)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if cpp_results:
        print(f"\nC++ Implementation (BitNet.cpp):")
        print(f"  Mean time:   {cpp_results['mean']:.3f}s")
        print(f"  Std dev:     {cpp_results['stdev']:.3f}s")
        print(f"  Min time:    {cpp_results['min']:.3f}s")
        print(f"  Max time:    {cpp_results['max']:.3f}s")

    if rust_results:
        print(f"\nRust Implementation (BitNet-rs):")
        print(f"  Mean time:   {rust_results['mean']:.3f}s")
        print(f"  Std dev:     {rust_results['stdev']:.3f}s")
        print(f"  Min time:    {rust_results['min']:.3f}s")
        print(f"  Max time:    {rust_results['max']:.3f}s")
        if 'actual_text_output' in rust_results and rust_results['actual_text_output']:
            output = rust_results['actual_text_output']
            if len(output) > 100:
                output = output[:100] + "..."
            print(f"  Sample output: {output}")
        if 'response_correctness' in rust_results:
            correctness = rust_results['response_correctness']
            status_emoji = {"correct": "✓", "generated": "~", "incorrect": "✗", "unknown": "?"}
            print(f"  Correctness: {status_emoji.get(correctness, '?')} {correctness}")

    # Analysis and comparison
    speedup = None
    improvement = None

    if cpp_results and rust_results:
        speedup = cpp_results['mean'] / rust_results['mean']
        improvement = (cpp_results['mean'] - rust_results['mean']) / cpp_results['mean'] * 100

        print(f"\nComparison:")
        print(f"  Speedup:     {speedup:.2f}x")
        if improvement > 0:
            print(f"  Improvement: {improvement:.1f}% faster")
        else:
            print(f"  Difference:  {-improvement:.1f}% slower")

        # Winner determination
        print(f"\n🏆 Winner: ", end="")
        if speedup > 1.05:
            print("BitNet-rs (Rust)")
        elif speedup < 0.95:
            print("BitNet.cpp (C++)")
        else:
            print("TIE (within 5% margin)")

    # Save results with metadata
    results = {
        "metadata": {
            "model": args.model,
            "prompt": args.prompt,
            "tokens": args.tokens,
            "iterations": args.iterations,
            "gpu_mode": args.gpu,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "cpp": cpp_results,
        "rust": rust_results,
        "comparison": {
            "speedup": speedup,
            "improvement_percent": improvement
        }
    }

    output_file = f"benchmark_results_{int(time.time())}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
