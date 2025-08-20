#!/usr/bin/env python3
"""
Performance comparison between BitNet.rs and BitNet.cpp
"""

import subprocess
import time
import json
import os
import statistics
from pathlib import Path

MODEL_PATH = "/home/steven/code/Rust/BitNet-rs/models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"
CPP_DIR = "/home/steven/.cache/bitnet_cpp"
PROMPT = "The capital of France is"
ITERATIONS = 3

def run_cpp_benchmark():
    """Run the C++ implementation benchmark"""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["GGML_NUM_THREADS"] = "1"
    
    cpp_bin = f"{CPP_DIR}/build/bin/llama-cli"
    if not Path(cpp_bin).exists():
        print(f"C++ binary not found at {cpp_bin}")
        return None
    
    times = []
    for i in range(ITERATIONS):
        start = time.time()
        try:
            result = subprocess.run(
                [cpp_bin, 
                 "-m", MODEL_PATH,
                 "-p", PROMPT,
                 "-n", "32",
                 "-t", "1",
                 "--no-display-prompt"],
                capture_output=True,
                text=True,
                timeout=30,
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

def run_rust_benchmark():
    """Run the Rust implementation benchmark"""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    
    times = []
    for i in range(ITERATIONS):
        start = time.time()
        try:
            result = subprocess.run(
                ["cargo", "run", "--release", "-p", "bitnet-cli", 
                 "--no-default-features", "--features", "cpu", "--",
                 "inference",
                 "-m", MODEL_PATH,
                 "-p", PROMPT,
                 "-n", "32"],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                cwd="/home/steven/code/Rust/BitNet-rs"
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Rust iteration {i+1}: {elapsed:.3f}s")
        except Exception as e:
            print(f"  Rust error: {e}")
            return None
    
    return {
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "iterations": len(times)
    }

def main():
    print("=" * 60)
    print("BitNet.rs vs BitNet.cpp Performance Comparison")
    print("=" * 60)
    print(f"Model: {Path(MODEL_PATH).name}")
    print(f"Prompt: '{PROMPT}'")
    print(f"Tokens to generate: 32")
    print(f"Threads: 1 (for deterministic comparison)")
    print(f"Iterations: {ITERATIONS}")
    print()
    
    print("Running C++ implementation benchmark...")
    cpp_results = run_cpp_benchmark()
    
    print("\nRunning Rust implementation benchmark...")
    rust_results = run_rust_benchmark()
    
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
        print(f"\nRust Implementation (BitNet.rs):")
        print(f"  Mean time:   {rust_results['mean']:.3f}s")
        print(f"  Std dev:     {rust_results['stdev']:.3f}s")
        print(f"  Min time:    {rust_results['min']:.3f}s")
        print(f"  Max time:    {rust_results['max']:.3f}s")
    
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
            print("BitNet.rs (Rust)")
        elif speedup < 0.95:
            print("BitNet.cpp (C++)")
        else:
            print("TIE (within 5% margin)")
    
    # Save results
    results = {
        "cpp": cpp_results,
        "rust": rust_results,
        "comparison": {
            "speedup": speedup if cpp_results and rust_results else None,
            "improvement_percent": improvement if cpp_results and rust_results else None
        }
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmark_results.json")

if __name__ == "__main__":
    main()