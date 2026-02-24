# How-To: Running Strict Mode Validation Workflows

**Problem-oriented** | **Goal: Validate quantized inference in different deployment scenarios**

This guide shows you how to run strict mode validation workflows for common BitNet.rs deployment scenarios. Each workflow ensures your inference uses real quantized computation without silent FP32 fallbacks.

## Prerequisites

- BitNet.rs installed with appropriate features (`cpu` or `gpu`)
- A valid GGUF model file
- Understanding of strict mode basics (see [Tutorial](../tutorials/strict-mode-quantization-validation.md))

## Workflow 1: Validate CPU Inference Pipeline

**Use case:** Verify CPU inference uses real I2S/TL1/TL2 quantization before production deployment.

### Step-by-Step

```bash
# 1. Build with CPU features
cargo build --no-default-features --release --features cpu

# 2. Set strict mode environment
export BITNET_STRICT_MODE=1

# 3. Run inference test
cargo run --release -p bitnet-cli --no-default-features --features cpu -- \
  infer \
  --model models/bitnet-model.gguf \
  --prompt "Test quantization path" \
  --max-tokens 32

# 4. Verify receipt
cargo run -p xtask -- verify-receipt ci/inference.json

# 5. Check kernel IDs
cat ci/inference.json | jq '.kernels[]'
# Expected: "i2s_gemv", "quantized_matmul_i2s", "tl1_neon_matmul", or "tl2_avx_matmul"
```

### Success Criteria

✓ Inference completes without strict mode errors
✓ Receipt shows `compute_path="real"`
✓ Kernels array contains quantized kernel IDs (not fallback IDs)
✓ Performance matches baseline (I2S: 10-20 tok/s, TL1: 12-18 tok/s, TL2: 10-15 tok/s)

### Common Issues

**Issue: "kernel_unavailable" error**
```bash
# Check architecture-specific kernels
# ARM: Expects TL1 with NEON
# x86: Expects TL2 with AVX2/AVX-512
uname -m  # Check architecture

# Verify SIMD features detected
cargo run -p xtask -- preflight
```

**Issue: Kernels show "scalar_fallback"**
```bash
# SIMD not available - rebuild with explicit SIMD features
export RUSTFLAGS="-C target-cpu=native"
cargo build --no-default-features --release --features cpu
```

## Workflow 2: Validate GPU Inference Pipeline

**Use case:** Verify GPU inference uses real GPU kernels (not CPU fallback) before production.

### Step-by-Step

```bash
# 1. Check GPU availability
nvidia-smi  # Should show your GPU
nvcc --version  # CUDA 11.0+ required

# 2. Build with GPU features
cargo build --no-default-features --release --features gpu

# 3. Verify GPU compilation
cargo run -p xtask -- preflight
# Expected: "✓ GPU: Available (CUDA 12.0, device 0)"

# 4. Run GPU inference with strict mode
BITNET_STRICT_MODE=1 \
cargo run --release -p bitnet-cli --no-default-features --features gpu -- \
  infer \
  --model models/bitnet-model.gguf \
  --prompt "GPU quantization test" \
  --max-tokens 64 \
  --device cuda:0

# 5. Verify GPU kernels used
cat ci/inference.json | jq '.backend, .kernels[]'
# Expected backend: "cuda"
# Expected kernels: "gemm_fp16", "i2s_gpu_quantize", "wmma_matmul"

# 6. Verify GPU kernel enforcement
cargo run -p xtask -- verify-receipt --require-gpu-kernels ci/inference.json
```

### Success Criteria

✓ Inference runs on GPU (not silently falling back to CPU)
✓ Receipt shows `backend="cuda"`
✓ Kernels include GPU prefixes: `gemm_*`, `i2s_gpu_*`, `wmma_*`
✓ Performance matches GPU baseline (50-100 tok/s with mixed precision)

### Common Issues

**Issue: Falls back to CPU despite --device cuda:0**
```bash
# Check GPU memory
nvidia-smi  # Look at "Memory-Usage"

# Reduce model size or batch size
# Or use smaller model for testing

# Check CUDA in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Issue: "GPU compiled but not available at runtime"**
```bash
# Verify CUDA runtime libraries
ldd target/release/bitnet-cli | grep cuda
# Should show libcuda.so, libcudart.so

# Install CUDA runtime if missing
# Ubuntu: sudo apt install nvidia-cuda-toolkit
# Fedora: sudo dnf install cuda
```

## Workflow 3: CI/CD Integration for Strict Mode Validation

**Use case:** Ensure all CI builds pass strict mode validation before merge.

### GitHub Actions Example

```yaml
# .github/workflows/strict-mode-validation.yml
name: Strict Mode Validation

on:
  pull_request:
    paths:
      - 'crates/**'
      - 'Cargo.toml'
      - 'Cargo.lock'

jobs:
  validate-cpu-strict-mode:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: actions-rust-lang/setup-rust-toolchain@v1
        with:
          toolchain: stable

      - name: Build with CPU features
        run: cargo build --no-default-features --release --features cpu

      - name: Run strict mode tests
        env:
          BITNET_STRICT_MODE: "1"
          BITNET_DETERMINISTIC: "1"
          BITNET_SEED: "42"
          RAYON_NUM_THREADS: "1"
        run: |
          cargo test --no-default-features --features cpu \
            -p bitnet-inference \
            test_ac5_16_token_decode_cpu_strict_mode

      - name: Verify receipts
        run: |
          for receipt in ci/inference-*.json; do
            cargo run -p xtask -- verify-receipt "$receipt"
          done

      - name: Check for fallback indicators
        run: |
          # Fail if any receipt contains fallback kernels
          if grep -r "dequant_fp32\|fp32_matmul\|fallback_" ci/*.json; then
            echo "ERROR: Fallback kernels detected in receipts"
            exit 1
          fi

  validate-gpu-strict-mode:
    runs-on: [self-hosted, gpu]  # Requires GPU runner
    steps:
      - uses: actions/checkout@v4

      - name: Check GPU availability
        run: nvidia-smi

      - name: Build with GPU features
        run: cargo build --no-default-features --release --features gpu

      - name: Run GPU strict mode tests
        env:
          BITNET_STRICT_MODE: "1"
          BITNET_DETERMINISTIC: "1"
          BITNET_SEED: "42"
        run: |
          cargo test --no-default-features --features gpu \
            -p bitnet-inference \
            test_ac5_16_token_decode_gpu_strict_mode

      - name: Verify GPU kernels used
        run: |
          cargo run -p xtask -- verify-receipt --require-gpu-kernels ci/inference.json
```

### Success Criteria

✓ CI passes for both CPU and GPU strict mode tests
✓ No fallback kernels in any receipt
✓ All receipts pass `verify-receipt` validation
✓ Performance matches established baselines

## Workflow 4: Cross-Validation with Strict Mode

**Use case:** Validate Rust implementation against C++ reference with strict quantization.

### Step-by-Step

```bash
# 1. Enable strict mode + deterministic inference
export BITNET_STRICT_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

# 2. Set model path for cross-validation
export BITNET_GGUF="models/bitnet-model.gguf"

# 3. Run cross-validation
cargo run -p xtask -- crossval

# Expected output:
# ✓ Model loaded: bitnet-b1.58-2B (I2S)
# ✓ Strict mode: Enabled (enforce_quantized_inference=true)
# ✓ Deterministic: Enabled (seed=42)
# ✓ Running cross-validation...
#   - Rust inference: 16 tokens in 0.87s (18.4 tok/s)
#   - C++ reference: 16 tokens in 0.89s (18.0 tok/s)
# ✓ Token-level accuracy: 100% match (16/16 tokens)
# ✓ Quantization accuracy: I2S 99.82% correlation with FP32 (target: ≥99.8%)
# ✓ Receipt validation: compute_path=real, kernels=["i2s_gemv"]
```

### Success Criteria

✓ Rust and C++ produce identical token sequences
✓ Quantization accuracy meets targets (I2S ≥99.8%, TL1/TL2 ≥99.6%)
✓ Strict mode passes without errors
✓ Receipts show real quantized computation

### Common Issues

**Issue: Token mismatch between Rust and C++**
```bash
# This usually indicates fallback in one implementation
# Check receipts from both:

# Rust receipt
cat ci/inference.json | jq '.kernels'

# C++ receipt (if available)
cat ci/cpp-inference.json | jq '.kernels'

# Verify both use same quantization path
```

## Workflow 5: Performance Baseline Establishment

**Use case:** Establish performance baselines with strict mode to ensure future regressions are detected.

### Step-by-Step

```bash
# 1. Create baseline script
cat > scripts/establish-baseline.sh << 'EOF'
#!/bin/bash
set -euo pipefail

export BITNET_STRICT_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

MODEL="${1:-models/bitnet-model.gguf}"
RUNS="${2:-10}"

echo "Establishing baseline for: $MODEL"
echo "Runs: $RUNS"

for i in $(seq 1 $RUNS); do
  echo "Run $i/$RUNS..."
  cargo run -p xtask -- benchmark \
    --model "$MODEL" \
    --tokens 128 \
    --quiet
done

# Aggregate results
jq -s '[.[] | {
  tps: .tokens_per_second,
  kernels: .kernels,
  compute_path: .compute_path
}]' ci/inference-*.json > ci/baseline-results.json

# Calculate statistics
jq '[.[] | .tps] | {
  mean: (add / length),
  min: min,
  max: max,
  std_dev: (
    (map(. - (add / length)) | map(. * .) | add / length) | sqrt
  )
}' ci/baseline-results.json > ci/baseline-stats.json

echo ""
echo "Baseline statistics:"
cat ci/baseline-stats.json | jq
EOF

chmod +x scripts/establish-baseline.sh

# 2. Run baseline establishment
./scripts/establish-baseline.sh models/bitnet-model.gguf 10

# 3. Save baseline for CI
cp ci/baseline-stats.json ci/baselines/cpu-i2s-baseline.json

# 4. Commit baseline to repository
git add ci/baselines/cpu-i2s-baseline.json
git commit -m "chore: establish CPU I2S performance baseline"
```

### Success Criteria

✓ All 10 runs complete without strict mode errors
✓ Performance variance <5% (indicates deterministic behavior)
✓ All receipts show real quantized kernels
✓ Baseline saved for future regression detection

## Workflow 6: Debugging Fallback Scenarios

**Use case:** Systematically diagnose why strict mode is failing.

### Step-by-Step

```bash
# 1. Enable verbose logging
export RUST_LOG=debug,bitnet_inference=trace
export BITNET_STRICT_MODE=1

# 2. Run inference with detailed logging
cargo run -p bitnet-cli --no-default-features --features cpu -- \
  infer \
  --model models/bitnet-model.gguf \
  --prompt "Debug test" \
  --max-tokens 16 2>&1 | tee strict-mode-debug.log

# 3. Search for fallback triggers
grep -i "fallback\|kernel.*unavailable\|fp32" strict-mode-debug.log

# 4. Check layer-specific issues
grep -i "layer.*failed\|quantization.*error" strict-mode-debug.log

# 5. Verify feature compilation
cargo tree --no-default-features --features cpu -e features

# 6. Test specific quantization types
cargo test --no-default-features --features cpu \
  -p bitnet-quantization \
  test_ac3_strict_mode_rejects_fallback -- --nocapture
```

### Common Diagnostic Patterns

**Pattern 1: Architecture mismatch**
```bash
# Log shows: "TL1 kernel unavailable on x86 architecture"
# Solution: TL1 is ARM-specific; use TL2 for x86

# Check model quantization type
cargo run -p bitnet-cli -- inspect model.gguf | grep "quantization"
```

**Pattern 2: Missing SIMD features**
```bash
# Log shows: "AVX2 not available, falling back to scalar"
# Solution: Rebuild with target-cpu=native or check CPU capabilities

cat /proc/cpuinfo | grep -i "avx2"  # x86
cat /proc/cpuinfo | grep -i "neon"  # ARM
```

**Pattern 3: GPU OOM (Out of Memory)**
```bash
# Log shows: "CUDA out of memory, attempting CPU fallback"
# Solution: Reduce batch size or model size

nvidia-smi  # Check GPU memory usage
```

## Workflow 7: Granular Strict Mode Control

**Use case:** Enable specific strict mode checks while allowing others to pass.

### Configuration Matrix

```bash
# Full strict mode (all checks)
export BITNET_STRICT_MODE=1

# Granular control
export BITNET_STRICT_FAIL_ON_MOCK=1              # Fail on mock computation
export BITNET_STRICT_REQUIRE_QUANTIZATION=1     # Require real quantization
export BITNET_STRICT_VALIDATE_PERFORMANCE=1     # Validate performance metrics

# Example: Require quantization but allow mock testing
unset BITNET_STRICT_FAIL_ON_MOCK
export BITNET_STRICT_REQUIRE_QUANTIZATION=1
```

### Use Cases

**Scenario 1: Integration testing with mock components**
```bash
# Allow mock tokenizers but require real quantization
export BITNET_STRICT_REQUIRE_QUANTIZATION=1
unset BITNET_STRICT_FAIL_ON_MOCK

cargo test -p bitnet-inference --features cpu integration_test_with_mock_tokenizer
```

**Scenario 2: Performance validation only**
```bash
# Skip quantization checks, focus on performance metrics
export BITNET_STRICT_VALIDATE_PERFORMANCE=1
unset BITNET_STRICT_REQUIRE_QUANTIZATION

cargo run -p xtask -- benchmark --model model.gguf --tokens 256
# Fails if tokens_per_second > 150 (suspicious, likely mock)
```

## Best Practices

1. **Always use strict mode in CI/CD:**
   ```yaml
   env:
     BITNET_STRICT_MODE: "1"
   ```

2. **Combine with deterministic inference for reproducibility:**
   ```bash
   export BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42
   ```

3. **Save receipts for audit trail:**
   ```bash
   # Unique receipt per run
   TIMESTAMP=$(date +%Y%m%d-%H%M%S)
   cargo run -p xtask -- benchmark --model model.gguf
   cp ci/inference.json ci/receipts/inference-$TIMESTAMP.json
   ```

4. **Document baseline expectations:**
   ```yaml
   # ci/baselines/README.md
   CPU I2S: 10-20 tok/s (AVX2), 15-25 tok/s (AVX-512)
   GPU I2S: 50-100 tok/s (mixed precision)
   TL1: 12-18 tok/s (ARM NEON)
   TL2: 10-15 tok/s (x86 AVX)
   ```

## Related Documentation

- **Tutorial:** [Getting Started with Strict Mode](../tutorials/strict-mode-quantization-validation.md)
- **How-To:** [Verifying Receipt Honesty](./receipt-verification.md)
- **Reference:** [Strict Mode Environment Variables](../environment-variables.md#strict-mode-variables)
- **Reference:** [Quantization Support](../reference/quantization-support.md#strict-mode-enforcement)
- **Explanation:** [Why Strict Mode Exists](../explanation/FEATURES.md#strict-mode)

## Summary

This guide covered seven practical workflows for strict mode validation:

✓ **CPU validation:** Ensure real I2S/TL1/TL2 quantization on CPU
✓ **GPU validation:** Verify GPU kernels (not CPU fallback)
✓ **CI/CD integration:** Automate strict mode checks in pipelines
✓ **Cross-validation:** Compare Rust and C++ with strict enforcement
✓ **Baseline establishment:** Create performance baselines with strict mode
✓ **Debugging:** Systematically diagnose fallback scenarios
✓ **Granular control:** Use specific strict mode checks as needed

All workflows ensure honest computation paths and reliable performance baselines for production deployments.
