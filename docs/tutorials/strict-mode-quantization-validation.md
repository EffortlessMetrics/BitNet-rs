# Tutorial: Getting Started with Strict Mode Quantization Validation

**Learning-oriented** | **Estimated time: 15 minutes** | **Prerequisites: Basic familiarity with bitnet-rs inference**

This tutorial will teach you how to use strict mode in bitnet-rs to ensure your neural network inference uses real quantized computation instead of silently falling back to FP32. By the end, you'll understand when and why to use strict mode, and how to interpret validation results.

## What You'll Learn

- What strict mode is and why it matters for production deployments
- How to enable strict mode for quantization validation
- How to interpret strict mode errors and fix common issues
- How to verify receipts show honest computation paths

## Why Strict Mode Matters

When running 1-bit neural network inference with bitnet-rs, you expect quantized computation (I2S, TL1, TL2) to be used. However, if required kernels are unavailable, the system may silently fall back to FP32 dequantization. This produces correct results but with misleading performance metrics.

**Problem scenario without strict mode:**
```bash
# You run inference expecting GPU-accelerated I2S quantization
cargo run -p xtask -- benchmark --model model.gguf --tokens 128

# Receipt claims: "87.5 tok/s with I2S quantization"
# Reality: Fell back to FP32 CPU, actually ~12 tok/s
# You deploy to production expecting 87.5 tok/s performance...
```

**With strict mode:**
```bash
BITNET_STRICT_MODE=1 \
cargo run -p xtask -- benchmark --model model.gguf --tokens 128

# If fallback would occur: Error immediately with detailed message
# Error: "Strict mode: FP32 fallback rejected - qtype=I2S, device=Cuda(0), reason=kernel_unavailable"
# You fix the issue before production deployment ✓
```

## Step 1: Understanding the Three Validation Tiers

bitnet-rs provides three layers of protection against silent fallbacks:

### Tier 1: Debug Assertions (Development Time)

**When:** Running debug builds (`cargo build` without `--release`)
**What:** Panics immediately if FP32 fallback would occur
**Purpose:** Catch issues early during development

```bash
# Debug build automatically includes assertions
cargo test -p bitnet-inference --no-default-features --features cpu

# If fallback occurs, you'll see:
# thread 'test' panicked at 'fallback to FP32 in debug mode: layer=blk.0.attn_q, qtype=I2S, reason=kernel_unavailable'
```

### Tier 2: Strict Mode Enforcement (Production Time)

**When:** Running release builds with `BITNET_STRICT_MODE=1`
**What:** Returns error instead of falling back to FP32
**Purpose:** Guarantee quantized inference in production

```bash
# Production inference with strict mode
BITNET_STRICT_MODE=1 \
cargo run --release -p xtask -- infer \
  --model model.gguf \
  --prompt "Explain quantum computing"

# If kernel unavailable: Fails with detailed error
# Otherwise: Succeeds with guaranteed quantized computation
```

### Tier 3: Receipt Validation (Verification Time)

**When:** After inference completes
**What:** Validates receipt claims match actual kernels used
**Purpose:** Audit trail for performance baselines

```bash
# Run benchmark
cargo run -p xtask -- benchmark --model model.gguf --tokens 128

# Verify receipt honesty
cargo run -p xtask -- verify-receipt ci/inference.json

# Checks:
# - compute_path="real" matches actual kernel IDs
# - GPU claims require GPU kernel IDs (gemm_*, i2s_gpu_*)
# - CPU claims require CPU kernel IDs (i2s_gemv, tl1_neon_*, tl2_avx_*)
```

## Step 2: Enable Strict Mode for Your First Test

Let's verify your bitnet-rs installation works with strict mode.

### 2.1 Download a Test Model

```bash
# Download a small test model
cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf

# Or use your own model
export BITNET_GGUF=/path/to/your/model.gguf
```

### 2.2 Run Inference with Strict Mode

```bash
# Enable strict mode and run inference
BITNET_STRICT_MODE=1 \
cargo run -p bitnet-cli --no-default-features --features cpu -- \
  infer \
  --model models/bitnet-model.gguf \
  --prompt "The future of AI is" \
  --max-tokens 16
```

**Expected output (success):**
```
Loaded model: bitnet-b1.58-2B (I2S quantized)
Generating 16 tokens...
The future of AI is transformative, enabling new possibilities in healthcare, education, and scientific research.

Performance: 18.5 tok/s (CPU I2S)
Receipt: ci/inference.json (compute_path=real, kernels=["i2s_gemv", "quantized_matmul_i2s"])
✓ Strict mode: Quantized inference validated
```

**Expected output (error - kernel unavailable):**
```
Error: Strict mode: FP32 fallback rejected - qtype=I2S, device=Cpu, layer_dims=[2048, 2048], reason=kernel_unavailable

This means:
- Your binary was not compiled with CPU quantization kernels
- Solution: Rebuild with --features cpu
  cargo build --no-default-features --features cpu
```

## Step 3: Understand Strict Mode Error Messages

Strict mode errors are designed to be actionable. Let's decode a typical error:

```
Error: Strict mode: FP32 fallback rejected - qtype=I2S, device=Cuda(0), layer_dims=[2048, 2048], reason=kernel_unavailable
       ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^  ^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^
       (1)           (2)                     (3)        (4)          (5)                 (6)
```

1. **Strict mode:** Indicates this is a strict mode validation failure
2. **FP32 fallback rejected:** System tried to fall back to FP32 but strict mode prevented it
3. **qtype=I2S:** The quantization type that was attempted (I2S, TL1, or TL2)
4. **device=Cuda(0):** The device where inference was attempted (GPU device 0)
5. **layer_dims=[2048, 2048]:** Layer dimensions (in_features × out_features)
6. **reason=kernel_unavailable:** Why fallback was needed

### Common Reasons and Solutions

| Reason | Meaning | Solution |
|--------|---------|----------|
| `kernel_unavailable` | Feature not compiled | `cargo build --no-default-features --features cpu` or `--features gpu` |
| `device_mismatch` | Tensor on wrong device | Ensure model loaded on same device as inference |
| `unsupported_dimensions` | Layer size not supported | Check model architecture compatibility |
| `gpu_oom` | GPU out of memory | Reduce batch size or use smaller model |

## Step 4: Validate Receipts for Honest Computation

After successful inference, verify the receipt shows real quantized computation:

### 4.1 Run Inference and Generate Receipt

```bash
# Run benchmark to generate receipt
BITNET_STRICT_MODE=1 \
cargo run -p xtask -- benchmark \
  --model models/bitnet-model.gguf \
  --tokens 128

# Receipt written to: ci/inference.json
```

### 4.2 Inspect Receipt Contents

```bash
# View receipt
cat ci/inference.json | jq

# Example output:
{
  "schema_version": "1.0.0",
  "backend": "cpu",
  "compute_path": "real",
  "kernels": [
    "i2s_gemv",           # ← Real quantized kernel
    "quantized_matmul_i2s" # ← Real quantized kernel
  ],
  "tokens_per_second": 18.5,
  "tokens_generated": 128,
  "timestamp": "2025-10-14T12:34:56.789Z"
}
```

### 4.3 Verify Receipt Honesty

```bash
# Automated verification
cargo run -p xtask -- verify-receipt ci/inference.json

# Expected output:
✓ Schema version: 1.0.0 (valid)
✓ Compute path: real (valid)
✓ Backend: cpu (valid)
✓ Kernel validation: 2 quantized kernels detected
  - i2s_gemv (CPU quantized matmul)
  - quantized_matmul_i2s (CPU quantized matmul)
✓ Receipt validation: PASS
```

### 4.4 Detect False Claims

```bash
# If receipt claims "real" but has only fallback kernels:
# {
#   "compute_path": "real",
#   "kernels": ["dequant_fp32", "fp32_matmul"]  # ← Fallback kernels!
# }

# verify-receipt will fail:
✗ Receipt validation: FAIL
Error: Receipt claims compute_path="real" but kernels contain only fallback indicators:
  - dequant_fp32 (FP32 dequantization fallback)
  - fp32_matmul (FP32 fallback computation)

This indicates silent FP32 fallback occurred despite receipt claiming "real" computation.
```

## Step 5: Combine Strict Mode with Deterministic Inference

For maximum reproducibility in testing and cross-validation:

```bash
# Enable strict mode + deterministic inference
export BITNET_STRICT_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

# Run inference twice
cargo run -p bitnet-cli --no-default-features --features cpu -- \
  infer \
  --model models/bitnet-model.gguf \
  --prompt "Test prompt" \
  --max-tokens 16 \
  --seed 42

# Outputs should be:
# 1. Identical across runs (deterministic)
# 2. Using real quantized kernels (strict mode)
# 3. Verified via receipt (honest computation)
```

## Step 6: Test Strict Mode with GPU Acceleration

If you have a CUDA-capable GPU:

```bash
# Build with GPU support
cargo build --no-default-features --release --features gpu

# Run with GPU strict mode
BITNET_STRICT_MODE=1 \
cargo run --release -p bitnet-cli --no-default-features --features gpu -- \
  infer \
  --model models/bitnet-model.gguf \
  --prompt "GPU-accelerated inference test" \
  --max-tokens 32 \
  --device cuda:0

# Expected receipt kernels (GPU):
# {
#   "backend": "cuda",
#   "kernels": [
#     "gemm_fp16",      # ← GPU mixed precision matmul
#     "i2s_gpu_quantize", # ← GPU quantized computation
#     "wmma_matmul"     # ← Tensor Core acceleration
#   ]
# }
```

## Troubleshooting

### Issue: Strict mode fails with "kernel_unavailable"

**Symptom:**
```
Error: Strict mode: FP32 fallback rejected - reason=kernel_unavailable
```

**Solution:**
```bash
# Check your build features
cargo tree --features

# Rebuild with correct features
cargo build --no-default-features --features cpu  # For CPU
cargo build --no-default-features --features gpu  # For GPU

# Verify GPU compilation (if using GPU)
cargo run -p xtask -- preflight
```

### Issue: Receipt shows "mock" kernels

**Symptom:**
```json
{
  "kernels": ["mock_kernel", "test_stub"]
}
```

**Solution:**
```bash
# This indicates test/development mode
# Ensure you're running production inference:

# 1. Use release build
cargo run --release -p bitnet-cli --no-default-features --features cpu

# 2. Disable mock testing flags
unset BITNET_MOCK_INFERENCE
unset BITNET_TEST_MODE
```

### Issue: GPU falls back to CPU despite strict mode

**Symptom:**
```
Expected GPU kernels, got CPU kernels
```

**Solution:**
```bash
# Check GPU availability
nvidia-smi  # Should show your GPU

# Check CUDA toolkit
nvcc --version  # Should show CUDA 11.0+

# Verify GPU detection
cargo run -p xtask -- preflight
# Look for: "✓ GPU: Available (CUDA 12.0, device 0)"

# Check GPU feature compilation
cargo build --no-default-features --features gpu --verbose
# Should see: "Running custom build command for `bitnet-kernels`"
```

## Next Steps

Now that you understand strict mode basics:

- **How-To Guides:**
  - [Running Strict Mode Validation Workflows](../how-to/strict-mode-validation-workflows.md) - Practical workflows for different scenarios
  - [Verifying Receipt Honesty](../how-to/receipt-verification.md) - Deep dive into receipt validation

- **Reference Documentation:**
  - [Strict Mode Environment Variables](../environment-variables.md#strict-mode-variables) - Complete variable reference
  - [Quantization Support](../reference/quantization-support.md#strict-mode-enforcement) - Technical details

- **Explanation:**
  - [Why Strict Mode Exists](../explanation/FEATURES.md#strict-mode) - Design rationale

## Summary

You've learned:

✓ **Three validation tiers:** Debug assertions, strict mode enforcement, receipt validation
✓ **Enable strict mode:** `BITNET_STRICT_MODE=1`
✓ **Interpret errors:** Detailed messages show qtype, device, dimensions, and reason
✓ **Verify receipts:** `cargo run -p xtask -- verify-receipt` checks kernel honesty
✓ **Combine with determinism:** Strict mode + deterministic inference = reproducible validation

Strict mode is your safety net for production deployments, ensuring that performance claims are backed by real quantized computation, not silent FP32 fallbacks.
