# Slow Inference Troubleshooting Guide

**Having slow inference performance with BitNet.rs?** This guide helps you diagnose and resolve performance issues across different quantization formats.

## Quick Diagnosis

### Check Your Quantization Format

First, identify which quantization format your model uses:

```bash
# Inspect GGUF metadata to check quantization format
cargo run -p bitnet-cli --features cpu,full-cli -- compat-check model.gguf --verbose

# Look for "Quantization Type" in output
# Common formats: I2_S (BitNet32-F16 or QK256), TL1, TL2, IQ2_S
```

### Expected Performance by Format

| Quantization Format | CPU Performance | Status | Use Case |
|---------------------|-----------------|--------|----------|
| **I2_S BitNet32-F16** | 10-20 tok/s | ✅ Production | Recommended for all use cases |
| **I2_S QK256 (GGML)** | ~0.1 tok/s | ⚠️ MVP Scalar | Validation only (NOT production) |
| **TL1 (Table Lookup)** | 8-15 tok/s | 🚧 Experimental | Research/testing |
| **TL2 (Table Lookup)** | 8-15 tok/s | 🚧 Experimental | Research/testing |

**Hardware reference:** Intel Core i7-12700K, 32GB RAM

## QK256 Slow Performance (Most Common Issue)

### Symptoms

- Inference generates ~0.1 tokens/second
- Each token takes 10+ seconds
- 128 tokens takes 20+ minutes
- Model file size matches QK256 characteristics (microsoft/bitnet-b1.58-2B-4T-gguf)

### Why Is QK256 Slow?

**QK256 currently uses scalar (non-SIMD) kernels for MVP correctness validation.**

This is **expected behavior**, not a bug:

- **Scalar implementation:** No SIMD vectorization (AVX2/AVX-512/NEON)
- **Correctness-first:** MVP focused on numerical accuracy before optimization
- **Foundation complete:** AVX2 base established (~1.2× uplift observed)
- **Roadmap:** v0.2.0 targets ≥3× with nibble-LUT + FMA tiling optimizations

### Immediate Workarounds

#### Option 1: Limit Token Generation (Recommended for QK256)

```bash
# ✅ Quick validation with QK256 (4-16 tokens)
RUST_LOG=warn cargo run -p bitnet-cli --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --prompt "What is 2+2?" \
  --max-tokens 8  # Keep this small

# ❌ Avoid long generations (128+ tokens)
# This will take 20+ minutes with QK256 scalar kernels
```

#### Option 2: Use I2_S BitNet32-F16 Format (10-20× Faster)

**Convert QK256 model to BitNet32-F16 for production use:**

```bash
# Convert model (if conversion tool available)
# Note: Direct QK256 → BitNet32-F16 conversion not yet implemented
# Alternative: Use BitNet32-F16 models directly from HuggingFace

# Download BitNet32-F16 model instead
cargo run -p xtask -- download-model --id <bitnet32-model-repo>

# Run with production performance
RUST_LOG=warn cargo run -p bitnet-cli --features cpu,full-cli -- run \
  --model models/bitnet32-model.gguf \
  --prompt "Production workload" \
  --max-tokens 128  # Much faster
```

#### Option 3: Wait for v0.2.0 SIMD Optimizations

**QK256 performance roadmap:**

- **v0.1.0 (current):** ~0.1 tok/s (scalar baseline)
- **v0.2.0 (planned):** ~0.3+ tok/s (≥3× with AVX2 SIMD)
- **v0.3.0+ (future):** Further optimizations (AVX-512, GPU support)

**Optimizations planned for v0.2.0:**

| Optimization | Expected Impact |
|--------------|-----------------|
| Nibble-LUT unpack (`pshufb`) | 1.5-2× |
| FMA tiling (8-16 rows) | 1.5-2× |
| Load combine (reduce AVX crossings) | 1.2-1.3× |
| Prefetch (code + input) | 1.1-1.2× |
| **Combined Target** | **≥3×** |

### QK256 Performance Validation

**Verify your QK256 performance is within expected range:**

```bash
# Run deterministic benchmark (8 tokens)
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
  cargo run -p bitnet-cli --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --prompt "Test" \
  --max-tokens 8 \
  --greedy

# Expected: ~80 seconds for 8 tokens (~0.1 tok/s)
```

**If performance is SLOWER than expected:**

1. Check CPU frequency scaling (performance vs powersave mode)
2. Ensure no background processes consuming CPU
3. Verify build with optimizations: `cargo build --release --features cpu`
4. Check for thermal throttling: `sensors` (Linux) or activity monitor

## General Performance Troubleshooting

### CPU Optimization Not Applied

#### Symptoms

- Performance slower than baseline (10-20 tok/s for BitNet32-F16)
- Receipt shows no native CPU features enabled
- Release build not used

#### Solutions

```bash
# 1. Build with native CPU optimizations
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features cpu,full-cli

# 2. Verify CPU features detected
cargo run -p xtask -- preflight

# 3. Run with optimized binary
RAYON_NUM_THREADS=$(nproc) RUST_LOG=warn \
  cargo run --release -p bitnet-cli --features cpu,full-cli -- run \
  --model model.gguf \
  --prompt "Test" \
  --max-tokens 32

# 4. Benchmark to verify improvements
cargo run -p xtask -- benchmark --model model.gguf --tokens 128
```

**Expected optimization impact:**

- Native CPU features: +20-30% throughput
- Link-time optimization: +10-15% throughput
- Release build: +50-100% vs debug

### Single-Threaded Execution

#### Symptoms

- CPU usage shows only one core active
- Performance scales poorly with token count
- `RAYON_NUM_THREADS=1` set in environment

#### Solutions

```bash
# 1. Remove single-threading constraint
unset RAYON_NUM_THREADS

# 2. Explicitly set to all cores
export RAYON_NUM_THREADS=$(nproc)

# 3. Verify parallel execution
RUST_LOG=warn cargo run --release -p bitnet-cli --features cpu,full-cli -- run \
  --model model.gguf \
  --prompt "Test" \
  --max-tokens 32

# Monitor CPU usage: should see multiple cores active
htop  # or top/Activity Monitor
```

**Note:** Single-threading (`RAYON_NUM_THREADS=1`) is required for deterministic output in tests. Remove for production inference.

### Debug Build Instead of Release

#### Symptoms

- Extremely slow performance (10-100× slower than expected)
- No optimizations applied
- Build directory is `target/debug/` not `target/release/`

#### Solutions

```bash
# Always use --release flag for performance testing
cargo build --release --no-default-features --features cpu,full-cli

# Run release binary
cargo run --release -p bitnet-cli --features cpu,full-cli -- run \
  --model model.gguf \
  --prompt "Test" \
  --max-tokens 32

# Verify release binary location
ls -lh target/release/bitnet-cli
```

### Memory Mapping Issues

#### Symptoms

- High memory usage
- Slow model loading
- Page faults during inference

#### Solutions

```bash
# 1. Enable memory mapping (usually automatic for GGUF)
cargo run -p bitnet-cli --features cpu,full-cli -- run \
  --model model.gguf \
  --mmap \  # Explicit mmap flag
  --prompt "Test" \
  --max-tokens 32

# 2. Check available RAM
free -h

# 3. Monitor memory usage during inference
RUST_LOG=debug cargo run -p bitnet-cli --features cpu,full-cli -- run \
  --model model.gguf \
  --prompt "Test" \
  --max-tokens 8 2>&1 | grep -i "memory"

# 4. Use smaller batch sizes if memory-constrained
cargo run -p bitnet-cli --features cpu,full-cli -- run \
  --model model.gguf \
  --batch-size 8 \
  --prompt "Test" \
  --max-tokens 32
```

## GPU Performance Issues

### GPU Not Being Used

#### Symptoms

- Receipt shows `backend: "cpu"` instead of `"cuda"`
- No GPU kernels in kernel list
- nvidia-smi shows 0% GPU utilization

#### Diagnosis

```bash
# 1. Check GPU compilation
cargo run -p xtask -- preflight

# 2. Verify GPU build
cargo build --release --no-default-features --features gpu

# 3. Check CUDA availability
nvidia-smi
nvcc --version

# 4. Run with explicit GPU flag
RUST_LOG=warn cargo run --release -p bitnet-cli --features gpu,full-cli -- run \
  --model model.gguf \
  --device cuda \
  --prompt "Test" \
  --max-tokens 32
```

#### Solutions

If GPU not available:

```bash
# Install CUDA toolkit
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Rebuild with GPU support
cargo clean
cargo build --release --no-default-features --features gpu
```

### GPU Memory Exhaustion

#### Symptoms

- CUDA out of memory errors
- Performance degrades during long generations
- nvidia-smi shows GPU memory at 100%

#### Solutions

```bash
# 1. Reduce batch size
cargo run -p bitnet-cli --features gpu,full-cli -- run \
  --model model.gguf \
  --device cuda \
  --batch-size 4 \
  --prompt "Test" \
  --max-tokens 32

# 2. Use smaller model
# Download smaller variant (e.g., 1B instead of 2B)

# 3. Clear GPU memory
nvidia-smi --gpu-reset

# 4. Monitor GPU memory usage
watch -n 1 nvidia-smi
```

## Alternative Quantization Formats

If QK256 performance is blocking your use case, consider alternative formats:

### I2_S BitNet32-F16 (Recommended)

**Best for:** Production inference, interactive applications

```bash
# Download BitNet32-F16 model
cargo run -p xtask -- download-model --id <bitnet32-repo>

# Run inference
cargo run -p bitnet-cli --features cpu,full-cli -- run \
  --model models/bitnet32-model.gguf \
  --prompt "Test" \
  --max-tokens 128  # 10-20 tok/s
```

**Performance:**
- CPU: 10-20 tok/s (2B models)
- GPU: 50-100 tok/s (2B models)
- Status: ✅ Production-ready

### TL1/TL2 (Experimental)

**Best for:** Research, experimentation

```bash
# Build with TL1/TL2 support
cargo build --release --no-default-features --features cpu

# Run TL1 inference
cargo run -p bitnet-cli --features cpu,full-cli -- run \
  --model models/tl1-model.gguf \
  --prompt "Test" \
  --max-tokens 64  # 8-15 tok/s
```

**Performance:**
- CPU: 8-15 tok/s (depends on architecture)
- Status: 🚧 Experimental

## Performance Benchmarking

### Measure Your Hardware

```bash
# Comprehensive benchmark
cargo run -p xtask -- benchmark --model model.gguf --tokens 128

# View receipt with performance metrics
cat ci/inference.json | jq '{
  throughput: .throughput_tokens_per_sec,
  backend: .backend,
  quantization: .quantization,
  kernels: .kernels
}'

# Compare against baselines
# - I2_S BitNet32-F16: 10-20 tok/s (CPU), 50-100 tok/s (GPU)
# - I2_S QK256: ~0.1 tok/s (CPU scalar, MVP)
# - TL1/TL2: 8-15 tok/s (CPU experimental)
```

### Optimization Checklist

- [ ] Build with `--release` flag
- [ ] Enable native CPU features: `RUSTFLAGS="-C target-cpu=native"`
- [ ] Use link-time optimization: `-C lto=thin`
- [ ] Set `RAYON_NUM_THREADS=$(nproc)` for parallel execution
- [ ] Use I2_S BitNet32-F16 format (not QK256) for production
- [ ] Limit QK256 to `--max-tokens 4-16` for validation only
- [ ] Check CPU frequency scaling (performance mode)
- [ ] Verify no thermal throttling
- [ ] Monitor with `htop`/`nvidia-smi` during inference

## Migration Path: QK256 → Production

### Short-term (v0.1.0)

1. **Validation only:** Use QK256 with `--max-tokens 4-16`
2. **Production:** Switch to I2_S BitNet32-F16 format (10-20× faster)
3. **Testing:** Use limited token budgets for CI/CD validation

### Medium-term (v0.2.0)

1. **QK256 SIMD:** ≥3× performance improvement with AVX2
2. **Production-ready QK256:** 0.3+ tok/s for 2B models
3. **Cross-format conversion:** QK256 ↔ BitNet32-F16 tools

### Long-term (v0.3.0+)

1. **AVX-512 support:** Additional 1.5-2× over AVX2
2. **GPU QK256:** 50-100× over scalar CPU
3. **Multi-threading:** Linear scaling with cores

## Getting Help

### Collect Performance Diagnostics

```bash
# 1. System information
cargo run -p bitnet-cli --features cpu,full-cli -- inspect --system > perf_debug.txt

# 2. Model information
cargo run -p bitnet-cli --features cpu,full-cli -- compat-check model.gguf >> perf_debug.txt

# 3. Performance baseline
cargo run -p xtask -- benchmark --model model.gguf --tokens 16
cat ci/inference.json >> perf_debug.txt

# 4. CPU/GPU features
cargo run -p xtask -- preflight >> perf_debug.txt

# 5. Environment
env | grep -E "(RUST|RAYON|BITNET|CUDA)" >> perf_debug.txt
```

### Report Performance Issues

When reporting performance issues, include:

1. **Quantization format:** (I2_S BitNet32-F16, QK256, TL1, TL2)
2. **Hardware:** CPU model, GPU model, RAM
3. **Performance metrics:** Measured tok/s, expected tok/s
4. **Build configuration:** Release/debug, features, RUSTFLAGS
5. **Performance diagnostics:** Output from commands above
6. **Receipt artifact:** `ci/inference.json` contents

**GitHub Issues:** https://github.com/microsoft/BitNet/issues

## Summary

**Key takeaways:**

1. **QK256 is slow by design** (~0.1 tok/s) in v0.1.0 MVP
2. **Use I2_S BitNet32-F16** for production (10-20 tok/s)
3. **Limit QK256 to 4-16 tokens** for quick validation
4. **v0.2.0 will bring ≥3× QK256 improvement** with AVX2 SIMD
5. **Always build with `--release`** and native CPU features
6. **This is expected behavior, not a bug**

For more details, see:
- [Performance Benchmarking Guide](../performance-benchmarking.md)
- [QK256 Usage Guide](../howto/use-qk256-models.md)
- [Troubleshooting Guide](troubleshooting.md)
