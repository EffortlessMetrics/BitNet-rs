# Phase 2 Timing Build Summary

## Build Status: SUCCESS ✓

### Build Details

**Date**: 2025-10-22T07:01:37Z

**Binary Location**: `/home/steven/code/Rust/BitNet-rs/target/release/bitnet`
- Size: 8.6 MB
- Type: ELF 64-bit LSB pie executable, x86-64
- Status: Stripped (release optimizations applied)

**Build Command**:
```bash
RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
  cargo build --release -p bitnet-cli --no-default-features --features cpu,full-cli
```

**Build Time**: ~1 minute 33 seconds

### Compiler Configuration

- **Rust**: rustc 1.92.0-nightly (4082d6a3f 2025-09-27)
- **Cargo**: cargo 1.92.0-nightly (f2932725b 2025-09-24)
- **Optimization Level**: -C opt-level=3 (maximum optimizations)
- **Target CPU**: native (AMD Ryzen 9 9950X3D with AVX-512 support)
- **LTO**: Not used (incompatible with embed-bitcode settings)

### SIMD Instructions Verified

Binary analysis confirms the following SIMD instructions are present:
- ✓ AVX (vpaddd and related instructions detected)
- ✓ AVX2 (confirmed through instruction patterns)
- ✓ Native CPU features enabled (target-cpu=native)

### Performance Receipt

**Receipt Location**: `/home/steven/code/Rust/BitNet-rs/docs/baselines/perf/phase2_timing_i2s.md`

**Key Results** (median of 3 runs):
- **Total time per token**: 1,950.925 ms (1.95 seconds)
- **Throughput**: 0.5126 tokens/second
- **Performance bottleneck**: Forward pass (95.61% of time)

**Breakdown**:
- Embedding: 26 μs (0.00%)
- Forward Pass: 1,865,375 μs (95.61%)
- Logits: 72,092 μs (3.70%)
- Sampling: 155 μs (0.01%)

### System Configuration

**Hardware**:
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor
- Cores: 16 physical / 32 logical
- L1d cache: 768 KiB
- L1i cache: 512 KiB
- L2 cache: 16 MiB
- L3 cache: 96 MiB

**SIMD Extensions Available**:
- AVX, AVX2
- AVX-512 (F, DQ, IFMA, CD, BW, VL, VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, VP2INTERSECT, BF16)
- FMA, BMI1, BMI2
- SHA-NI, VAES, VPCLMULQDQ

**Operating System**: Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2)

### Model Configuration

**Model**: microsoft-bitnet-b1.58-2B-4T-gguf
- Path: `models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf`
- Size: 1.2 GB
- Format: I2_S (BitNet32-F16 2-bit signed quantization)
- Parameters: ~2B

**Tokenizer**: `models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json`
- Size: 8.7 MB

### Test Configuration

- **Prompt**: "2+2="
- **Max tokens**: 1
- **Sampling**: Greedy (temperature=0.0)
- **Iterations**: 3 (median reported)
- **Environment**: BITNET_TRACE_TIMING=1, RUST_LOG=warn

### Notes

1. **LTO Not Applied**: The original script requested `-C lto=thin`, but this is incompatible with the current build configuration (`-C embed-bitcode=no`). The build proceeded without LTO but with full optimizations (-C opt-level=3).

2. **SIMD Utilization**: AVX/AVX2 instructions are present in the binary. AVX-512 support is available on the CPU but may not be fully utilized in the current implementation.

3. **Performance Baseline**: The ~0.5 tok/s performance establishes a baseline for I2_S CPU inference. The forward pass dominates (95.61%), indicating optimization focus should be on transformer computation kernels.

4. **WSL2 Environment**: Performance measurements are on WSL2, which may have slight overhead compared to native Linux.

5. **Binary Stripping**: The release binary is stripped, reducing size from potentially larger debug builds.

### Next Steps

- Consider SIMD optimization of the forward pass (95.61% bottleneck)
- Investigate AVX-512 utilization for further performance gains
- Profile specific kernel operations within the forward pass
- Compare performance with alternative quantization formats

---

**Generated**: 2025-10-22T07:01:37Z
**Script**: `scripts/perf_phase2_timing.sh` (modified to skip LTO)
**Verification**: Manual execution with pre-built release binary
