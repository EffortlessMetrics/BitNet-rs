# GPU Performance Expectations

Expected inference throughput targets and memory requirements for BitNet-rs
GPU backends. These are **targets and estimates** — actual performance depends
on model quantization format, batch size, prompt length, and driver version.

> **Current status (v0.1.0-qna-mvp)**: CUDA is the production GPU backend.
> QK256 scalar kernels are ~0.1 tok/s (MVP limitation). I2_S BitNet32-F16
> format is 10–20× faster. See [COMPATIBILITY.md](../COMPATIBILITY.md) and
> [GPU_COMPATIBILITY_MATRIX.md](GPU_COMPATIBILITY_MATRIX.md) for backend status.

## Performance Targets by GPU Tier

### BitNet 1.58B-2B Models (I2_S BitNet32-F16)

| Tier | Example GPUs | Target Throughput | Target Latency | Notes |
|------|-------------|-------------------|----------------|-------|
| **High-end** | RTX 4090, A100, Arc A770 | ≥ 50 tok/s | < 20 ms/tok | Tensor Cores / high EU count |
| **Mid-range** | RTX 4060, RTX 3070, Arc A750 | ≥ 25 tok/s | < 40 ms/tok | Good balance of compute and VRAM |
| **Entry** | GTX 1060 6 GB, Arc A380 | ≥ 8 tok/s | < 125 ms/tok | VRAM-constrained |
| **CPU baseline** | 8-core x86_64 (AVX2) | ≥ 10 tok/s | < 100 ms/tok | Reference baseline |

### QK256 Format (Current MVP — Scalar Kernels)

| Tier | Example GPUs | Current Throughput | Notes |
|------|-------------|-------------------|-------|
| All GPU tiers | Any supported | ~0.1 tok/s | Scalar kernels only (MVP) |
| CPU (AVX2) | 8-core x86_64 | ~0.1 tok/s | Scalar path; AVX2 nibble-LUT in v0.2 |

**QK256 is not production-ready.** The v0.2.0 release targets ≥ 3× improvement
with AVX2 nibble-LUT + FMA tiling dequantization. Limit to `--max-tokens 4-16`
for validation only.

### Projected 7B+ Models (Future)

| Tier | Example GPUs | Projected Throughput | Notes |
|------|-------------|---------------------|-------|
| **High-end** | RTX 4090 (24 GB), A100 (80 GB) | 15–30 tok/s | VRAM sufficient |
| **Mid-range** | RTX 4070 Ti (12 GB) | 8–15 tok/s | May require offloading |
| **Entry** | RTX 3060 (12 GB) | 3–8 tok/s | Partial GPU offload |

## Memory Requirements

### GPU VRAM by Model Size

| Model Size | Weights (1.58-bit) | KV Cache (seq 2048) | Activation Overhead | **Total Estimate** |
|-----------|-------------------|---------------------|--------------------|--------------------|
| 1.58B–2B | ~0.4 GB | ~0.3 GB | ~0.3 GB | **~1.0 GB** |
| 3B | ~0.6 GB | ~0.4 GB | ~0.5 GB | **~1.5 GB** |
| 7B | ~1.4 GB | ~0.8 GB | ~0.8 GB | **~3.0 GB** |
| 13B | ~2.6 GB | ~1.2 GB | ~1.2 GB | **~5.0 GB** |

1.58-bit quantization dramatically reduces weight memory compared to FP16/FP32.
The dominant memory consumers at larger scales are KV cache and activations.

### Recommended GPU VRAM

| VRAM | Suitable For |
|------|-------------|
| 6 GB | 2B models, short contexts (≤ 1024 tokens) |
| 8 GB | 2B models, standard contexts (≤ 2048 tokens) |
| 12 GB | 7B models, short contexts |
| 16 GB | 7B models, standard contexts |
| 24 GB+ | 13B+ models or long contexts (≥ 4096 tokens) |

## Batch Size Impact on Throughput

Batching multiple requests improves GPU utilization. The effect is most
pronounced on high-end GPUs with spare compute capacity.

| Batch Size | Relative Throughput (High-end) | Relative Throughput (Entry) |
|-----------|-------------------------------|----------------------------|
| 1 | 1.0× (baseline) | 1.0× (baseline) |
| 2 | ~1.6× | ~1.3× |
| 4 | ~2.5× | ~1.5× |
| 8 | ~3.5× | ~1.6× |
| 16 | ~4.0× | ~1.7× (VRAM-limited) |

Diminishing returns appear earlier on entry GPUs due to VRAM and
memory-bandwidth constraints. The BitNet-rs server (`bitnet-server`)
supports dynamic batching via its concurrency manager.

## Comparison: GPU vs CPU Baseline

For the 2B I2_S BitNet32-F16 model:

| Platform | Throughput | Relative | Power Draw |
|----------|-----------|----------|------------|
| CPU 8-core AVX2 | ~10 tok/s | 1.0× | ~65 W (TDP) |
| CPU 8-core AVX-512 | ~15 tok/s | 1.5× | ~80 W (TDP) |
| RTX 3060 12 GB | ~20 tok/s | 2.0× | ~170 W |
| RTX 4060 8 GB | ~30 tok/s | 3.0× | ~115 W |
| RTX 4090 24 GB | ~55 tok/s | 5.5× | ~350 W |
| Arc A770 16 GB | ~25 tok/s | 2.5× | ~225 W |
| Arc A750 8 GB | ~20 tok/s | 2.0× | ~225 W |

> **Note**: These are projected targets. 1.58-bit models are extremely
> memory-bandwidth-bound; GPU advantages come primarily from higher
> memory bandwidth rather than raw compute throughput.

## Performance Tuning Tips

### CUDA

1. **Use native target-cpu** for the CPU portions of the pipeline:
   ```bash
   RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
     cargo build --release --no-default-features --features gpu,full-cli
   ```

2. **Pin GPU clock frequency** for consistent benchmarks:
   ```bash
   nvidia-smi -lgc 1800,1800   # Lock clocks
   nvidia-smi -rgc              # Reset after benchmarking
   ```

3. **Monitor utilization**:
   ```bash
   nvidia-smi dmon -s pucvmet -d 1   # Real-time monitoring
   ```

### Intel Arc (OpenCL)

1. **Ensure PCIe x16 link** — reduced link width halves bandwidth:
   ```bash
   lspci -vv | grep -A 20 "VGA.*Intel"   # Check LnkSta width
   ```

2. **Monitor GPU utilization**:
   ```bash
   sudo intel_gpu_top
   ```

3. **Use latest Compute Runtime** — each release includes kernel
   compiler improvements that can affect throughput.

### General

- Set `RUST_LOG=warn` to reduce logging overhead during inference.
- Use `--max-tokens` to bound generation length for benchmarking.
- The `BackendStartupSummary` log line confirms which backend was
  actually selected — verify it matches your expectation.
- Receipt files (`ci/inference.json`) capture benchmark results with
  `compute_path: "real"` to prevent mock data from polluting metrics.

## Benchmarking

BitNet-rs includes Criterion benchmarks for kernel-level regression
detection:

```bash
# Run all kernel benchmarks
cargo bench -p bitnet-kernels

# Run SRP microcrate benchmarks (logits, top-k, RoPE, KV cache)
cargo bench --bench srp_ops
```

See [performance-benchmarking.md](performance-benchmarking.md) for
the full benchmarking methodology and [performance-tracking.md](performance-tracking.md)
for CI-integrated tracking.

## Related Documentation

- [GPU_COMPATIBILITY_MATRIX.md](GPU_COMPATIBILITY_MATRIX.md) — Hardware and feature support
- [performance-guide.md](performance-guide.md) — General optimization guide
- [cuda-configuration-guide.md](cuda-configuration-guide.md) — CUDA tuning
- [INTEL_GPU_SETUP.md](INTEL_GPU_SETUP.md) — Intel Arc setup
- [gpu-kernel-architecture.md](gpu-kernel-architecture.md) — Kernel design
