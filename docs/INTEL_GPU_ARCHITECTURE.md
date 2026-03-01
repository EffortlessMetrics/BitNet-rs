# Intel GPU Backend Architecture

## Overview

BitNet-rs supports Intel Arc GPUs (A770, A750, A580) via OpenCL for compute acceleration.
This document describes the architecture of the Intel GPU backend.

## Architecture Diagram

```
┌──────────────────────────────────────────┐
│           bitnet-inference               │
│  ┌─────────────────────────────────────┐ │
│  │     InferenceEngine                 │ │
│  │  ┌──────────┐  ┌─────────────────┐ │ │
│  │  │ CPU Path │  │   GPU Path      │ │ │
│  │  │ (SIMD)   │  │  ┌───────────┐  │ │ │
│  │  │          │  │  │ CUDA      │  │ │ │
│  │  │          │  │  │ OpenCL    │  │ │ │
│  │  │          │  │  │ Vulkan    │  │ │ │
│  │  │          │  │  └───────────┘  │ │ │
│  │  └──────────┘  └─────────────────┘ │ │
│  └─────────────────────────────────────┘ │
└──────────────────────────────────────────┘
         │                    │
    ┌────▼────┐         ┌────▼──────────┐
    │ bitnet- │         │  bitnet-      │
    │ kernels │         │  kernels      │
    │ (CPU)   │         │  (OpenCL)     │
    │         │         │               │
    │ AVX2    │         │ matmul.cl     │
    │ AVX-512 │         │ softmax.cl    │
    │ NEON    │         │ layer_norm.cl │
    │         │         │ rope.cl       │
    │         │         │ elementwise.cl│
    │         │         │ quantized.cl  │
    └─────────┘         └───────────────┘
```

## Backend Selection

The backend is selected at runtime with this priority:
1. **CUDA** — NVIDIA GPUs (via cudarc)
2. **OpenCL** — Intel Arc / AMD GPUs (via opencl3 or dynamic loading)
3. **CPU** — SIMD-optimized fallback (AVX2/AVX-512/NEON)

### Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `BITNET_GPU_FAKE` | `opencl`, `intel`, `cuda`, `none` | Simulate GPU detection for testing |
| `BITNET_STRICT_MODE` | `1` | Disable BITNET_GPU_FAKE, require real hardware |
| `BITNET_DEVICE` | `cpu`, `cuda:0`, `opencl:0` | Force specific device |

## OpenCL Kernel Design

### Kernel Categories

1. **Matrix Operations** — matmul (naive, tiled 16×16, batched)
2. **Normalization** — LayerNorm, RMSNorm
3. **Attention** — softmax with temperature scaling
4. **Position Encoding** — RoPE (Rotary Position Embedding)
5. **Activations** — SiLU, GELU (approximate), ReLU
6. **Quantization** — I2_S dequantization, quantized matrix-vector multiply

### Intel Xe-HPG Optimization Notes

- A770 has 32 Xe-cores, 512 EUs
- 16 GB GDDR6 @ 560 GB/s bandwidth
- Prefer tile sizes that are multiples of 16 (SIMD lane width)
- Use `__local` memory for tiled matmul
- Use barrier synchronization for shared memory access
- Prefer `float` over `double` (2:1 FP32:FP64 ratio)

## Testing Strategy

### CPU Reference Tests

Every OpenCL kernel has a corresponding CPU reference implementation in
`opencl_reference.rs` that implements the exact same algorithm. Tests verify:
- Numerical accuracy (within floating-point tolerance)
- Edge cases (empty inputs, single element, very large values)
- Property tests (softmax sums to 1, layer norm mean ≈ 0)

### Hardware Tests

When Intel GPU hardware is available:
- Build with `--features opencl`
- Run `BITNET_GPU_FAKE=opencl cargo test` for simulated testing
- Run `cargo test` on actual A770 hardware for integration testing

## Performance Profiling

Use these tools on Intel Arc:
- `intel_gpu_top` — real-time GPU utilization
- `clinfo` — OpenCL device capabilities
- OpenCL event timing — per-kernel execution time
- `RenderDoc` — compute shader debugging (Vulkan path)

## Related Docs

- [`docs/GPU_SETUP.md`](GPU_SETUP.md) — CUDA setup guide
- [`docs/GPU_SETUP_INTEL.md`](GPU_SETUP_INTEL.md) — Intel Arc setup guide (PR #1686)
- [`docs/architecture-overview.md`](architecture-overview.md) — Overall system architecture
