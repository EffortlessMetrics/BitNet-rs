# GPU Architecture Guide

This document provides a comprehensive overview of the BitNet-rs multi-GPU
backend system: how backends are selected, which kernels run on which device,
and how memory flows through the inference pipeline.

> **See also:**
> - [GPU Setup (CUDA)](GPU_SETUP.md) — driver & toolchain installation
> - [Intel GPU Setup](INTEL_GPU_SETUP.md) — Intel Arc via OpenCL/oneAPI
> - [CUDA Configuration Guide](cuda-configuration-guide.md) — memory pools & tuning
> - [GPU Kernel Architecture](gpu-kernel-architecture.md) — low-level kernel design

---

## Overview

BitNet-rs supports multiple GPU backends through a **two-level Hardware
Abstraction Layer (HAL)** implemented entirely via Rust traits and Cargo feature
gates — there are no separate per-backend crates. All GPU code lives in
`bitnet-kernels` (kernel providers) and `bitnet-inference` (orchestration
backends).

### Backend Status

| Backend | API / Crate | Feature Flag | Status | Notes |
|---------|-------------|-------------|--------|-------|
| CUDA | cudarc 0.17 | `gpu` / `cuda` | **Production** | Primary GPU path; PTX kernels |
| ROCm | HIP | `rocm` | **Weekly CI** | AMD GPUs; compile + smoke tests |
| OpenCL (Intel) | opencl3 | `oneapi` | **Alpha** | Intel Arc A-series via Compute Runtime |
| Metal | NPU backend | `metal` | **Experimental** | macOS Apple Silicon; NPU fallback |
| Vulkan | — | `vulkan` | **Feature-gated** | Maps to `gpu`; no separate kernels yet |
| Level Zero | — | — | **Planned** | Intel discrete GPUs; future optimization |

### Feature Flags

GPU features are defined in the workspace root `Cargo.toml`:

```toml
# GPU umbrella (enables CUDA path)
gpu = ["kernels", "inference", "tokenizers",
       "bitnet-kernels/gpu", "bitnet-inference/gpu", "bitnet-quantization/gpu"]
cuda = ["gpu"]          # backward-compat alias

# Alternative backends
rocm    = ["kernels", "inference", "tokenizers",
           "bitnet-kernels/rocm", "bitnet-inference/rocm", "bitnet-quantization/rocm"]
vulkan  = ["kernels", "inference", "tokenizers",
           "bitnet-kernels/vulkan", "bitnet-inference/gpu", "bitnet-quantization/gpu"]
metal   = ["kernels", "inference", "tokenizers",
           "bitnet-common/metal", "bitnet-inference/metal"]
npu     = ["dep:bitnet-device-probe", "bitnet-device-probe/npu"]

# CPU SIMD (always available as fallback)
avx2    = ["bitnet-kernels/avx2"]
avx512  = ["bitnet-kernels/avx512"]
neon    = ["bitnet-kernels/neon"]
```

> **Convention:** GPU code paths must use the **unified predicate** —
> `#[cfg(any(feature = "gpu", feature = "cuda"))]` — never
> `#[cfg(feature = "cuda")]` alone.  Runtime checks use
> `bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime}`.

---

## Two-Level HAL

### Level 1: Kernel Provider (`bitnet-kernels`)

Low-level compute operations. Every backend that can run math implements this
trait:

```rust
// crates/bitnet-kernels/src/lib.rs
pub trait KernelProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn is_available(&self) -> bool;
    fn matmul_i2s(
        &self, a: &[i8], b: &[u8], c: &mut [f32],
        m: usize, n: usize, k: usize,
    ) -> Result<()>;
    fn quantize(
        &self, input: &[f32], output: &mut [u8],
        scales: &mut [f32], qtype: QuantizationType,
    ) -> Result<()>;
}
```

**Implementations:**

| Provider | Crate location | Selection priority |
|----------|---------------|-------------------|
| `CudaKernel` | `bitnet-kernels/src/gpu/cuda.rs` | 1 (highest) |
| ROCm kernel | `bitnet-kernels` (feature-gated) | 2 |
| NPU (Metal) | `bitnet-kernels` (feature-gated) | 3 |
| AVX-512 | `bitnet-kernels/src/cpu/` | 4 |
| AVX2 | `bitnet-kernels/src/cpu/` | 5 |
| NEON (ARM) | `bitnet-kernels/src/cpu/` | 6 |
| FFI bridge | `bitnet-kernels/src/ffi/` | 7 |
| `FallbackKernel` | `bitnet-kernels/src/cpu/` | 8 (always available) |

The **`KernelManager`** holds a `Vec<Box<dyn KernelProvider>>`, selects the
best available provider at startup via `OnceLock<usize>`, and caches the
selection. Call `KernelManager::select_best()` to get the active provider, or
`list_available_providers()` to enumerate all.

### Level 2: Inference Backend (`bitnet-inference`)

Orchestrates full model layers using the kernel provider:

```rust
// crates/bitnet-inference/src/backends.rs
#[async_trait]
pub trait Backend: Send + Sync {
    fn backend_type(&self) -> String;
    fn clone_backend(&self) -> Box<dyn Backend>;
    async fn forward(
        &self, input: &ConcreteTensor, cache: &mut KVCache,
    ) -> Result<ConcreteTensor>;
    fn capabilities(&self) -> BackendCapabilities { /* default */ }
    async fn warmup(&self) -> Result<()> { /* default */ }
}
```

**Concrete backends:**

| Backend | Description | Max batch |
|---------|-------------|-----------|
| `CpuBackend` | Multi-threaded CPU, memory-efficient | 8 |
| `GpuBackend` | CUDA/Metal mixed precision | 32 |
| `NpuBackend` | Apple Neural Engine with CPU fallback | 16 |

---

## Backend Selection Flow

### Request → Detection → Selection

```
CLI / env-var / code
        │
        ▼
  BackendRequest        ← Auto | Cpu | Gpu | Cuda | Hip | OneApi
        │
        ▼
  KernelManager         ← probes providers in priority order
        │
        ▼
  BackendStartupSummary ← "requested=auto detected=[cuda,cpu-rust] selected=cuda"
        │
        ▼
  Backend trait impl    ← CpuBackend / GpuBackend / NpuBackend
```

**Backend request sources (highest to lowest priority):**

1. **CLI flag:** `--device {cpu|cuda|gpu|vulkan|opencl|ocl|npu|auto}`
2. **Environment:** `BITNET_DEVICE=cuda`
3. **Default:** `Auto` — tries GPU first, falls back to best CPU SIMD

The selection logic lives in `bitnet-common/src/backend_selection.rs`:

```rust
pub enum BackendRequest {
    Auto,     // Automatically select best available
    Cpu,      // Force CPU
    Gpu,      // Require GPU (error if unavailable)
    Cuda,     // Require CUDA specifically
    Hip,      // Require AMD HIP specifically
    OneApi,   // Require Intel oneAPI specifically
}
```

### Startup Summary

At boot, the runtime logs a structured summary:

```
requested=auto detected=[cuda,cpu-rust] selected=cuda
```

This is captured in `BackendStartupSummary` and persisted into inference
receipts for auditability.

---

## Kernel Inventory

### CUDA Kernels (`crates/bitnet-kernels/src/cuda/`)

| Kernel | File | Description |
|--------|------|-------------|
| `bitnet_matmul_i2s` | `gpu/cuda.rs` | Quantized int8 × uint8 → f32 matmul |
| `bitnet_quantize_i2s` | `gpu/cuda.rs` | Float → 2-bit I2S quantization |
| `bitnet_quantize_tl1` | `gpu/cuda.rs` | Float → ternary TL1 quantization |
| `bitnet_quantize_tl2` | `gpu/cuda.rs` | Float → ternary TL2 quantization |
| `qk256_gemv` | `cuda/qk256_gemv.rs` | QK256 2-bit dequant + GEMV |
| `attention` | `cuda/attention.rs` | Scaled dot-product attention (causal) |
| `rmsnorm` | `cuda/rmsnorm.rs` | RMS normalization (warp reduction) |

### Mixed Precision Kernels (CC ≥ 6.0)

| Kernel | Precision | Min CC | Description |
|--------|-----------|--------|-------------|
| `bitnet_matmul_fp16` | FP16 | 6.1 | Half-precision matmul |
| `bitnet_matmul_bf16` | BF16 | 8.0 | BFloat16 matmul |
| `bitnet_matmul_tensor_core` | FP16 (WMMA) | 7.0 | Tensor Core matmul |
| `convert_fp32_to_fp16` | — | 6.1 | Precision conversion |
| `convert_fp32_to_bf16` | — | 8.0 | Precision conversion |

### CPU Fallback Coverage

| Operation | GPU (CUDA) | CPU SIMD | Generic fallback |
|-----------|-----------|----------|------------------|
| matmul_i2s | ✅ | ✅ AVX2/AVX-512 | ✅ |
| quantize (I2S/TL1/TL2) | ✅ | ✅ | ✅ |
| attention (fused SDPA) | ✅ | via Candle | ✅ |
| rmsnorm | ✅ | ✅ | ✅ |
| RoPE | ❌ (CPU) | ✅ | ✅ |
| softmax | ⚠️ (in attention) | ✅ | ✅ |
| elementwise (SiLU, etc.) | ❌ (CPU) | ✅ | ✅ |

---

## Data Flow

### End-to-End Inference

```
Token IDs
  │
  ▼
Embedding lookup
  │
  ▼
┌──────────────────── Transformer Layer × N ─────────────────────┐
│                                                                 │
│  RMSNorm ──► Attention ──► Residual Add                        │
│              │                                                  │
│              ├─ Q/K/V projection (matmul_i2s)                  │
│              ├─ RoPE (rotary position embeddings)              │
│              ├─ Scaled dot-product attention                   │
│              └─ Output projection (matmul_i2s)                 │
│                                                                 │
│  RMSNorm ──► Feed-Forward Network ──► Residual Add             │
│              │                                                  │
│              ├─ Gate + Up projection (matmul_i2s)              │
│              ├─ SiLU activation                                │
│              └─ Down projection (matmul_i2s)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
Output projection (matmul_i2s)
  │
  ▼
Sampling (temperature, top-k, top-p)
  │
  ▼
Next Token
```

### GPU Data Transfer

```
Host (CPU)                        Device (GPU)
──────────                        ────────────
Model weights ──── mmap load ───► VRAM (persistent)
Input tokens  ──── H2D copy ────► Input buffer
                                  ↓
                                  Kernel execution
                                  ↓
Output logits ◄─── D2H copy ──── Output buffer
```

---

## Memory Management

### Memory Pool (`OptimizedMemoryPool`)

Pre-allocated GPU buffer pool with size-based reuse, defined in
`crates/bitnet-kernels/src/gpu/memory_optimization.rs`:

```rust
pub struct MemoryPoolConfig {
    pub max_pool_size: usize,           // Default 2 GB
    pub max_cached_buffers: usize,      // Default 1000
    pub enable_memory_tracking: bool,   // Leak detection
    pub cleanup_interval: Duration,     // Default 30 s
}
```

**Allocation strategy:**

1. Check free-buffer cache (keyed by size) → cache hit
2. Allocate new buffer → cache miss, tracked in `MemoryStats`
3. On deallocation → return to cache for reuse
4. Periodic cleanup removes expired buffers

**Access pattern optimization:**

```rust
pub enum AccessPattern {
    Sequential,              // Optimal for GPU coalescing
    Random,                  // No pattern detected
    Strided { stride: usize }, // Regular stride
}
```

### KV Cache

Managed by `bitnet-inference` with configurable eviction:

```rust
pub struct CacheConfig {
    pub max_size_bytes: usize,           // Default 1 GB
    pub max_sequence_length: usize,      // Default 2048
    pub enable_compression: bool,
    pub eviction_policy: EvictionPolicy, // LRU | FIFO | LFU
    pub block_size: usize,               // Default 64
}
```

### Weight Loading

Model weights are loaded via the GGUF loader in `bitnet-models`, memory-mapped
from disk, and transferred to GPU on demand. QK256 format uses 256-element
blocks; BitNet32-F16 uses 32-element blocks with inline FP16 scales.

---

## Configuration Reference

### CLI

```bash
# Auto-detect best backend
cargo run -p bitnet-cli --features gpu,full-cli -- run \
  --model model.gguf --prompt "Hello"

# Force specific device
cargo run -p bitnet-cli --features gpu,full-cli -- run \
  --device cuda --model model.gguf --prompt "Hello"

# Intel Arc via OpenCL
cargo run -p bitnet-cli --features oneapi,full-cli -- run \
  --device opencl --model model.gguf --prompt "Hello"
```

### Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `BITNET_DEVICE` | `cpu`, `cuda`, `gpu`, `opencl`, `npu`, `auto` | Override device selection |
| `BITNET_GPU_FAKE` | `cuda`, `metal`, `rocm`, `none` | Mock GPU detection (testing) |
| `BITNET_STRICT_MODE` | `1` | Reject fake GPU, enforce real hardware |
| `BITNET_STRICT_NO_FAKE_GPU` | `1` | Panic if fake GPU + strict mode |
| `CUDA_VISIBLE_DEVICES` | `0,1` | Limit visible NVIDIA GPUs |
| `CUDA_ARCH` | `sm_80` | Target compute capability |

### Strict Mode

When `BITNET_STRICT_MODE=1`:
- `BITNET_GPU_FAKE` is ignored (real hardware required)
- Mock inference paths are rejected
- Validation fails with **exit code 8** on suspicious weights
- Receipts must use `compute_path: "real"`

---

## Architecture Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                    bitnet-cli / bitnet-server                  │
│                     (user-facing entry points)                │
├───────────────────────────────────────────────────────────────┤
│                      bitnet-inference                         │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│   │CpuBackend│  │GpuBackend│  │NpuBackend│  ← Backend trait  │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
├────────┼──────────────┼─────────────┼─────────────────────────┤
│        │         bitnet-kernels     │                         │
│        │     ┌────────┴────────┐    │                         │
│        │     │  KernelManager  │    │  ← KernelProvider trait │
│        │     └───────┬─────────┘    │                         │
│   ┌────┴──┐  ┌───────┴───────┐  ┌──┴────┐                   │
│   │ CPU   │  │     CUDA      │  │ Metal │                    │
│   │ SIMD  │  │  (cudarc PTX) │  │ (NPU) │                   │
│   │AVX2/  │  │  matmul_i2s   │  │       │                   │
│   │AVX512 │  │  attention    │  │       │                   │
│   │NEON   │  │  rmsnorm      │  │       │                   │
│   └───────┘  │  qk256_gemv   │  └───────┘                   │
│              │  quantize_*   │                                │
│              └───────────────┘                                │
├───────────────────────────────────────────────────────────────┤
│  bitnet-common    bitnet-quantization    bitnet-models        │
│  (BackendRequest, (I2S, TL1, TL2, QK256) (GGUF loader)      │
│   KernelBackend,                                              │
│   device_features)                                            │
└───────────────────────────────────────────────────────────────┘
```

---

## Extending with a New Backend

To add a new GPU backend (e.g., Level Zero):

1. **Implement `KernelProvider`** in `bitnet-kernels` behind a new feature flag.
2. **Register** the provider in `KernelManager::default()` with appropriate
   priority.
3. **Add a `BackendRequest` variant** in `bitnet-common/src/backend_selection.rs`.
4. **Wire the feature flag** in workspace `Cargo.toml` following the `gpu`
   umbrella pattern.
5. **Add CI smoke workflow** (see `.github/workflows/gpu-smoke.yml` as
   template).
6. **Add `--device` CLI option** in `bitnet-cli/src/config.rs`.

> No separate crate is required — backends are feature-gated modules within
> `bitnet-kernels`.
