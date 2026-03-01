# Intel GPU / A770 OpenCL Backend Architecture

> Comprehensive architecture reference for the BitNet-rs Intel GPU backend.
> For setup instructions see [SETUP.md](SETUP.md); for the development roadmap
> see [ROADMAP.md](ROADMAP.md).

---

## 1. Overview

BitNet-rs provides an OpenCL-based inference backend targeting Intel Arc
discrete GPUs (Xe-HPG architecture). The backend is organised as a set of
`opencl_*` modules inside `crates/bitnet-kernels/` together with a dedicated
`bitnet-opencl` crate that owns the SPIR-V compilation pipeline, backend
dispatch, and runtime context management.

**Design principles:**

* **CPU-reference first** — every kernel has a scalar CPU implementation used
  for correctness testing before hardware dispatch.
* **Backend-agnostic dispatch** — a unified `BackendDispatcher` routes
  operations (matmul, attention, RoPE, …) to the best available provider
  (CUDA → OpenCL → CPU fallback).
* **Offline SPIR-V compilation** — `.cl` kernel sources are compiled to SPIR-V
  binaries at build time via `clang` / `ocloc`, with a runtime `.cl` fallback
  for development.
* **Hardware-tuned defaults** — buffer alignment, work-group sizes, and tile
  dimensions default to optimal values for the Intel Arc A770.

---

## 2. Module Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Inference Pipeline                              │
│  bitnet-inference  ──►  KernelProvider trait  ──►  opencl_provider      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
          ┌──────────────────────▼──────────────────────┐
          │            opencl_pipeline                   │
          │  Orchestrates end-to-end inference:          │
          │  embedding → layers × N → lm_head           │
          │  PipelineConfig (dims, heads, eps, …)       │
          └──┬────────┬────────┬────────┬───────────────┘
             │        │        │        │
    ┌────────▼──┐ ┌───▼────┐ ┌▼──────┐ │
    │ opencl_   │ │opencl_ │ │opencl_│ │
    │ embedding │ │attention│ │ ffn   │ │
    │           │ │  (sdpa) │ │(gated)│ │
    └─────┬─────┘ └───┬────┘ └──┬────┘ │
          │           │         │      │
          ▼           ▼         ▼      ▼
    ┌─────────────────────────────────────────────┐
    │         opencl_quantized                     │
    │  I2_S quantized matmul (BitNet 1-bit)       │
    └─────────────────────┬───────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────┐
    │        Kernel Source & Dispatch              │
    │                                              │
    │  opencl_kernel_sources   .cl source mgmt    │
    │  opencl_work_size        dispatch tuning     │
    │  opencl_cache            binary caching      │
    └────────────────────┬────────────────────────┘
                         │
    ┌────────────────────▼────────────────────────┐
    │         Memory Subsystem                     │
    │                                              │
    │  opencl_context   platform/device lifecycle  │
    │  opencl_buffer    A770-aligned pool mgmt     │
    │  opencl_memory    transfer tracking (H2D/    │
    │                   D2H/D2D), pinned/zero-copy │
    └─────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────┐
    │         bitnet-opencl  (dedicated crate)     │
    │                                              │
    │  backend_registry    capability discovery    │
    │  backend_dispatcher  op → backend routing    │
    │  context_pool        fast model switching    │
    │  spirv               .cl → SPIR-V pipeline   │
    │  kv_cache            paged KV cache          │
    │  paged_attention     GQA paged engine        │
    │  quantized_kernels   quantized op kernels    │
    │  model_validator     GPU cap checks          │
    │  numerical_validator FP tolerance gates      │
    │  diagnostics         driver/runtime status   │
    └─────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────┐
    │         bitnet-device-probe                   │
    │                                              │
    │  opencl.rs   dynamic libOpenCL loading       │
    │              (Windows: OpenCL.dll,            │
    │               Linux: libOpenCL.so)           │
    └─────────────────────────────────────────────┘
```

### Kernel source files (`crates/bitnet-kernels/src/gpu/kernels/`)

| File                | Purpose                               |
|---------------------|---------------------------------------|
| `activations.cl`    | SiLU, GELU, ReLU activation functions |
| `attention.cl`      | Scaled dot-product attention          |
| `embedding.cl`      | Token embedding lookup                |
| `rope.cl`           | Rotary Position Embeddings            |
| `normalization.cl`  | RMSNorm, LayerNorm                    |
| `matmul_i2s.cl`     | I2_S matrix–vector multiply           |
| `tiled_matmul.cl`   | Tiled / blocked matmul                |
| `elementwise.cl`    | Pointwise arithmetic                  |
| `quantize_i2s.cl`   | On-device quantisation                |

CUDA counterparts (`bitnet_matmul.cu`, `mixed_precision_kernels.cu`) live in the
same directory and share the tiling strategy.

---

## 3. A770 Hardware Profile

The Intel Arc A770 uses the **Xe-HPG** (High-Performance Graphics) micro-
architecture, which is the primary target for this backend.

| Parameter                  | Value                              |
|----------------------------|------------------------------------|
| Xe-cores                   | 32                                 |
| Execution Units (EUs)      | 512 (16 per Xe-core)              |
| Threads per EU             | 8                                  |
| SIMD width                 | 16 (native), 32 (emulated)        |
| Subgroup sizes             | 8, 16, 32                         |
| Shared Local Memory (SLM)  | 64 KB per Xe-core                 |
| L1 / L3 Cache              | 192 KB / ~16 MB (partitioned)     |
| VRAM                       | 16 GB GDDR6, 560 GB/s bandwidth   |
| FP32 peak                  | ~19.66 TFLOPS                     |
| FP16 peak                  | ~39.32 TFLOPS (2× rate)           |
| INT8 peak (DP4A)           | ~78.64 TOPS (4× rate)             |

### Key hardware features for inference

* **FP16 2× throughput** — mixed-precision kernels should keep accumulation in
  FP32 and operands in FP16 wherever possible.
* **INT8 DP4A** — four-way dot product available via `intel_subgroup_dot_product`
  extension; enables efficient 1-bit/2-bit quantised matmul.
* **Subgroup operations** — use `intel_subgroups` for shuffle, broadcast, and
  block read/write; prefer subgroup width of **16** on Xe-HPG.
* **SLM (64 KB)** — use for tile buffering in tiled matmul; fits 8 K×FP16 or
  16 K×INT8 elements per Xe-core.
* **L3 partitioning** — ~262 KB partitions; align large buffers to partition
  boundaries for best cache utilisation.

---

## 4. Kernel Implementation Guide

### 4.1 Write the OpenCL kernel source

Create a new `.cl` file under `crates/bitnet-kernels/src/gpu/kernels/`:

```c
// crates/bitnet-kernels/src/gpu/kernels/my_kernel.cl
__kernel void my_kernel(
    __global const float* input,
    __global float* output,
    const int N
) {
    int gid = get_global_id(0);
    if (gid < N) {
        output[gid] = /* compute */ input[gid];
    }
}
```

**Guidelines:**

* Declare all buffers `__global`; use `__local` only for SLM tiles.
* Guard out-of-bounds access (`if (gid < N)`).
* Prefer `half` / `half4` types where FP16 2× rate is beneficial.
* Use `__attribute__((intel_reqd_sub_group_size(16)))` to lock subgroup width.

### 4.2 Register in the kernel source manager

Add the source to `opencl_kernel_sources.rs` so the pipeline can discover it:

```rust
// crates/bitnet-kernels/src/opencl_kernel_sources.rs
pub const MY_KERNEL_SRC: &str = include_str!("gpu/kernels/my_kernel.cl");
```

For SPIR-V pre-compilation, also add the path to the manifest consumed by
`bitnet-opencl/src/spirv.rs`.

### 4.3 Write a CPU reference implementation

Every OpenCL kernel **must** have a scalar CPU equivalent used for testing:

```rust
// crates/bitnet-kernels/src/opencl_my_kernel.rs
pub fn my_kernel_cpu(input: &[f32], output: &mut [f32]) {
    for (o, i) in output.iter_mut().zip(input.iter()) {
        *o = /* same computation */;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_kernel_cpu_reference() {
        let input = vec![1.0_f32; 256];
        let mut output = vec![0.0_f32; 256];
        my_kernel_cpu(&input, &mut output);
        // assert correctness …
    }
}
```

### 4.4 Wire into the pipeline

1. Add an operation variant to `BackendDispatcher` in
   `bitnet-opencl/src/backend_dispatcher.rs`.
2. Implement the dispatch logic that selects OpenCL → CPU fallback.
3. Call the operation from `opencl_pipeline.rs` at the appropriate stage.

### 4.5 Add hardware-gated tests

```rust
#[test]
#[ignore = "requires Intel Arc GPU — run with --ignored on A770 hardware"]
fn test_my_kernel_on_hardware() {
    // real OpenCL dispatch …
}
```

---

## 5. Performance Considerations

### Memory alignment

`opencl_buffer.rs` defines A770-specific alignment constants:

| Constant              | Value     | Purpose                          |
|-----------------------|-----------|----------------------------------|
| Cache line            | 64 B      | Minimum allocation alignment     |
| L3 partition          | 262 KB    | Large-buffer partition alignment |
| Optimal DMA transfer  | 4 096 B   | Host ↔ device transfer unit      |
| Subgroup width        | 32        | Vector-load alignment            |

Use `align_size(byte_count, alignment)` to round up allocations.

### Workgroup sizing

`opencl_work_size.rs` provides `WorkSizeConfig`:

* **SIMD width** — 16 (Xe-HPG native).
* **Max workgroup size** — 1 024 work-items.
* **Preferred workgroup** — multiple of 16 × compute-unit count.
* **Dispatch dimensions** — 1-D for vector ops; 2-D for matmul tiles; 3-D for
  batched attention.

Rule of thumb: local work size should be a multiple of the subgroup size (16)
and total work size should saturate all 32 Xe-cores.

### FP16 mixed precision

* Store weights and KV cache in FP16 to halve memory bandwidth.
* Accumulate in FP32 to preserve numerical accuracy.
* Use `vload_half4` / `vstore_half4` for coalesced 64-bit memory transactions.

### SLM usage

* 64 KB per Xe-core → budget ~48 KB for data tiles, 16 KB for scratch.
* Tiled matmul: tile size of 16×16 FP16 (512 B per tile) fits comfortably.
* Always call `barrier(CLK_LOCAL_MEM_FENCE)` between SLM writes and reads.

### Transfer strategy

`opencl_memory.rs` supports four buffer allocation strategies:

| Strategy          | When to use                                     |
|-------------------|-------------------------------------------------|
| `AllocateOnWrite` | Default; lazy allocation on first write          |
| `PreAllocate`     | Known-size tensors (weights, KV cache)           |
| `PinnedMemory`    | Frequent H2D/D2H (streaming tokens)             |
| `ZeroCopy`        | Shared-memory / integrated GPU paths             |

---

## 6. Testing Strategy

### Tier 1 — CPU reference tests (always run)

Every kernel module (`opencl_attention.rs`, `opencl_ffn.rs`, …) contains unit
tests that exercise the CPU-reference path. These run in every PR CI job with
`--features cpu` and require **no GPU hardware**.

```
cargo nextest run --workspace --no-default-features --features cpu
```

### Tier 2 — Hardware-gated tests (run on A770 machines)

Tests marked `#[ignore = "requires Intel Arc GPU …"]` perform real OpenCL
dispatch. Run them locally on an A770 workstation:

```
cargo nextest run --workspace --no-default-features --features cpu \
    --run-ignored ignored-only
```

### Tier 3 — Numerical validation

`bitnet-opencl/src/numerical_validator.rs` compares GPU kernel output against
CPU reference within configurable FP tolerances. This catches precision
regressions introduced by FP16 mixed-precision or driver updates.

### Tier 4 — CI integration

The `gpu-smoke.yml` workflow runs weekly on GPU-enabled runners, uploading
receipt artifacts for post-hoc analysis. The `ci-core.yml` workflow runs
CPU-reference tests on every push.

---

## See Also

* [SETUP.md](SETUP.md) — Driver installation and environment verification
* [ROADMAP.md](ROADMAP.md) — Phased development plan
* [`docs/INTEL_GPU_SETUP.md`](../INTEL_GPU_SETUP.md) — Legacy setup guide
* [`docs/INTEL_GPU_ARCHITECTURE.md`](../INTEL_GPU_ARCHITECTURE.md) — Legacy
  architecture overview
