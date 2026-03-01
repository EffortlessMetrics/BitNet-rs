# GPU Backend Contributor Guide

This guide covers everything you need to contribute GPU backend code to BitNet-rs,
including adding new kernels, implementing new backends, and testing without hardware.

## Getting Started

### Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | 1.92+ (see `rust-toolchain.toml`) | Compiler toolchain |
| CUDA Toolkit | 12.x | NVIDIA GPU backend (optional) |
| ROCm | 6.x | AMD GPU backend (optional) |
| `clinfo` | any | OpenCL device enumeration (optional) |
| `nvidia-smi` | any | NVIDIA GPU diagnostics (optional) |
| `cargo-nextest` | latest | Test runner with timeout support |

You do **not** need GPU hardware to contribute — see [Testing Without Hardware](#testing-without-hardware).

### Project Structure

GPU support is integrated across several crates rather than isolated into a single GPU crate:

```
crates/
├── bitnet-common/
│   └── src/
│       ├── kernel_registry.rs    ← KernelBackend enum (CpuRust, Cuda, Hip, OneApi, CppFfi)
│       └── backend_selection.rs  ← BackendRequest enum (Auto, Cpu, Gpu, Cuda, Hip, OneApi)
├── bitnet-kernels/
│   └── src/
│       ├── lib.rs                ← KernelProvider trait, KernelManager
│       ├── gpu/
│       │   ├── mod.rs            ← GPU module organisation
│       │   ├── cuda.rs           ← CudaKernel (cudarc 0.17)
│       │   ├── kernels/          ← Native CUDA kernel sources (.cu)
│       │   ├── memory_optimization.rs
│       │   ├── mixed_precision.rs
│       │   ├── validation.rs     ← GPU testing framework
│       │   ├── benchmark.rs
│       │   └── tests.rs
│       ├── cuda/
│       │   ├── mod.rs
│       │   ├── attention.rs
│       │   ├── qk256_gemv.rs
│       │   └── rmsnorm.rs
│       ├── rocm/                 ← AMD ROCm/HIP backend
│       └── cpu/                  ← CPU kernels (x86 SIMD, ARM NEON)
├── bitnet-device-probe/          ← Runtime device detection & capability probing
└── bitnet-inference/             ← Inference engine (GPU-aware execution)
```

### Key Abstractions

**`KernelProvider` trait** (`bitnet-kernels/src/lib.rs`) — The core abstraction every backend implements:

```rust
pub trait KernelProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn is_available(&self) -> bool;
    fn matmul_i2s(&self, a: &[i8], b: &[u8], c: &mut [f32],
                   m: usize, n: usize, k: usize) -> Result<()>;
    fn quantize(&self, input: &[f32], output: &mut [u8],
                scales: &mut [f32], qtype: QuantizationType) -> Result<()>;
}
```

**`KernelBackend` enum** (`bitnet-common/src/kernel_registry.rs`) — Registered backends:

```rust
pub enum KernelBackend {
    CpuRust,   // Pure Rust CPU with optional SIMD
    Cuda,      // NVIDIA CUDA via cudarc
    Hip,       // AMD ROCm/HIP
    OneApi,    // Intel oneAPI (SYCL/Level Zero)
    CppFfi,    // C++ FFI bridge (bitnet.cpp / llama.cpp)
}
```

**`BackendRequest` enum** (`bitnet-common/src/backend_selection.rs`) — User-facing selection:

```rust
pub enum BackendRequest {
    Auto,    // Auto-select best available
    Cpu,     // Force CPU
    Gpu,     // Require any GPU
    Cuda,    // Require NVIDIA CUDA
    Hip,     // Require AMD HIP
    OneApi,  // Require Intel oneAPI
}
```

### How to Build

```bash
# CPU-only (no GPU hardware required)
cargo build --no-default-features --features cpu

# GPU (CUDA)
cargo build --no-default-features --features gpu

# ROCm/HIP
cargo build --no-default-features --features rocm

# Vulkan
cargo build --no-default-features --features vulkan
```

> **Important**: Default features are empty. Always specify `--no-default-features --features <target>`.

### How to Test

```bash
# CPU tests (always works, no hardware needed)
cargo nextest run --workspace --no-default-features --features cpu

# GPU tests (requires CUDA hardware + toolkit)
cargo nextest run --workspace --no-default-features --features gpu

# Single crate
cargo test -p bitnet-kernels --no-default-features --features cpu

# CI profile (4 threads, 5-min timeout)
cargo nextest run --profile ci
```

---

## Adding a New Kernel

Follow these steps to add a new GPU compute kernel:

### Step 1: Write the CUDA Kernel Source

Create or edit a `.cu` file in `crates/bitnet-kernels/src/gpu/kernels/`:

```cuda
// crates/bitnet-kernels/src/gpu/kernels/my_kernel.cu
extern "C" __global__
void my_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // Example operation
    }
}
```

### Step 2: Register the Kernel

Load the kernel source in the GPU module using `include_str!`:

```rust
// In crates/bitnet-kernels/src/gpu/mod.rs or relevant submodule
const MY_KERNEL_SRC: &str = include_str!("kernels/my_kernel.cu");
```

Wire it into `CudaKernel` initialisation so it's compiled and available at runtime.

### Step 3: Add a CPU Reference Implementation

Every GPU kernel **must** have a corresponding CPU reference for validation and
fallback. Add it alongside existing CPU kernels:

```rust
// In crates/bitnet-kernels/src/cpu/ (appropriate file)
pub fn my_kernel_cpu(input: &[f32], output: &mut [f32]) {
    for (i, val) in input.iter().enumerate() {
        output[i] = val * 2.0;
    }
}
```

### Step 4: Add Unit Tests

Write tests that compare the CPU reference against expected outputs:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_kernel_cpu_reference() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        my_kernel_cpu(&input, &mut output);
        assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn test_my_kernel_gpu_matches_cpu() {
        let input = vec![1.0f32; 1024];
        let cpu_output = /* ... */;
        let gpu_output = /* ... */;
        for (cpu, gpu) in cpu_output.iter().zip(gpu_output.iter()) {
            assert!((cpu - gpu).abs() < 1e-5, "CPU/GPU mismatch");
        }
    }
}
```

> **Note**: GPU tests that require hardware must use `#[ignore = "requires CUDA hardware"]`
> with a justification string (bare `#[ignore]` is rejected by pre-commit hooks).

### Step 5: Add Property Tests

Add property-based tests to verify invariants hold across random inputs:

```rust
#[cfg(test)]
mod proptests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn my_kernel_output_length_matches_input(len in 1..4096usize) {
            let input = vec![1.0f32; len];
            let mut output = vec![0.0f32; len];
            my_kernel_cpu(&input, &mut output);
            prop_assert_eq!(output.len(), input.len());
        }
    }
}
```

### Step 6: Wire Into KernelProvider

If the kernel introduces a new operation (not just an optimisation of an existing one),
add it to the `KernelProvider` trait and implement it for each backend.

### Step 7: Update Documentation

- Add the kernel to the inventory in `docs/gpu-kernel-architecture.md`
- Document performance characteristics and any precision trade-offs
- Update this guide if the workflow changed

---

## Adding a New Backend

To add support for a new GPU platform (e.g., WebGPU, SYCL):

### Step 1: Create the Microcrate

```bash
cargo init --lib crates/bitnet-<backend>
```

Add it to the workspace in the root `Cargo.toml` under `[workspace.members]`.

### Step 2: Implement KernelProvider

Your backend must implement the `KernelProvider` trait from `bitnet-kernels`:

```rust
use bitnet_kernels::KernelProvider;

pub struct MyBackendKernel { /* ... */ }

impl KernelProvider for MyBackendKernel {
    fn name(&self) -> &'static str { "<backend>" }
    fn is_available(&self) -> bool { /* runtime probe */ }
    fn matmul_i2s(&self, a: &[i8], b: &[u8], c: &mut [f32],
                   m: usize, n: usize, k: usize) -> Result<()> { /* ... */ }
    fn quantize(&self, input: &[f32], output: &mut [u8],
                scales: &mut [f32], qtype: QuantizationType) -> Result<()> { /* ... */ }
}
```

### Step 3: Add a Feature Flag

In the root `Cargo.toml`, add a feature that follows the existing pattern:

```toml
[features]
my-backend = [
    "kernels", "inference", "tokenizers",
    "bitnet-kernels/my-backend",
    "bitnet-inference/gpu",
    "bitnet-quantization/gpu",
]
```

### Step 4: Register in KernelBackend

Add a variant to the `KernelBackend` enum in `crates/bitnet-common/src/kernel_registry.rs`:

```rust
pub enum KernelBackend {
    // ... existing variants ...
    MyBackend,
}
```

And optionally extend `BackendRequest` in `backend_selection.rs`.

### Step 5: Wire Into Backend Selection

Update the backend selection logic so that `BackendRequest::Auto` can discover and
prefer your backend when appropriate. The selection priority is typically:
CUDA > ROCm/HIP > oneAPI > CPU.

### Step 6: Add CI Matrix Entry

Add your backend to the CI matrix in `.github/workflows/`:

1. Create or update a workflow file (e.g., `my-backend-smoke.yml`)
2. Add a label-gated trigger (see `gpu.yml` for the pattern)
3. Ensure the workflow runs on appropriate self-hosted runners if hardware is needed

---

## Testing Without Hardware

You can develop and test GPU code without physical GPU hardware.

### CPU Reference Implementations

Every GPU kernel has a CPU reference. Run the full test suite with:

```bash
cargo nextest run --workspace --no-default-features --features cpu
```

This validates all kernel logic through CPU code paths.

### BITNET_GPU_FAKE

Set `BITNET_GPU_FAKE=1` to enable a mock GPU backend that simulates GPU execution
using CPU code paths. This is useful for testing GPU code paths (dispatch, memory
management, error handling) without hardware:

```bash
BITNET_GPU_FAKE=1 cargo test -p bitnet-kernels --no-default-features --features gpu
```

> **Warning**: `BITNET_GPU_FAKE` is **ignored** when `BITNET_STRICT_MODE=1` to prevent
> mock paths from leaking into production validation.

### Mock Backend Usage

For integration tests that exercise GPU dispatch without real hardware:

```rust
#[test]
fn test_gpu_dispatch_fallback() {
    // BackendRequest::Auto falls back to CPU when no GPU is detected
    let backend = select_backend(BackendRequest::Auto);
    assert!(backend.is_available());
}
```

### Feature Flag Combinations

Test different feature flag combinations to ensure correct conditional compilation:

```bash
# CPU only
cargo check --no-default-features --features cpu

# GPU feature gate (compiles GPU code paths)
cargo check --no-default-features --features gpu

# Both (tests fallback logic)
cargo check --no-default-features --features cpu,gpu
```

---

## Code Style

### Feature Gate Conventions

GPU code **must** use the unified predicate — never `#[cfg(feature = "cuda")]` alone:

```rust
// ✅ Correct: unified GPU predicate
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_function() { /* ... */ }

// ❌ Wrong: standalone cuda feature check
#[cfg(feature = "cuda")]
pub fn gpu_function() { /* ... */ }
```

Use runtime checks from `bitnet_kernels::device_features`:

```rust
use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};

if gpu_compiled() && gpu_available_runtime() {
    // GPU path
} else {
    // CPU fallback
}
```

### Error Handling

- Use `bitnet_common::BitNetError` for all public-facing errors
- GPU-specific errors should wrap the underlying SDK error with context
- Always provide a CPU fallback path or a clear error message

```rust
pub fn run_kernel(input: &[f32]) -> Result<Vec<f32>> {
    match gpu_kernel(input) {
        Ok(result) => Ok(result),
        Err(e) => {
            warn_once!("gpu_fallback", "GPU kernel failed, falling back to CPU: {e}");
            cpu_kernel(input)
        }
    }
}
```

### CUDA Kernel Naming Conventions

- Kernel function names: `snake_case` (e.g., `matmul_i2s_kernel`)
- Source files: `snake_case.cu` (e.g., `qk256_gemv.cu`)
- Rust wrapper functions: match the kernel name without `_kernel` suffix
- Constants: `UPPER_SNAKE_CASE` (e.g., `BLOCK_SIZE`, `WARP_SIZE`)

### Documentation Requirements

- All public GPU functions must have rustdoc with:
  - Brief description of the operation
  - `# Panics` section if applicable
  - `# Errors` section listing failure modes
  - `# Safety` section for any `unsafe` code
- CUDA kernel sources should include a header comment describing:
  - Grid/block dimensions expected
  - Memory access patterns
  - Precision characteristics

### Rate-Limited Logging

Use `warn_once!` from `bitnet_common` for hot-path GPU warnings:

```rust
use bitnet_common::warn_once;

warn_once!("cuda_init", "CUDA context initialisation took {elapsed:?}");
```

### Ignored Tests

All `#[ignore]` attributes must include a justification string:

```rust
// ✅ Valid
#[ignore = "requires CUDA hardware - run with `gpu` CI label"]
fn test_cuda_matmul() { /* ... */ }

// ❌ Rejected by pre-commit hook
#[ignore]
fn test_cuda_matmul() { /* ... */ }
```

---

## Further Reading

- [GPU Setup Guide](GPU_SETUP.md) — CUDA toolkit installation and verification
- [Intel GPU Setup](INTEL_GPU_SETUP.md) — Intel Arc / OpenCL setup
- [GPU Kernel Architecture](gpu-kernel-architecture.md) — Design decisions and phase roadmap
- [CUDA Configuration Guide](cuda-configuration-guide.md) — Runtime CUDA configuration
- [Performance Guide](performance-guide.md) — GPU performance optimisation tips
- [GPU Development Workflow](GPU_DEVELOPMENT_WORKFLOW.md) — Branching, PRs, CI, and reviews
- [CONTRIBUTING.md](../CONTRIBUTING.md) — General contribution guidelines
