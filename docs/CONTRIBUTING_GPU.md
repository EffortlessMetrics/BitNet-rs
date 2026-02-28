# Contributing GPU Backends to BitNet-rs

This guide walks you through adding a new GPU backend, developing
kernels, and meeting the project's testing requirements.

> **Prerequisites** — Familiarity with `crates/bitnet-kernels/src/lib.rs`
> (the `KernelProvider` trait) and the existing CPU fallback kernel.

---

## Table of Contents

1. [Adding a New GPU Backend](#adding-a-new-gpu-backend)
2. [Kernel Development Workflow](#kernel-development-workflow)
3. [Testing Requirements](#testing-requirements)
4. [Code Style](#code-style)
5. [CI Integration](#ci-integration)
6. [Debugging Tips](#debugging-tips)

---

## Adding a New GPU Backend

### Step 1 — Create a microcrate

```bash
mkdir -p crates/bitnet-gpu-<backend>/src
```

Create `crates/bitnet-gpu-<backend>/Cargo.toml`:

```toml
[package]
name = "bitnet-gpu-<backend>"
version.workspace = true
edition.workspace = true
license.workspace = true
description = "<Backend> GPU backend for BitNet inference"

[dependencies]
bitnet-common = { path = "../bitnet-common", version = "0.2.1-dev" }
bitnet-kernels = { path = "../bitnet-kernels", version = "0.2.1-dev" }
log.workspace = true
thiserror.workspace = true

[dev-dependencies]
insta.workspace = true
proptest.workspace = true

[features]
default = []
```

### Step 2 — Implement device enumeration

Create `src/device.rs`:

```rust
use bitnet_common::Device;

/// Enumerate available <backend> devices.
///
/// Returns an empty vec if the <backend> runtime is not installed.
pub fn enumerate_devices() -> Vec<Device> {
    // 1. Dynamically load the <backend> library
    // 2. Query available devices
    // 3. Map each to bitnet_common::Device
    //
    // Return Vec::new() on any error — never panic.
    Vec::new()
}
```

### Step 3 — Implement `KernelProvider`

Create `src/provider.rs`:

```rust
use bitnet_kernels::KernelProvider;
use bitnet_common::BitNetError;

pub struct MyBackendProvider {
    // Device handle, compiled kernels, memory pool, etc.
}

impl KernelProvider for MyBackendProvider {
    fn name(&self) -> &'static str {
        "<backend>"
    }

    fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BitNetError> {
        // Dispatch to GPU kernel
        todo!("Implement GPU matmul")
    }

    // Implement remaining trait methods...
}
```

### Step 4 — Add feature flag to workspace `Cargo.toml`

```toml
# In the root Cargo.toml [features] section:
<backend> = [
    "kernels",
    "inference",
    "bitnet-kernels/<backend>",
    "bitnet-inference/gpu",
]
```

### Step 5 — Wire into `KernelManager` selection

In `crates/bitnet-kernels/src/lib.rs`, add the backend to the priority
list in `KernelManager::select_best()`:

```rust
#[cfg(feature = "<backend>")]
if let Some(provider) = bitnet_gpu_<backend>::try_create() {
    candidates.push(provider);
}
```

The `KernelManager` picks the highest-priority available backend.
Priority order: CUDA > OpenCL > Vulkan > CPU fallback.

### Step 6 — Add CI workflow

Create `.github/workflows/<backend>-smoke.yml`:

```yaml
name: <Backend> Smoke
on:
  push:
    branches: [main, "intel-gpu/**"]
    paths: ["crates/bitnet-gpu-<backend>/**"]
  pull_request:
    paths: ["crates/bitnet-gpu-<backend>/**"]

jobs:
  check:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo check -p bitnet-gpu-<backend> --locked
```

---

## Kernel Development Workflow

### 1. Write the kernel source

Place kernel sources in `crates/bitnet-gpu-<backend>/csrc/`:

```
csrc/
  matmul.cl          # OpenCL
  matmul.wgsl        # WebGPU / Vulkan
  matmul.metal        # Metal
```

Embed in Rust via `include_str!`:

```rust
const MATMUL_SOURCE: &str = include_str!("../csrc/matmul.cl");
```

### 2. Create CPU reference implementation

Every GPU kernel needs a scalar CPU equivalent for testing:

```rust
/// CPU reference: triple-loop matmul. Correct, not fast.
fn cpu_matmul_reference(
    a: &[f32], b: &[f32],
    m: usize, n: usize, k: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for t in 0..k {
                sum += a[i * k + t] * b[t * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}
```

### 3. Write cross-validation tests (GPU vs CPU)

```rust
#[test]
#[cfg(feature = "<backend>")]
#[ignore = "Requires <backend> hardware"]
fn matmul_gpu_matches_cpu_reference() {
    let a = deterministic_input(m * k, 42);
    let b = deterministic_input(k * n, 84);

    let expected = cpu_matmul_reference(&a, &b, m, n, k);
    let actual = gpu_matmul(&a, &b, m, n, k);

    // Correlation > 0.999, max relative error < 1e-3
    assert_outputs_close(&expected, &actual, 0.999, 1e-3);
}
```

### 4. Add property tests for invariants

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn softmax_sums_to_one(input in prop::collection::vec(-10.0f32..10.0, 1..256)) {
        let result = cpu_softmax_reference(&input);
        let sum: f32 = result.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5);
    }
}
```

### 5. Add golden output tests for regression

```rust
#[test]
fn matmul_golden_output() {
    let a = deterministic_input(4 * 8, 42);
    let b = deterministic_input(8 * 4, 84);
    let result = cpu_matmul_reference(&a, &b, 4, 4, 8);
    insta::assert_debug_snapshot!(result);
}
```

### 6. Benchmark with Criterion

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_matmul(c: &mut Criterion) {
    let a = deterministic_input(256 * 512, 42);
    let b = deterministic_input(512 * 256, 84);

    c.bench_function("matmul_256x512x256", |bencher| {
        bencher.iter(|| cpu_matmul_reference(&a, &b, 256, 256, 512))
    });
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
```

---

## Testing Requirements

Every GPU kernel **must** satisfy all of these:

| Requirement | Description |
|------------|-------------|
| **CPU reference** | Scalar implementation for hardware-free testing |
| **Property tests** | `proptest` invariants (e.g., softmax sums to 1) |
| **Golden output tests** | `insta` snapshots for regression detection |
| **Numerical stability** | Tests with extreme inputs (±1e30, zeros, NaN) |
| **Stress tests** | Repeated launches, concurrent threads, large allocations |
| **Feature contract** | Verify the feature flag compiles without conflicts |

### Tolerance guidelines

| Kernel | Correlation | Max Relative Error |
|--------|------------|-------------------|
| matmul (f32) | ≥ 0.9999 | < 1e-5 |
| matmul (mixed) | ≥ 0.999 | < 1e-3 |
| softmax | ≥ 0.9999 | < 1e-5 |
| rmsnorm | ≥ 0.999 | < 1e-4 |
| quantize/dequant | ≥ 0.99 | < 1e-2 |

---

## Code Style

### Feature gating

GPU modules must be gated with the backend feature flag:

```rust
#[cfg(feature = "<backend>")]
pub mod <backend>;
```

For the GPU umbrella (CUDA + Vulkan), use the unified predicate:

```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_only_function() { /* ... */ }
```

**Never** use `#[cfg(feature = "cuda")]` alone — always include the
`gpu` umbrella.

### Kernel sources

- Use `include_str!` for kernel embedding
- Place sources in `csrc/` within the backend crate
- One file per kernel (e.g., `matmul.cl`, `rmsnorm.cl`)

### Dynamic linking

All GPU libraries must be loaded at runtime:

```rust
// Good: dynamic loading
let lib = unsafe { libloading::Library::new("libOpenCL.so") }?;

// Bad: static linking (breaks compile-everywhere goal)
// extern "C" { fn clCreateContext(...); }
```

### Error recovery

Always fall back to CPU on GPU errors:

```rust
match gpu_matmul(&a, &b, m, n, k) {
    Ok(result) => result,
    Err(e) => {
        warn_once!("gpu_matmul_fallback", "GPU matmul failed: {e}, using CPU");
        cpu_matmul_reference(&a, &b, m, n, k)
    }
}
```

### Logging

Use `warn_once!` from `bitnet_common` for hot-path GPU warnings:

```rust
use bitnet_common::warn_once;
warn_once!("opencl_subgroup", "Device lacks subgroup support, using scalar path");
```

---

## CI Integration

### Compile checks (every PR)

The `feature-matrix.yml` workflow verifies that your backend compiles
with `cargo check`.  Add your feature combination to the matrix.

### GPU smoke tests (weekly / manual)

The `gpu-smoke.yml` workflow runs on GPU-equipped runners.  Add a job
for your backend's smoke tests.

### Receipt verification

If your backend produces inference receipts, add it to the
`verify-receipt` check in `xtask`.

---

## Debugging Tips

### No GPU detected

```bash
# OpenCL
clinfo

# Vulkan
vulkaninfo --summary

# CUDA
nvidia-smi
```

### Kernel compilation errors

Set `RUST_LOG=debug` to see the full OpenCL/Vulkan compiler output:

```bash
RUST_LOG=debug cargo test -p bitnet-gpu-<backend> --features <backend>
```

### Numerical mismatches

1. Reduce input size to the smallest reproducing case
2. Print intermediate values from both GPU and CPU paths
3. Check if the mismatch is consistent (driver bug) or random (race condition)
4. Use `oclgrind` (OpenCL) or `compute-sanitizer` (CUDA) for memory errors

### Performance profiling

```bash
# OpenCL
cargo run --features <backend> -- --profile-kernels

# Or use vendor tools:
# Intel: vtune, GPU Profiler
# NVIDIA: nsight-compute
# AMD: rocprof
```
