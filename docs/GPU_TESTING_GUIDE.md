# GPU Testing Guide

How to write, run, and debug GPU tests in BitNet-rs.

> **Prerequisites:** Read [GPU Setup](GPU_SETUP.md) for CUDA toolchain
> installation, or [Intel GPU Setup](INTEL_GPU_SETUP.md) for Intel Arc.

---

## Quick Reference

```bash
# Compile-check GPU code (no hardware needed)
cargo check --workspace --no-default-features --features gpu,cpu

# Run GPU tests (requires NVIDIA GPU)
cargo test -p bitnet-kernels --no-default-features --features gpu -- --nocapture

# Run GPU tests with nextest
cargo nextest run -p bitnet-kernels --no-default-features --features gpu

# Run with mock GPU (no hardware)
BITNET_GPU_FAKE=cuda cargo test -p bitnet-kernels --no-default-features --features gpu

# ROCm compile check
cargo check --workspace --no-default-features --features rocm,cpu
```

---

## Test Files

All GPU-specific tests live in `crates/bitnet-kernels/tests/`:

| File | Purpose | Hardware? |
|------|---------|-----------|
| `gpu_smoke.rs` | Fast parity check: GPU vs CPU matmul | Yes |
| `gpu_real_compute.rs` | Real GPU compute validation (strict, no mocks) | Yes |
| `gpu_integration.rs` | Numerical accuracy, performance, memory | Yes |
| `gpu_infra_smoke.rs` | Mock-based infrastructure detection | No |
| `gpu_quantization.rs` | Quantization kernel correctness | Yes |
| `gpu_quantization_parity.rs` | GPU vs CPU quantization parity | Yes |
| `gpu_info_mock.rs` | GPU info detection with mocks | No |
| `mixed_precision_gpu_kernels.rs` | FP16/BF16 validation | Yes |
| `strict_gpu_mode.rs` | Strict mode prevents fake GPU | No |

Cross-crate parity tests:
- `crates/bitnet-quantization/tests/gpu_parity.rs`

Shared utilities:
- `tests/common/gpu.rs`

---

## Mock Backend

The mock backend lets you test GPU code paths **without hardware**. It's
controlled by the `BITNET_GPU_FAKE` environment variable.

### How It Works

`BITNET_GPU_FAKE` overrides hardware detection in
`bitnet-kernels/src/gpu_utils.rs`:

```rust
// When BITNET_GPU_FAKE is set, gpu_utils::get_gpu_info() returns
// a synthetic GpuInfo based on the variable's value:
//   BITNET_GPU_FAKE=cuda   → reports CUDA available
//   BITNET_GPU_FAKE=metal  → reports Metal available
//   BITNET_GPU_FAKE=rocm   → reports ROCm available
//   BITNET_GPU_FAKE=none   → reports no GPU available
```

### Using Mocks in Tests

```rust
use serial_test::serial;

#[test]
#[serial(bitnet_env)]
fn test_gpu_capability_detection() {
    temp_env::with_var("BITNET_GPU_FAKE", Some("cuda"), || {
        let info = gpu_utils::get_gpu_info();
        assert!(info.cuda);
    });
}
```

**Important:** Always use `#[serial(bitnet_env)]` and `temp_env::with_var` (or
`EnvGuard`) when mutating environment variables. See
[environment-variables.md](environment-variables.md) for details.

### Mock Limitations

- Mock does **not** execute real GPU kernels — only detects capability
- Real compute tests skip automatically when no hardware is detected:

```rust
#[test]
fn gpu_smoke_test() {
    if !check_gpu_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }
    // ... actual GPU test
}
```

### Strict Mode Blocks Mocks

When `BITNET_STRICT_MODE=1`, the runtime rejects `BITNET_GPU_FAKE`:

```rust
// BITNET_STRICT_NO_FAKE_GPU=1 causes a panic if both
// BITNET_GPU_FAKE and strict mode are set simultaneously.
```

This prevents accidental use of mock GPUs in production or receipt-validated
benchmarks.

---

## Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `BITNET_GPU_FAKE` | `cuda`, `metal`, `rocm`, `none` | unset | Override GPU detection |
| `BITNET_ENABLE_GPU_TESTS` | `1` | unset | Enable GPU tests (off by default) |
| `GPU_TEST_SIZE` | `tiny`, `small`, `medium` | `tiny` | Control test matrix dimensions |
| `GPU_TEST_TOLERANCE` | `0.0`–`1.0` | `0.99` | Cosine similarity threshold |
| `BITNET_GPU_CACHE` | `0` | `1` | Disable GPU info caching |
| `BITNET_STRICT_MODE` | `1` | unset | Require real hardware |
| `BITNET_STRICT_NO_FAKE_GPU` | `1` | unset | Panic on fake + strict |

---

## CI Pipelines

### GPU Smoke (`gpu-smoke.yml`)

- **Schedule:** Weekly, Monday 4:00 AM UTC
- **Jobs:**
  - `gpu-compile` — always-on Ubuntu runner, `cargo check --features gpu,cpu`
  - `gpu-runtime` — self-hosted NVIDIA runner, kernel + inference smoke tests
- **Environment:** `BITNET_GPU_FAKE=none`, `RUST_LOG=warn`
- **Artifacts:** GPU receipts uploaded on success

### ROCm Smoke (`rocm-smoke.yml`)

- **Schedule:** Weekly, Tuesday 4:00 AM UTC (offset from GPU)
- **Jobs:**
  - `rocm-compile` — always-on Ubuntu runner, `cargo check --features rocm,cpu`
  - `rocm-runtime` — self-hosted ROCm runner, kernel + inference smoke tests
- **Environment:** `BITNET_GPU_FAKE=none`, `RUST_LOG=warn`
- **Artifacts:** ROCm receipts uploaded on success

### Core CI (`ci-core.yml`)

- CPU-only tests run on every PR
- GPU compile checks may be included in matrix builds with `--features gpu,cpu`

### What CI Validates

1. **Compile check** — GPU feature flags compile without hardware (every week)
2. **Kernel smoke** — matmul_i2s runs and matches CPU reference
3. **Inference smoke** — 4-token generation completes without error
4. **Receipt upload** — backend summary + performance data captured

---

## Writing New GPU Tests

### Step 1: Choose the Right File

- **Kernel correctness** → add to `gpu_smoke.rs` or `gpu_integration.rs`
- **Quantization** → add to `gpu_quantization.rs` or `gpu_quantization_parity.rs`
- **Infrastructure / mock** → add to `gpu_infra_smoke.rs`
- **Mixed precision** → add to `mixed_precision_gpu_kernels.rs`

### Step 2: Gate on Hardware

Always check GPU availability at the start of tests that need real hardware:

```rust
#[test]
fn my_new_gpu_test() {
    if !gpu_utils::gpu_available() {
        eprintln!("Skipping: no GPU");
        return;
    }

    let gpu = CudaKernel::new().expect("GPU init failed");
    // ... test body
}
```

Or use `#[ignore]` with a justification:

```rust
#[test]
#[ignore = "requires NVIDIA GPU — run manually or on GPU runner"]
fn my_gpu_test() {
    // ...
}
```

### Step 3: Validate with CPU Parity

The gold standard for GPU kernel tests is **cross-backend parity**: run the
same operation on CPU and GPU, then compare results.

```rust
#[test]
fn test_matmul_gpu_cpu_parity() {
    if !gpu_utils::gpu_available() { return; }

    let cpu_kernel = FallbackKernel::new();
    let gpu_kernel = CudaKernel::new().unwrap();

    let (a, b) = generate_test_data(m, n, k);
    let mut cpu_result = vec![0.0f32; m * n];
    let mut gpu_result = vec![0.0f32; m * n];

    cpu_kernel.matmul_i2s(&a, &b, &mut cpu_result, m, n, k).unwrap();
    gpu_kernel.matmul_i2s(&a, &b, &mut gpu_result, m, n, k).unwrap();

    // Compare via cosine similarity (tolerance from GPU_TEST_TOLERANCE)
    let similarity = cosine_similarity(&cpu_result, &gpu_result);
    assert!(similarity >= 0.99, "Parity check failed: {similarity}");
}
```

### Step 4: Test Determinism

GPU operations must be deterministic for reproducible inference:

```rust
#[test]
fn test_gpu_determinism() {
    if !gpu_utils::gpu_available() { return; }

    let kernel = CudaKernel::new().unwrap();
    let (a, b) = generate_test_data(16, 16, 16);

    let results: Vec<Vec<f32>> = (0..3).map(|_| {
        let mut c = vec![0.0f32; 16 * 16];
        kernel.matmul_i2s(&a, &b, &mut c, 16, 16, 16).unwrap();
        c
    }).collect();

    assert_eq!(results[0], results[1]);
    assert_eq!(results[1], results[2]);
}
```

### Step 5: Use Feature Gates

Ensure your test compiles only when GPU features are enabled:

```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[test]
fn test_cuda_specific_feature() {
    // ...
}
```

---

## Cross-Backend Validation

### Quantization Parity (`gpu_parity.rs`)

Tests in `crates/bitnet-quantization/tests/gpu_parity.rs` validate that
quantization produces identical results on CPU and GPU:

```rust
#[test]
fn test_i2s_cpu_gpu_parity() -> Result<()> {
    let cpu = Device::Cpu;
    let gpu = prepare_devices()?;  // returns None if no GPU

    let qt_cpu = q.quantize(&t_cpu, &cpu)?;
    let qt_gpu = q.quantize(&t_gpu, &gpu)?;
    assert_eq!(qt_cpu.data, qt_gpu.data);
    Ok(())
}
```

### Cross-Validation with C++ Reference

The `crossval` crate provides a framework for validating Rust GPU results
against the bitnet.cpp C++ reference implementation. See
`crossval/README.md` for setup instructions.

---

## Debugging GPU Tests

### Increase Verbosity

```bash
RUST_LOG=debug cargo test -p bitnet-kernels --features gpu -- gpu_smoke --nocapture
```

### Enable Memory Tracking

```bash
# Tracks allocations and detects leaks
RUST_LOG=debug BITNET_GPU_CACHE=0 cargo test -p bitnet-kernels --features gpu
```

### Use Smaller Test Sizes

```bash
GPU_TEST_SIZE=tiny cargo test -p bitnet-kernels --features gpu -- gpu_smoke
```

### Check CUDA Setup

```bash
# Verify CUDA is working
nvidia-smi
nvcc --version

# Check device visibility
CUDA_VISIBLE_DEVICES=0 cargo test -p bitnet-kernels --features gpu -- gpu_smoke
```

### Common Failures

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| "CUDA not available" | No driver or device | Install NVIDIA driver, check `nvidia-smi` |
| "Failed to transfer to device" | OOM | Reduce `GPU_TEST_SIZE` |
| Parity check < 0.99 | Precision / kernel bug | Check FP32 vs mixed precision path |
| "Feature not compiled" | Wrong feature flags | Add `--features gpu` |
| Test hangs | GPU deadlock | Use `--test-threads=1`, check CUDA context |
