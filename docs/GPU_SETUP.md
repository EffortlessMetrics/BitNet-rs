# GPU Setup (CUDA 12.x)

This guide is the canonical setup for running BitNet-rs with the `gpu` feature.

> **Status:** CUDA inference is implemented, but full GPU receipt-validation parity is still in progress.

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA driver compatible with your CUDA toolkit
- CUDA Toolkit **12.x** (12.0+ minimum)
- Rust toolchain matching project MSRV (see `rust-toolchain.toml`)

## 1) Install CUDA Toolkit

Install CUDA 12.x using NVIDIA packages for your OS.

After installation, ensure toolkit binaries and shared libraries are discoverable:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

Verify the environment:

```bash
nvcc --version
nvidia-smi
```

## 2) Build with GPU support

Use explicit feature selection (default features are intentionally empty):

```bash
cargo build --no-default-features --features gpu
```

Run GPU tests where hardware is available:

```bash
cargo test --workspace --no-default-features --features gpu
```

If you only need compilation validation in non-GPU environments:

```bash
cargo check --workspace --no-default-features --features gpu,cpu
```

## 3) Runtime notes

- Prefer `gpu` feature in new code.
- `cuda` is a compatibility alias for `gpu`.
- Use unified cfg predicates for GPU code paths:

```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
```

## 4) CI recommendations

For reliable CUDA validation in CI:

1. Keep a compile-only GPU job (works without GPU hardware).
2. Add a runtime GPU smoke job on self-hosted NVIDIA runners.
3. Test at least one CUDA 12.x version used in production.

The repository already includes a dedicated workflow at `.github/workflows/gpu-smoke.yml`.

## Troubleshooting

### Linker cannot find CUDA libraries

Symptoms include missing `-lcuda`, `-lnvrtc`, `-lcublas`, or `-lcurand`.

- Confirm `$CUDA_HOME/lib64` exists.
- Export `LD_LIBRARY_PATH` to include CUDA `lib64`.
- Re-run with verbose build output:

```bash
cargo build -vv --no-default-features --features gpu
```

### No CUDA device found at runtime

- Ensure NVIDIA driver is loaded (`nvidia-smi`).
- Confirm the process can access `/dev/nvidia*` devices.
- In containers, pass GPU devices/runtime correctly (e.g. NVIDIA Container Toolkit).

### GPU tests are flaky in CI

- Serialize GPU integration tests if they share device-global state.
- Reduce parallelism (`-- --test-threads=1`) for smoke checks.
- Split compile-only and runtime GPU jobs.

## Related docs

- `docs/development/gpu-setup-guide.md` (extended examples)
- `docs/howto/receipt-verification.md`
- `docs/reference/validation-gates.md`
