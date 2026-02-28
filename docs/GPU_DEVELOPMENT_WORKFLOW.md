# GPU Development Workflow

This document describes the development cycle, branching conventions, PR process,
CI pipeline, and review workflow for GPU backend contributions to BitNet-rs.

## Development Cycle

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌───────────┐
│  1. Design   │────▶│  2. Implement │────▶│  3. Test     │────▶│  4. Review │
│  & Plan      │     │  & Iterate    │     │  & Validate  │     │  & Merge   │
└─────────────┘     └──────────────┘     └─────────────┘     └───────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
  Open issue or        Write kernel +       CPU ref tests       PR review +
  discussion with      CPU reference        GPU hardware        CI green +
  design sketch        implementation       tests (if avail)    benchmarks
                                            Property tests      checked
```

### Phase 1: Design & Plan

- Open a GitHub Issue describing the kernel or backend you plan to add
- For significant changes, start a Discussion to gather feedback
- Reference the [GPU Kernel Architecture](gpu-kernel-architecture.md) for design context

### Phase 2: Implement & Iterate

- Follow the [Adding a New Kernel](GPU_CONTRIBUTOR_GUIDE.md#adding-a-new-kernel) or
  [Adding a New Backend](GPU_CONTRIBUTOR_GUIDE.md#adding-a-new-backend) guide
- Write the CPU reference implementation first — it serves as the ground truth
- Iterate locally with `cargo check` and CPU tests before touching GPU code

### Phase 3: Test & Validate

- Run `cargo nextest run --workspace --no-default-features --features cpu` for CPU validation
- If you have GPU hardware, run `cargo nextest run --workspace --no-default-features --features gpu`
- Use `BITNET_GPU_FAKE=1` for GPU code-path testing without hardware
- Add property tests for numeric invariants (e.g., output length, value bounds)

### Phase 4: Review & Merge

- Open a PR following the conventions below
- CI runs automatically; add labels for GPU-specific checks
- Address review feedback; maintainer merges when CI is green

---

## Branch Naming

GPU feature branches follow the `intel-gpu/<feature-name>` convention:

```
intel-gpu/contributor-guide-v2    # Documentation
intel-gpu/opencl-matmul           # New kernel
intel-gpu/rocm-backend            # New backend
intel-gpu/cuda-memory-pool        # Optimisation
intel-gpu/ci-vulkan-smoke         # CI infrastructure
```

**Pattern**: `intel-gpu/<short-descriptive-name>`

For non-GPU changes that touch GPU code (e.g., refactoring `KernelProvider`),
use the standard branch naming: `feature/<name>` or `refactor/<name>`.

---

## PR Template for GPU Features

When opening a PR for GPU backend work, include:

```markdown
## Summary

Brief description of what this PR does.

## Type of Change

- [ ] New kernel
- [ ] New backend
- [ ] Kernel optimisation
- [ ] Bug fix
- [ ] CI / infrastructure
- [ ] Documentation

## Checklist

### Code Quality
- [ ] CPU reference implementation included
- [ ] Unit tests pass: `cargo nextest run --workspace --no-default-features --features cpu`
- [ ] Clippy clean: `cargo clippy --all-targets --no-default-features --features cpu -- -D warnings`
- [ ] `cargo fmt --all` applied
- [ ] Feature gates use unified `#[cfg(any(feature = "gpu", feature = "cuda"))]` predicate

### Testing
- [ ] CPU reference tests added
- [ ] GPU hardware tests added (with `#[ignore = "requires <hardware>"]`)
- [ ] Property tests for numeric invariants
- [ ] Tested with `BITNET_GPU_FAKE=1` (if applicable)
- [ ] No bare `#[ignore]` (all have justification strings)

### Documentation
- [ ] Rustdoc on all public items
- [ ] Kernel inventory updated (if new kernel)
- [ ] Architecture docs updated (if structural change)

### Performance (if applicable)
- [ ] Benchmark results included in PR description
- [ ] No regression on existing benchmarks
- [ ] Receipt verification passes: `cargo run -p xtask -- verify-receipt`

## Hardware Tested On

- [ ] No hardware (CPU reference only)
- [ ] NVIDIA GPU (model: _______)
- [ ] AMD GPU (model: _______)
- [ ] Intel GPU (model: _______)

## Related Issues

Closes #___
```

---

## Review Checklist

Reviewers of GPU PRs should verify:

### Correctness
- [ ] CPU reference matches expected mathematical behaviour
- [ ] GPU kernel produces results within tolerance of CPU reference
- [ ] Edge cases handled (empty input, single element, non-power-of-2 sizes)
- [ ] Error paths tested (device unavailable, out of memory)

### Safety
- [ ] `unsafe` blocks are minimal and well-documented
- [ ] No undefined behaviour in CUDA/HIP kernel launches
- [ ] Memory allocations are bounded and cleaned up
- [ ] Feature gates are correct (unified GPU predicate)

### Performance
- [ ] No unnecessary host↔device memory transfers
- [ ] Appropriate block/grid dimensions for the workload
- [ ] Memory access patterns are coalesced where possible
- [ ] No performance regression on existing benchmarks

### Style
- [ ] Follows [Code Style](GPU_CONTRIBUTOR_GUIDE.md#code-style) conventions
- [ ] `warn_once!` used for hot-path warnings (not `warn!`)
- [ ] Ignored tests have justification strings
- [ ] Environment-mutating tests use `EnvGuard` + `#[serial(bitnet_env)]`

---

## CI Pipeline

### Core Checks (Always Run)

These run on every PR and push to `main`:

| Check | Workflow | Description |
|-------|----------|-------------|
| Build & Test | `ci-core.yml` | Workspace build + CPU tests |
| Clippy | `ci-core.yml` | Lint with `-D warnings` |
| Formatting | `ci-core.yml` | `cargo fmt --check` |
| Guards | `guards.yml` | SHA-pinned actions, `--locked` flags, MSRV |

### GPU-Specific Checks (Label-Gated)

Add CI labels to trigger GPU validation:

| Label | Workflow | What It Checks | Runner |
|-------|----------|----------------|--------|
| `gpu` | `gpu.yml` | CUDA kernel tests, GPU dispatch | Self-hosted (GPU) |
| `quant` | Quantization CI | I2S, TL1, TL2, QK256 feature matrix | Standard |
| `perf` | Performance Gate | Benchmark regression detection | Standard |
| `crossval` | Cross-Validation | Rust vs C++ reference comparison | Standard |

**How to add labels:**

```bash
# When creating the PR
gh pr create --title "feat: ..." --label gpu,quant

# After PR creation
gh pr edit <number> --add-label gpu
```

### Weekly Smoke Tests

These run on a schedule regardless of PRs:

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `gpu-smoke.yml` | Weekly | CUDA compile check + smoke test |
| `rocm-smoke.yml` | Weekly | ROCm/HIP compile check |
| `nightly-fuzz.yml` | Nightly | Fuzz targets (15 targets × 60s) |

### Pipeline Flow

```
Push / PR opened
       │
       ▼
┌─────────────────┐
│   Core Checks   │  ← Always runs (build, test, clippy, fmt, guards)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Label Checks   │  ← Only if labels present (gpu, quant, perf, crossval)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Branch Protect  │  ← Core checks required; label checks informational
└─────────────────┘
```

---

## Performance Regression Detection

### Baselines

Performance baselines are stored in `baselines/` and `ci/inference.json`. The receipt
system tracks inference performance with schema v1.0.0.

### Detecting Regressions

1. **Add the `perf` label** to your PR to trigger benchmark validation
2. CI compares current benchmarks against stored baselines
3. Significant regressions (>5% slowdown) are flagged in the PR

### Running Benchmarks Locally

```bash
# Criterion benchmarks (SRP operations)
cargo bench --bench srp_ops

# Full inference benchmark (requires model file)
BITNET_GGUF=models/model.gguf cargo run -p xtask -- verify-receipt

# With native SIMD optimisation
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo bench --bench srp_ops
```

### Receipt Verification

After any inference-affecting change, verify the receipt:

```bash
# CPU receipt
cargo run -p xtask -- verify-receipt --path ci/inference.json

# GPU receipt (requires GPU kernels)
cargo run -p xtask -- verify-receipt --path ci/inference.json --require-gpu-kernels
```

The receipt enforces 8 validation gates including `compute_path = "real"` (never `"mock"`).

---

## Quick Reference

| Task | Command |
|------|---------|
| Build (CPU) | `cargo build --no-default-features --features cpu` |
| Build (GPU) | `cargo build --no-default-features --features gpu` |
| Test (CPU) | `cargo nextest run --workspace --no-default-features --features cpu` |
| Test (GPU) | `cargo nextest run --workspace --no-default-features --features gpu` |
| Lint | `cargo clippy --all-targets --no-default-features --features cpu -- -D warnings` |
| Format | `cargo fmt --all` |
| Benchmark | `cargo bench --bench srp_ops` |
| Guards | `make guards` |
| Verify receipt | `cargo run -p xtask -- verify-receipt --path ci/inference.json` |

---

## Further Reading

- [GPU Contributor Guide](GPU_CONTRIBUTOR_GUIDE.md) — Kernel and backend development
- [GPU Setup Guide](GPU_SETUP.md) — CUDA toolkit installation
- [Intel GPU Setup](INTEL_GPU_SETUP.md) — Intel Arc / OpenCL setup
- [GPU Kernel Architecture](gpu-kernel-architecture.md) — Design and phase roadmap
- [CONTRIBUTING.md](../CONTRIBUTING.md) — General contribution guidelines
