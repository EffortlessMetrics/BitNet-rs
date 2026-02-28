# macOS 26 / Apple Silicon Integration Roadmap

This document translates the current BitNet-rs architecture into a concrete plan for Apple macOS 26 (Tahoe) and recent Apple Silicon generations (M4/M5).

## Scope

BitNet-rs already has:
- CPU SIMD kernels (including ARM NEON).
- CUDA acceleration for NVIDIA GPUs.

BitNet-rs does **not yet** have:
- A production Apple GPU backend (Metal / MPS / MLX integration).
- Apple-specific accelerator capability probing.
- Multi-node validation for Thunderbolt 5 RDMA scenarios.

## Goals

1. Add an Apple GPU backend for inference kernels.
2. Detect Apple GPU and accelerator capabilities at runtime.
3. Keep CPU/NEON path performant as fallback on all macOS hardware.
4. Add macOS-specific CI/test coverage for backend correctness.
5. Add optional cluster-oriented validation for Thunderbolt 5 links.

## Workstreams

### 1) Apple GPU backend (critical path)

Primary crate impact:
- `crates/bitnet-kernels` (new Metal compute kernels and dispatch code).

Suggested implementation shape:
- Add a new module tree such as `crates/bitnet-kernels/src/metal/`.
- Gate via a new Cargo feature, for example `apple-gpu`.
- Keep CPU fallback available when Metal is unavailable.

Near-term deliverables:
- Kernel parity for the highest-impact inference operations first.
- Correctness tests that compare Metal outputs against CPU reference outputs.
- Performance benchmarks on Apple Silicon hardware.

### 2) Device probing and runtime backend selection

Primary crate impact:
- `crates/bitnet-device-probe`.

Suggested additions:
- Detect whether the host has an Apple GPU usable via Metal.
- Report capabilities in a backend-neutral structure used by inference/runtime layers.
- Distinguish backend availability from backend preference (availability should not force usage).

### 3) Build flags and platform wiring

Primary crate/config impact:
- Workspace `Cargo.toml` feature wiring.
- Potential crate-level `build.rs` updates where native framework linking is needed.

Suggested behavior:
- `cpu` remains the default recommended development path.
- `gpu` remains CUDA-specific.
- Add `apple-gpu` for Metal path, available on macOS targets.

### 4) Testing and CI

Primary impact:
- Existing test suites plus CI workflow matrices.

Suggested additions:
- Build and test jobs on macOS for `--features cpu` and `--features cpu,apple-gpu`.
- Deterministic correctness tests that validate Apple backend against CPU backend.
- Performance checks recorded as benchmark artifacts (non-gating initially).

### 5) Thunderbolt 5 / RDMA validation (optional extension)

Primary impact:
- Integration test scripts and documentation.

Suggested scope:
- Start with multi-node data transfer and inference service tests.
- Keep this as optional and non-blocking for core inference milestones.
- Treat larger-scale distributed inference as follow-up after backend parity.

## Priority Order

1. **Apple GPU backend prototype** (compile + run + correctness for small cases).
2. **Device probe updates** (runtime backend visibility).
3. **CI enablement** (macOS build/test lanes).
4. **Performance tuning** (throughput and memory behavior).
5. **RDMA/cluster validation** (optional expansion path).

## Risks and mitigations

- **Metal kernel complexity (high):**
  - Mitigate by implementing parity for a narrow op subset first.
  - Add golden-reference tests before optimization.
- **Hardware availability (medium):**
  - Ensure graceful CPU fallback and keep Apple-GPU tests opt-in where needed.
- **Backend divergence (medium):**
  - Enforce shared correctness fixtures across CPU/CUDA/Apple paths.

## Documentation updates to keep in sync

When implementation starts, update:
- `README.md` feature and status tables.
- `docs/GPU_SETUP.md` with Apple backend setup notes.
- `COMPATIBILITY.md` hardware/backend matrix.

## Notes on drivers and system policy

BitNet-rs is a user-space inference engine. No custom kernel extension is required for planned Apple support. Future hardware integration work should remain compatible with current macOS driver and security expectations.
