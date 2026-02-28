# ADR-006: Feature flag structure for multi-backend

- **Status:** Accepted
- **Date:** 2025-07-14
- **Context:** bitnet-rs already uses a feature-flag hierarchy:

  ```
  gpu  (umbrella) ─┬─ cuda
                    ├─ vulkan
                    └─ rocm
  cpu               ─ kernels + SIMD auto-detection
  metal             ─ Apple-only
  ```

  Adding OpenCL, Level Zero, WebGPU, and oneAPI backends requires extending this
  scheme without breaking existing users or CI.  Features must be **additive** (Cargo
  requirement) and compose cleanly for multi-backend builds (e.g.,
  `--features cpu,opencl` for CPU primary + OpenCL acceleration).

  The root `Cargo.toml` already propagates features through the workspace:
  `gpu` enables `bitnet-kernels/gpu`, `bitnet-inference/gpu`, and
  `bitnet-quantization/gpu`.

- **Decision:** Extend the feature hierarchy as follows:

  ```
  gpu  (umbrella) ─┬─ cuda       → bitnet-cuda
                    ├─ opencl     → bitnet-opencl     [NEW]
                    ├─ vulkan     → bitnet-vulkan
                    ├─ rocm       → bitnet-rocm
                    ├─ oneapi     → bitnet-level-zero  [NEW]
                    └─ webgpu     → bitnet-webgpu      [NEW]
  cpu               ─ SIMD auto (AVX-512 / AVX2 / NEON / scalar)
  metal             ─ bitnet-metal (Apple-only, not under gpu umbrella)
  ```

  Design rules:

  1. **`gpu` is the umbrella** — enables all GPU backends compiled for the current
     platform.  Users who want a single backend use the specific flag
     (`--features opencl`).
  2. **Backend flags are additive** — `--features cuda,opencl` compiles both; runtime
     selection picks the best available ([ADR-005](./ADR-005-cpu-fallback-strategy.md)).
  3. **`cpu` is always available** — it is the fallback; GPU features add acceleration
     on top.  `cpu` is the minimal useful feature set.
  4. **`metal` stays outside `gpu`** — Apple platforms require different CI
     infrastructure; `metal` implies Apple-only dependencies.
  5. **Backend crates own their dependencies** — `bitnet-opencl` declares
     `dep:opencl3`; the root `Cargo.toml` only forwards the feature flag.
  6. **`default = []`** — unchanged; users must opt in explicitly (matches existing
     convention documented in CLAUDE.md).

  Feature propagation through the workspace:

  ```toml
  # Root Cargo.toml (excerpt)
  [features]
  opencl = ["kernels", "inference", "tokenizers",
            "bitnet-kernels/opencl", "bitnet-inference/gpu",
            "bitnet-quantization/gpu"]
  oneapi = ["kernels", "inference", "tokenizers",
            "bitnet-kernels/oneapi", "bitnet-inference/gpu",
            "bitnet-quantization/gpu"]
  ```

- **Consequences:**
  - *Positive:* Fine-grained compilation — CI runners without CUDA can still test
    OpenCL; WebAssembly builds enable only `webgpu`.
  - *Positive:* Clean dependency tree — no user pulls in `cudarc` when they only
    want OpenCL.
  - *Positive:* Composable — `--features cpu,opencl` gives CPU fallback + OpenCL
    acceleration in one binary.
  - *Positive:* Backward compatible — existing `--features gpu` and `--features cuda`
    continue to work unchanged.
  - *Negative:* Feature matrix grows; CI must test key combinations (mitigated by
    the `ci-core.yml` matrix strategy).
  - *Negative:* Potential for feature-flag conflicts if two backends claim the same
    kernel name; mitigated by the HAL trait boundary
    ([ADR-003](./ADR-003-gpu-hal-abstraction.md)).

## Alternatives considered

- **Single `gpu` feature, select backend via env var:** Simpler feature surface but
  compiles all backends unconditionally; slow builds; pulls in every GPU SDK.
- **Separate top-level crates (not features):** Maximum isolation but users must
  choose the right crate at dependency declaration time; loses workspace coherence.
- **`cfg_if!` mega-crate:** One `bitnet-gpu` crate with `cfg_if!` dispatching; tested
  in early CUDA integration and found to be fragile and hard to test in isolation.

## How to revert

Collapse backend-specific features into a single `gpu` feature and use runtime
detection only.  Remove per-backend crates and move code into `bitnet-kernels/gpu/`.
