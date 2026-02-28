# ADR-002: One microcrate per GPU backend

- **Status:** Accepted
- **Date:** 2025-07-14
- **Context:** bitnet-rs follows an SRP microcrate architecture (see `bitnet-logits`,
  `bitnet-gguf`, `bitnet-generation`, `bitnet-device-probe`, `bitnet-engine-core`).
  GPU support requires multiple backend implementations (CUDA, OpenCL, Vulkan, ROCm,
  Metal, WebGPU) that differ in dependencies, platform support, and build
  requirements.  Mixing them into a single crate would create complex conditional
  compilation, slow builds for users who only need one backend, and hard-to-isolate
  test failures.

- **Decision:** Each GPU backend lives in its own workspace crate:

  | Crate              | Backend     | Primary target hardware          |
  |--------------------|-------------|----------------------------------|
  | `bitnet-cuda`      | CUDA        | NVIDIA GPUs (existing)           |
  | `bitnet-opencl`    | OpenCL 3.0  | Intel Arc, AMD, cross-vendor     |
  | `bitnet-vulkan`    | Vulkan      | Cross-platform compute           |
  | `bitnet-rocm`      | ROCm/HIP    | AMD Radeon / Instinct            |
  | `bitnet-metal`     | Metal       | Apple Silicon                    |
  | `bitnet-webgpu`    | WebGPU/wgpu | Browser & portable GPU           |

  All backend crates implement the common HAL traits from `bitnet-gpu-hal`
  ([ADR-003](./ADR-003-gpu-hal-abstraction.md)) and are wired into the workspace via
  Cargo features ([ADR-006](./ADR-006-feature-flag-design.md)).

  Backend crates are excluded from `default-members` in the root `Cargo.toml` and
  must be opted into explicitly, matching the existing pattern for wasm, Python
  bindings, and FFI crates.

- **Consequences:**
  - *Positive:* Clean dependency isolation — `bitnet-opencl` pulls in `opencl3` but
    never `cudarc`; `bitnet-metal` only compiles on macOS.
  - *Positive:* Independent CI lanes — each backend can have its own test matrix
    (e.g., `gpu-smoke.yml` for CUDA, a separate workflow for OpenCL on Intel runners).
  - *Positive:* Users only compile the backends they need; build times stay fast.
  - *Positive:* New backends can be added without touching existing crates.
  - *Negative:* More crates to maintain; version alignment across backends requires
    discipline (mitigated by workspace dependency inheritance).
  - *Negative:* Cross-backend integration tests need a dedicated harness that
    conditionally enables multiple features.

## Alternatives considered

- **Single `bitnet-gpu` crate with cfg-gated modules:** Fewer crates but entangles
  dependencies; a CUDA build failure blocks OpenCL development; harder to
  conditionally compile on CI runners that lack specific hardware.
- **External plugin crates (out-of-tree):** Maximum isolation but loses workspace
  coherence, CI integration, and version synchronization.

## How to revert

Merge the backend crates back into a single `bitnet-gpu` crate with
`#[cfg(feature = "...")]` modules.  The HAL trait boundary
([ADR-003](./ADR-003-gpu-hal-abstraction.md)) remains unchanged.
