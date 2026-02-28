# ADR-001: Use OpenCL as initial Intel GPU backend

- **Status:** Accepted
- **Date:** 2025-07-14
- **Context:** bitnet-rs needs Intel Arc GPU support.  Four candidates were evaluated:
  OpenCL (via Khronos), Level Zero (Intel oneAPI native), Vulkan compute shaders,
  and SYCL/DPC++ (Intel's C++-based model).  The selection criteria were: driver
  maturity on Intel Arc (A-series) and integrated Xe graphics, Rust crate ecosystem
  quality, cross-vendor portability, and time-to-first-kernel.

  Intel ships production OpenCL 3.0 drivers for both Arc discrete and integrated GPUs
  on Windows and Linux.  The `opencl3` Rust crate provides safe, idiomatic bindings
  with full 3.0 coverage.  Level Zero offers lower overhead but has no maintained Rust
  bindings.  Vulkan compute is viable but requires SPIR-V toolchains and has a steeper
  bring-up cost.  SYCL requires a C++ host compiler (icpx) and FFI bridging.

- **Decision:** Use OpenCL 3.0 via the `opencl3` crate for the initial Intel GPU
  backend (`bitnet-opencl`).  Kernel source is embedded as `.cl` files (see
  [ADR-004](./ADR-004-kernel-compilation-strategy.md)).  The implementation targets
  Intel Arc A-series and Xe integrated GPUs but will also work on any conformant
  OpenCL 3.0 device (AMD, NVIDIA with `cl_nv` driver, Qualcomm Adreno).

- **Consequences:**
  - *Positive:* Cross-vendor compatibility from day one; mature driver ecosystem;
    `opencl3` crate is well-maintained and idiomatic; same `.cl` kernels run on
    Intel, AMD, and NVIDIA without rewrite.
  - *Positive:* Aligns with the microcrate pattern ([ADR-002](./ADR-002-microcrate-per-backend.md)) —
    `bitnet-opencl` can be developed and tested independently.
  - *Negative:* Slightly lower performance ceiling than Level Zero on Intel hardware
    (estimated 5–15% overhead for memory management and kernel dispatch).
  - *Negative:* OpenCL lacks some modern features (e.g., cooperative groups, native
    sub-group shuffles on all vendors) that Level Zero / Vulkan provide.
  - *Risk:* Intel could deprecate OpenCL in favor of Level Zero long-term; mitigated
    by the HAL abstraction ([ADR-003](./ADR-003-gpu-hal-abstraction.md)) which allows
    swapping backends without inference code changes.

## Alternatives considered

- **Level Zero (oneAPI):** Lowest-latency path to Intel GPUs; no stable Rust crate;
  would require raw FFI bindings and ongoing maintenance.  Planned as a future
  `bitnet-level-zero` crate once Rust bindings mature.
- **Vulkan Compute:** Portable and well-supported in Rust (`ash`, `vulkano`), but
  requires SPIR-V pre-compilation or `glslangValidator` at build time; higher API
  complexity for compute-only workloads (descriptor sets, pipeline barriers).
- **SYCL/DPC++:** Best Intel optimization story; C++ only; requires FFI bridge;
  tight coupling to Intel icpx compiler version.

## How to revert

Replace `bitnet-opencl` with `bitnet-level-zero` or `bitnet-vulkan`.  The HAL
abstraction ([ADR-003](./ADR-003-gpu-hal-abstraction.md)) ensures inference code is
backend-agnostic, so the change is confined to the backend crate and Cargo feature
wiring.
