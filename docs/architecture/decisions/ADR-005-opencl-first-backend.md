# ADR-005: OpenCL-First GPU Backend

**Status**: ACCEPTED
**Date**: 2025-06-24
**Context**: Intel Arc GPU support for BitNet-rs inference

---

## Context

BitNet-rs needs Intel Arc A-series GPU support for accelerated 1-bit LLM
inference.  Multiple compute APIs are available on Intel hardware:

| API | Rust Ecosystem | Intel ICD | Maturity |
|-----|---------------|-----------|----------|
| OpenCL 3.0 | `opencl3` crate (stable) | Shipped in driver | Production |
| Vulkan Compute | `ash` / `vulkano` | Shipped in driver | Production |
| Level-Zero (oneAPI) | Minimal bindings | Separate install | Emerging |
| SYCL (via DPC++) | No native Rust crate | oneAPI toolkit | C++ only |

### Constraints

- Must compile without GPU libraries installed (dynamic linking only)
- Must coexist with the existing CUDA backend
- Must support Intel Arc A770/A750 (Xe-HPG, 32 EUs / 512 cores)
- Must be testable on CI runners without GPU hardware

---

## Decision

**Use OpenCL 3.0 as the first Intel GPU backend.**

Vulkan Compute and Level-Zero are deferred to follow-up milestones.

---

## Rationale

### 1. Mature Rust Crate
The `opencl3` crate provides safe, well-maintained bindings with buffer
management, kernel compilation, and command-queue abstractions.  It has
>600k total downloads and is actively maintained.

### 2. Intel ICD Ships in the Driver
Intel's OpenCL ICD (`intel-opencl-icd`) is bundled with the standard GPU
driver on Windows and Linux.  No additional SDK install is required at
runtime, reducing user friction.

### 3. Simpler Programming Model
OpenCL's buffer + kernel + queue model maps cleanly onto the existing
`KernelProvider` trait.  A single `.cl` source file per kernel is easy to
review, embed via `include_str!`, and compile at runtime.

### 4. Cross-Vendor Portability
The same OpenCL kernels can run on NVIDIA (via pocl or vendor ICD), AMD
(via ROCm ICD), and Intel GPUs, giving users a fallback when CUDA is
unavailable.

---

## Consequences

### Positive
- ✅ Fast time-to-prototype — working matmul kernel in <1 week
- ✅ No build-time GPU dependency (dynamic linking)
- ✅ Reusable across Intel, AMD, and NVIDIA hardware
- ✅ Well-understood debugging tools (`clinfo`, `oclgrind`)

### Negative
- ⚠️ Slightly lower peak throughput compared to Level-Zero (no
  direct GPU memory mapping, higher dispatch overhead)
- ⚠️ OpenCL 3.0 feature support varies by vendor — need runtime feature
  queries for subgroups, fp16, etc.
- ⚠️ No SPIR-V pre-compilation in OpenCL 1.2 fallback — must compile from
  source on first launch

### Mitigations
1. **SPIR-V Cache**: Pre-compile kernels to SPIR-V and cache the binary in
   `$XDG_CACHE_HOME/bitnet/kernels/` for subsequent launches.
2. **Feature Probing**: Use `CL_DEVICE_EXTENSIONS` to query subgroup and
   fp16 support at device enumeration time.
3. **Level-Zero Upgrade Path**: The `KernelProvider` trait is
   backend-agnostic — a Level-Zero provider can replace OpenCL without
   changing upper layers.

---

## Alternatives Considered

### Vulkan Compute
**Deferred**: Higher API complexity (descriptor sets, pipeline layouts,
memory barriers).  Better suited for graphics-adjacent workloads.  Will be
evaluated for v0.3.

### Level-Zero
**Deferred**: Requires separate oneAPI toolkit installation.  Rust
bindings are immature.  Best reserved for when sub-microsecond dispatch
latency matters.

### SYCL via DPC++
**Rejected**: No native Rust integration.  Would require FFI to a C++
runtime, conflicting with the project's pure-Rust philosophy.

---

## References

- Intel OpenCL ICD: https://github.com/intel/compute-runtime
- `opencl3` crate: https://crates.io/crates/opencl3
- BitNet-rs KernelProvider trait: `crates/bitnet-kernels/src/lib.rs`

---

## Changelog

- **2025-06-24**: Initial decision for Intel Arc GPU support
