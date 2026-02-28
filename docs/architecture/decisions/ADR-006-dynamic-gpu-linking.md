# ADR-006: Dynamic GPU Library Linking

**Status**: ACCEPTED
**Date**: 2025-06-24
**Context**: GPU backend compilation and distribution strategy

---

## Context

BitNet-rs must compile on machines **without** GPU libraries installed
(CI runners, developer laptops, ARM devices).  At the same time, users
with a GPU driver should get hardware acceleration automatically.

Two linking strategies exist:

| Strategy | Compile Requirement | Runtime Requirement | Error Mode |
|----------|-------------------|-------------------|------------|
| Static / build-time | GPU SDK headers + libs | GPU driver | Linker error if SDK absent |
| Dynamic / runtime | None | GPU driver (optional) | Graceful runtime fallback |

---

## Decision

**Use dynamic library loading (dlopen / LoadLibrary) for all GPU backends.**

GPU libraries (OpenCL ICD, CUDA runtime, Vulkan loader) are loaded at
runtime via the platform's dynamic linker.  If loading fails, the backend
reports "unavailable" and the `KernelManager` falls back to CPU.

---

## Rationale

### 1. Compile Everywhere
A single `cargo build --features cpu,gpu` produces the same binary whether
the build machine has GPU libraries or not.  This simplifies CI (no GPU
SDK in the build container) and distribution (one binary for GPU and
non-GPU machines).

### 2. Graceful Degradation
Users who install BitNet-rs on a headless server or a Raspberry Pi get
CPU inference automatically.  The absence of `libOpenCL.so` or
`nvcuda.dll` is a soft warning, not a hard error.

### 3. Multi-Backend Coexistence
The same binary can probe for CUDA, OpenCL, and Vulkan at startup,
selecting the best available backend.  This is impossible with static
linking when only one SDK is installed.

---

## Consequences

### Positive
- ✅ Single binary for all platforms (CPU + optional GPU)
- ✅ No GPU SDK required at build time
- ✅ CI can test GPU code paths (compile check) without GPU hardware
- ✅ Users get automatic CPU fallback

### Negative
- ⚠️ Runtime errors instead of link-time errors — a missing function
  pointer is detected at first kernel launch, not at compile time
- ⚠️ Slightly more complex initialisation code (symbol resolution,
  version checks)
- ⚠️ Harder to diagnose "driver too old" issues versus "driver missing"

### Mitigations
1. **Startup Diagnostics**: `BackendStartupSummary` logs
   `requested=X detected=[…] selected=Y` at launch.
2. **Feature Probing**: Each backend crate exposes a
   `gpu_available_runtime()` function that performs a full health check
   (load library → query devices → verify compute capability).
3. **Strict Mode**: `BITNET_STRICT_MODE=1` escalates a missing GPU to an
   error instead of a silent fallback.

---

## Alternatives Considered

### Static Linking Against GPU SDKs
**Rejected**: Would require CUDA Toolkit and Intel oneAPI SDK in every CI
job and on every developer machine.  Breaks the "compile everywhere" goal.

### Optional Crate Features Without Dynamic Loading
**Rejected**: Cargo features select code at compile time, but the linked
library must still be present.  Dynamic loading is the only way to defer
the requirement to runtime.

---

## References

- `bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime}`
- `BackendStartupSummary` in `bitnet-inference/src/backend.rs`
- `KernelManager::select_best()` fallback logic

---

## Changelog

- **2025-06-24**: Initial decision
