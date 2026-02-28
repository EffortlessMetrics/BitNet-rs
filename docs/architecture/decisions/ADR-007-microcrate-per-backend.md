# ADR-007: Microcrate per GPU Backend

**Status**: ACCEPTED
**Date**: 2025-06-24
**Context**: Workspace organisation for multi-GPU-backend support

---

## Context

BitNet-rs targets multiple GPU backends: CUDA (NVIDIA), OpenCL (Intel /
AMD), Vulkan (cross-vendor), and potentially Metal (Apple).  Each backend
has distinct dependencies, build requirements, and feature-flag surfaces.

Two workspace structures were considered:

### Option A: Single `bitnet-kernels` Crate
All backends live under `crates/bitnet-kernels/src/gpu/{cuda,opencl,vulkan}/`.
Feature flags gate each sub-module.

### Option B: Separate Microcrate per Backend
Each backend gets its own crate:
```
crates/bitnet-kernels/          (core trait + CPU fallback)
crates/bitnet-gpu-cuda/         (NVIDIA CUDA)
crates/bitnet-gpu-opencl/       (Intel/AMD OpenCL)
crates/bitnet-gpu-vulkan/       (Vulkan compute)
```

---

## Decision

**Option B — one microcrate per GPU backend.**

Each backend is a standalone workspace member that depends on
`bitnet-common` for shared types and implements the `KernelProvider` trait
from `bitnet-kernels`.

---

## Rationale

### 1. Single Responsibility
Each crate owns exactly one backend.  Its `Cargo.toml` declares only the
dependencies needed for that backend (e.g., `opencl3` for OpenCL,
`cudarc` for CUDA).  No conditional `dep:` gymnastics.

### 2. Independent Compilation
`cargo check -p bitnet-gpu-opencl` compiles only OpenCL code.  Developers
working on one backend are not slowed by changes to another.  CI can
parallelise backend checks across matrix jobs.

### 3. Clear Dependency Graph
The workspace dependency graph makes it obvious which crates pull in GPU
SDKs.  Auditing `Cargo.lock` for unexpected native dependencies is
straightforward.

### 4. Testability
Each backend crate can have its own integration tests, property tests, and
snapshot tests without polluting the core kernel test suite.

---

## Consequences

### Positive
- ✅ Minimal compile scope per backend
- ✅ Clean `Cargo.toml` per crate (no conditional deps)
- ✅ Parallel CI matrix across backends
- ✅ Easy to add new backends (copy template, implement trait)

### Negative
- ⚠️ More crates to maintain (one per backend + the core trait crate)
- ⚠️ Inter-crate coordination for shared kernel utilities (solved by
  `bitnet-common` and `bitnet-kernels` as the trait host)
- ⚠️ Version bumps touch more `Cargo.toml` files (mitigated by
  `workspace.version`)

### Mitigations
1. **Workspace Version**: All backend crates inherit
   `version.workspace = true` to keep versions in sync.
2. **Shared Utilities**: Common GPU helpers (memory pool, profiling,
   error mapping) live in `bitnet-kernels` under a `gpu-common` module
   that backends can depend on.
3. **Template**: A `crates/bitnet-gpu-template/` skeleton accelerates
   new backend creation.

---

## Alternatives Considered

### Single Mega-Crate (Option A)
**Rejected**: Feature-flag complexity grows quadratically with backends.
Conditional `dep:` attributes make `Cargo.toml` hard to audit.  A
compile-time error in one backend blocks `cargo check` for all.

### Runtime Plugin System
**Deferred**: A `libloading`-based plugin architecture is more flexible
but introduces ABI stability requirements.  May be revisited for
third-party backend contributions.

---

## References

- Workspace members: `Cargo.toml` `[workspace]` section
- `KernelProvider` trait: `crates/bitnet-kernels/src/lib.rs`
- SRP microcrate pattern: `bitnet-logits`, `bitnet-gguf`, `bitnet-device-probe`

---

## Changelog

- **2025-06-24**: Initial decision
