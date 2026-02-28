# ADR-008: Kernel Embedding Strategy

**Status**: ACCEPTED
**Date**: 2025-06-24
**Context**: How GPU kernel source code is shipped and compiled

---

## Context

GPU kernels (OpenCL `.cl`, Vulkan `.glsl` / SPIR-V, CUDA `.cu`) must be
available at runtime for compilation or dispatch.  Three strategies exist:

| Strategy | Ship Format | First-Launch Cost | Cross-Platform |
|----------|-----------|------------------|---------------|
| Embed source (`include_str!`) | Text in binary | Compile on first use | ✅ |
| Pre-compile to SPIR-V / PTX | Binary blob in binary | Near-zero | Partial |
| External files | Separate `.cl` / `.spv` files | Compile on first use | ✅ |

---

## Decision

**Embed kernel source via `include_str!` and compile at runtime.**

Each kernel is stored as a `.cl` (or `.glsl`) file in the backend crate's
`csrc/` directory and embedded into the Rust binary at compile time.
Runtime compilation produces device-specific binaries that are cached to
disk.

---

## Rationale

### 1. Simplicity
No external build tools (SPIR-V compiler, `clang`, `glslc`) required at
build time.  The Rust toolchain is sufficient — `include_str!` is a
built-in macro.

### 2. Cross-Platform by Default
OpenCL source compiles to the target device's ISA at runtime.  A single
binary works on Intel, AMD, and NVIDIA GPUs without separate builds per
vendor.

### 3. No Distribution Complexity
The kernel source is baked into the executable.  There are no separate
`.cl` files to package, no `KERNEL_DIR` environment variable to configure,
and no risk of version mismatch between binary and kernel files.

### 4. Developer Experience
Kernel authors edit a `.cl` file and re-run `cargo build` — the updated
source is embedded automatically.  No separate compilation step.

---

## Consequences

### Positive
- ✅ Zero external build dependencies for GPU kernels
- ✅ Single-file distribution (binary includes all kernels)
- ✅ Runtime compilation targets the exact device present
- ✅ Easy to inspect embedded kernels (`strings binary | grep __kernel`)

### Negative
- ⚠️ First-launch latency: OpenCL runtime compilation takes 50–500 ms per
  kernel depending on driver and kernel complexity
- ⚠️ No compile-time validation of kernel syntax — errors surface at
  runtime
- ⚠️ Binary size increases slightly (kernel sources are small — typically
  <10 KB per kernel)

### Mitigations
1. **SPIR-V Pre-Compilation Cache**: After first runtime compilation, the
   compiled binary is cached in
   `$XDG_CACHE_HOME/bitnet/kernels/<hash>.bin`.  Subsequent launches load
   the cached binary directly, reducing startup to <1 ms per kernel.
2. **Build-Script Validation** (optional): A `build.rs` step can invoke
   `clang -cl-std=CL3.0 -fsyntax-only` to catch syntax errors at build
   time on machines with the OpenCL SDK installed.
3. **SPIR-V Embedding** (future): For performance-critical deployments,
   SPIR-V blobs can be pre-compiled and embedded alongside the source as
   a fast-path.  The runtime tries the SPIR-V blob first, falling back to
   source compilation if the blob is incompatible.

---

## Alternatives Considered

### Pre-Compile to SPIR-V at Build Time
**Deferred**: Requires `spirv-tools` or `clang` in the build environment.
Breaks the "no GPU SDK at build time" principle (ADR-006).  Will be
offered as an opt-in `build.rs` step for release builds.

### Ship External Kernel Files
**Rejected**: Adds deployment complexity (file paths, packaging, version
skew).  Contradicts the single-binary distribution goal.

### Ahead-of-Time PTX for CUDA
**Not applicable**: The CUDA backend uses `cudarc` which already handles
JIT compilation from PTX.  This ADR focuses on OpenCL and Vulkan.

---

## References

- `include_str!` usage: `crates/bitnet-kernels/src/gpu/opencl/kernels/`
- Kernel cache implementation: `crates/bitnet-kernels/src/gpu/kernel_cache.rs`
- ADR-006: Dynamic GPU Library Linking

---

## Changelog

- **2025-06-24**: Initial decision
