# ADR-004: Runtime kernel compilation with caching

- **Status:** Accepted
- **Date:** 2025-07-14
- **Context:** OpenCL and Vulkan backends need compiled kernel programs.  Three
  strategies were evaluated:

  1. **Build-time compilation** — compile `.cl` → SPIR-V or binary during `cargo build`
     via `build.rs`.
  2. **SPIR-V pre-compilation** — ship pre-compiled SPIR-V blobs in the crate.
  3. **Runtime compilation with caching** — embed `.cl` source, compile on first launch,
     cache the binary for subsequent runs.

  Build-time compilation requires the OpenCL SDK and/or `clang` on every developer
  machine and CI runner, complicating the build matrix.  SPIR-V pre-compilation is
  portable but loses device-specific optimizations and requires a SPIR-V toolchain at
  development time.  Runtime compilation is the standard practice in GPU computing and
  matches how NVIDIA's CUDA driver compiles PTX.

- **Decision:** Embed OpenCL kernel source files via `include_str!("kernels/matmul.cl")`
  at compile time.  At runtime:

  1. Compute a cache key: `sha256(kernel_source || device_name || driver_version)`.
  2. Look up the cache key in `$BITNET_CACHE_DIR/kernels/` (default:
     `~/.cache/bitnet/kernels/` on Linux, `%LOCALAPPDATA%\bitnet\kernels\` on Windows).
  3. If a cached binary exists, load it via `clCreateProgramWithBinary`.
  4. If not, compile from source via `clBuildProgram`, store the binary, and log the
     compilation time.

  The same pattern applies to Vulkan (GLSL → SPIR-V via `shaderc` at runtime) and
  future backends.

  A `BITNET_NO_KERNEL_CACHE=1` environment variable forces recompilation (useful for
  kernel development).

- **Consequences:**
  - *Positive:* Zero build-time dependencies on GPU SDKs — `cargo build` works on any
    machine; only the GPU driver is needed at runtime.
  - *Positive:* Device-specific optimizations — the OpenCL driver can apply
    architecture-specific code generation for the exact GPU present.
  - *Positive:* Cache makes subsequent launches near-instant; first-launch penalty is
    typically 100–500 ms for the bitnet kernel set.
  - *Positive:* Kernel source is auditable in the repository (no opaque blobs).
  - *Negative:* First launch is slower (compilation overhead); mitigated by cache and
    by a progress indicator in `bitnet-cli`.
  - *Negative:* Cache invalidation adds complexity (driver updates change the cache
    key, correctly triggering recompilation).
  - *Negative:* Kernel source is visible to users (IP consideration); acceptable for
    an open-source project.

## Alternatives considered

- **Build-time compilation (`build.rs`):** Guarantees cached binaries at install time
  but requires OpenCL SDK / `clang` for every `cargo build`; breaks `cargo install`
  on machines without the SDK.
- **SPIR-V pre-compilation:** Portable intermediate format; loses device-specific
  optimizations; requires `spirv-tools` or `clspv` in the build pipeline; SPIR-V
  support varies across vendors.
- **Hybrid (ship SPIR-V + runtime fallback):** Best of both worlds but doubles
  maintenance; considered for a future optimization if first-launch latency becomes
  a user complaint.

## How to revert

Switch to build-time compilation by adding a `build.rs` that invokes `clBuildProgram`
(or an offline compiler) and embeds the binary via `include_bytes!`.  Remove the
runtime cache directory logic.
