# Dual-Backend Support Implementation Roadmap

> **Last updated**: reflects implementation state after PRs #608–#636.
> Items marked ✅ are **done**; items marked 🔲 are **planned**.

---

## Implementation Status

### ✅ What's Implemented

| Capability | Location | PR |
|---|---|---|
| Library discovery (bitnet.cpp + llama.cpp) | `crossval/build.rs`, `crates/bitnet-sys/` | #608 |
| Feature lattice: `gpu` umbrella, `cuda = ["gpu"]` backward-compat alias | workspace `Cargo.toml` | #611 |
| Orthogonal runtime reporting: `gpu`/`cuda` never conflated | `crates/bitnet-runtime-feature-flags/src/lib.rs` | #611 |
| Kernel capability registry (single source of truth) | `crates/bitnet-common/src/kernel_registry.rs` | #611 |
| `xtask analyze-library` symbol inspection command | `xtask/src/analyze_library.rs` | #611 |
| `bitnet-device-probe` SRP microcrate | `crates/bitnet-device-probe/` | #629 |
| `bitnet-logits` SRP microcrate | `crates/bitnet-logits/` | #629 |
| `bitnet-generation` SRP microcrate | `crates/bitnet-generation/` | #629 |
| `bitnet-engine-core` SRP microcrate | `crates/bitnet-engine-core/` | #629 |
| `bitnet-gguf` SRP microcrate | `crates/bitnet-gguf/` | #629 |
| `KernelBackend`, `KernelCapabilities`, `SimdLevel` types | `crates/bitnet-common/src/kernel_registry.rs` | #611 |
| CodeRabbit path exclusions (archive, reports) | `.coderabbit.yaml` | #611 |
| GPU smoke workflow (compile-only PR lane) | `.github/workflows/gpu-smoke.yml` | #611 |
| Preflight backend availability checks | `xtask/` | #611 |
| `BITNET_GPU_FAKE`, `BITNET_STRICT_MODE` env guards | `crates/bitnet-device-probe/` | #609 |

### 🔲 What's Planned

1. **Build-time symbol detection → rustc-cfg flags** (`bitnet-sys`)
   - `build.rs` runs `nm`/`objdump -T` on found libs
   - Emits `bitnet_cpp_has_cuda`, `bitnet_cpp_has_bitnet_shim`
   - Code can compile-time gate "CUDA-backed C++ path" vs "CPU-only C++ path"

2. **Runtime backend validation + enforcement**
   - `BackendCapabilities` snapshot at startup: compiled + runtime available backends
   - `BackendRequest::Cuda` → hard error if CUDA runtime unavailable
   - Startup output: `requested=X detected=[…] selected=Y` (deterministic in logs + receipts)

3. **BDD grid as compile-coverage contract** (`xtask grid-check`)
   - Enumerate supported cells → `cargo check` per cell
   - Unsupported cells require a reason token (no silent gaps)

4. **CPU golden path E2E test** (always-on in PR CI)
   - Tiny synthetic GGUF fixture
   - Deterministic `generate` with fixed seed → assert output slice + receipt invariants
   - No model download needed

5. **CUDA smoke lane** (self-hosted runner, nightly/manual)
   - Allocate device, run small inference, upload parity receipt

6. **Scheduled fuzz/crossval evidence**
   - Nightly timeboxed fuzz runs with artifact upload
   - Real-model crossval producing receipts

7. **TL1/TL2 quantizer round-trip accuracy**
   - Known issue: `pack_2bit_values` clamps codes 2–3 due to LUT offset mismatch
   - `dequantize_scalar` gives max error ~2.0 on ternary inputs
   - Fix: shift LUT output to `[-2, 1]` before packing, OR update dequantize to subtract `num_levels/2`

---

## Feature Lattice Design

### Current (CUDA-first, non-CUDA-ready)

```toml
[features]
gpu = ["kernels", "inference", "tokenizers", "bitnet-kernels/gpu", ...]
cuda = ["gpu"]      # backward-compat alias — CUDA is the only GPU backend today
```

### Code Gating Rules

```rust
// CUDA-specific imports:
#[cfg(feature = "cuda")]
use cudarc::...;

// Generic "any GPU compiled":
#[cfg(feature = "gpu")]
pub fn gpu_dispatch() { ... }

use bitnet_device_probe::{gpu_compiled, gpu_available_runtime};
```

### Adding ROCm / Metal (future, additive only)

Adding a new GPU backend only requires new feature entries and new
backend-specific modules. Existing `#[cfg(feature = "gpu")]` code
continues to work without modification.

---

## Phase A: Build-time Symbol Detection (Planned)

Wire `xtask analyze-library` symbol analysis into `bitnet-sys/build.rs` to
emit rustc-cfg flags (`bitnet_cpp_has_cuda`, `bitnet_cpp_has_bitnet_shim`)
that let downstream code distinguish "CUDA-backed C++ path" vs "CPU-only C++ path"
at compile time.

---

## Phase B: Runtime Validation + Enforcement (Planned)

Add a `BackendCapabilities` snapshot at CLI/server startup that reports:
compiled backends, runtime CUDA availability, FFI library loadability + symbol
presence. Enforce `BackendRequest::Cuda` → hard error when CUDA unavailable;
`BackendRequest::Gpu`/`Auto` → graceful fallback, always recorded in receipts.

Startup output: `requested=X detected=[…] selected=Y` (deterministic, in logs + receipts).

---

## Phase C: BDD Grid as Compile-Coverage Contract (Planned)

`xtask grid-check` enumerates every supported BDD grid cell and runs
`cargo check --features <cell>` for each. Unsupported
cells require a reason token — no silent gaps. CPU cells in PR lane;
GPU cells compile-only (runtime tests on CUDA runner).

---

## Phase D: End-to-End Proofs (Planned)

- **CPU golden path**: tiny synthetic GGUF fixture (checked in), deterministic generate,
  assert output slice + receipt invariants. Runs on every PR with no downloads.
- **CUDA smoke lane**: self-hosted runner, small inference, parity receipt uploaded
  as CI artifact.

---

## Known Issues

| Issue | Affected Component | Status |
|---|---|---|
| TL1/TL2 round-trip max error ~2.0 | `bitnet-quantization` | Known; tests use loose tolerance |
| Shape mismatch in layer-norm (#254) | `bitnet-inference` | In analysis |
| Mock elimination (#260) | `bitnet-inference` | Pending refactor |
| Tokenizer parity + FFI hygiene (#469) | `bitnet-tokenizers`, `crossval` | Active development |

---

## Related Documentation

- `docs/architecture-overview.md` — overall system design and crate relationships
- `docs/howto/cpp-setup.md` — building the C++ reference for cross-validation
- `docs/explanation/i2s-dual-flavor.md` — I2S QK256 vs BitNet32 format details
- `crates/bitnet-common/src/kernel_registry.rs` — canonical kernel/backend type definitions
