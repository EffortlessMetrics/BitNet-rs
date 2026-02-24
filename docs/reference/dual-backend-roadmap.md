# Dual-Backend Support Implementation Roadmap

> **Last updated**: reflects implementation state after PRs #608–#645.
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
| TL1/TL2 quantizer round-trip accuracy fix | `crates/bitnet-quantization/` | #641 |
| Runtime backend validation + enforcement wired into CLI/server | `crates/bitnet-cli/`, `crates/bitnet-server/` | #642 |
| `BackendCapabilities` snapshot: `requested=X detected=[…] selected=Y` | `crates/bitnet-kernels/src/device_features.rs` | #642 |
| CPU golden path E2E tests (5 deterministic, no model download) | `crates/bitnet-inference/tests/cpu_golden_path.rs` | #643 |
| SRP microcrates wired into CI matrix | `.github/workflows/ci-core.yml` | #644 |
| Security audit fixes (bytes, time CVEs; audit.toml accepted-risk) | `Cargo.lock`, `.cargo/audit.toml` | #645 |
| Nightly fuzz runs with artifact upload | `.github/workflows/fuzz-ci.yml` | #609 |
| Cross-validation scheduled lanes | `.github/workflows/crossval.yml` | #611 |
| Build-time symbol detection → rustc-cfg flags | `crates/bitnet-sys/build.rs` (feature: `symbol-analysis`) | #611 |

### 🔲 What's Planned

1. **CUDA smoke lane** (self-hosted runner, nightly/manual)
   - Allocate device, run small inference, upload parity receipt
   - Blocked on having a CUDA-capable self-hosted runner

2. **Scheduled fuzz/crossval evidence expansion**
   - Nightly timeboxed fuzz runs with artifact upload (partially done via `fuzz-ci.yml`)
   - Real-model crossval producing receipts (gated on model download infrastructure)

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

### 🔲 What Remains (Future Work)

1. **Build-time symbol detection → rustc-cfg flags** (`bitnet-sys`)
   - `build.rs` runs `nm`/`objdump -T` on found libs
   - Emits `bitnet_cpp_has_cuda`, `bitnet_cpp_has_bitnet_shim`
   - Code can compile-time gate "CUDA-backed C++ path" vs "CPU-only C++ path"

2. **CUDA smoke lane** (self-hosted runner, nightly/manual)
   - Allocate device, run small inference, upload parity receipt

---

## Known Issues

| Issue | Affected Component | Status |
|---|---|---|
| Shape mismatch in layer-norm (#254) | `bitnet-inference` | In analysis |
| Mock elimination (#260) | `bitnet-inference` | Pending refactor |
| Tokenizer parity + FFI hygiene (#469) | `bitnet-tokenizers`, `crossval` | Active development |

---

## Related Documentation

- `docs/architecture-overview.md` — overall system design and crate relationships
- `docs/howto/cpp-setup.md` — building the C++ reference for cross-validation
- `docs/explanation/i2s-dual-flavor.md` — I2S QK256 vs BitNet32 format details
- `crates/bitnet-common/src/kernel_registry.rs` — canonical kernel/backend type definitions
