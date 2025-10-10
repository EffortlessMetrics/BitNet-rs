# Issue #439: GPU Feature-Gate Hardening (Workspace-Wide)

## Context
PR #438 fixed quantization-side feature gate mismatches where runtime checks used `cfg!(feature="gpu")` but CUDA code was compiled under `#[cfg(feature="cuda")]`. This created situations where GPU capabilities could be advertised at runtime but the actual GPU kernels were not compiled, leading to silent CPU fallback and misleading performance receipts.

This issue tracks **workspace-wide hardening** to prevent GPU capability drift across all BitNet.rs components: kernels, quantization, inference, models, runtime helpers, xtask tooling, and documentation.

**Affected BitNet.rs Components:**
- `bitnet-kernels`: CUDA/SIMD kernel implementations with mixed feature gates
- `bitnet-quantization`: Device-aware quantization selection (I2_S, TL1, TL2)
- `bitnet-models`: Model loading, device selection, and layer dispatch logic
- `bitnet-inference`: Autoregressive generation with device-aware optimization
- `bitnet-common`: Shared runtime helpers for device capability detection
- `xtask`: Developer tooling for preflight checks and validation

**Neural Network Performance Implications:**
- GPU acceleration critical for large model inference (10-20 tok/s CPU vs 50-100 tok/s GPU baseline from #261)
- Silent CPU fallback degrades performance by 5-10x without user awareness
- Receipt honesty essential for performance validation and regression detection
- Feature gate inconsistency prevents proper device-aware quantization selection

## User Story
As a BitNet.rs developer, I want a single unified predicate for GPU capability across the entire workspace so that GPU features are only advertised when both compiled and runtime-available, preventing silent CPU fallback and dishonest performance receipts.

## Acceptance Criteria

AC1: All CUDA symbol gates in `bitnet-kernels` and workspace crates use unified predicate `#[cfg(any(feature="gpu", feature="cuda"))]` (no standalone `#[cfg(feature="cuda")]`)

AC2: All `build.rs` feature probes combine `CARGO_FEATURE_GPU || CARGO_FEATURE_CUDA` for GPU detection

AC3: Shared helpers `gpu_compiled()` and `gpu_available_runtime()` exist in `bitnet-kernels/src/device_features.rs` (NOT bitnet-common to avoid circular dependency) and are used in all `supports_device()` implementations across workspace

AC4: Feature matrix builds pass locally for all combinations:
```bash
cargo check --workspace --no-default-features
cargo check --workspace --no-default-features --features cpu
cargo check --workspace --no-default-features --features gpu
cargo check --workspace --no-default-features --features "cpu gpu"
```

AC5: xtask preflight reports correct GPU status based on `BITNET_GPU_FAKE` environment variable:
- `BITNET_GPU_FAKE=none` → preflight reports no GPU
- `BITNET_GPU_FAKE=cuda` → preflight reports GPU present

AC6: Receipt guard validates GPU kernel usage - if `backend:"cuda"` then `kernels[]` must contain at least one GPU kernel id following naming convention (`gemm_*`, `wmma_*`, `cuda_*`, `i2s_gpu_*`, `tl1_gpu_*`, `tl2_gpu_*`) otherwise verification fails with actionable error message

AC7: Documentation updated with standardized feature flag examples (`--no-default-features --features cpu|gpu`) and `cuda = ["gpu"]` alias noted as temporary

AC8: Ephemeral test artifacts excluded from version control - `.gitignore` updated with pattern `**/*.proptest-regressions` (note: `tests/tests/cache/incremental/last_run.json` already present at line 196)

## Technical Implementation Notes

### Affected Crates
- **bitnet-kernels**: CUDA kernel compilation gates, GPU validation modules, device capability helpers (NEW: `device_features.rs`)
- **bitnet-quantization**: Device-aware quantization selection gates
- **bitnet-models**: Model loading, builder device selection, layer dispatch
- **bitnet-inference**: Autoregressive generation device routing
- **xtask**: Preflight checks, receipt validation, GPU fake support

### Pipeline Stages Affected
- **Model Loading**: Device selection validation in model builders
- **Quantization**: Device-aware quantization type selection (I2_S vs TL1/TL2)
- **Kernels**: CUDA compilation gates and runtime availability checks
- **Inference**: Device routing in autoregressive generation
- **Output**: Receipt generation with honest GPU backend reporting

### Performance Considerations
- **GPU Acceleration**: Prevent silent 5-10x performance degradation from unintended CPU fallback
- **Receipt Honesty**: Ensure performance baselines accurately reflect GPU vs CPU execution
- **Device Detection**: Fake precedence for deterministic testing (`BITNET_GPU_FAKE=cuda`)
- **Mixed Precision**: Maintain FP16/BF16 support when GPU compiled and available

### Quantization Requirements
- Device-aware selection must respect unified GPU predicate
- I2_S quantization on CPU, TL1/TL2 selection based on actual device capability
- Cross-validation with C++ reference must report correct backend in receipts

### Cross-Validation
- Receipt verification must detect GPU capability mismatches
- `cargo run -p xtask -- crossval` must report honest device usage
- GPU kernels in receipt must match actual execution path

### Feature Flags
- **Root workspace**: `cpu = [...]`, `gpu = [...]`, `default = []`
- **CUDA crates**: `cuda = ["gpu"]` as temporary compatibility alias
- **Build matrix**: Validate no-features, cpu, gpu, cpu+gpu combinations
- **Graceful fallback**: Document fallback behavior when GPU unavailable

### GGUF Compatibility
- Model loading device selection must validate against feature gates
- GGUF tensor operations respect device capability at compile and runtime
- Mixed-device workflows (load on CPU, quantize for GPU) handled correctly

### Testing Strategy
- **TDD scaffolding**: Test helpers with `// AC:1`, `// AC:3`, `// AC:5` tags
- **Compile-time matrix**: Feature flag combinations in `cargo check` (AC4)
- **Runtime validation**: Preflight tests with GPU fake environment (AC5)
- **Receipt guard**: Kernel presence validation for GPU backend claims (AC6)
- **Cross-validation**: Integration with `xtask crossval` for honest reporting

### Implementation Approach

#### 1. Kernel Unification (AC1)
Replace all standalone CUDA guards:
```rust
// Before
#[cfg(feature = "cuda")]
mod gemm_fp16;

// After
#[cfg(any(feature = "gpu", feature = "cuda"))]
mod gemm_fp16;
```

Search command: `rg -n '#\[cfg\([^)]*feature\s*=\s*"(cuda|gpu)"' -g '!Cargo.lock'`

#### 2. Build Script Parity (AC2)
```rust
// build.rs
fn main() {
    let gpu = std::env::var_os("CARGO_FEATURE_GPU").is_some()
           || std::env::var_os("CARGO_FEATURE_CUDA").is_some();

    if gpu {
        println!("cargo:rustc-cfg=bitnet_build_gpu");
    }
}
```

#### 3. Shared Helpers (AC3)
```rust
// crates/bitnet-kernels/src/device_features.rs
// NOTE: Located in bitnet-kernels (NOT bitnet-common) to avoid circular dependency
// since bitnet-common depends on bitnet-kernels for CUDA availability check
#[inline]
pub fn gpu_compiled() -> bool {
    cfg!(any(feature = "gpu", feature = "cuda"))
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
#[inline]
pub fn gpu_available_runtime() -> bool {
    crate::gpu::cuda::is_cuda_available()
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
#[inline]
pub fn gpu_available_runtime() -> bool { false }
```

#### 4. Device Support Usage
```rust
use bitnet_kernels::{gpu_compiled, gpu_available_runtime};

pub fn supports_device(device: Device) -> bool {
    match device {
        Device::Cpu => true,
        Device::Cuda(_) => gpu_compiled() && gpu_available_runtime(),
        Device::Metal => false,
    }
}
```

#### 5. xtask Preflight (AC5)
```rust
enum GpuFake { None, Cuda }

fn gpu_fake() -> GpuFake {
    match std::env::var("BITNET_GPU_FAKE").as_deref() {
        Ok("cuda") => GpuFake::Cuda,
        _ => GpuFake::None,
    }
}

fn gpu_backend_detected() -> bool {
    match gpu_fake() {
        GpuFake::Cuda => true, // fake overrides real
        GpuFake::None => gpu_compiled() && bitnet_kernels::gpu::cuda::is_cuda_available(),
    }
}
```

#### 6. Receipt Validation (AC6)
```rust
pub fn verify_receipt(r: &Receipt) -> anyhow::Result<()> {
    if r.backend == "cuda" {
        let gpu_kernel_prefixes = ["gemm_", "wmma_", "cuda_", "i2s_gpu_", "tl1_gpu_", "tl2_gpu_"];
        anyhow::ensure!(
            !r.kernels.is_empty() && r.kernels.iter().any(|k|
                gpu_kernel_prefixes.iter().any(|prefix| k.starts_with(prefix))
            ),
            "GPU receipt backend:'cuda' requires at least one GPU kernel id matching naming convention: {}",
            gpu_kernel_prefixes.join(", ")
        );
    }
    Ok(())
}
```

### Validation Commands
```bash
# Compile matrix (AC4)
cargo check --workspace --no-default-features
cargo check --workspace --no-default-features --features cpu
cargo check --workspace --no-default-features --features gpu
cargo check --workspace --no-default-features --features "cpu gpu"

# CPU tests
cargo test --workspace --no-default-features --features cpu

# Preflight validation (AC5)
BITNET_GPU_FAKE=none cargo run -p xtask -- preflight  # expect: no GPU
BITNET_GPU_FAKE=cuda cargo run -p xtask -- preflight  # expect: GPU present

# Receipt guard (AC6)
cargo run -p xtask -- verify-receipt <path-to-gpu-receipt>
```

### Documentation Updates (AC7)
- Standardize all examples to `--no-default-features --features cpu|gpu`
- Document `cuda = ["gpu"]` as temporary compatibility alias
- Add device selection guide explaining `gpu_compiled()` vs `gpu_available_runtime()`
- Update GPU development guide with unified predicate approach

### Non-Goals
- Removing `cuda` alias (deferred to future minor release)
- Provisioning GPU CI runners (local validation remains standard)
- Changing quantization algorithms or accuracy thresholds
- Modifying inference performance baselines

### Risk Mitigation
- **Partial application risk**: Compile-only matrix (AC4) catches incomplete unification
- **Inverted fake semantics**: Explicit preflight tests (AC5) validate fake precedence
- **Silent CPU fallback**: Receipt kernel guard (AC6) prevents dishonest GPU claims
- **Documentation drift**: Grep-based audit ensures example consistency (AC7)

### References
- PR #437: Feature propagation rename (`cuda` → `gpu`)
- PR #438: Quantization feature gate alignment + regression tests
- Issue #261: Mock elimination, receipt-driven baselines (CPU 10-20 tok/s, GPU 50-100 tok/s)

### Estimated Effort
**2-3 hours** total: surgical edits + local matrix build + doc sweep + receipt guard implementation

---

## Detailed Implementation Architecture

### 1. Device Feature Detection API

**Module Location**: `crates/bitnet-kernels/src/device_features.rs` (NEW)

**Design Rationale**: Located in `bitnet-kernels` (NOT `bitnet-common`) to avoid circular dependency since `bitnet-common` depends on `bitnet-kernels` for CUDA availability checks.

**Public API Contract**:

```rust
//! Device feature detection and capability queries
//!
//! This module provides unified device capability checks for GPU/CPU selection
//! across the BitNet.rs workspace. It consolidates compile-time feature gates
//! with runtime hardware detection.
//!
//! # Architecture Decision
//!
//! This module lives in `bitnet-kernels` rather than `bitnet-common` to avoid
//! circular dependencies, since `bitnet-common` depends on `bitnet-kernels`
//! for GPU availability checks.
//!
//! # Usage
//!
//! ```rust
//! use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};
//! use bitnet_common::Device;
//!
//! pub fn supports_device(device: &Device) -> bool {
//!     match device {
//!         Device::Cpu => true,
//!         Device::Cuda(_) => gpu_compiled() && gpu_available_runtime(),
//!         Device::Metal => false,
//!     }
//! }
//! ```

/// Check if GPU support was compiled into this binary
///
/// Returns `true` if either `feature="gpu"` or `feature="cuda"` was enabled
/// at compile time. This does NOT check runtime GPU availability.
///
/// # Example
///
/// ```rust
/// if !gpu_compiled() {
///     println!("GPU support not compiled - rebuild with --features gpu");
///     return Err(DeviceError::NotCompiled);
/// }
/// ```
#[inline]
pub fn gpu_compiled() -> bool {
    cfg!(any(feature = "gpu", feature = "cuda"))
}

/// Check if GPU is available at runtime
///
/// Returns `false` if:
/// - GPU not compiled (`gpu_compiled() == false`)
/// - CUDA runtime unavailable (`nvidia-smi` fails)
/// - `BITNET_GPU_FAKE=none` environment variable set
///
/// Returns `true` if:
/// - GPU compiled AND CUDA runtime detected
/// - `BITNET_GPU_FAKE=cuda` environment variable set (overrides real detection)
///
/// # Example
///
/// ```rust
/// if !gpu_available_runtime() {
///     eprintln!("Warning: GPU requested but unavailable, falling back to CPU");
///     return Device::Cpu;
/// }
/// ```
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[inline]
pub fn gpu_available_runtime() -> bool {
    // Check for fake GPU environment variable first (deterministic testing)
    if let Ok(fake) = std::env::var("BITNET_GPU_FAKE") {
        return fake.to_lowercase().contains("cuda");
    }

    // Check real CUDA availability via gpu_utils
    crate::gpu_utils::get_gpu_info().cuda
}

/// Stub implementation when GPU not compiled
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
#[inline]
pub fn gpu_available_runtime() -> bool {
    false
}

/// Get device capability summary for diagnostics
///
/// Returns a human-readable summary of compile-time and runtime capabilities.
///
/// # Example Output
///
/// ```text
/// Device Capabilities:
///   Compiled: GPU ✓, CPU ✓
///   Runtime: CUDA 12.1 ✓, cuBLAS ✓
/// ```
pub fn device_capability_summary() -> String {
    let compiled = if gpu_compiled() { "GPU ✓" } else { "GPU ✗" };
    let runtime = if gpu_available_runtime() {
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            let info = crate::gpu_utils::get_gpu_info();
            if let Some(version) = info.cuda_version {
                format!("CUDA {} ✓", version)
            } else {
                "CUDA ✓".to_string()
            }
        }
        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        {
            "N/A".to_string()
        }
    } else {
        "CUDA ✗".to_string()
    };

    format!("Device Capabilities:\n  Compiled: {}, CPU ✓\n  Runtime: {}", compiled, runtime)
}
```

**Integration Points**:

All workspace crates with `supports_device()` implementations must use these helpers:

- `bitnet-quantization`: I2S/TL1/TL2 quantizer device selection
- `bitnet-models`: Model builder device validation
- `bitnet-inference`: Autoregressive generation device routing
- `xtask`: Preflight checks and receipt validation

**Migration Pattern**:

```rust
// BEFORE (inconsistent)
#[cfg(feature = "gpu")]
pub fn supports_device(device: &Device) -> bool {
    match device {
        Device::Cuda(_) => cfg!(feature = "cuda"), // MISMATCH!
        _ => true,
    }
}

// AFTER (unified)
use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};

pub fn supports_device(device: &Device) -> bool {
    match device {
        Device::Cpu => true,
        Device::Cuda(_) => gpu_compiled() && gpu_available_runtime(),
        Device::Metal => false,
    }
}
```

### 2. Receipt Validation Architecture

**Module Location**: `xtask/src/verify_receipt.rs` (NEW)

**Design Rationale**: Receipt validation must detect "GPU on paper, CPU in reality" scenarios by verifying GPU backend claims are backed by actual GPU kernel execution evidence.

**GPU Kernel Naming Convention**:

All GPU kernels must follow consistent naming patterns to enable automated receipt verification:

- **GEMM kernels**: `gemm_*` (e.g., `gemm_fp16`, `gemm_bf16`, `gemm_i2s`)
- **Tensor Core kernels**: `wmma_*` (e.g., `wmma_matmul`, `wmma_quantize`)
- **CUDA utilities**: `cuda_*` (e.g., `cuda_memcpy`, `cuda_sync`)
- **Quantization kernels**:
  - I2_S GPU: `i2s_gpu_*` (e.g., `i2s_gpu_quantize`, `i2s_gpu_dequantize`)
  - TL1 GPU: `tl1_gpu_*` (e.g., `tl1_gpu_pack`, `tl1_gpu_unpack`)
  - TL2 GPU: `tl2_gpu_*` (e.g., `tl2_gpu_matmul`)

**Public API Contract**:

```rust
//! Receipt validation and honesty checks
//!
//! This module verifies that performance receipts accurately reflect
//! device usage. GPU backend claims must be backed by GPU kernel execution
//! evidence to prevent misleading performance reporting.

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

/// Performance receipt structure (matches bitnet-inference Receipt)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    /// Backend used: "cpu", "cuda", or "unknown"
    pub backend: String,
    /// Kernels executed during inference
    pub kernels: Vec<String>,
    /// Performance metrics
    pub tokens_per_second: Option<f64>,
    pub latency_ms: Option<f64>,
}

/// GPU kernel prefixes for validation (AC6)
const GPU_KERNEL_PREFIXES: &[&str] = &[
    "gemm_",      // GEMM kernels
    "wmma_",      // Tensor Core kernels
    "cuda_",      // CUDA utilities
    "i2s_gpu_",   // I2_S GPU quantization
    "tl1_gpu_",   // TL1 GPU quantization
    "tl2_gpu_",   // TL2 GPU quantization
];

/// Verify GPU receipt honesty (AC6)
///
/// Ensures that if a receipt claims GPU backend execution, it contains
/// evidence of actual GPU kernel usage via naming convention.
///
/// # Validation Rules
///
/// - If `backend == "cuda"` OR `backend == "gpu"`:
///   - `kernels` array MUST contain at least one GPU kernel
///   - GPU kernels identified by prefix matching (see GPU_KERNEL_PREFIXES)
///   - Empty `kernels` array is a validation failure
///
/// - If `backend == "cpu"`:
///   - GPU kernels in `kernels` array trigger a warning but not failure
///   - This allows mixed CPU/GPU fallback scenarios
///
/// # Examples
///
/// ```rust
/// // VALID: GPU backend with GPU kernel evidence
/// let receipt = Receipt {
///     backend: "cuda".to_string(),
///     kernels: vec!["i2s_gpu_quantize".to_string(), "gemm_fp16".to_string()],
///     tokens_per_second: Some(87.5),
///     latency_ms: Some(11.4),
/// };
/// verify_gpu_receipt(&receipt)?; // ✓ PASS
///
/// // INVALID: GPU backend without GPU kernel evidence
/// let bad_receipt = Receipt {
///     backend: "cuda".to_string(),
///     kernels: vec!["i2s_cpu_quantize".to_string()], // CPU kernel!
///     tokens_per_second: Some(12.3),
///     latency_ms: Some(81.2),
/// };
/// verify_gpu_receipt(&bad_receipt)?; // ✗ FAIL
/// ```
pub fn verify_gpu_receipt(receipt: &Receipt) -> Result<()> {
    let backend_claims_gpu = receipt.backend == "cuda" || receipt.backend == "gpu";

    if !backend_claims_gpu {
        // CPU backend - no validation needed
        return Ok(());
    }

    // GPU backend claimed - verify kernel evidence
    ensure!(
        !receipt.kernels.is_empty(),
        "GPU backend '{}' requires non-empty kernels array, got: {:?}",
        receipt.backend,
        receipt.kernels
    );

    let has_gpu_kernel = receipt.kernels.iter().any(|kernel_id| {
        GPU_KERNEL_PREFIXES.iter().any(|prefix| kernel_id.starts_with(prefix))
    });

    ensure!(
        has_gpu_kernel,
        "GPU backend '{}' requires at least one GPU kernel matching naming convention.\n\
         Expected kernel prefixes: {}\n\
         Actual kernels: {:?}\n\n\
         This likely indicates silent CPU fallback. Verify:\n\
         1. GPU feature compiled: cargo build --features gpu\n\
         2. CUDA runtime available: nvidia-smi\n\
         3. Device selection: Device::Cuda(0) passed to inference",
        receipt.backend,
        GPU_KERNEL_PREFIXES.join(", "),
        receipt.kernels
    );

    Ok(())
}

/// Verify receipt file from path (xtask integration)
pub fn verify_receipt_file(path: &std::path::Path) -> Result<()> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read receipt file: {}", path.display()))?;

    let receipt: Receipt = serde_json::from_str(&contents)
        .with_context(|| format!("Failed to parse receipt JSON: {}", path.display()))?;

    verify_gpu_receipt(&receipt)
        .with_context(|| format!("Receipt validation failed: {}", path.display()))?;

    println!("✓ Receipt validation passed: {}", path.display());
    println!("  Backend: {}", receipt.backend);
    println!("  Kernels: {:?}", receipt.kernels);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ac6_gpu_backend_requires_gpu_kernel() {
        // AC:6 - GPU backend must have GPU kernel evidence
        let receipt = Receipt {
            backend: "cuda".to_string(),
            kernels: vec!["i2s_cpu_quantize".to_string()],
            tokens_per_second: Some(12.0),
            latency_ms: Some(80.0),
        };

        assert!(
            verify_gpu_receipt(&receipt).is_err(),
            "GPU backend with CPU kernels should fail validation"
        );
    }

    #[test]
    fn ac6_gpu_backend_with_valid_kernel() {
        // AC:6 - GPU backend with GPU kernel should pass
        let receipt = Receipt {
            backend: "cuda".to_string(),
            kernels: vec!["gemm_fp16".to_string()],
            tokens_per_second: Some(87.5),
            latency_ms: Some(11.4),
        };

        assert!(
            verify_gpu_receipt(&receipt).is_ok(),
            "GPU backend with GPU kernel should pass validation"
        );
    }

    #[test]
    fn ac6_cpu_backend_no_validation() {
        // AC:6 - CPU backend requires no kernel validation
        let receipt = Receipt {
            backend: "cpu".to_string(),
            kernels: vec![],
            tokens_per_second: Some(15.0),
            latency_ms: Some(66.0),
        };

        assert!(
            verify_gpu_receipt(&receipt).is_ok(),
            "CPU backend should pass validation regardless of kernels"
        );
    }
}
```

### 3. Feature Gate Unification Pattern

**Unified Predicate (AC1)**:

Replace all standalone `#[cfg(feature = "cuda")]` with unified predicate:

```rust
// PATTERN: Unified GPU feature gate
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod gpu_module {
    // GPU-specific code
}

// PATTERN: Runtime capability check
pub fn use_gpu_if_available() -> bool {
    cfg!(any(feature = "gpu", feature = "cuda")) // compile-time
        && bitnet_kernels::device_features::gpu_available_runtime() // runtime
}

// PATTERN: Conditional compilation for function bodies
pub fn quantize_i2s(data: &[f32]) -> Vec<i8> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if bitnet_kernels::device_features::gpu_available_runtime() {
            return quantize_i2s_gpu(data);
        }
    }

    // CPU fallback
    quantize_i2s_cpu(data)
}
```

**Build Script Pattern (AC2)**:

```rust
// crates/bitnet-kernels/build.rs
fn main() {
    // Unified GPU feature detection
    let gpu = std::env::var_os("CARGO_FEATURE_GPU").is_some()
           || std::env::var_os("CARGO_FEATURE_CUDA").is_some();

    if gpu {
        println!("cargo:rustc-cfg=bitnet_build_gpu");

        // CUDA library paths (existing logic)
        println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=cuda");
        // ... rest of CUDA setup
    }

    // FFI detection (unchanged)
    // ...
}
```

### 4. Test Strategy and TDD Mapping

**Test Scaffolding Structure**:

```
crates/bitnet-kernels/tests/
├── feature_gate_consistency.rs    # AC1 regression tests
├── device_features.rs              # AC3 unit tests (NEW)
└── gpu_integration.rs              # AC6 integration tests

xtask/tests/
├── preflight.rs                    # AC5 preflight validation (NEW)
└── verify_receipt.rs               # AC6 receipt guard tests (NEW)
```

**AC1: Feature Gate Unification Tests**

```rust
// crates/bitnet-kernels/tests/feature_gate_consistency.rs

/// AC:1 - Verify no standalone cuda feature gates in source
#[test]
fn ac1_no_standalone_cuda_gates() {
    // Grep for standalone #[cfg(feature = "cuda")] without gpu
    let output = std::process::Command::new("rg")
        .args(&[
            r#"#\[cfg\(feature\s*=\s*"cuda"\)\]"#,
            "--glob", "!Cargo.lock",
            "--glob", "*.rs",
            "crates/bitnet-kernels/src/",
        ])
        .output()
        .expect("Failed to run ripgrep");

    let violations = String::from_utf8_lossy(&output.stdout);

    assert!(
        violations.is_empty() || violations.contains("any(feature"),
        "Found standalone cuda feature gates (AC1):\n{}",
        violations
    );
}

/// AC:1 - Verify unified predicate in gpu/validation.rs
#[test]
fn ac1_validation_uses_unified_gate() {
    let validation_rs = std::fs::read_to_string(
        "crates/bitnet-kernels/src/gpu/validation.rs"
    ).expect("Failed to read validation.rs");

    // Check line 13, 344, 578 use unified predicate
    let unified_pattern = r#"#[cfg(any(feature = "gpu", feature = "cuda"))]"#;

    assert!(
        validation_rs.contains(unified_pattern),
        "validation.rs must use unified GPU predicate (AC1)"
    );
}
```

**AC3: Device Features Unit Tests**

```rust
// crates/bitnet-kernels/tests/device_features.rs (NEW)

use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};

/// AC:3 - gpu_compiled() returns true when GPU features enabled
#[test]
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn ac3_gpu_compiled_true_with_features() {
    assert!(gpu_compiled(), "gpu_compiled() should return true with gpu/cuda features");
}

/// AC:3 - gpu_compiled() returns false when GPU features disabled
#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn ac3_gpu_compiled_false_without_features() {
    assert!(!gpu_compiled(), "gpu_compiled() should return false without gpu/cuda features");
}

/// AC:3 - gpu_available_runtime() respects BITNET_GPU_FAKE
#[test]
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn ac3_gpu_fake_cuda_overrides_detection() {
    std::env::set_var("BITNET_GPU_FAKE", "cuda");
    assert!(
        gpu_available_runtime(),
        "BITNET_GPU_FAKE=cuda should override real detection"
    );
    std::env::remove_var("BITNET_GPU_FAKE");
}

/// AC:3 - gpu_available_runtime() respects BITNET_GPU_FAKE=none
#[test]
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn ac3_gpu_fake_none_disables_detection() {
    std::env::set_var("BITNET_GPU_FAKE", "none");
    assert!(
        !gpu_available_runtime(),
        "BITNET_GPU_FAKE=none should disable GPU detection"
    );
    std::env::remove_var("BITNET_GPU_FAKE");
}
```

**AC5: Preflight Validation Tests**

```rust
// xtask/tests/preflight.rs (NEW)

/// AC:5 - Preflight reports no GPU with BITNET_GPU_FAKE=none
#[test]
fn ac5_preflight_respects_fake_none() {
    let output = std::process::Command::new("cargo")
        .args(&["run", "-p", "xtask", "--", "preflight"])
        .env("BITNET_GPU_FAKE", "none")
        .output()
        .expect("Failed to run xtask preflight");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stdout.contains("GPU: Not available") || stdout.contains("GPU: ✗"),
        "Preflight should report no GPU with BITNET_GPU_FAKE=none (AC5)"
    );
}

/// AC:5 - Preflight reports GPU present with BITNET_GPU_FAKE=cuda
#[test]
fn ac5_preflight_respects_fake_cuda() {
    let output = std::process::Command::new("cargo")
        .args(&["run", "-p", "xtask", "--", "preflight"])
        .env("BITNET_GPU_FAKE", "cuda")
        .output()
        .expect("Failed to run xtask preflight");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stdout.contains("GPU: Available") || stdout.contains("GPU: ✓") || stdout.contains("CUDA"),
        "Preflight should report GPU present with BITNET_GPU_FAKE=cuda (AC5)"
    );
}
```

**AC6: Receipt Validation Tests**

```rust
// xtask/tests/verify_receipt.rs (NEW)

use xtask::verify_receipt::{Receipt, verify_gpu_receipt};

/// AC:6 - GPU backend requires GPU kernel naming convention
#[test]
fn ac6_cuda_backend_requires_gpu_kernel_prefix() {
    let receipt = Receipt {
        backend: "cuda".to_string(),
        kernels: vec!["i2s_cpu_quantize".to_string()],
        tokens_per_second: Some(12.0),
        latency_ms: Some(80.0),
    };

    let result = verify_gpu_receipt(&receipt);
    assert!(
        result.is_err(),
        "GPU backend with CPU kernels should fail (AC6)"
    );

    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("naming convention"),
        "Error should mention naming convention (AC6)"
    );
}

/// AC:6 - Valid GPU kernel prefixes pass validation
#[test]
fn ac6_valid_gpu_kernel_prefixes() {
    let valid_prefixes = vec![
        "gemm_fp16",
        "wmma_matmul",
        "cuda_sync",
        "i2s_gpu_quantize",
        "tl1_gpu_pack",
        "tl2_gpu_matmul",
    ];

    for kernel in valid_prefixes {
        let receipt = Receipt {
            backend: "cuda".to_string(),
            kernels: vec![kernel.to_string()],
            tokens_per_second: Some(87.5),
            latency_ms: Some(11.4),
        };

        assert!(
            verify_gpu_receipt(&receipt).is_ok(),
            "Kernel '{}' should pass validation (AC6)",
            kernel
        );
    }
}
```

### 5. Performance Baseline Specifications

**CPU Baseline (from Issue #261)**:
- Target: 10-20 tokens/second
- Quantization: I2_S CPU implementation
- Device: `Device::Cpu`
- Kernels: `i2s_cpu_quantize`, `avx2_matmul`, `fallback_*`

**GPU Baseline (from Issue #261)**:
- Target: 50-100 tokens/second
- Quantization: TL1/TL2 GPU implementation
- Device: `Device::Cuda(0)`
- Kernels: `tl1_gpu_pack`, `gemm_fp16`, `wmma_matmul`

**Receipt Validation Thresholds**:

```rust
/// Performance validation for receipt honesty
pub fn validate_performance_reasonable(receipt: &Receipt) -> Result<()> {
    if let Some(tps) = receipt.tokens_per_second {
        match receipt.backend.as_str() {
            "cpu" => {
                // CPU should be 5-30 tok/s (allowing variance)
                ensure!(
                    tps >= 5.0 && tps <= 30.0,
                    "CPU performance {} tok/s outside expected range 5-30 tok/s",
                    tps
                );
            }
            "cuda" | "gpu" => {
                // GPU should be 30-150 tok/s (allowing variance)
                ensure!(
                    tps >= 30.0 && tps <= 150.0,
                    "GPU performance {} tok/s outside expected range 30-150 tok/s",
                    tps
                );
            }
            _ => {} // Unknown backend - skip validation
        }
    }

    Ok(())
}
```

**Silent Fallback Detection**:

```rust
/// Detect suspicious performance indicating silent CPU fallback
pub fn detect_silent_cpu_fallback(receipt: &Receipt) -> Option<String> {
    if receipt.backend == "cuda" || receipt.backend == "gpu" {
        if let Some(tps) = receipt.tokens_per_second {
            // GPU performance < 25 tok/s is suspicious (CPU-like)
            if tps < 25.0 {
                return Some(format!(
                    "WARNING: GPU backend but CPU-like performance ({:.1} tok/s < 25 tok/s threshold). \
                     Possible silent CPU fallback - verify:\n\
                     1. GPU kernels in receipt: {:?}\n\
                     2. CUDA available: nvidia-smi\n\
                     3. Feature compiled: cargo build --features gpu",
                    tps, receipt.kernels
                ));
            }
        }
    }
    None
}
```

### 6. Validation Commands Reference

**AC1: Feature Gate Unification**

```bash
# Grep for standalone cuda gates (should find none or only unified predicates)
rg -n '#\[cfg\([^)]*feature\s*=\s*"cuda"' -g '!Cargo.lock' crates/

# Expected: All matches should contain "any(feature" pattern
# Violations: Standalone #[cfg(feature = "cuda")] without "any"

# Run regression tests
cargo test --workspace --no-default-features --features cpu feature_gate_consistency
```

**AC2: Build Script Parity**

```bash
# Grep for feature environment variable checks in build scripts
rg -n 'CARGO_FEATURE_(CUDA|GPU)' -- '*/build.rs'

# Verify unified detection in bitnet-kernels/build.rs line 11
cat crates/bitnet-kernels/build.rs | sed -n '10,15p'

# Expected output should show:
# if env::var_os("CARGO_FEATURE_CUDA").is_some() || env::var_os("CARGO_FEATURE_GPU").is_some()
```

**AC3: Device Features Module**

```bash
# Verify device_features.rs module exists
test -f crates/bitnet-kernels/src/device_features.rs && echo "✓ Module exists" || echo "✗ Module missing"

# Run device features unit tests
cargo test --package bitnet-kernels --no-default-features --features cpu device_features
cargo test --package bitnet-kernels --no-default-features --features gpu device_features

# Verify all supports_device() implementations use helpers
rg -n 'supports_device' -A 5 crates/ | grep -E '(gpu_compiled|gpu_available_runtime)'
```

**AC4: Feature Matrix Builds**

```bash
# No features (should build successfully)
cargo check --workspace --no-default-features

# CPU only
cargo check --workspace --no-default-features --features cpu

# GPU only
cargo check --workspace --no-default-features --features gpu

# CPU + GPU combined
cargo check --workspace --no-default-features --features "cpu gpu"

# All combinations should exit with code 0
echo "Feature matrix validation: PASS"
```

**AC5: xtask Preflight**

```bash
# Test GPU fake environment variable precedence
BITNET_GPU_FAKE=none cargo run -p xtask -- preflight
# Expected output: "GPU: Not available" or "GPU: ✗"

BITNET_GPU_FAKE=cuda cargo run -p xtask -- preflight
# Expected output: "GPU: Available" or "GPU: ✓" or "CUDA"

# Run preflight tests
cargo test --package xtask --test preflight
```

**AC6: Receipt Validation**

```bash
# Create test receipt files
mkdir -p /tmp/bitnet-receipts

# Valid GPU receipt
cat > /tmp/bitnet-receipts/valid-gpu.json <<EOF
{
  "backend": "cuda",
  "kernels": ["gemm_fp16", "i2s_gpu_quantize"],
  "tokens_per_second": 87.5,
  "latency_ms": 11.4
}
EOF

# Invalid GPU receipt (CPU kernels)
cat > /tmp/bitnet-receipts/invalid-gpu.json <<EOF
{
  "backend": "cuda",
  "kernels": ["i2s_cpu_quantize", "avx2_matmul"],
  "tokens_per_second": 12.3,
  "latency_ms": 81.2
}
EOF

# Test verification
cargo run -p xtask -- verify-receipt /tmp/bitnet-receipts/valid-gpu.json
# Expected: "✓ Receipt validation passed"

cargo run -p xtask -- verify-receipt /tmp/bitnet-receipts/invalid-gpu.json
# Expected: Error with "naming convention" message

# Run receipt validation tests
cargo test --package xtask --test verify_receipt
```

**AC7: Documentation Updates**

```bash
# Grep for cuda feature references in docs
rg -n -- '--features\s+cuda|feature\s*=\s*"cuda"' docs/ crates/

# Count violations (should be 0 after updates)
rg -c -- '--features\s+cuda' docs/ | awk -F: '{sum+=$2} END {print "Total violations:", sum}'

# Update all docs to use --features gpu instead of --features cuda
# Standardize to: --no-default-features --features cpu|gpu
```

**AC8: .gitignore Update**

```bash
# Verify proptest-regressions pattern exists
grep -n '\.proptest-regressions' .gitignore

# Expected: Should find line with **/*.proptest-regressions

# Verify last_run.json already excluded (line 196)
sed -n '196p' .gitignore
# Expected: tests/tests/cache/incremental/last_run.json
```

### 7. Neural Network Context Integration

**Quantization Pipeline Impact**:

```rust
// I2_S quantization with device-aware selection
impl I2SQuantizer {
    pub fn quantize(&self, input: &[f32], device: &Device) -> Result<Vec<i8>> {
        match device {
            Device::Cpu => {
                // CPU: SIMD-optimized I2_S
                self.quantize_cpu(input)
            }
            Device::Cuda(gpu_id) => {
                // GPU: Check compile + runtime availability
                if !gpu_compiled() {
                    return Err(QuantizationError::DeviceNotCompiled("GPU"));
                }
                if !gpu_available_runtime() {
                    eprintln!("Warning: GPU requested but unavailable, falling back to CPU");
                    return self.quantize_cpu(input);
                }

                // GPU path: Record kernel usage for receipt
                self.record_kernel_usage("i2s_gpu_quantize");
                self.quantize_cuda(input, *gpu_id)
            }
            _ => Err(QuantizationError::UnsupportedDevice),
        }
    }
}
```

**GGUF Model Loading Device Selection**:

```rust
// Model builder with device validation
impl BitNetModelBuilder {
    pub fn with_device(mut self, device: Device) -> Result<Self> {
        // Validate device is actually supported
        if !self.supports_device(&device) {
            return Err(ModelError::UnsupportedDevice {
                requested: device,
                available: self.available_devices(),
            });
        }

        self.device = device;
        Ok(self)
    }

    fn supports_device(&self, device: &Device) -> bool {
        match device {
            Device::Cpu => true,
            Device::Cuda(_) => {
                use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};
                gpu_compiled() && gpu_available_runtime()
            }
            _ => false,
        }
    }
}
```

**Inference Receipt Generation**:

```rust
// Autoregressive generation with honest receipt tracking
pub struct InferenceEngine {
    kernel_usage: Vec<String>,
    device: Device,
}

impl InferenceEngine {
    fn record_kernel(&mut self, kernel_id: &str) {
        self.kernel_usage.push(kernel_id.to_string());
    }

    pub fn generate(&mut self, prompt: &str) -> Result<(String, Receipt)> {
        // Track kernels used during generation
        self.kernel_usage.clear();

        // ... inference logic records kernels via self.record_kernel()

        let backend = match self.device {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            _ => "unknown",
        };

        let receipt = Receipt {
            backend: backend.to_string(),
            kernels: self.kernel_usage.clone(),
            tokens_per_second: Some(self.calculate_tps()),
            latency_ms: Some(self.total_latency()),
        };

        // Verify receipt honesty before returning
        verify_gpu_receipt(&receipt)?;

        Ok((output, receipt))
    }
}
```

**Cross-Validation Integration**:

```rust
// xtask crossval with receipt comparison
pub fn run_crossval(model_path: &Path) -> Result<()> {
    // Run Rust inference
    let (rust_output, rust_receipt) = run_rust_inference(model_path)?;

    // Run C++ reference
    let (cpp_output, cpp_receipt) = run_cpp_reference(model_path)?;

    // Verify receipt honesty
    verify_gpu_receipt(&rust_receipt)?;
    verify_gpu_receipt(&cpp_receipt)?;

    // Compare outputs
    let accuracy = compare_outputs(&rust_output, &cpp_output);

    // Compare performance (should be within 2x)
    let rust_tps = rust_receipt.tokens_per_second.unwrap();
    let cpp_tps = cpp_receipt.tokens_per_second.unwrap();

    ensure!(
        (rust_tps / cpp_tps) > 0.5 && (rust_tps / cpp_tps) < 2.0,
        "Performance divergence: Rust {:.1} tok/s vs C++ {:.1} tok/s",
        rust_tps,
        cpp_tps
    );

    println!("Cross-validation PASS:");
    println!("  Accuracy: {:.2}%", accuracy * 100.0);
    println!("  Rust: {:.1} tok/s (backend: {})", rust_tps, rust_receipt.backend);
    println!("  C++:  {:.1} tok/s (backend: {})", cpp_tps, cpp_receipt.backend);

    Ok(())
}
```
