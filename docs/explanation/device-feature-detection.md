# Device Feature Detection API

**Status**: Draft
**Related Issue**: #439 (GPU feature-gate hardening)
**Audience**: BitNet.rs developers implementing device-aware operations
**Type**: Explanation (Diátaxis)

## Overview

The Device Feature Detection API provides unified device capability checks for GPU/CPU selection across the BitNet.rs workspace. It consolidates compile-time feature gates with runtime hardware detection to prevent "GPU on paper, CPU in reality" scenarios that cause silent performance degradation.

## Problem Statement

Prior to this API, device capability checks were inconsistent across the workspace:

**Inconsistent Feature Gates**:
- Runtime checks used `cfg!(feature = "gpu")`
- Compile-time gates used `#[cfg(feature = "cuda")]`
- Build scripts checked only `CARGO_FEATURE_CUDA`

**Result**: GPU capabilities advertised but kernels not compiled → silent CPU fallback → 5-10x performance degradation.

## Architecture Decision

### Module Location

**Location**: `crates/bitnet-kernels/src/device_features.rs`

**Rationale**: Located in `bitnet-kernels` (NOT `bitnet-common`) to avoid circular dependency:
- `bitnet-common` depends on `bitnet-kernels` for CUDA availability checks
- Placing device detection in `bitnet-common` would create circular dependency
- `bitnet-kernels` is the lowest-level crate with GPU integration

### API Design Principles

1. **Separation of Concerns**: Compile-time capability vs. runtime availability
2. **Deterministic Testing**: Environment variable override for testing
3. **Graceful Fallback**: Clear distinction between "not compiled" vs "not available"
4. **Diagnostic Support**: Human-readable capability summaries

## Public API Specification

### Core Functions

#### `gpu_compiled() -> bool`

**Purpose**: Check if GPU support was compiled into this binary.

**Returns**: `true` if either `feature="gpu"` or `feature="cuda"` was enabled at compile time.

**Usage Pattern**:
```rust
use bitnet_kernels::device_features::gpu_compiled;

pub fn validate_device_request(device: &Device) -> Result<()> {
    if matches!(device, Device::Cuda(_)) && !gpu_compiled() {
        return Err(DeviceError::NotCompiled {
            device: "GPU",
            hint: "Rebuild with --features gpu",
        });
    }
    Ok(())
}
```

**Implementation Note**: Uses `cfg!(any(feature = "gpu", feature = "cuda"))` to unify both feature flags.

#### `gpu_available_runtime() -> bool`

**Purpose**: Check if GPU is available at runtime (requires GPU compiled first).

**Returns**:
- `false` if:
  - GPU not compiled (`gpu_compiled() == false`)
  - CUDA runtime unavailable (`nvidia-smi` fails)
  - `BITNET_GPU_FAKE=none` environment variable set
- `true` if:
  - GPU compiled AND CUDA runtime detected
  - `BITNET_GPU_FAKE=cuda` environment variable set (overrides real detection)

**Usage Pattern**:
```rust
use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};

pub fn select_optimal_device() -> Device {
    if gpu_compiled() && gpu_available_runtime() {
        Device::Cuda(0) // GPU available
    } else {
        Device::Cpu // Fallback to CPU
    }
}
```

**Environment Variable Precedence**:

1. **`BITNET_GPU_FAKE=cuda`**: Force GPU available (for testing)
2. **`BITNET_GPU_FAKE=none`**: Force GPU unavailable (for testing)
3. **No fake set**: Check real CUDA availability via `gpu_utils::get_gpu_info()`

**Implementation Detail**:
```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_available_runtime() -> bool {
    // Fake precedence for deterministic testing
    if let Ok(fake) = std::env::var("BITNET_GPU_FAKE") {
        return fake.to_lowercase().contains("cuda");
    }

    // Real CUDA detection
    crate::gpu_utils::get_gpu_info().cuda
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
pub fn gpu_available_runtime() -> bool {
    false // Stub when GPU not compiled
}
```

#### `device_capability_summary() -> String`

**Purpose**: Get human-readable diagnostic summary of device capabilities.

**Returns**: Multi-line string with compile-time and runtime capabilities.

**Example Output**:
```text
Device Capabilities:
  Compiled: GPU ✓, CPU ✓
  Runtime: CUDA 12.1 ✓
```

**Usage Pattern**:
```rust
use bitnet_kernels::device_features::device_capability_summary;

pub fn diagnostic_report() {
    println!("{}", device_capability_summary());
    println!("Model: bitnet-b1.58-2B");
    println!("Quantization: I2_S");
}
```

## Integration Points

### Quantization (AC3)

**File**: `crates/bitnet-quantization/src/i2s.rs`

```rust
use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};

impl I2SQuantizer {
    pub fn supports_device(&self, device: &Device) -> bool {
        match device {
            Device::Cpu => true,
            Device::Cuda(_) => gpu_compiled() && gpu_available_runtime(),
            Device::Metal => false,
        }
    }
}
```

### Model Loading

**File**: `crates/bitnet-models/src/builder.rs`

```rust
use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};

impl BitNetModelBuilder {
    pub fn with_device(mut self, device: Device) -> Result<Self> {
        match device {
            Device::Cpu => {
                self.device = device;
                Ok(self)
            }
            Device::Cuda(_) => {
                if !gpu_compiled() {
                    return Err(ModelError::DeviceNotCompiled("GPU"));
                }
                if !gpu_available_runtime() {
                    return Err(ModelError::DeviceNotAvailable("CUDA"));
                }
                self.device = device;
                Ok(self)
            }
            _ => Err(ModelError::UnsupportedDevice(device)),
        }
    }
}
```

### Inference Engine

**File**: `crates/bitnet-inference/src/engine.rs`

```rust
use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};

impl InferenceEngine {
    pub fn new(device: Device) -> Result<Self> {
        // Validate device before creating engine
        match device {
            Device::Cpu => Ok(Self { device, ..Default::default() }),
            Device::Cuda(gpu_id) => {
                if !gpu_compiled() || !gpu_available_runtime() {
                    eprintln!("Warning: GPU requested but unavailable, using CPU");
                    Ok(Self { device: Device::Cpu, ..Default::default() })
                } else {
                    Ok(Self { device, ..Default::default() })
                }
            }
            _ => Err(InferenceError::UnsupportedDevice),
        }
    }
}
```

### xtask Preflight (AC5)

**File**: `xtask/src/preflight.rs`

```rust
use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime, device_capability_summary};

pub fn run_preflight() -> anyhow::Result<()> {
    println!("BitNet.rs Preflight Check");
    println!("========================");
    println!();
    println!("{}", device_capability_summary());
    println!();

    if gpu_compiled() {
        if gpu_available_runtime() {
            println!("✓ GPU: Available for inference");
        } else {
            println!("⚠ GPU: Compiled but not available at runtime");
            println!("  Run 'nvidia-smi' to check CUDA installation");
        }
    } else {
        println!("✗ GPU: Not compiled (rebuild with --features gpu)");
    }

    Ok(())
}
```

## Testing Strategy

### Unit Tests (AC3)

**File**: `crates/bitnet-kernels/tests/device_features.rs`

```rust
use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};

#[test]
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn ac3_gpu_compiled_true_with_features() {
    // AC:3 - Verify gpu_compiled() returns true when features enabled
    assert!(gpu_compiled(), "gpu_compiled() should return true with gpu/cuda features");
}

#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn ac3_gpu_compiled_false_without_features() {
    // AC:3 - Verify gpu_compiled() returns false when features disabled
    assert!(!gpu_compiled(), "gpu_compiled() should return false without gpu/cuda features");
}

#[test]
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn ac3_gpu_fake_cuda_overrides_detection() {
    // AC:3 - Verify BITNET_GPU_FAKE=cuda forces GPU available
    std::env::set_var("BITNET_GPU_FAKE", "cuda");
    assert!(
        gpu_available_runtime(),
        "BITNET_GPU_FAKE=cuda should override real detection"
    );
    std::env::remove_var("BITNET_GPU_FAKE");
}

#[test]
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn ac3_gpu_fake_none_disables_detection() {
    // AC:3 - Verify BITNET_GPU_FAKE=none forces GPU unavailable
    std::env::set_var("BITNET_GPU_FAKE", "none");
    assert!(
        !gpu_available_runtime(),
        "BITNET_GPU_FAKE=none should disable GPU detection"
    );
    std::env::remove_var("BITNET_GPU_FAKE");
}
```

### Integration Tests

**File**: `xtask/tests/preflight.rs`

```rust
#[test]
fn ac5_preflight_respects_fake_none() {
    // AC:5 - Verify preflight reports no GPU with BITNET_GPU_FAKE=none
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
```

## Validation Commands

### Verify Module Exists

```bash
# Check module file exists
test -f crates/bitnet-kernels/src/device_features.rs && echo "✓ Module exists" || echo "✗ Module missing"

# Verify module exported in lib.rs
grep -n "pub mod device_features" crates/bitnet-kernels/src/lib.rs
```

### Run Unit Tests

```bash
# Test with CPU features
cargo test --package bitnet-kernels --no-default-features --features cpu device_features

# Test with GPU features
cargo test --package bitnet-kernels --no-default-features --features gpu device_features

# Test without features (should compile with stubs)
cargo test --package bitnet-kernels --no-default-features device_features
```

### Verify Integration Usage

```bash
# Find all supports_device() implementations using helpers
rg -n 'supports_device' -A 5 crates/ | grep -E '(gpu_compiled|gpu_available_runtime)'

# Verify quantization uses device features
grep -n "use bitnet_kernels::device_features" crates/bitnet-quantization/src/*.rs

# Verify xtask uses device features
grep -n "use bitnet_kernels::device_features" xtask/src/*.rs
```

## Performance Impact

### Compile-Time Impact

- **Zero runtime cost**: `gpu_compiled()` uses `cfg!()` macro (compile-time constant)
- **Inlined functions**: All functions marked `#[inline]` for zero call overhead
- **No allocations**: All functions return stack values (bool, String)

### Runtime Impact

- **GPU detection cached**: `gpu_utils::get_gpu_info()` caches CUDA detection result
- **Fake precedence fast**: Environment variable check is O(1) hash lookup
- **Minimal overhead**: Device validation adds <1μs per inference request

## Migration Guide

### Before (Inconsistent)

```rust
// Runtime check using gpu feature
#[cfg(feature = "gpu")]
pub fn supports_device(device: &Device) -> bool {
    match device {
        // Compile gate using cuda feature - MISMATCH!
        Device::Cuda(_) => cfg!(feature = "cuda"),
        _ => true,
    }
}
```

### After (Unified)

```rust
use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};

pub fn supports_device(device: &Device) -> bool {
    match device {
        Device::Cpu => true,
        Device::Cuda(_) => gpu_compiled() && gpu_available_runtime(),
        Device::Metal => false,
    }
}
```

### Migration Checklist

- [ ] Add `use bitnet_kernels::device_features::*` import
- [ ] Replace `cfg!(feature = "cuda")` with `gpu_compiled()`
- [ ] Replace `cfg!(feature = "gpu")` with `gpu_compiled()`
- [ ] Add runtime check with `gpu_available_runtime()` where needed
- [ ] Update tests to use `BITNET_GPU_FAKE` for deterministic testing

## Related Documentation

- **Main Spec**: `docs/explanation/issue-439-spec.md` - Full feature gate hardening specification
- **Receipt Validation**: `docs/explanation/receipt-validation.md` - GPU kernel verification
- **GPU Architecture**: `docs/gpu-kernel-architecture.md` - CUDA kernel design patterns
- **Feature Flags**: `docs/explanation/FEATURES.md` - Feature flag documentation

## References

- **Issue #439**: GPU feature-gate hardening (workspace-wide)
- **Issue #437**: Feature propagation rename (`cuda` → `gpu`)
- **Issue #438**: Quantization feature gate alignment
- **AC3**: Shared helpers requirement for device capability detection
