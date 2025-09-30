# Stub code: `NeonKernel` in `arm.rs` is conditionally compiled

The `NeonKernel` struct and its implementation in `crates/bitnet-kernels/src/cpu/arm.rs` are conditionally compiled with `#[cfg(target_arch = "aarch64")]`. If ARM64 is not detected, it uses a stub implementation. This is a form of stubbing.

**File:** `crates/bitnet-kernels/src/cpu/arm.rs`

**Struct:** `NeonKernel`

**Code:**
```rust
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON optimized kernel for ARM64 architectures
///
/// This kernel leverages ARM NEON SIMD instructions for vectorized operations,
/// providing significant performance improvements over the fallback implementation.
/// It's specifically optimized for TL1 quantization patterns.
///
/// Performance characteristics:
/// - Matrix multiplication: Vectorized using NEON with 4x float32 operations per instruction
/// - TL1 quantization: Optimized lookup table generation and vectorized processing
/// - Memory access: Optimized for ARM cache hierarchy and memory bandwidth
///
/// Requirements:
/// - ARM64 architecture with NEON support
/// - Target feature "neon" must be available at runtime
#[cfg(target_arch = "aarch64")]
pub struct NeonKernel;
```

## Proposed Fix

The `NeonKernel` struct and its implementation should not be conditionally compiled. Instead, the NEON kernel should be integrated directly into the `arm.rs` file without relying on feature flags. This would involve:

1.  **Removing conditional compilation:** Remove the `#[cfg(target_arch = "aarch64")]` attributes.
2.  **Implementing NEON intrinsics:** Implement the NEON intrinsics directly in the functions.
3.  **Providing a clear error message:** If NEON is not available, provide a clear error message instead of falling back to a stub implementation.

### Example Implementation

```rust
use std::arch::aarch64::*;

pub struct NeonKernel;

impl KernelProvider for NeonKernel {
    fn name(&self) -> &'static str {
        "neon"
    }

    fn is_available(&self) -> bool {
        // NEON is mandatory on ARM64, but check for safety
        std::arch::is_aarch64_feature_detected!("neon")
    }

    // ...
}
```
