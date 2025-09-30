# Stub code: `select_cpu_kernel` in `lib.rs` has conditional compilation for various kernels

The `select_cpu_kernel` function in `crates/bitnet-kernels/src/lib.rs` has conditional compilation for `Avx2Kernel` and `NeonKernel`. If these features are not enabled or detected, the corresponding kernels are not added to the providers. This is a form of stubbing.

**File:** `crates/bitnet-kernels/src/lib.rs`

**Function:** `select_cpu_kernel`

**Code:**
```rust
pub fn select_cpu_kernel() -> Result<Box<dyn KernelProvider>> {
    #[allow(unused_mut)]
    let mut providers: Vec<Box<dyn KernelProvider>> = vec![Box::new(cpu::FallbackKernel)];

    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        if is_x86_feature_detected!("avx2") {
            providers.insert(0, Box::new(cpu::Avx2Kernel));
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            providers.insert(0, Box::new(cpu::NeonKernel));
        }
    }

    for provider in providers {
        if provider.is_available() {
            return Ok(provider);
        }
    }

    Err(bitnet_common::BitNetError::Kernel(bitnet_common::KernelError::NoProvider))
}
```

## Proposed Fix

The `select_cpu_kernel` function should not have conditional compilation for various kernels. Instead, the kernel providers should be dynamically loaded based on the available hardware features at runtime. This would involve:

1.  **Removing conditional compilation:** Remove the `#[cfg(...)]` attributes.
2.  **Dynamic kernel loading:** Dynamically load kernel providers based on available hardware features.
3.  **Providing a clear error message:** If a kernel is not available, provide a clear error message instead of silently skipping it.

### Example Implementation

```rust
pub fn select_cpu_kernel() -> Result<Box<dyn KernelProvider>> {
    #[allow(unused_mut)]
    let mut providers: Vec<Box<dyn KernelProvider>> = vec![Box::new(cpu::FallbackKernel)];

    if is_x86_feature_detected!("avx2") {
        providers.insert(0, Box::new(cpu::Avx2Kernel));
    }

    if std::arch::is_aarch64_feature_detected!("neon") {
        providers.insert(0, Box::new(cpu::NeonKernel));
    }

    for provider in providers {
        if provider.is_available() {
            return Ok(provider);
        }
    }

    Err(bitnet_common::BitNetError::Kernel(bitnet_common::KernelError::NoProvider))
}
```
