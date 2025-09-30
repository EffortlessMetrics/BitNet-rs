# Stub code: `KernelManager::new` in `lib.rs` has conditional compilation for various kernels

The `KernelManager::new` function in `crates/bitnet-kernels/src/lib.rs` has conditional compilation for `cuda_kernel`, `Avx512Kernel`, `Avx2Kernel`, `NeonKernel`, and `FfiKernel`. If these features are not enabled or detected, the corresponding kernels are not added to the providers. This is a form of stubbing.

**File:** `crates/bitnet-kernels/src/lib.rs`

**Function:** `KernelManager::new`

**Code:**
```rust
impl KernelManager {
    pub fn new() -> Self {
        #[allow(unused_mut)]
        let mut providers: Vec<Box<dyn KernelProvider>> = vec![Box::new(cpu::FallbackKernel)];

        // Add GPU kernels first (highest priority)
        #[cfg(feature = "gpu")]
        {
            if let Ok(cuda_kernel) = gpu::CudaKernel::new() {
                if cuda_kernel.is_available() {
                    log::info!("CUDA kernel available, adding to providers");
                    providers.insert(0, Box::new(cuda_kernel));
                }
            } else {
                log::debug!("CUDA kernel not available");
            }
        }

        // Add optimized CPU kernels in order of preference (best first)
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                let insert_pos = if providers.is_empty() { 0 } else { providers.len() - 1 };
                providers.insert(insert_pos, Box::new(cpu::Avx512Kernel));
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            if is_x86_feature_detected!("avx2") {
                let insert_pos = if providers.len() > 1 { providers.len() - 1 } else { 0 };
                providers.insert(insert_pos, Box::new(cpu::Avx2Kernel));
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                let insert_pos = if providers.len() > 1 { providers.len() - 1 } else { 0 };
                providers.insert(insert_pos, Box::new(cpu::NeonKernel));
            }
        }

        // Add FFI kernel as a fallback option (lower priority than optimized kernels)
        #[cfg(feature = "ffi")]
        {
            if let Ok(ffi_kernel) = ffi::FfiKernel::new()
                && ffi_kernel.is_available()
            {
                providers.push(Box::new(ffi_kernel));
            }
        }

        Self { providers, selected: OnceLock::new() }
    }
```

## Proposed Fix

The `KernelManager::new` function should not have conditional compilation for various kernels. Instead, the kernel providers should be dynamically loaded based on the available hardware features at runtime. This would involve:

1.  **Removing conditional compilation:** Remove the `#[cfg(...)]` attributes.
2.  **Dynamic kernel loading:** Dynamically load kernel providers based on available hardware features.
3.  **Providing a clear error message:** If a kernel is not available, provide a clear error message instead of silently skipping it.

### Example Implementation

```rust
impl KernelManager {
    pub fn new() -> Self {
        let mut providers: Vec<Box<dyn KernelProvider>> = vec![Box::new(cpu::FallbackKernel)];

        // Add GPU kernels first (highest priority)
        if let Ok(cuda_kernel) = gpu::CudaKernel::new() {
            if cuda_kernel.is_available() {
                log::info!("CUDA kernel available, adding to providers");
                providers.insert(0, Box::new(cuda_kernel));
            }
        } else {
            log::debug!("CUDA kernel not available");
        }

        // Add optimized CPU kernels in order of preference (best first)
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            let insert_pos = if providers.is_empty() { 0 } else { providers.len() - 1 };
            providers.insert(insert_pos, Box::new(cpu::Avx512Kernel));
        }

        if is_x86_feature_detected!("avx2") {
            let insert_pos = if providers.len() > 1 { providers.len() - 1 } else { 0 };
            providers.insert(insert_pos, Box::new(cpu::Avx2Kernel));
        }

        if std::arch::is_aarch64_feature_detected!("neon") {
            let insert_pos = if providers.len() > 1 { providers.len() - 1 } else { 0 };
            providers.insert(insert_pos, Box::new(cpu::NeonKernel));
        }

        // Add FFI kernel as a fallback option (lower priority than optimized kernels)
        if let Ok(ffi_kernel) = ffi::FfiKernel::new()
            && ffi_kernel.is_available()
        {
            providers.push(Box::new(ffi_kernel));
        }

        Self { providers, selected: OnceLock::new() }
    }
```
