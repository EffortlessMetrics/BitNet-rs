# [Architecture] Remove conditional compilation from KernelManager for dynamic kernel detection

## Problem Description

The `KernelManager::new` function in `crates/bitnet-kernels/src/lib.rs` uses conditional compilation (`#[cfg(...)]`) for various kernels, which prevents runtime detection of available hardware features and limits deployment flexibility.

## Environment

- **File**: `crates/bitnet-kernels/src/lib.rs`
- **Function**: `KernelManager::new`
- **Component**: Kernel management and hardware feature detection
- **Type**: Architecture improvement
- **MSRV**: Rust 1.90.0

## Current Implementation Issues

```rust
impl KernelManager {
    pub fn new() -> Self {
        let mut providers: Vec<Box<dyn KernelProvider>> = vec![Box::new(cpu::FallbackKernel)];

        #[cfg(feature = "gpu")]
        {
            // GPU kernels only available at compile time
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            // AVX2 kernels only available at compile time
        }
        // ... more conditional compilation
    }
}
```

**Problems:**
- Kernels determined at compile time, not runtime
- Cannot adapt to actual hardware capabilities
- Limits deployment flexibility
- Prevents dynamic optimization selection

## Proposed Solution

Implement runtime hardware detection with graceful fallbacks:

```rust
impl KernelManager {
    pub fn new() -> Self {
        let mut providers: Vec<Box<dyn KernelProvider>> = vec![Box::new(cpu::FallbackKernel)];

        // Runtime GPU detection
        if let Ok(cuda_kernel) = gpu::CudaKernel::new() {
            if cuda_kernel.is_available() {
                providers.insert(0, Box::new(cuda_kernel));
            }
        }

        // Runtime CPU feature detection
        if Self::supports_avx512() {
            providers.insert(-1, Box::new(cpu::Avx512Kernel));
        }

        if Self::supports_avx2() {
            providers.insert(-1, Box::new(cpu::Avx2Kernel));
        }

        if Self::supports_neon() {
            providers.insert(-1, Box::new(cpu::NeonKernel));
        }

        Self { providers, selected: OnceLock::new() }
    }

    fn supports_avx512() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
        }
        #[cfg(not(target_arch = "x86_64"))]
        false
    }

    fn supports_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        false
    }

    fn supports_neon() -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            std::arch::is_aarch64_feature_detected!("neon")
        }
        #[cfg(not(target_arch = "aarch64"))]
        false
    }
}
```

## Implementation Plan

### Phase 1: Runtime Detection (1 day)
- [ ] Replace conditional compilation with runtime detection
- [ ] Implement hardware capability checking functions
- [ ] Add graceful fallback mechanisms
- [ ] Test on different hardware configurations

### Phase 2: Dynamic Optimization (0.5 days)
- [ ] Add kernel performance benchmarking
- [ ] Implement automatic kernel selection
- [ ] Add kernel switching based on workload
- [ ] Create kernel capability reporting

## Acceptance Criteria

- [ ] Kernels detected at runtime, not compile time
- [ ] Proper fallback to available kernels
- [ ] Hardware detection works across architectures
- [ ] No performance regression from dynamic detection

## Labels

`architecture`, `kernel-management`, `runtime-detection`, `medium-priority`