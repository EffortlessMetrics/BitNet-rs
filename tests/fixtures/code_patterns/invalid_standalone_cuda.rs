// ANTI-PATTERN: Standalone cuda feature gate (Issue #439 AC1)
// This pattern will FAIL AC1 validation tests

/// WRONG: Only checks feature="cuda", ignores feature="gpu"
#[cfg(feature = "cuda")]
pub mod gpu_module {
    /// GPU matmul - only compiles with --features cuda
    /// BREAKS with --features gpu
    pub fn gpu_matmul(a: &[f32], b: &[f32], c: &mut [f32]) {
        println!("Running GPU matmul");
    }
}

/// CPU fallback - only compiles when cuda NOT enabled
#[cfg(not(feature = "cuda"))]
pub mod gpu_module {
    /// This fallback is used even when --features gpu is set!
    /// This is the COMPILE-TIME DRIFT issue
    pub fn gpu_matmul(_a: &[f32], _b: &[f32], _c: &mut [f32]) {
        panic!("GPU not available");
    }
}

/// Runtime check - also wrong
pub fn has_gpu_support() -> bool {
    // WRONG: Only checks cuda feature
    cfg!(feature = "cuda")
}

// Expected unified pattern:
// #[cfg(any(feature = "gpu", feature = "cuda"))]
// pub mod gpu_module { ... }
//
// #[cfg(not(any(feature = "gpu", feature = "cuda")))]
// pub mod gpu_module { ... }
