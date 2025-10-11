// ANTI-PATTERN: Standalone gpu feature gate without cuda alias
// This pattern will FAIL AC1 validation for backward compatibility

/// WRONG: Only checks feature="gpu", ignores feature="cuda" alias
#[cfg(feature = "gpu")]
pub mod gpu_module {
    /// GPU matmul - only compiles with --features gpu
    /// BREAKS with --features cuda (backward compatibility issue)
    pub fn gpu_matmul(a: &[f32], b: &[f32], c: &mut [f32]) {
        println!("Running GPU matmul");
    }
}

/// CPU fallback
#[cfg(not(feature = "gpu"))]
pub mod gpu_module {
    /// This fallback is used when --features cuda is set!
    /// Breaks backward compatibility
    pub fn gpu_matmul(_a: &[f32], _b: &[f32], _c: &mut [f32]) {
        panic!("GPU not available");
    }
}

/// Runtime check - also incomplete
pub fn has_gpu_support() -> bool {
    // WRONG: Only checks gpu feature, ignores cuda alias
    cfg!(feature = "gpu")
}

// Expected unified pattern:
// #[cfg(any(feature = "gpu", feature = "cuda"))]
// This ensures both --features gpu AND --features cuda work
