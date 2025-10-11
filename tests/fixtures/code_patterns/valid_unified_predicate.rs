// VALID: Unified GPU feature predicate for Issue #439 AC1
// Tests specification: docs/explanation/issue-439-spec.md#ac1-kernel-gate-unification

/// GPU-specific module with unified feature gate
///
/// This pattern ensures GPU code compiles when EITHER feature="gpu" OR feature="cuda"
/// is enabled, preventing compile-time drift.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod gpu_module {
    use super::*;

    /// GPU-accelerated matrix multiplication
    pub fn gpu_matmul(a: &[f32], b: &[f32], c: &mut [f32]) {
        // CUDA kernel implementation
        println!("Running GPU matmul");
    }

    /// I2S quantization on GPU
    pub fn i2s_gpu_quantize(input: &[f32]) -> Vec<i8> {
        println!("Running I2S GPU quantization");
        vec![0; input.len()]
    }

    /// GPU device initialization
    pub fn init_gpu() -> Result<(), Box<dyn std::error::Error>> {
        println!("Initializing GPU");
        Ok(())
    }
}

/// CPU fallback implementation (always compiled)
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
pub mod gpu_module {
    /// Stub implementation when GPU not compiled
    pub fn gpu_matmul(_a: &[f32], _b: &[f32], _c: &mut [f32]) {
        panic!("GPU module not compiled - use --features gpu");
    }

    pub fn i2s_gpu_quantize(_input: &[f32]) -> Vec<i8> {
        panic!("GPU module not compiled - use --features gpu");
    }

    pub fn init_gpu() -> Result<(), Box<dyn std::error::Error>> {
        Err("GPU not available".into())
    }
}

/// Runtime GPU check using unified cfg! macro
pub fn has_gpu_support() -> bool {
    cfg!(any(feature = "gpu", feature = "cuda"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_gpu_compiled() {
        assert!(has_gpu_support(), "GPU should be available");
    }

    #[test]
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    fn test_gpu_not_compiled() {
        assert!(!has_gpu_support(), "GPU should NOT be available");
    }
}
