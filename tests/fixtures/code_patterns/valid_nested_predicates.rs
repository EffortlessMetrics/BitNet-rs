// VALID: Nested feature predicates with unified GPU detection
// Tests specification: docs/explanation/issue-439-spec.md#ac1

/// Top-level GPU module with unified predicate
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod gpu {
    /// Mixed precision sub-module (GPU + specific precision feature)
    #[cfg(all(
        any(feature = "gpu", feature = "cuda"),
        feature = "mixed_precision"
    ))]
    pub mod mixed_precision {
        pub fn gemm_fp16() {
            println!("FP16 GEMM kernel");
        }

        pub fn wmma_bf16() {
            println!("BF16 Tensor Core kernel");
        }
    }

    /// I2S GPU quantization (always available when GPU compiled)
    pub mod i2s {
        pub fn quantize_gpu(input: &[f32]) -> Vec<i8> {
            println!("I2S GPU quantization");
            vec![0; input.len()]
        }
    }

    /// Validation module with unified predicate
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    pub mod validation {
        pub fn validate_gpu_receipt() -> bool {
            println!("Validating GPU receipt");
            true
        }
    }
}

/// FFI bridge (requires both GPU and FFI features)
#[cfg(all(
    any(feature = "gpu", feature = "cuda"),
    feature = "ffi"
))]
pub mod ffi_bridge {
    pub fn call_cpp_kernel() {
        println!("Calling C++ kernel via FFI");
    }
}

/// Runtime capability checks
pub mod capabilities {
    /// Check if GPU compiled
    pub fn gpu_compiled() -> bool {
        cfg!(any(feature = "gpu", feature = "cuda"))
    }

    /// Check if mixed precision compiled
    pub fn mixed_precision_compiled() -> bool {
        cfg!(all(
            any(feature = "gpu", feature = "cuda"),
            feature = "mixed_precision"
        ))
    }
}
