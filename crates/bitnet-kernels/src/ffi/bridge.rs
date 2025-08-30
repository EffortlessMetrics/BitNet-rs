//! FFI bridge implementation for C++ kernel integration
//!
//! This module provides safe Rust wrappers around existing C++ kernel functions,
//! enabling gradual migration from C++ to native Rust implementations while
//! maintaining performance and correctness during the transition period.

use crate::KernelProvider;
use bitnet_common::{BitNetError, KernelError, QuantizationType, Result};

// Keep the public module path stable for callers
pub mod cpp {
    // Real bridge only when the crate feature is on AND build.rs detected the C++ lib
    #[cfg(all(feature = "ffi", have_cpp))]
    mod imp {
        use std::ffi::{c_char, c_float, c_int, c_uchar};

        extern "C" {
            fn bitnet_cpp_init() -> c_int;
            fn bitnet_cpp_cleanup();
            fn bitnet_cpp_is_available() -> c_int;
            fn bitnet_cpp_matmul_i2s(
                a: *const i8,
                b: *const c_uchar,
                c: *mut c_float,
                m: c_int,
                n: c_int,
                k: c_int,
            ) -> c_int;
            fn bitnet_cpp_quantize(
                input: *const c_float,
                input_len: c_int,
                output: *mut c_uchar,
                output_len: c_int,
                scales: *mut c_float,
                scales_len: c_int,
                qtype: c_int,
            ) -> c_int;
            fn bitnet_cpp_get_last_error() -> *const c_char;
        }

        pub fn init() -> i32 {
            unsafe { bitnet_cpp_init() }
        }
        pub fn cleanup() {
            unsafe { bitnet_cpp_cleanup() }
        }
        pub fn is_available() -> bool {
            unsafe { bitnet_cpp_is_available() != 0 }
        }

        pub fn matmul_i2s(
            a: &[i8],
            b: &[u8],
            c: &mut [f32],
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<(), &'static str> {
            let rc = unsafe {
                bitnet_cpp_matmul_i2s(
                    a.as_ptr(),
                    b.as_ptr() as *const c_uchar,
                    c.as_mut_ptr() as *mut c_float,
                    m as c_int,
                    n as c_int,
                    k as c_int,
                )
            };
            if rc == 0 { Ok(()) } else { Err("cpp matmul failed") }
        }

        pub fn quantize(
            input: &[f32],
            output: &mut [u8],
            scales: &mut [f32],
            qtype: i32,
        ) -> Result<(), &'static str> {
            let rc = unsafe {
                bitnet_cpp_quantize(
                    input.as_ptr() as *const c_float,
                    input.len() as c_int,
                    output.as_mut_ptr() as *mut c_uchar,
                    output.len() as c_int,
                    scales.as_mut_ptr() as *mut c_float,
                    scales.len() as c_int,
                    qtype as c_int,
                )
            };
            if rc == 0 { Ok(()) } else { Err("cpp quantize failed") }
        }

        pub fn get_last_error() -> &'static str {
            unsafe {
                let ptr = bitnet_cpp_get_last_error();
                if ptr.is_null() {
                    "unknown error"
                } else {
                    std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("invalid error string")
                }
            }
        }
    }

    // Fallback stubs when ffi is off OR the C++ library is missing
    #[cfg(any(not(feature = "ffi"), not(have_cpp)))]
    mod imp {
        pub fn init() -> i32 {
            -1
        }
        pub fn cleanup() {}
        pub fn is_available() -> bool {
            false
        }

        pub fn matmul_i2s(
            _a: &[i8],
            _b: &[u8],
            _c: &mut [f32],
            _m: usize,
            _n: usize,
            _k: usize,
        ) -> Result<(), &'static str> {
            Err("ffi bridge unavailable")
        }

        pub fn quantize(
            _input: &[f32],
            _output: &mut [u8],
            _scales: &mut [f32],
            _qtype: i32,
        ) -> Result<(), &'static str> {
            Err("ffi bridge unavailable")
        }

        pub fn get_last_error() -> &'static str {
            "ffi bridge unavailable"
        }
    }

    pub use imp::*;
}

/// FFI kernel that bridges to existing C++ implementations
///
/// This kernel provides a safe interface to existing C++ kernel functions,
/// allowing for gradual migration while maintaining performance. It includes
/// comprehensive error handling and memory safety guarantees.
///
/// Performance characteristics:
/// - Matrix multiplication: Delegates to optimized C++ implementations
/// - Quantization: Uses existing C++ quantization algorithms
/// - Memory management: Safe Rust wrappers with automatic cleanup
///
/// Migration path:
/// 1. Use FFI kernel to maintain functionality during migration
/// 2. Gradually replace C++ functions with native Rust implementations
/// 3. Performance test each replacement to ensure no regressions
/// 4. Remove FFI bridge once all kernels are migrated
pub struct FfiKernel {
    initialized: bool,
}

impl FfiKernel {
    /// Create a new FFI kernel instance
    pub fn new() -> Result<Self> {
        // Initialize C++ library if available
        if cpp::init() == 0 {
            Ok(Self { initialized: true })
        } else {
            Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: "BitNet C++ FFI not available (build with --features ffi and ensure C++ library is present)".to_string(),
            }))
        }
    }
}

impl Default for FfiKernel {
    fn default() -> Self {
        Self::new().unwrap_or(Self { initialized: false })
    }
}

impl KernelProvider for FfiKernel {
    fn name(&self) -> &'static str {
        "ffi"
    }

    fn is_available(&self) -> bool {
        self.initialized && cpp::is_available()
    }

    fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if !self.is_available() {
            return Err(BitNetError::Kernel(KernelError::NoProvider));
        }

        // Validate input dimensions
        if a.len() != m * k {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!("Matrix A dimension mismatch: expected {}, got {}", m * k, a.len()),
            }));
        }
        if b.len() != k * n {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!("Matrix B dimension mismatch: expected {}, got {}", k * n, b.len()),
            }));
        }
        if c.len() != m * n {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!("Matrix C dimension mismatch: expected {}, got {}", m * n, c.len()),
            }));
        }

        // Call C++ implementation
        cpp::matmul_i2s(a, b, c, m, n, k).map_err(|msg| {
            BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!("C++ kernel error: {}", msg),
            })
        })
    }

    fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()> {
        if !self.is_available() {
            return Err(BitNetError::Kernel(KernelError::NoProvider));
        }

        // Convert quantization type to C++ enum
        let cpp_qtype = match qtype {
            QuantizationType::I2S => 0,
            QuantizationType::TL1 => 1,
            QuantizationType::TL2 => 2,
        };

        // Call C++ implementation
        cpp::quantize(input, output, scales, cpp_qtype).map_err(|msg| {
            BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!("C++ quantization error: {}", msg),
            })
        })
    }
}

impl Drop for FfiKernel {
    fn drop(&mut self) {
        if self.initialized {
            cpp::cleanup();
        }
    }
}

/// Performance comparison utilities for migration validation
pub struct PerformanceComparison {
    pub rust_time_ns: u64,
    pub cpp_time_ns: u64,
    pub accuracy_match: bool,
    pub max_error: f32,
}

impl PerformanceComparison {
    /// Compare performance between Rust and C++ implementations
    pub fn compare_matmul(
        rust_kernel: &dyn KernelProvider,
        cpp_kernel: &FfiKernel,
        a: &[i8],
        b: &[u8],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        let mut rust_result = vec![0.0f32; m * n];
        let mut cpp_result = vec![0.0f32; m * n];

        // Time Rust implementation
        let rust_start = std::time::Instant::now();
        rust_kernel.matmul_i2s(a, b, &mut rust_result, m, n, k)?;
        let rust_time_ns = rust_start.elapsed().as_nanos() as u64;

        // Time C++ implementation
        let cpp_start = std::time::Instant::now();
        cpp_kernel.matmul_i2s(a, b, &mut cpp_result, m, n, k)?;
        let cpp_time_ns = cpp_start.elapsed().as_nanos() as u64;

        // Compare accuracy
        let mut max_error = 0.0f32;
        let mut accuracy_match = true;
        const TOLERANCE: f32 = 1e-5;

        for (rust_val, cpp_val) in rust_result.iter().zip(cpp_result.iter()) {
            let error = (rust_val - cpp_val).abs();
            max_error = max_error.max(error);
            if error > TOLERANCE {
                accuracy_match = false;
            }
        }

        Ok(Self { rust_time_ns, cpp_time_ns, accuracy_match, max_error })
    }

    /// Compare quantization performance
    pub fn compare_quantize(
        rust_kernel: &dyn KernelProvider,
        cpp_kernel: &FfiKernel,
        input: &[f32],
        qtype: QuantizationType,
    ) -> Result<Self> {
        let output_len = input.len() / 4;
        let scales_len = input.len().div_ceil(32); // Assuming 32-element blocks

        let mut rust_output = vec![0u8; output_len];
        let mut rust_scales = vec![0.0f32; scales_len];
        let mut cpp_output = vec![0u8; output_len];
        let mut cpp_scales = vec![0.0f32; scales_len];

        // Time Rust implementation
        let rust_start = std::time::Instant::now();
        rust_kernel.quantize(input, &mut rust_output, &mut rust_scales, qtype)?;
        let rust_time_ns = rust_start.elapsed().as_nanos() as u64;

        // Time C++ implementation
        let cpp_start = std::time::Instant::now();
        cpp_kernel.quantize(input, &mut cpp_output, &mut cpp_scales, qtype)?;
        let cpp_time_ns = cpp_start.elapsed().as_nanos() as u64;

        // Compare accuracy (scales are more important than exact bit patterns)
        let mut max_error = 0.0f32;
        let mut accuracy_match = true;
        const TOLERANCE: f32 = 1e-4;

        for (rust_scale, cpp_scale) in rust_scales.iter().zip(cpp_scales.iter()) {
            let error = (rust_scale - cpp_scale).abs();
            max_error = max_error.max(error);
            if error > TOLERANCE {
                accuracy_match = false;
            }
        }

        Ok(Self { rust_time_ns, cpp_time_ns, accuracy_match, max_error })
    }

    /// Get performance improvement ratio (positive means Rust is faster)
    pub fn performance_improvement(&self) -> f64 {
        if self.rust_time_ns == 0 {
            return 0.0;
        }

        (self.cpp_time_ns as f64 / self.rust_time_ns as f64) - 1.0
    }

    /// Check if migration is recommended based on performance and accuracy
    pub fn migration_recommended(&self) -> bool {
        // Recommend migration if:
        // 1. Accuracy matches (within tolerance)
        // 2. Performance is at least as good (no more than 10% slower)
        self.accuracy_match && self.performance_improvement() >= -0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_kernel_creation() {
        // This test will fail if C++ library is not available, which is expected
        match FfiKernel::new() {
            Ok(kernel) => {
                assert_eq!(kernel.name(), "ffi");
                // If creation succeeds, test basic functionality
                println!("FFI kernel available: {}", kernel.is_available());
            }
            Err(_) => {
                // Expected when C++ library is not available
                println!("FFI kernel not available (expected in test environment)");
            }
        }
    }

    #[test]
    fn test_performance_comparison_structure() {
        // Test the performance comparison structure without actual C++ calls
        let comparison = PerformanceComparison {
            rust_time_ns: 1000,
            cpp_time_ns: 1200,
            accuracy_match: true,
            max_error: 1e-6,
        };

        assert!(comparison.performance_improvement() > 0.0); // Rust is faster
        assert!(comparison.migration_recommended());

        let comparison_slow = PerformanceComparison {
            rust_time_ns: 1200,
            cpp_time_ns: 1000,
            accuracy_match: true,
            max_error: 1e-6,
        };

        assert!(comparison_slow.performance_improvement() < 0.0); // Rust is slower
        assert!(comparison_slow.migration_recommended()); // But still within tolerance

        let comparison_inaccurate = PerformanceComparison {
            rust_time_ns: 800,
            cpp_time_ns: 1000,
            accuracy_match: false,
            max_error: 1e-2,
        };

        assert!(!comparison_inaccurate.migration_recommended()); // Accuracy mismatch
    }

    #[test]
    fn test_stub_implementation() {
        // Test that stub implementation works when FFI is not available
        if !cpp::is_available() {
            let kernel = FfiKernel::default();
            assert_eq!(kernel.name(), "ffi");
            assert!(!kernel.is_available());

            let result = kernel.matmul_i2s(&[0], &[0], &mut [0.0], 1, 1, 1);
            assert!(result.is_err());
        }
    }
}
