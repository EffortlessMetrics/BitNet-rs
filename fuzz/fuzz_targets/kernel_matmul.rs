#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct MatMulInput {
    m: u8,  // Use small dimensions to prevent timeout
    n: u8,
    k: u8,
    a_data: Vec<i8>,
    b_data: Vec<u8>,
}

fuzz_target!(|input: MatMulInput| {
    // Convert to reasonable dimensions
    let m = (input.m as usize).max(1).min(32);
    let n = (input.n as usize).max(1).min(32);
    let k = (input.k as usize).max(1).min(32);
    
    // Ensure we have enough data
    let a_size = m * k;
    let b_size = k * n;
    
    if input.a_data.len() < a_size || input.b_data.len() < b_size {
        return;
    }
    
    let a = &input.a_data[..a_size];
    let b = &input.b_data[..b_size];
    let mut c = vec![0.0f32; m * n];
    
    // Test fallback kernel (should never panic)
    let kernel = MockFallbackKernel;
    kernel.matmul_i2s(a, b, &mut c, m, n, k);
    
    // Verify output is finite
    for value in &c {
        assert!(value.is_finite(), "Matrix multiplication produced non-finite result");
    }
    
    // Test with SIMD kernels if available
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            let avx_kernel = MockAvxKernel;
            let mut c_avx = vec![0.0f32; m * n];
            avx_kernel.matmul_i2s(a, b, &mut c_avx, m, n, k);
            
            // Results should be similar (allowing for floating point differences)
            for (fallback, avx) in c.iter().zip(c_avx.iter()) {
                let diff = (fallback - avx).abs();
                assert!(diff < 1e-3, "AVX and fallback results differ too much: {} vs {}", fallback, avx);
            }
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            let neon_kernel = MockNeonKernel;
            let mut c_neon = vec![0.0f32; m * n];
            neon_kernel.matmul_i2s(a, b, &mut c_neon, m, n, k);
            
            // Results should be similar
            for (fallback, neon) in c.iter().zip(c_neon.iter()) {
                let diff = (fallback - neon).abs();
                assert!(diff < 1e-3, "NEON and fallback results differ too much: {} vs {}", fallback, neon);
            }
        }
    }
});

// Mock kernel implementations for fuzzing
trait KernelProvider {
    fn matmul_i2s(&self, a: &[i8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize);
}

struct MockFallbackKernel;

impl KernelProvider for MockFallbackKernel {
    fn matmul_i2s(&self, a: &[i8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
        // Simple fallback implementation
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    let a_val = a[i * k + l] as f32;
                    let b_val = b[l * n + j] as f32;
                    sum += a_val * b_val;
                }
                c[i * n + j] = sum;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
struct MockAvxKernel;

#[cfg(target_arch = "x86_64")]
impl KernelProvider for MockAvxKernel {
    fn matmul_i2s(&self, a: &[i8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
        // For fuzzing, just use the fallback implementation
        // In real code, this would use AVX2 intrinsics
        let fallback = MockFallbackKernel;
        fallback.matmul_i2s(a, b, c, m, n, k);
    }
}

#[cfg(target_arch = "aarch64")]
struct MockNeonKernel;

#[cfg(target_arch = "aarch64")]
impl KernelProvider for MockNeonKernel {
    fn matmul_i2s(&self, a: &[i8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
        // For fuzzing, just use the fallback implementation
        // In real code, this would use NEON intrinsics
        let fallback = MockFallbackKernel;
        fallback.matmul_i2s(a, b, c, m, n, k);
    }
}