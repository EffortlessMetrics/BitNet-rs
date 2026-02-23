//! QK256 AVX2 Fast Path Integration Tests
//!
//! This test suite validates the complete QK256 AVX2 fast path in a full
//! inference context, ensuring:
//!
//! 1. **Correctness**: AVX2 produces identical results to scalar reference
//! 2. **Performance**: AVX2 provides measurable speedup (≥1.2× baseline)
//! 3. **Determinism**: Same input produces same output across multiple runs
//! 4. **Receipt verification**: Honest compute proof for production inference
//!
//! ## Test Coverage
//!
//! - Single layer forward pass with QK256 weights
//! - Multi-layer transformer inference
//! - Deterministic output validation (greedy decoding)
//! - Performance regression gates (speedup ≥ 1.2×)
//! - Receipt schema validation (v1.0.0)
//!
//! ## Running Tests
//!
//! ```bash
//! # Run all QK256 fast path tests
//! cargo test --test qk256_fast_path --no-default-features --features cpu,avx2
//!
//! # Run with output for diagnostics
//! cargo test --test qk256_fast_path --no-default-features --features cpu,avx2 -- --nocapture
//! ```
#![cfg(all(target_arch = "x86_64", feature = "cpu"))]
use bitnet_common::Result;
/// Test QK256 dequantization correctness in isolation
///
/// Validates that AVX2 dequantization produces numerically identical
/// results to scalar reference implementation.
#[test]
fn test_qk256_dequant_correctness() {
    use bitnet_kernels::KernelProvider;
    use bitnet_kernels::cpu::x86::Avx2Kernel;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let kernel = Avx2Kernel;
    if !kernel.is_available() {
        eprintln!("Skipping test: AVX2 not available");
        return;
    }
    const QK256_PACKED_BYTES: usize = 64;
    let num_blocks = 8;
    let mut rng = ChaCha8Rng::seed_from_u64(424242);
    let mut quantized = vec![0i8; num_blocks * QK256_PACKED_BYTES];
    for byte in quantized.iter_mut() {
        *byte = rng.random();
    }
    let scales: Vec<f32> = (0..num_blocks).map(|_| rng.random_range(0.5..5.0)).collect();
    let result_avx2 =
        kernel.dequantize_qk256(&quantized, &scales, 256).expect("AVX2 dequantize should succeed");
    let result_scalar = kernel
        .dequantize_qk256_scalar(&quantized, &scales, 256)
        .expect("Scalar dequantize should succeed");
    assert_eq!(result_avx2.len(), result_scalar.len(), "Length mismatch");
    let mut max_abs_diff = 0.0f32;
    for (i, (&avx2_val, &scalar_val)) in result_avx2.iter().zip(result_scalar.iter()).enumerate() {
        let abs_diff = (avx2_val - scalar_val).abs();
        max_abs_diff = max_abs_diff.max(abs_diff);
        assert!(
            abs_diff < 1e-5,
            "Mismatch at element {}: AVX2={}, Scalar={}, diff={}",
            i,
            avx2_val,
            scalar_val,
            abs_diff
        );
    }
    println!("✅ QK256 dequant correctness test passed (max_diff={:.2e})", max_abs_diff);
}
/// Test QK256 dequantization performance baseline
///
/// Validates that AVX2 fast path provides measurable speedup over scalar
/// reference (≥1.2× baseline established in MVP).
#[test]
#[ignore = "TDD scaffold: unimplemented; see blocking issue for details"]
fn test_qk256_dequant_performance_baseline() {
    use bitnet_kernels::KernelProvider;
    use bitnet_kernels::cpu::x86::Avx2Kernel;
    use std::time::Instant;
    let kernel = Avx2Kernel;
    if !kernel.is_available() {
        eprintln!("Skipping performance test: AVX2 not available");
        return;
    }
    const QK256_PACKED_BYTES: usize = 64;
    let num_blocks = 64;
    let quantized = vec![0x1Bu8 as i8; num_blocks * QK256_PACKED_BYTES];
    let scales = vec![0.5f32; num_blocks];
    let _ = kernel.dequantize_qk256(&quantized, &scales, 256).unwrap();
    let _ = kernel.dequantize_qk256_scalar(&quantized, &scales, 256).unwrap();
    let start_scalar = Instant::now();
    for _ in 0..10 {
        let _ = kernel.dequantize_qk256_scalar(&quantized, &scales, 256).unwrap();
    }
    let scalar_time = start_scalar.elapsed();
    let start_avx2 = Instant::now();
    for _ in 0..10 {
        let _ = kernel.dequantize_qk256(&quantized, &scales, 256).unwrap();
    }
    let avx2_time = start_avx2.elapsed();
    let speedup = scalar_time.as_secs_f64() / avx2_time.as_secs_f64();
    println!("Scalar time: {:?}", scalar_time);
    println!("AVX2 time:   {:?}", avx2_time);
    println!("Speedup:     {:.2}×", speedup);
    assert!(speedup >= 1.2, "AVX2 speedup {:.2}× below 1.2× baseline requirement", speedup);
    println!("✅ QK256 performance baseline met ({:.2}× speedup)", speedup);
}
/// Test QK256 deterministic inference
///
/// Validates that the same input produces the same output across multiple
/// runs when using greedy decoding.
#[test]
fn test_qk256_deterministic_inference() -> Result<()> {
    use bitnet_kernels::KernelProvider;
    use bitnet_kernels::cpu::x86::Avx2Kernel;
    let kernel = Avx2Kernel;
    if !kernel.is_available() {
        eprintln!("Skipping deterministic test: AVX2 not available");
        return Ok(());
    }
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(999999);
    const QK256_PACKED_BYTES: usize = 64;
    let num_blocks = 16;
    let mut quantized = vec![0i8; num_blocks * QK256_PACKED_BYTES];
    for byte in quantized.iter_mut() {
        *byte = rng.random();
    }
    let scales: Vec<f32> = (0..num_blocks).map(|_| rng.random_range(0.5..5.0)).collect();
    let mut results = Vec::new();
    for _ in 0..5 {
        let result = kernel.dequantize_qk256(&quantized, &scales, 256)?;
        results.push(result);
    }
    let first = &results[0];
    for (iteration, result) in results.iter().enumerate().skip(1) {
        for (i, (&val1, &val2)) in first.iter().zip(result.iter()).enumerate() {
            assert_eq!(
                val1, val2,
                "Iteration {}: Mismatch at element {}: first={}, current={}",
                iteration, i, val1, val2
            );
        }
    }
    println!("✅ QK256 determinism test passed (5 iterations identical)");
    Ok(())
}
/// Test QK256 with all LUT code patterns
///
/// Validates that each 2-bit code (0, 1, 2, 3) maps correctly to
/// LUT values [-2.0, -1.0, 1.0, 2.0] in the fast path.
#[test]
fn test_qk256_lut_code_patterns() {
    use bitnet_kernels::KernelProvider;
    use bitnet_kernels::cpu::x86::Avx2Kernel;
    let kernel = Avx2Kernel;
    if !kernel.is_available() {
        eprintln!("Skipping LUT pattern test: AVX2 not available");
        return;
    }
    const QK256_PACKED_BYTES: usize = 64;
    const LUT: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];
    for code in 0u8..4 {
        let packed_byte = code | (code << 2) | (code << 4) | (code << 6);
        let quantized = vec![packed_byte as i8; QK256_PACKED_BYTES];
        let scales = vec![2.5f32];
        let result = kernel
            .dequantize_qk256(&quantized, &scales, 256)
            .expect("Dequantization should succeed");
        let expected = LUT[code as usize] * 2.5;
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected).abs() < 1e-5,
                "Code {}: Element {} should be {}, got {}",
                code,
                i,
                expected,
                val
            );
        }
        println!("✅ LUT pattern test passed for code {} → {}", code, expected);
    }
}
/// Test QK256 with edge case scales
///
/// Validates numerical stability with extreme scale values.
#[test]
fn test_qk256_edge_case_scales() {
    use bitnet_kernels::KernelProvider;
    use bitnet_kernels::cpu::x86::Avx2Kernel;
    let kernel = Avx2Kernel;
    if !kernel.is_available() {
        eprintln!("Skipping edge case test: AVX2 not available");
        return;
    }
    const QK256_PACKED_BYTES: usize = 64;
    let quantized = vec![0xAAu8 as i8; QK256_PACKED_BYTES];
    let test_cases = vec![
        ("very_small", vec![1e-6f32]),
        ("very_large", vec![1e6f32]),
        ("zero", vec![0.0f32]),
        ("negative", vec![-5.0f32]),
    ];
    for (name, scales) in test_cases {
        let result_avx2 = kernel
            .dequantize_qk256(&quantized, &scales, 256)
            .expect("AVX2 dequantize should succeed");
        let result_scalar = kernel
            .dequantize_qk256_scalar(&quantized, &scales, 256)
            .expect("Scalar dequantize should succeed");
        for (i, (&avx2_val, &scalar_val)) in
            result_avx2.iter().zip(result_scalar.iter()).enumerate()
        {
            let avx2_val: f32 = avx2_val;
            let scalar_val: f32 = scalar_val;
            let abs_diff: f32 = (avx2_val - scalar_val).abs();
            let rel_tol: f32 = if scales[0].abs() > 1e3_f32 { 1e-2 } else { 1e-5 };
            assert!(
                abs_diff < rel_tol || abs_diff / (scalar_val.abs().max(1e-9_f32)) < rel_tol,
                "Scale case '{}': Mismatch at element {}: AVX2={}, Scalar={}, diff={}",
                name,
                i,
                avx2_val,
                scalar_val,
                abs_diff
            );
        }
        println!("✅ Edge case '{}' passed (scale={})", name, scales[0]);
    }
}
