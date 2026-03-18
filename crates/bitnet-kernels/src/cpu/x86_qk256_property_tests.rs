//! Property-based tests for QK256 AVX2 dequantization
//!
//! This module contains comprehensive property-based tests for the AVX2
//! QK256 fast path, validating correctness across a wide range of inputs.
//!
//! ## Test Coverage
//!
//! 1. **Block count variation**: 1, 2, 4, 8, 16, 32 blocks
//! 2. **Scale ranges**: Very small (1e-6), very large (1e6), zero, negative, mixed
//! 3. **Code mapping**: All 4 LUT codes (0, 1, 2, 3) → [-2.0, -1.0, 1.0, 2.0]
//! 4. **Alignment handling**: Unaligned memory access (offsets 0, 1, 3, 7 bytes)
//!
//! ## Usage
//!
//! ```bash
//! # Run all property tests
//! cargo test --lib --no-default-features --features cpu,avx2 -- x86_qk256_property
//!
//! # Run specific property test
//! cargo test --lib --no-default-features --features cpu,avx2 -- property_block_counts
//! ```

#![cfg(all(target_arch = "x86_64", test))]

use super::Avx2Kernel;
use crate::KernelProvider;

/// Property-based test: QK256 dequantization with various block counts
///
/// Tests the AVX2 QK256 dequantization across multiple block sizes
/// to ensure correctness for different tensor shapes.
#[test]
fn test_avx2_dequantize_qk256_property_block_counts() {
    let kernel = Avx2Kernel;

    if !kernel.is_available() {
        eprintln!("Skipping QK256 property test: AVX2 not available");
        return;
    }

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(12345);

    // Test with various block counts: 1, 2, 4, 8, 16, 32
    let block_counts = [1, 2, 4, 8, 16, 32];

    for &num_blocks in &block_counts {
        const QK256_PACKED_BYTES: usize = 64;

        // Generate random quantized data
        let mut quantized = vec![0i8; num_blocks * QK256_PACKED_BYTES];
        for byte in quantized.iter_mut() {
            *byte = rng.random();
        }

        // Generate random scales
        let scales: Vec<f32> = (0..num_blocks).map(|_| rng.random_range(0.1..10.0)).collect();

        // Compute AVX2 result
        let result_avx2 = kernel
            .dequantize_qk256(&quantized, &scales, 256)
            .expect("AVX2 dequantize should succeed");

        // Compute scalar reference
        let result_scalar = kernel
            .dequantize_qk256_scalar(&quantized, &scales, 256)
            .expect("Scalar dequantize should succeed");

        // Verify lengths
        assert_eq!(
            result_avx2.len(),
            num_blocks * 256,
            "AVX2 output length mismatch for {} blocks",
            num_blocks
        );
        assert_eq!(
            result_scalar.len(),
            num_blocks * 256,
            "Scalar output length mismatch for {} blocks",
            num_blocks
        );

        // Verify numerical equivalence
        for (i, (&avx2_val, &scalar_val)) in
            result_avx2.iter().zip(result_scalar.iter()).enumerate()
        {
            let abs_diff = (avx2_val - scalar_val).abs();
            assert!(
                abs_diff < 1e-5,
                "Block count {}: Mismatch at element {}: AVX2={}, Scalar={}, diff={}",
                num_blocks,
                i,
                avx2_val,
                scalar_val,
                abs_diff
            );
        }
    }

    println!("✅ AVX2 QK256 property test passed for {} block counts", block_counts.len());
}

/// Property-based test: QK256 dequantization with edge case scales
///
/// Tests AVX2 dequantization with extreme scale values to ensure
/// numerical stability (very small, very large, zero scales).
#[test]
fn test_avx2_dequantize_qk256_property_scale_ranges() {
    let kernel = Avx2Kernel;

    if !kernel.is_available() {
        eprintln!("Skipping QK256 scale range test: AVX2 not available");
        return;
    }

    const QK256_PACKED_BYTES: usize = 64;
    const NUM_BLOCKS: usize = 4;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(54321);

    // Generate random quantized data (reused for all scale tests)
    let mut quantized = vec![0i8; NUM_BLOCKS * QK256_PACKED_BYTES];
    for byte in quantized.iter_mut() {
        *byte = rng.random();
    }

    // Test cases: different scale ranges
    let scale_test_cases = vec![
        ("very_small", vec![1e-6f32; NUM_BLOCKS]),
        ("very_large", vec![1e6f32; NUM_BLOCKS]),
        ("zero", vec![0.0f32; NUM_BLOCKS]),
        ("negative", vec![-1.5f32; NUM_BLOCKS]),
        ("mixed", vec![1e-5, 1.0, 1e5, -0.5]),
    ];

    for (name, scales) in scale_test_cases {
        // Compute AVX2 result
        let result_avx2 = kernel
            .dequantize_qk256(&quantized, &scales, 256)
            .expect("AVX2 dequantize should succeed");

        // Compute scalar reference
        let result_scalar = kernel
            .dequantize_qk256_scalar(&quantized, &scales, 256)
            .expect("Scalar dequantize should succeed");

        // Verify numerical equivalence (with appropriate tolerance for extreme values)
        let max_scale = scales.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let tolerance = if max_scale > 1e3 {
            1e-2 // Relaxed tolerance for very large scales
        } else if max_scale < 1e-3 {
            1e-8 // Tighter tolerance for very small scales
        } else {
            1e-5 // Standard tolerance
        };

        for (i, (&avx2_val, &scalar_val)) in
            result_avx2.iter().zip(result_scalar.iter()).enumerate()
        {
            let abs_diff = (avx2_val - scalar_val).abs();
            assert!(
                abs_diff < tolerance,
                "Scale case '{}': Mismatch at element {}: AVX2={}, Scalar={}, diff={} (tolerance={})",
                name,
                i,
                avx2_val,
                scalar_val,
                abs_diff,
                tolerance
            );
        }

        println!("✅ Scale range test '{}' passed (tolerance={})", name, tolerance);
    }
}

/// Property-based test: QK256 dequantization preserves code patterns
///
/// Verifies that each 2-bit code (0, 1, 2, 3) maps correctly to
/// LUT values [-2.0, -1.0, 1.0, 2.0] when multiplied by scale.
#[test]
fn test_avx2_dequantize_qk256_property_code_mapping() {
    let kernel = Avx2Kernel;

    if !kernel.is_available() {
        eprintln!("Skipping QK256 code mapping test: AVX2 not available");
        return;
    }

    const QK256_PACKED_BYTES: usize = 64;
    const LUT: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];

    // Test each code (0, 1, 2, 3) in isolation
    for code in 0u8..4 {
        // Create packed data with all bytes set to this code
        let packed_byte = code | (code << 2) | (code << 4) | (code << 6);
        let quantized = vec![packed_byte as i8; QK256_PACKED_BYTES];

        // Use scale = 3.5 to test LUT scaling
        let scales = vec![3.5f32];

        // Compute AVX2 result
        let result = kernel
            .dequantize_qk256(&quantized, &scales, 256)
            .expect("AVX2 dequantize should succeed");

        // Verify all elements match expected LUT value * scale
        let expected = LUT[code as usize] * 3.5;
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected).abs() < 1e-5,
                "Code {}: Element {} should be {} (LUT[{}] * 3.5), got {}",
                code,
                i,
                expected,
                code,
                val
            );
        }

        println!("✅ Code {} mapping verified: {} * 3.5 = {}", code, LUT[code as usize], expected);
    }
}

/// Property-based test: QK256 dequantization alignment handling
///
/// Tests that AVX2 implementation correctly handles unaligned memory
/// accesses (scales and quantized data at odd addresses).
#[test]
fn test_avx2_dequantize_qk256_property_alignment() {
    let kernel = Avx2Kernel;

    if !kernel.is_available() {
        eprintln!("Skipping QK256 alignment test: AVX2 not available");
        return;
    }

    const QK256_PACKED_BYTES: usize = 64;
    const NUM_BLOCKS: usize = 3;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(99999);

    // Generate random data with extra padding to test unaligned access
    let mut padded_quantized = vec![0i8; NUM_BLOCKS * QK256_PACKED_BYTES + 7];
    for byte in padded_quantized.iter_mut() {
        *byte = rng.random();
    }

    // Test with different offset alignments (0, 1, 3, 7 bytes)
    let offsets = [0, 1, 3, 7];

    for offset in offsets {
        let quantized = &padded_quantized[offset..offset + NUM_BLOCKS * QK256_PACKED_BYTES];
        let scales: Vec<f32> = (0..NUM_BLOCKS).map(|_| rng.random_range(0.5..5.0)).collect();

        // Compute AVX2 result (should handle unaligned access)
        let result_avx2 = kernel
            .dequantize_qk256(quantized, &scales, 256)
            .expect("AVX2 dequantize should succeed");

        // Compute scalar reference
        let result_scalar = kernel
            .dequantize_qk256_scalar(quantized, &scales, 256)
            .expect("Scalar dequantize should succeed");

        // Verify numerical equivalence
        for (i, (&avx2_val, &scalar_val)) in
            result_avx2.iter().zip(result_scalar.iter()).enumerate()
        {
            let abs_diff = (avx2_val - scalar_val).abs();
            assert!(
                abs_diff < 1e-5,
                "Alignment offset {}: Mismatch at element {}: AVX2={}, Scalar={}, diff={}",
                offset,
                i,
                avx2_val,
                scalar_val,
                abs_diff
            );
        }

        println!("✅ Alignment test passed for offset {} bytes", offset);
    }
}
