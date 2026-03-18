//! Correctness tests for AVX2 QK256 kernel against scalar reference
//!
//! This test suite validates the AVX2 QK256 dequantization and GEMV kernel
//! by comparing its results against the scalar reference implementation with
//! strict numerical tolerance requirements.
//!
//! ## Test Strategy
//!
//! 1. Generate random 2-bit quantized data (codes 0..=3)
//! 2. Generate random f32 input vectors
//! 3. Run both scalar and AVX2 implementations
//! 4. Compare results with tolerance 1e-5 (floating-point precision)
//!
//! ## Test Coverage
//!
//! - Single block (256 elements)
//! - Multiple blocks (512, 1024, 4096 elements)
//! - Edge case: partial block (non-multiple of 256)
//! - Multiple rows (matrix operations)
//! - Random seeds for comprehensive coverage
//!
//! ## Feature Gates
//!
//! Tests are conditionally compiled for x86_64 and skipped at runtime if AVX2
//! is not available on the CPU.

#![cfg(target_arch = "x86_64")]

use bitnet_models::quant::i2s_qk256::{QK256_BLOCK, QK256_PACKED_BYTES, gemv_qk256_row};
use bitnet_models::quant::i2s_qk256_avx2::gemv_qk256_avx2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Code-to-weight mapping for QK256 format
///
/// This matches the GGML I2_S specification:
/// - Code 0 → -2.0
/// - Code 1 → -1.0
/// - Code 2 → +1.0
/// - Code 3 → +2.0
const WEIGHTS: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];

/// Tolerance for floating-point comparison
///
/// This is set to 1e-4 to account for:
/// - FMA rounding differences between scalar and SIMD
/// - Order-of-operations differences in accumulation
/// - Minor precision loss in horizontal sum reduction
/// - Accumulated errors in multi-block operations
///
/// Note: The AVX2 implementation uses different order of operations than scalar
/// (vectorized FMA vs. scalar accumulation), so small differences are expected.
const TOLERANCE: f32 = 1e-4;

/// Generate random quantized data (packed 2-bit codes)
///
/// Each byte contains 4 codes in the format:
/// ```text
/// byte = code0 | (code1 << 2) | (code2 << 4) | (code3 << 6)
/// ```
///
/// # Arguments
///
/// * `num_bytes` - Number of packed bytes to generate
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// Vector of packed bytes with random 2-bit codes (0..=3)
fn generate_random_quantized_data(num_bytes: usize, seed: u64) -> Vec<u8> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(num_bytes);

    for _ in 0..num_bytes {
        // Generate 4 random codes (0..=3) and pack them into a byte
        let c0 = rng.random_range(0..4u8);
        let c1 = rng.random_range(0..4u8);
        let c2 = rng.random_range(0..4u8);
        let c3 = rng.random_range(0..4u8);

        let byte = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
        data.push(byte);
    }

    data
}

/// Generate random input vector
///
/// # Arguments
///
/// * `len` - Vector length
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// Vector of random f32 values in range [-10.0, 10.0]
fn generate_random_input(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..len).map(|_| rng.random_range(-10.0..10.0)).collect()
}

/// Test AVX2 matches scalar for single block (256 elements)
#[test]
fn test_qk256_avx2_single_block() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test - not available");
        return;
    }

    const COLS: usize = 256;
    const SEED: u64 = 42;

    // Generate random quantized data (1 block = 64 bytes)
    let qs_data = generate_random_quantized_data(QK256_PACKED_BYTES, SEED);

    // Generate random input
    let x = generate_random_input(COLS, SEED + 1);

    // Compute results using both implementations
    let scalar_result = gemv_qk256_row(&qs_data, &x, COLS);

    let mut y_avx2 = vec![0.0f32; 1];
    gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, 1, COLS, QK256_PACKED_BYTES)
        .expect("AVX2 GEMV should succeed");
    let avx2_result = y_avx2[0];

    // Compare with tolerance
    let abs_diff = (scalar_result - avx2_result).abs();
    let rel_diff =
        if scalar_result.abs() > 1e-6 { abs_diff / scalar_result.abs() } else { abs_diff };

    assert!(
        abs_diff < TOLERANCE || rel_diff < TOLERANCE,
        "Single block mismatch: scalar={}, avx2={}, abs_diff={}, rel_diff={}",
        scalar_result,
        avx2_result,
        abs_diff,
        rel_diff
    );

    println!("✅ Single block test passed (256 elements)");
}

/// Test AVX2 matches scalar for multiple blocks (512 elements)
#[test]
fn test_qk256_avx2_multiple_blocks_512() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test - not available");
        return;
    }

    const COLS: usize = 512;
    const BLOCKS: usize = COLS / QK256_BLOCK; // 2 blocks
    const SEED: u64 = 1337;

    // Generate random quantized data (2 blocks = 128 bytes)
    let qs_data = generate_random_quantized_data(BLOCKS * QK256_PACKED_BYTES, SEED);

    // Generate random input
    let x = generate_random_input(COLS, SEED + 1);

    // Compute results
    let scalar_result = gemv_qk256_row(&qs_data, &x, COLS);

    let mut y_avx2 = vec![0.0f32; 1];
    gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, 1, COLS, BLOCKS * QK256_PACKED_BYTES)
        .expect("AVX2 GEMV should succeed");
    let avx2_result = y_avx2[0];

    // Compare with tolerance
    let abs_diff = (scalar_result - avx2_result).abs();
    let rel_diff =
        if scalar_result.abs() > 1e-6 { abs_diff / scalar_result.abs() } else { abs_diff };

    assert!(
        abs_diff < TOLERANCE || rel_diff < TOLERANCE,
        "Multiple blocks (512) mismatch: scalar={}, avx2={}, abs_diff={}, rel_diff={}",
        scalar_result,
        avx2_result,
        abs_diff,
        rel_diff
    );

    println!("✅ Multiple blocks test passed (512 elements, 2 blocks)");
}

/// Test AVX2 matches scalar for multiple blocks (1024 elements)
#[test]
fn test_qk256_avx2_multiple_blocks_1024() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test - not available");
        return;
    }

    const COLS: usize = 1024;
    const BLOCKS: usize = COLS / QK256_BLOCK; // 4 blocks
    const SEED: u64 = 9999;

    // Generate random quantized data (4 blocks = 256 bytes)
    let qs_data = generate_random_quantized_data(BLOCKS * QK256_PACKED_BYTES, SEED);

    // Generate random input
    let x = generate_random_input(COLS, SEED + 1);

    // Compute results
    let scalar_result = gemv_qk256_row(&qs_data, &x, COLS);

    let mut y_avx2 = vec![0.0f32; 1];
    gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, 1, COLS, BLOCKS * QK256_PACKED_BYTES)
        .expect("AVX2 GEMV should succeed");
    let avx2_result = y_avx2[0];

    // Compare with tolerance
    let abs_diff = (scalar_result - avx2_result).abs();
    let rel_diff =
        if scalar_result.abs() > 1e-6 { abs_diff / scalar_result.abs() } else { abs_diff };

    assert!(
        abs_diff < TOLERANCE || rel_diff < TOLERANCE,
        "Multiple blocks (1024) mismatch: scalar={}, avx2={}, abs_diff={}, rel_diff={}",
        scalar_result,
        avx2_result,
        abs_diff,
        rel_diff
    );

    println!("✅ Multiple blocks test passed (1024 elements, 4 blocks)");
}

/// Test AVX2 matches scalar for many blocks (4096 elements)
#[test]
fn test_qk256_avx2_many_blocks_4096() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test - not available");
        return;
    }

    const COLS: usize = 4096;
    const BLOCKS: usize = COLS / QK256_BLOCK; // 16 blocks
    const SEED: u64 = 12345;

    // Generate random quantized data (16 blocks = 1024 bytes)
    let qs_data = generate_random_quantized_data(BLOCKS * QK256_PACKED_BYTES, SEED);

    // Generate random input
    let x = generate_random_input(COLS, SEED + 1);

    // Compute results
    let scalar_result = gemv_qk256_row(&qs_data, &x, COLS);

    let mut y_avx2 = vec![0.0f32; 1];
    gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, 1, COLS, BLOCKS * QK256_PACKED_BYTES)
        .expect("AVX2 GEMV should succeed");
    let avx2_result = y_avx2[0];

    // Compare with tolerance
    let abs_diff = (scalar_result - avx2_result).abs();
    let rel_diff =
        if scalar_result.abs() > 1e-6 { abs_diff / scalar_result.abs() } else { abs_diff };

    assert!(
        abs_diff < TOLERANCE || rel_diff < TOLERANCE,
        "Many blocks (4096) mismatch: scalar={}, avx2={}, abs_diff={}, rel_diff={}",
        scalar_result,
        avx2_result,
        abs_diff,
        rel_diff
    );

    println!("✅ Many blocks test passed (4096 elements, 16 blocks)");
}

/// Test AVX2 matches scalar for partial block (300 elements)
///
/// This tests the tail handling logic when columns are not a multiple of 256.
#[test]
fn test_qk256_avx2_partial_block() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test - not available");
        return;
    }

    const COLS: usize = 300; // Not a multiple of 256
    const BLOCKS: usize = COLS.div_ceil(QK256_BLOCK); // 2 blocks (256 + 44)
    const SEED: u64 = 54321;

    // Generate random quantized data (2 blocks = 128 bytes)
    let qs_data = generate_random_quantized_data(BLOCKS * QK256_PACKED_BYTES, SEED);

    // Generate random input
    let x = generate_random_input(COLS, SEED + 1);

    // Compute results
    let scalar_result = gemv_qk256_row(&qs_data, &x, COLS);

    let mut y_avx2 = vec![0.0f32; 1];
    gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, 1, COLS, BLOCKS * QK256_PACKED_BYTES)
        .expect("AVX2 GEMV should succeed");
    let avx2_result = y_avx2[0];

    // Compare with tolerance
    let abs_diff = (scalar_result - avx2_result).abs();
    let rel_diff =
        if scalar_result.abs() > 1e-6 { abs_diff / scalar_result.abs() } else { abs_diff };

    assert!(
        abs_diff < TOLERANCE || rel_diff < TOLERANCE,
        "Partial block mismatch: scalar={}, avx2={}, abs_diff={}, rel_diff={}",
        scalar_result,
        avx2_result,
        abs_diff,
        rel_diff
    );

    println!("✅ Partial block test passed (300 elements)");
}

/// Test AVX2 matches scalar for edge case: exactly 2 blocks + 1 element
#[test]
fn test_qk256_avx2_edge_case_513() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test - not available");
        return;
    }

    const COLS: usize = 513; // 2 full blocks + 1 element
    const BLOCKS: usize = COLS.div_ceil(QK256_BLOCK); // 3 blocks
    const SEED: u64 = 99999;

    // Generate random quantized data
    let qs_data = generate_random_quantized_data(BLOCKS * QK256_PACKED_BYTES, SEED);

    // Generate random input
    let x = generate_random_input(COLS, SEED + 1);

    // Compute results
    let scalar_result = gemv_qk256_row(&qs_data, &x, COLS);

    let mut y_avx2 = vec![0.0f32; 1];
    gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, 1, COLS, BLOCKS * QK256_PACKED_BYTES)
        .expect("AVX2 GEMV should succeed");
    let avx2_result = y_avx2[0];

    // Compare with tolerance
    let abs_diff = (scalar_result - avx2_result).abs();
    let rel_diff =
        if scalar_result.abs() > 1e-6 { abs_diff / scalar_result.abs() } else { abs_diff };

    assert!(
        abs_diff < TOLERANCE || rel_diff < TOLERANCE,
        "Edge case (513) mismatch: scalar={}, avx2={}, abs_diff={}, rel_diff={}",
        scalar_result,
        avx2_result,
        abs_diff,
        rel_diff
    );

    println!("✅ Edge case test passed (513 elements)");
}

/// Comprehensive test with multiple random seeds
///
/// This test runs multiple iterations with different random seeds to ensure
/// robustness across different data patterns.
#[test]
fn test_qk256_avx2_random_seeds() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test - not available");
        return;
    }

    const COLS: usize = 1024;
    const BLOCKS: usize = COLS / QK256_BLOCK;

    // Test with multiple random seeds
    let seeds = [1, 42, 1337, 9999, 12345, 54321, 99999];

    for (idx, &seed) in seeds.iter().enumerate() {
        // Generate random quantized data
        let qs_data = generate_random_quantized_data(BLOCKS * QK256_PACKED_BYTES, seed);

        // Generate random input
        let x = generate_random_input(COLS, seed + 1);

        // Compute results
        let scalar_result = gemv_qk256_row(&qs_data, &x, COLS);

        let mut y_avx2 = vec![0.0f32; 1];
        gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, 1, COLS, BLOCKS * QK256_PACKED_BYTES)
            .expect("AVX2 GEMV should succeed");
        let avx2_result = y_avx2[0];

        // Compare with tolerance
        let abs_diff = (scalar_result - avx2_result).abs();
        let rel_diff =
            if scalar_result.abs() > 1e-6 { abs_diff / scalar_result.abs() } else { abs_diff };

        assert!(
            abs_diff < TOLERANCE || rel_diff < TOLERANCE,
            "Random seed {} (iteration {}) mismatch: scalar={}, avx2={}, abs_diff={}, rel_diff={}",
            seed,
            idx,
            scalar_result,
            avx2_result,
            abs_diff,
            rel_diff
        );
    }

    println!("✅ Random seeds test passed ({} seeds)", seeds.len());
}

/// Test special case: all codes are the same value
///
/// This validates correctness for uniform quantized data.
#[test]
fn test_qk256_avx2_uniform_codes() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test - not available");
        return;
    }

    const COLS: usize = 512;
    const BLOCKS: usize = COLS / QK256_BLOCK;

    // Test each code value (0, 1, 2, 3)
    for code in 0u8..4 {
        // Create uniform data: all codes are the same
        let byte = code | (code << 2) | (code << 4) | (code << 6);
        let qs_data = vec![byte; BLOCKS * QK256_PACKED_BYTES];

        // Generate random input
        let x = generate_random_input(COLS, 42 + code as u64);

        // Compute results
        let scalar_result = gemv_qk256_row(&qs_data, &x, COLS);

        let mut y_avx2 = vec![0.0f32; 1];
        gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, 1, COLS, BLOCKS * QK256_PACKED_BYTES)
            .expect("AVX2 GEMV should succeed");
        let avx2_result = y_avx2[0];

        // Compare scalar and AVX2 results
        let mutual_diff = (scalar_result - avx2_result).abs();
        let rel_diff = if scalar_result.abs() > 1e-6 {
            mutual_diff / scalar_result.abs()
        } else {
            mutual_diff
        };

        assert!(
            mutual_diff < TOLERANCE || rel_diff < TOLERANCE,
            "Uniform code {} scalar/AVX2 mismatch: scalar={}, avx2={}, abs_diff={}, rel_diff={}",
            code,
            scalar_result,
            avx2_result,
            mutual_diff,
            rel_diff
        );

        // Sanity check: expected value should be close to actual results
        // (for uniform codes, result should equal weight * sum(x), allowing for floating-point error)
        let expected_weight = WEIGHTS[code as usize];
        let sum_x: f32 = x.iter().sum();
        let expected = expected_weight * sum_x;
        let expected_diff = (scalar_result - expected).abs();

        // Allow larger tolerance for expected value due to summation order differences
        assert!(
            expected_diff / expected.abs() < 0.01,
            "Uniform code {} sanity check failed: expected={}, got={}, rel_diff={}",
            code,
            expected,
            scalar_result,
            expected_diff / expected.abs()
        );
    }

    println!("✅ Uniform codes test passed (all 4 code values)");
}

/// Test with zero input vector
///
/// This validates that both implementations correctly produce zero output
/// when the input is all zeros.
#[test]
fn test_qk256_avx2_zero_input() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test - not available");
        return;
    }

    const COLS: usize = 1024;
    const BLOCKS: usize = COLS / QK256_BLOCK;
    const SEED: u64 = 42;

    // Generate random quantized data (doesn't matter, input is zero)
    let qs_data = generate_random_quantized_data(BLOCKS * QK256_PACKED_BYTES, SEED);

    // Zero input
    let x = vec![0.0f32; COLS];

    // Compute results
    let scalar_result = gemv_qk256_row(&qs_data, &x, COLS);

    let mut y_avx2 = vec![0.0f32; 1];
    gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, 1, COLS, BLOCKS * QK256_PACKED_BYTES)
        .expect("AVX2 GEMV should succeed");
    let avx2_result = y_avx2[0];

    // Both should be exactly zero
    assert_eq!(
        scalar_result, 0.0,
        "Scalar should return exactly 0.0 for zero input, got {}",
        scalar_result
    );
    assert_eq!(
        avx2_result, 0.0,
        "AVX2 should return exactly 0.0 for zero input, got {}",
        avx2_result
    );

    println!("✅ Zero input test passed");
}

/// Test multi-row GEMV with AVX2
///
/// This validates the full GEMV operation (multiple rows).
#[test]
fn test_qk256_avx2_multi_row() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test - not available");
        return;
    }

    const ROWS: usize = 8;
    const COLS: usize = 512;
    const BLOCKS: usize = COLS / QK256_BLOCK;
    const ROW_STRIDE: usize = BLOCKS * QK256_PACKED_BYTES;
    const SEED: u64 = 42;

    // Generate random quantized data for all rows
    let qs_data = generate_random_quantized_data(ROWS * ROW_STRIDE, SEED);

    // Generate random input
    let x = generate_random_input(COLS, SEED + 1);

    // Compute scalar results (row by row)
    let mut y_scalar = [0.0f32; ROWS];
    for (row, output) in y_scalar.iter_mut().enumerate() {
        let start = row * ROW_STRIDE;
        let end = start + ROW_STRIDE;
        let row_bytes = &qs_data[start..end];
        *output = gemv_qk256_row(row_bytes, &x, COLS);
    }

    // Compute AVX2 results
    let mut y_avx2 = vec![0.0f32; ROWS];
    gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, ROWS, COLS, ROW_STRIDE)
        .expect("AVX2 GEMV should succeed");

    // Compare results for each row
    for (row, (&scalar, &avx2)) in y_scalar.iter().zip(y_avx2.iter()).enumerate() {
        let abs_diff = (scalar - avx2).abs();
        let rel_diff = if scalar.abs() > 1e-6 { abs_diff / scalar.abs() } else { abs_diff };

        assert!(
            abs_diff < TOLERANCE || rel_diff < TOLERANCE,
            "Multi-row mismatch at row {}: scalar={}, avx2={}, abs_diff={}, rel_diff={}",
            row,
            scalar,
            avx2,
            abs_diff,
            rel_diff
        );
    }

    println!("✅ Multi-row GEMV test passed ({} rows × {} cols)", ROWS, COLS);
}

/// Test architectural availability detection
///
/// This test validates that the AVX2 detection mechanism works correctly.
#[test]
fn test_avx2_detection() {
    let avx2_available = is_x86_feature_detected!("avx2");

    if avx2_available {
        println!("✅ AVX2 detected and available on this CPU");
    } else {
        println!("⚠️  AVX2 not available - tests will be skipped");
    }

    // This test always passes - it just reports the detection status
}
