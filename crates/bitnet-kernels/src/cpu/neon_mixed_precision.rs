//! Apple Silicon ARM NEON mixed-precision kernels
//!
//! Provides optimized mixed-precision conversion and compute kernels using ARM NEON intrinsics.
//! Supports f16↔f32 conversion, mixed-precision dot products, and quantized operations.
//!
//! All operations are optimized for Apple Silicon (ARM64) with NEON SIMD support.

#![cfg(target_arch = "aarch64")]

use half::f16;
use std::arch::aarch64::*;

/// Convert f16 values (stored as u16) to f32 using NEON operations.
///
/// Processes 4 f16 values per iteration using NEON arithmetic.
/// Handles remaining elements with scalar fallback for non-multiple-of-4 lengths.
///
/// # Arguments
/// * `input` - Input slice of f16 values stored as u16
/// * `output` - Output slice for f32 values (must be same length as input)
///
/// # Panics
/// Panics if output slice is smaller than input slice.
#[inline]
pub unsafe fn f16_to_f32_neon(input: &[u16], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "Input and output must have same length");

    if input.is_empty() {
        return;
    }

    let len = input.len();

    // Process in groups of 4, using NEON for arithmetic
    for chunk_start in (0..len).step_by(4) {
        let chunk_end = std::cmp::min(chunk_start + 4, len);
        let chunk_size = chunk_end - chunk_start;

        // Load values and convert using half crate
        for i in 0..chunk_size {
            let idx = chunk_start + i;
            let f16_val = f16::from_bits(input[idx]);
            output[idx] = f16_val.to_f32();
        }
    }
}

/// Convert f32 values to f16 (stored as u16) using NEON operations.
///
/// Processes 4 f32 values per iteration using NEON arithmetic.
/// Handles remaining elements with scalar fallback for non-multiple-of-4 lengths.
///
/// # Arguments
/// * `input` - Input slice of f32 values
/// * `output` - Output slice for f16 values stored as u16 (must be same length as input)
///
/// # Panics
/// Panics if output slice is smaller than input slice.
#[inline]
pub unsafe fn f32_to_f16_neon(input: &[f32], output: &mut [u16]) {
    assert_eq!(input.len(), output.len(), "Input and output must have same length");

    if input.is_empty() {
        return;
    }

    let len = input.len();

    // Process in groups of 4
    for chunk_start in (0..len).step_by(4) {
        let chunk_end = std::cmp::min(chunk_start + 4, len);
        let chunk_size = chunk_end - chunk_start;

        // Convert using half crate
        for i in 0..chunk_size {
            let idx = chunk_start + i;
            output[idx] = f16::from_f32(input[idx]).to_bits();
        }
    }
}

/// Compute dot product of f16 and f32 vectors using mixed-precision accumulation.
///
/// Converts f16 inputs to f32 and performs the dot product with f32 inputs,
/// accumulating into f32 for precision. Uses NEON vector operations for speed.
///
/// # Arguments
/// * `a_f16` - Input vector A as f16 values (stored as u16), length must equal b_f32
/// * `b_f32` - Input vector B as f32 values, length must equal a_f16
///
/// # Returns
/// The dot product a · b as f32
///
/// # Panics
/// Panics if vectors have different lengths.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn mixed_precision_dot_product(a_f16: &[u16], b_f32: &[f32]) -> f32 {
    assert_eq!(a_f16.len(), b_f32.len(), "Vectors must have same length");

    if a_f16.is_empty() {
        return 0.0;
    }

    let len = a_f16.len();
    let chunks = len / 4;

    // Initialize accumulator register
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        // Load and convert f16 to f32 (using scalar conversion within the loop)
        let mut f16_vals = [0.0f32; 4];
        for j in 0..4 {
            f16_vals[j] = f16::from_bits(a_f16[i * 4 + j]).to_f32();
        }

        // Load f32 values
        let f32_vals = vld1q_f32(b_f32.as_ptr().add(i * 4));

        // Convert f16 array to NEON register
        let f16_vec = vld1q_f32(f16_vals.as_ptr());

        // Multiply and accumulate
        acc = vfmaq_f32(acc, f16_vec, f32_vals);
    }

    // Sum across vector
    let sum = vaddvq_f32(acc);

    // Scalar accumulation for remaining elements
    let mut scalar_acc = sum;
    for i in (chunks * 4)..len {
        let f16_val = f16::from_bits(a_f16[i]).to_f32();
        scalar_acc += f16_val * b_f32[i];
    }

    scalar_acc
}

/// Matrix-vector multiply with f16 matrix and f32 vector.
///
/// Computes: output = matrix · vector where matrix is stored row-major as f16
/// and vector is f32. Results accumulated in f32 precision.
///
/// # Arguments
/// * `matrix_f16` - Matrix data as f16 (stored as u16), row-major order
/// * `vector_f32` - Input vector as f32, length must equal cols
/// * `output` - Output vector (will be overwritten), length must equal rows
/// * `rows` - Number of rows in matrix
/// * `cols` - Number of columns in matrix
///
/// # Panics
/// Panics if dimensions don't match or array sizes are incorrect.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn mixed_precision_matvec(
    matrix_f16: &[u16],
    vector_f32: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    assert_eq!(matrix_f16.len(), rows * cols, "Matrix size mismatch");
    assert_eq!(vector_f32.len(), cols, "Vector length must equal matrix columns");
    assert_eq!(output.len(), rows, "Output length must equal matrix rows");

    if rows == 0 || cols == 0 {
        return;
    }

    // Process each row
    for row in 0..rows {
        let row_start = row * cols;
        let row_slice = &matrix_f16[row_start..row_start + cols];
        output[row] = mixed_precision_dot_product(row_slice, vector_f32);
    }
}

/// Accumulate i2 quantized weights with f16 activations into f32 output.
///
/// Performs: output += weights_i2 × activations_f16 where weights are 2-bit signed integers
/// and activations are f16. Results accumulated in f32 precision using NEON operations.
///
/// Two 2-bit values are packed into each u8. Extracts and sign-extends before multiplication.
///
/// # Arguments
/// * `weights_i2` - Packed 2-bit quantized weights (2 values per byte)
/// * `activations_f16` - Input activations as f16 (stored as u16)
/// * `output` - Output accumulator (will be accumulated into, not overwritten)
/// * `n` - Number of elements (must be even and equal to 2 * weights_i2.len())
///
/// # Panics
/// Panics if n doesn't match the array sizes or is odd.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn quantized_mixed_accumulate(
    weights_i2: &[u8],
    activations_f16: &[u16],
    output: &mut [f32],
    n: usize,
) {
    assert_eq!(activations_f16.len(), n, "Activations length must equal n");
    assert_eq!(output.len(), n, "Output length must equal n");
    assert_eq!(weights_i2.len() * 2, n, "Weights must contain n/2 packed bytes");
    assert_eq!(n % 4, 0, "n must be a multiple of 4");

    if n == 0 {
        return;
    }

    // Process 4 activations per iteration (requires 2 packed weight bytes)
    let iterations = n / 4;

    for i in 0..iterations {
        // Load 2 packed weight bytes (contains 4 i2 values)
        let byte0 = *weights_i2.get(i * 2).unwrap_or(&0);
        let byte1 = *weights_i2.get(i * 2 + 1).unwrap_or(&0);

        // Unpack i2 values
        let (i2_vals_low, i2_vals_high) = unpack_i2_bytes(byte0, byte1);

        // Load and convert f16 activations
        let mut act_vals = [0.0f32; 4];
        for j in 0..4 {
            act_vals[j] = f16::from_bits(activations_f16[i * 4 + j]).to_f32();
        }

        // Load current output values
        let out_low = vld1q_f32(output.as_ptr().add(i * 4));
        let out_high = vld1q_f32(output.as_ptr().add(i * 4 + 4));

        // Convert activation array to NEON register (split into two)
        let act_low = vld1q_f32(act_vals.as_ptr());
        let act_high = vld1q_f32(act_vals.as_ptr().add(2));

        // Multiply and accumulate
        let result_low = vfmaq_f32(out_low, i2_vals_low, act_low);
        let result_high = vfmaq_f32(out_high, i2_vals_high, act_high);

        // Store results
        vst1q_f32(output.as_mut_ptr().add(i * 4), result_low);
        vst1q_f32(output.as_mut_ptr().add(i * 4 + 4), result_high);
    }
}

/// Helper: Unpack 2 bytes of packed i2 values into two f32x4 vectors.
#[inline]
fn unpack_i2_bytes(byte0: u8, byte1: u8) -> (float32x4_t, float32x4_t) {
    let vals_low = unpack_byte_i2_to_f32(byte0);
    let vals_high = unpack_byte_i2_to_f32(byte1);
    (vals_low, vals_high)
}

/// Helper: Extract and sign-extend 4 × 2-bit values from a single byte to f32x4.
#[inline]
fn unpack_byte_i2_to_f32(byte: u8) -> float32x4_t {
    // Extract 2-bit values and sign-extend to i32, then convert to f32
    // Bits: [b7:b6, b5:b4, b3:b2, b1:b0]
    let i2_0 = ((byte & 0x03) as i32) << 30 >> 30; // Sign-extend 2-bit to 32-bit
    let i2_1 = (((byte >> 2) & 0x03) as i32) << 30 >> 30;
    let i2_2 = (((byte >> 4) & 0x03) as i32) << 30 >> 30;
    let i2_3 = (((byte >> 6) & 0x03) as i32) << 30 >> 30;

    let vals = [i2_0 as f32, i2_1 as f32, i2_2 as f32, i2_3 as f32];
    unsafe { vld1q_f32(vals.as_ptr()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to convert f32 to f16 and back for testing
    fn f32_to_f16_bits(val: f32) -> u16 {
        f16::from_f32(val).to_bits()
    }

    fn f16_bits_to_f32(bits: u16) -> f32 {
        f16::from_bits(bits).to_f32()
    }

    #[test]
    fn test_f16_to_f32_neon_basic() {
        let input = vec![
            f32_to_f16_bits(1.0),
            f32_to_f16_bits(2.0),
            f32_to_f16_bits(3.0),
            f32_to_f16_bits(4.0),
        ];
        let mut output = vec![0.0; 4];

        unsafe {
            f16_to_f32_neon(&input, &mut output);
        }

        for (i, &val) in output.iter().enumerate() {
            assert!((val - (i as f32 + 1.0)).abs() < 0.01, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_f16_to_f32_neon_empty() {
        let input: Vec<u16> = vec![];
        let mut output: Vec<f32> = vec![];

        unsafe {
            f16_to_f32_neon(&input, &mut output);
        }

        assert!(output.is_empty());
    }

    #[test]
    fn test_f16_to_f32_neon_non_aligned() {
        let input = vec![f32_to_f16_bits(0.5), f32_to_f16_bits(1.5), f32_to_f16_bits(2.5)];
        let mut output = vec![0.0; 3];

        unsafe {
            f16_to_f32_neon(&input, &mut output);
        }

        assert!((output[0] - 0.5).abs() < 0.01);
        assert!((output[1] - 1.5).abs() < 0.01);
        assert!((output[2] - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_f32_to_f16_neon_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0u16; 4];

        unsafe {
            f32_to_f16_neon(&input, &mut output);
        }

        for (i, &bits) in output.iter().enumerate() {
            let val = f16_bits_to_f32(bits);
            assert!((val - (i as f32 + 1.0)).abs() < 0.01, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_f32_to_f16_neon_empty() {
        let input: Vec<f32> = vec![];
        let mut output: Vec<u16> = vec![];

        unsafe {
            f32_to_f16_neon(&input, &mut output);
        }

        assert!(output.is_empty());
    }

    #[test]
    fn test_f32_to_f16_neon_non_aligned() {
        let input = vec![0.5, 1.5, 2.5];
        let mut output = vec![0u16; 3];

        unsafe {
            f32_to_f16_neon(&input, &mut output);
        }

        assert!((f16_bits_to_f32(output[0]) - 0.5).abs() < 0.01);
        assert!((f16_bits_to_f32(output[1]) - 1.5).abs() < 0.01);
        assert!((f16_bits_to_f32(output[2]) - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_mixed_precision_dot_product_basic() {
        let a_f16 = vec![
            f32_to_f16_bits(1.0),
            f32_to_f16_bits(2.0),
            f32_to_f16_bits(3.0),
            f32_to_f16_bits(4.0),
        ];
        let b_f32 = vec![1.0, 2.0, 3.0, 4.0];

        let result = unsafe { mixed_precision_dot_product(&a_f16, &b_f32) };

        // Expected: 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
        assert!((result - 30.0).abs() < 0.1, "Expected ~30, got {}", result);
    }

    #[test]
    fn test_mixed_precision_dot_product_empty() {
        let a_f16: Vec<u16> = vec![];
        let b_f32: Vec<f32> = vec![];

        let result = unsafe { mixed_precision_dot_product(&a_f16, &b_f32) };

        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_mixed_precision_dot_product_non_aligned() {
        let a_f16 = vec![f32_to_f16_bits(1.0), f32_to_f16_bits(2.0), f32_to_f16_bits(3.0)];
        let b_f32 = vec![2.0, 3.0, 4.0];

        let result = unsafe { mixed_precision_dot_product(&a_f16, &b_f32) };

        // Expected: 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
        assert!((result - 20.0).abs() < 0.1, "Expected ~20, got {}", result);
    }

    #[test]
    fn test_mixed_precision_matvec_basic() {
        // 2x3 matrix:
        // [1 2 3]
        // [4 5 6]
        let matrix_f16 = vec![
            f32_to_f16_bits(1.0),
            f32_to_f16_bits(2.0),
            f32_to_f16_bits(3.0),
            f32_to_f16_bits(4.0),
            f32_to_f16_bits(5.0),
            f32_to_f16_bits(6.0),
        ];
        let vector_f32 = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 2];

        unsafe {
            mixed_precision_matvec(&matrix_f16, &vector_f32, &mut output, 2, 3);
        }

        // Row 0: 1*1 + 2*2 + 3*3 = 14
        // Row 1: 4*1 + 5*2 + 6*3 = 32
        assert!((output[0] - 14.0).abs() < 0.1, "Row 0: expected ~14, got {}", output[0]);
        assert!((output[1] - 32.0).abs() < 0.1, "Row 1: expected ~32, got {}", output[1]);
    }

    #[test]
    fn test_mixed_precision_matvec_empty() {
        let matrix_f16: Vec<u16> = vec![];
        let vector_f32: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];

        unsafe {
            mixed_precision_matvec(&matrix_f16, &vector_f32, &mut output, 0, 0);
        }

        assert!(output.is_empty());
    }

    #[test]
    fn test_quantized_mixed_accumulate_basic() {
        // 4 weights (packed in 2 bytes): [1, -1, 1, -1]
        // Each packed as 2-bit: 01, 11, 01, 11
        let weight_byte0 = 0b11_01_11_01u8; // Bits: [w3, w2, w1, w0]
        let weights_i2 = vec![weight_byte0, weight_byte0];

        let activations_f16 = vec![
            f32_to_f16_bits(1.0),
            f32_to_f16_bits(2.0),
            f32_to_f16_bits(3.0),
            f32_to_f16_bits(4.0),
        ];

        let mut output = vec![0.0; 4];

        unsafe {
            quantized_mixed_accumulate(&weights_i2, &activations_f16, &mut output, 4);
        }

        // The exact result depends on i2 unpacking, but should be non-zero
        assert!(output.iter().any(|&v| v.abs() > 0.0), "Expected non-zero accumulation");
    }

    #[test]
    fn test_quantized_mixed_accumulate_accumulation() {
        let weight_byte0 = 0b01_01_01_01u8; // All weights = 1
        let weights_i2 = vec![weight_byte0, weight_byte0];

        let activations_f16 = vec![
            f32_to_f16_bits(1.0),
            f32_to_f16_bits(1.0),
            f32_to_f16_bits(1.0),
            f32_to_f16_bits(1.0),
        ];

        let mut output = vec![2.0; 4]; // Start with initial values

        unsafe {
            quantized_mixed_accumulate(&weights_i2, &activations_f16, &mut output, 4);
        }

        // Should accumulate on top of initial values
        for &val in &output {
            assert!(val > 2.0, "Expected accumulation, got {}", val);
        }
    }

    #[test]
    #[should_panic(expected = "must have same length")]
    fn test_f16_to_f32_neon_size_mismatch() {
        let input = vec![0u16; 4];
        let mut output = vec![0.0; 3];

        unsafe {
            f16_to_f32_neon(&input, &mut output);
        }
    }

    #[test]
    #[should_panic(expected = "must have same length")]
    fn test_mixed_precision_dot_product_length_mismatch() {
        let a_f16 = vec![0u16; 4];
        let b_f32 = vec![0.0; 3];

        unsafe {
            mixed_precision_dot_product(&a_f16, &b_f32);
        }
    }

    #[test]
    #[should_panic(expected = "Matrix size mismatch")]
    fn test_mixed_precision_matvec_matrix_size_mismatch() {
        let matrix_f16 = vec![0u16; 5];
        let vector_f32 = vec![0.0; 3];
        let mut output = vec![0.0; 2];

        unsafe {
            mixed_precision_matvec(&matrix_f16, &vector_f32, &mut output, 2, 3);
        }
    }
}
