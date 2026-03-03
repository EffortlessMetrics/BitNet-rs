#![cfg(all(feature = "cpu", target_arch = "aarch64"))]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::useless_vec)]
#![allow(clippy::approx_constant)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::manual_is_multiple_of)]

//! Regression tests for NEON matrix multiplication on Apple Silicon.
//!
//! These tests validate a naive CPU reference matmul and provide TDD
//! scaffolds for when the NEON-accelerated path is wired up.

/// Naive triple-loop matrix multiplication (row-major).
///
/// Computes C = A * B where A is (m x k), B is (k x n), C is (m x n).
fn reference_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A dimensions mismatch");
    assert_eq!(b.len(), k * n, "B dimensions mismatch");
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Transpose a row-major matrix (m x n) into (n x m).
fn transpose(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(mat.len(), rows * cols);
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = mat[i * cols + j];
        }
    }
    out
}

/// Pack four 2-bit signed values (-1, 0, 1) into a single byte.
///
/// Each value occupies 2 bits in little-endian order within the byte.
fn pack_i2s(vals: &[i8; 4]) -> u8 {
    let mut byte = 0u8;
    for (i, &v) in vals.iter().enumerate() {
        let bits = (v & 0x03) as u8;
        byte |= bits << (i * 2);
    }
    byte
}

/// Unpack a byte into four 2-bit signed values (-1, 0, 1).
fn unpack_i2s(byte: u8) -> [i8; 4] {
    let mut vals = [0i8; 4];
    for i in 0..4 {
        let bits = ((byte >> (i * 2)) & 0x03) as i8;
        // Sign-extend: 0b11 → -1, 0b00 → 0, 0b01 → 1
        vals[i] = if bits >= 2 { bits - 4 } else { bits };
    }
    vals
}

fn assert_matrices_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "mismatch at index {i}: actual={a}, expected={e}, diff={}",
            (a - e).abs()
        );
    }
}

// ─── CPU reference matmul tests ─────────────────────────────────────

#[test]
fn test_matmul_1x1() {
    let a = vec![3.0f32];
    let b = vec![5.0f32];
    let c = reference_matmul(&a, &b, 1, 1, 1);
    assert_eq!(c, vec![15.0]);
}

#[test]
fn test_matmul_2x2() {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = reference_matmul(&a, &b, 2, 2, 2);
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_matmul_identity() {
    // A * I = A for 3x3
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let c = reference_matmul(&a, &identity, 3, 3, 3);
    assert_eq!(c, a);
}

#[test]
fn test_matmul_non_square() {
    // A(2x3) * B(3x4) = C(2x4)
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![
        7.0, 8.0, 9.0, 10.0, //
        11.0, 12.0, 13.0, 14.0, //
        15.0, 16.0, 17.0, 18.0,
    ];
    let c = reference_matmul(&a, &b, 2, 3, 4);
    // Row 0: [1*7+2*11+3*15, 1*8+2*12+3*16, 1*9+2*13+3*17, 1*10+2*14+3*18]
    //      = [74, 80, 86, 92]
    // Row 1: [4*7+5*11+6*15, 4*8+5*12+6*16, 4*9+5*13+6*17, 4*10+5*14+6*18]
    //      = [173, 188, 203, 218]
    assert_eq!(c, vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0]);
}

#[test]
fn test_matmul_zero_matrix() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let zero = vec![0.0f32; 4];
    let c = reference_matmul(&a, &zero, 2, 2, 2);
    assert_eq!(c, vec![0.0; 4]);

    // Zero * B = Zero
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c2 = reference_matmul(&zero, &b, 2, 2, 2);
    assert_eq!(c2, vec![0.0; 4]);
}

#[test]
fn test_matmul_transpose_correctness() {
    // (A * B)^T = B^T * A^T
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

    let ab = reference_matmul(&a, &b, 2, 3, 2);
    let ab_t = transpose(&ab, 2, 2);

    let bt = transpose(&b, 3, 2); // 2x3
    let at = transpose(&a, 2, 3); // 3x2
    let bt_at = reference_matmul(&bt, &at, 2, 3, 2);

    assert_matrices_close(&ab_t, &bt_at, 1e-6);
}

#[test]
fn test_matmul_large_64x64() {
    let n = 64;
    // Fill A with row index, B with column index (simple deterministic pattern).
    let mut a = vec![0.0f32; n * n];
    let mut b = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = (i + 1) as f32;
            b[i * n + j] = (j + 1) as f32;
        }
    }
    let c = reference_matmul(&a, &b, n, n, n);

    // C[i][j] = sum_{p=0}^{63} A[i][p] * B[p][j]
    //         = (i+1) * (j+1) * sum_{p=0}^{63} 1
    //         = (i+1) * (j+1) * 64
    for i in 0..n {
        for j in 0..n {
            let expected = (i + 1) as f32 * (j + 1) as f32 * n as f32;
            assert!(
                (c[i * n + j] - expected).abs() < 1e-3,
                "mismatch at [{i}][{j}]: got {}, expected {expected}",
                c[i * n + j]
            );
        }
    }
}

#[test]
fn test_i2s_quantized_matmul_pattern() {
    // Simulate I2S quantized matmul: pack ternary weights, dequantize, multiply.
    // Weight row: [-1, 0, 1, 1] with scale 0.5
    let weights_raw: [i8; 4] = [-1, 0, 1, 1];
    let scale = 0.5f32;

    let packed = pack_i2s(&weights_raw);
    let unpacked = unpack_i2s(packed);
    assert_eq!(unpacked, weights_raw);

    // Dequantize: float_weight = int_weight * scale
    let dequantized: Vec<f32> = unpacked.iter().map(|&w| w as f32 * scale).collect();
    assert_eq!(dequantized, vec![-0.5, 0.0, 0.5, 0.5]);

    // Multiply dequantized 1x4 weight row by 4x1 activation column.
    let activations = vec![2.0f32, 3.0, 4.0, 5.0];
    let result = reference_matmul(&dequantized, &activations, 1, 4, 1);
    // (-0.5)*2 + 0*3 + 0.5*4 + 0.5*5 = -1 + 0 + 2 + 2.5 = 3.5
    assert_matrices_close(&result, &[3.5], 1e-6);

    // Multi-row: 2 weight rows, each packed separately.
    let row0: [i8; 4] = [-1, 0, 1, 1];
    let row1: [i8; 4] = [1, 1, 0, -1];
    let scales = [0.5f32, 0.25];

    let mut weight_matrix = vec![0.0f32; 8];
    for (i, &w) in unpack_i2s(pack_i2s(&row0)).iter().enumerate() {
        weight_matrix[i] = w as f32 * scales[0];
    }
    for (i, &w) in unpack_i2s(pack_i2s(&row1)).iter().enumerate() {
        weight_matrix[4 + i] = w as f32 * scales[1];
    }

    let result2 = reference_matmul(&weight_matrix, &activations, 2, 4, 1);
    // Row 0: same as above = 3.5
    // Row 1: 0.25*2 + 0.25*3 + 0*4 + (-0.25)*5 = 0.5 + 0.75 + 0 - 1.25 = 0.0
    assert_matrices_close(&result2, &[3.5, 0.0], 1e-6);
}

// ─── TDD scaffolds: NEON-accelerated matmul (wired later) ──────────

#[test]
#[ignore = "TDD scaffold: requires NEON matmul kernel wired into KernelProvider"]
fn test_neon_vs_cpu_reference_parity_small() {
    // Verify NEON matmul matches CPU reference for small matrices (1x1..8x8).
    panic!("not yet implemented");
}

#[test]
#[ignore = "TDD scaffold: requires NEON matmul kernel wired into KernelProvider"]
fn test_neon_vs_cpu_reference_parity_medium() {
    // Verify NEON matmul matches CPU reference for medium matrices (16x16..128x128).
    panic!("not yet implemented");
}

#[test]
#[ignore = "TDD scaffold: requires NEON matmul kernel wired into KernelProvider"]
fn test_neon_vs_cpu_reference_parity_non_square() {
    // Verify NEON matmul matches CPU reference for non-square shapes (MxK * KxN).
    panic!("not yet implemented");
}

#[test]
#[ignore = "TDD scaffold: requires NEON quantized matmul kernel for I2S dequant+FMLA fusion"]
fn test_neon_quantized_matmul_performance() {
    // Verify NEON fused dequant-matmul produces correct results and is faster than scalar.
    panic!("not yet implemented");
}

#[test]
#[ignore = "TDD scaffold: requires NEON batch matmul dispatch in KernelProvider"]
fn test_neon_batch_matmul() {
    // Verify batched matmul dispatches to NEON and produces correct per-batch results.
    panic!("not yet implemented");
}
