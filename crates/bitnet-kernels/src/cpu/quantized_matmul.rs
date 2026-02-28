//! Quantized I2_S matrix multiplication kernels
//!
//! Provides optimized matrix multiplication where weights are stored in I2_S
//! (2-bit signed) packed format. Each weight element is one of {-1, 0, +1},
//! packed 4 values per byte (2 bits each).
//!
//! Two block sizes are supported:
//! - **32**: BitNet32-F16 format (32-element blocks with inline F16 scales)
//! - **256**: QK256 / GGML format (256-element blocks with separate scales)
//!
//! Encoding: bits `0b00` → 0, `0b01` → +1, `0b10` → unused, `0b11` → -1.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Encoding helpers ───────────────────────────────────────────────────

/// Decode a 2-bit I2_S code to its signed integer value.
#[inline(always)]
fn decode_i2s(bits: u8) -> i8 {
    match bits & 0x03 {
        0b00 => 0,
        0b01 => 1,
        0b11 => -1,
        // 0b10 is unused in the I2_S spec; treat as 0.
        _ => 0,
    }
}

/// Pack four ternary values ({-1, 0, +1}) into one byte, LSB-first.
pub fn pack_i2s(vals: [i8; 4]) -> u8 {
    let mut byte = 0u8;
    for (i, &v) in vals.iter().enumerate() {
        let code: u8 = match v {
            1 => 0b01,
            -1 => 0b11,
            _ => 0b00,
        };
        byte |= code << (i * 2);
    }
    byte
}

// ── Scalar reference implementation ────────────────────────────────────

/// Scalar I2_S matrix multiplication (reference / fallback).
///
/// Computes `C[m×n] = A[m×k] · B_packed[k×n]` where `B_packed` stores
/// each column-major weight in 2-bit I2_S encoding (4 values per byte).
///
/// # Layout
/// - `activations`: row-major `[m, k]` f32
/// - `weights_packed`: packed I2_S, `ceil(k/4) * n` bytes, column-major
///   within each output column (byte `col * packed_k + byte_idx`)
/// - `scales`: one f32 per block of `block_size` elements along `k` per
///   output column → `n * num_blocks_k` entries
/// - `out`: row-major `[m, n]` f32
pub fn i2s_matmul_f32(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    validate_matmul_args(activations, weights_packed, scales, out, m, n, k, block_size)?;

    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);

    out.fill(0.0);

    for row in 0..m {
        let a_row = &activations[row * k..(row + 1) * k];
        for col in 0..n {
            let mut acc = 0.0f32;
            for blk in 0..num_blocks_k {
                let blk_start = blk * block_size;
                let blk_end = (blk_start + block_size).min(k);
                let scale = scales[col * num_blocks_k + blk];

                for idx in blk_start..blk_end {
                    let byte_idx = col * packed_k + idx / 4;
                    let bit_off = (idx % 4) * 2;
                    let bits = (weights_packed[byte_idx] >> bit_off) & 0x03;
                    let w = decode_i2s(bits) as f32 * scale;
                    acc += a_row[idx] * w;
                }
            }
            out[row * n + col] = acc;
        }
    }
    Ok(())
}

// ── Block-oriented dequantize + matmul ─────────────────────────────────

/// Dequantize an I2_S weight matrix and multiply against activations.
///
/// This is a two-pass approach: first dequantize the full weight matrix
/// to `f32`, then run a standard f32 GEMM.  Useful as a correctness oracle
/// and for small matrices where the extra allocation is acceptable.
pub fn dequantize_and_matmul(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    validate_matmul_args(activations, weights_packed, scales, out, m, n, k, block_size)?;

    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);

    // Dequantize into a k×n f32 weight matrix (column of weights per output).
    let mut weights_f32 = vec![0.0f32; k * n];
    for col in 0..n {
        for blk in 0..num_blocks_k {
            let blk_start = blk * block_size;
            let blk_end = (blk_start + block_size).min(k);
            let scale = scales[col * num_blocks_k + blk];
            for idx in blk_start..blk_end {
                let byte_idx = col * packed_k + idx / 4;
                let bit_off = (idx % 4) * 2;
                let bits = (weights_packed[byte_idx] >> bit_off) & 0x03;
                weights_f32[idx * n + col] = decode_i2s(bits) as f32 * scale;
            }
        }
    }

    // Standard f32 GEMM: C = A · W_dequant
    out.fill(0.0);
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for idx in 0..k {
                acc += activations[row * k + idx] * weights_f32[idx * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    Ok(())
}

/// Block-based I2_S matmul that processes one block at a time.
///
/// Semantically identical to [`i2s_matmul_f32`] but structures the loop
/// to process weights one block at a time, which is friendlier to future
/// SIMD tiling where the inner loop can be vectorised per-block.
pub fn i2s_matmul_blocked(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    validate_matmul_args(activations, weights_packed, scales, out, m, n, k, block_size)?;

    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);

    out.fill(0.0);

    // Outer loop over blocks — enables future SIMD tiling.
    for blk in 0..num_blocks_k {
        let blk_start = blk * block_size;
        let blk_end = (blk_start + block_size).min(k);

        for col in 0..n {
            let scale = scales[col * num_blocks_k + blk];

            // Dequantize this block slice for the current column.
            let mut w_blk = [0i8; 256]; // max block_size supported
            for idx in blk_start..blk_end {
                let byte_idx = col * packed_k + idx / 4;
                let bit_off = (idx % 4) * 2;
                let bits = (weights_packed[byte_idx] >> bit_off) & 0x03;
                w_blk[idx - blk_start] = decode_i2s(bits);
            }

            for row in 0..m {
                let mut acc = 0.0f32;
                let a_row = &activations[row * k + blk_start..row * k + blk_end];
                for (i, &a_val) in a_row.iter().enumerate() {
                    acc += a_val * w_blk[i] as f32;
                }
                out[row * n + col] += acc * scale;
            }
        }
    }
    Ok(())
}

// ── Validation ─────────────────────────────────────────────────────────

fn validate_matmul_args(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &[f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    if block_size == 0 {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: "block_size must be > 0".into(),
        }));
    }
    if m == 0 || n == 0 || k == 0 {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("dimensions must be > 0: m={m}, n={n}, k={k}"),
        }));
    }
    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);

    if activations.len() < m * k {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("activations too small: expected {}, got {}", m * k, activations.len()),
        }));
    }
    if weights_packed.len() < packed_k * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "weights_packed too small: expected {}, got {}",
                packed_k * n,
                weights_packed.len()
            ),
        }));
    }
    if scales.len() < n * num_blocks_k {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "scales too small: expected {}, got {}",
                n * num_blocks_k,
                scales.len()
            ),
        }));
    }
    if out.len() < m * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("output too small: expected {}, got {}", m * n, out.len()),
        }));
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    /// Pack a full weight matrix (k×n, row-major ternary) into I2_S bytes
    /// with column-major packing and uniform scale = 1.0.
    fn pack_weight_matrix(
        weights: &[i8],
        k: usize,
        n: usize,
        block_size: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        let packed_k = k.div_ceil(4);
        let num_blocks_k = k.div_ceil(block_size);
        let mut packed = vec![0u8; packed_k * n];
        for col in 0..n {
            for row in 0..k {
                let val = weights[row * n + col];
                let code: u8 = match val {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                let byte_idx = col * packed_k + row / 4;
                let bit_off = (row % 4) * 2;
                packed[byte_idx] |= code << bit_off;
            }
        }
        let scales = vec![1.0f32; n * num_blocks_k];
        (packed, scales)
    }

    /// Naive f32 matmul: C = A · W (A is m×k, W is k×n, both row-major).
    fn naive_f32_matmul(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for l in 0..k {
                    s += a[i * k + l] * w[l * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at index {i}: {x} vs {y} (tol {tol})");
        }
    }

    // Run the same test body through all three kernel functions.
    fn run_all_kernels(
        act: &[f32],
        packed: &[u8],
        scales: &[f32],
        m: usize,
        n: usize,
        k: usize,
        bs: usize,
        expected: &[f32],
        tol: f32,
    ) {
        let mut out = vec![0.0f32; m * n];

        i2s_matmul_f32(act, packed, scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, expected, tol);

        i2s_matmul_blocked(act, packed, scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, expected, tol);

        dequantize_and_matmul(act, packed, scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, expected, tol);
    }

    // ── correctness against naive f32 matmul ──────────────────────────

    #[test]
    fn test_identity_2x2_block32() {
        let m = 2;
        let n = 2;
        let k = 2;
        let bs = 32;
        // Identity-like ternary weight matrix
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act = vec![3.0f32, -2.0, 5.0, 7.0];
        let expected = naive_f32_matmul(&act, &[1.0, 0.0, 0.0, 1.0], m, n, k);
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-6);
    }

    #[test]
    fn test_identity_2x2_block256() {
        let m = 2;
        let n = 2;
        let k = 2;
        let bs = 256;
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act = vec![3.0, -2.0, 5.0, 7.0];
        let expected = naive_f32_matmul(&act, &[1.0, 0.0, 0.0, 1.0], m, n, k);
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-6);
    }

    #[test]
    fn test_all_ones_weight_4x4_block32() {
        let m = 4;
        let n = 4;
        let k = 4;
        let bs = 32;
        let w = vec![1i8; k * n];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-5);
    }

    #[test]
    fn test_all_neg_ones_weight() {
        let m = 3;
        let n = 3;
        let k = 4;
        let bs = 32;
        let w = vec![-1i8; k * n];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act = vec![1.0f32; m * k];
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-5);
    }

    #[test]
    fn test_zero_weight_matrix() {
        let m = 4;
        let n = 4;
        let k = 8;
        let bs = 32;
        let w = vec![0i8; k * n];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act = vec![42.0f32; m * k];
        let expected = vec![0.0f32; m * n];
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-6);
    }

    #[test]
    fn test_zero_activation_matrix() {
        let m = 3;
        let n = 5;
        let k = 8;
        let bs = 32;
        let w = vec![1i8; k * n];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act = vec![0.0f32; m * k];
        let expected = vec![0.0f32; m * n];
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-6);
    }

    // ── various matrix sizes ──────────────────────────────────────────

    #[test]
    fn test_power_of_two_16x16_block32() {
        let m = 16;
        let n = 16;
        let k = 16;
        let bs = 32;
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-4);
    }

    #[test]
    fn test_non_power_of_two_7x5_k11_block32() {
        let m = 7;
        let n = 5;
        let k = 11;
        let bs = 32;
        let w: Vec<i8> = (0..k * n).map(|i| [-1, 0, 1][i % 3]).collect();
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.05 - 1.0).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-4);
    }

    #[test]
    fn test_non_power_of_two_13x9_k17_block256() {
        let m = 13;
        let n = 9;
        let k = 17;
        let bs = 256;
        let w: Vec<i8> = (0..k * n).map(|i| [0, 1, -1][i % 3]).collect();
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 10) as f32 - 5.0).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-4);
    }

    #[test]
    fn test_medium_32x64_block32() {
        let m = 32;
        let n = 64;
        let k = 32;
        let bs = 32;
        let w: Vec<i8> = (0..k * n).map(|i| [1, 0, -1, 0][i % 4]).collect();
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32).sin()).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-3);
    }

    #[test]
    fn test_medium_64x32_block256() {
        let m = 64;
        let n = 32;
        let k = 256;
        let bs = 256;
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1][i % 2]).collect();
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-3);
    }

    // ── edge cases ────────────────────────────────────────────────────

    #[test]
    fn test_1x1_matrix() {
        let m = 1;
        let n = 1;
        let k = 1;
        let bs = 32;
        let w = vec![1i8];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act = vec![7.5f32];
        let expected = vec![7.5f32];
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-6);
    }

    #[test]
    fn test_1xn_row_vector() {
        let m = 1;
        let n = 8;
        let k = 4;
        let bs = 32;
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act = vec![1.0, 2.0, 3.0, 4.0];
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-5);
    }

    #[test]
    fn test_nx1_column_vector() {
        let m = 6;
        let n = 1;
        let k = 4;
        let bs = 32;
        let w = vec![1i8, -1, 1, -1]; // k×1
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-5);
    }

    #[test]
    fn test_k_not_multiple_of_4() {
        let m = 3;
        let n = 2;
        let k = 5;
        let bs = 32;
        let w: Vec<i8> = vec![1, 0, -1, 1, 0, 1, -1, 0, 1, -1];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32 + 0.5).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 1e-5);
    }

    // ── scale factor tests ────────────────────────────────────────────

    #[test]
    fn test_non_unit_scales_block32() {
        let m: usize = 2;
        let n: usize = 2;
        let k: usize = 4;
        let bs: usize = 32;
        let packed_k = k.div_ceil(4);
        let num_blocks_k = k.div_ceil(bs);

        // Manually build packed weights: all +1.
        let w_ternary = vec![1i8; k * n];
        let mut packed = vec![0u8; packed_k * n];
        for col in 0..n {
            for row in 0..k {
                let code = 0b01u8;
                let byte_idx = col * packed_k + row / 4;
                let bit_off = (row % 4) * 2;
                packed[byte_idx] |= code << bit_off;
            }
        }
        // Scale col0 = 2.0, col1 = 0.5
        let mut scales = vec![0.0f32; n * num_blocks_k];
        scales[0] = 2.0;
        scales[1] = 0.5;

        let act = vec![1.0f32; m * k];
        let mut out = vec![0.0f32; m * n];
        i2s_matmul_f32(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();

        // Each row: col0 = sum(1.0*1*2.0, 4 times) = 8.0, col1 = 4*0.5 = 2.0
        let _ = w_ternary; // suppress unused
        assert_close(&out, &[8.0, 2.0, 8.0, 2.0], 1e-5);
    }

    #[test]
    fn test_multi_block_scales_block32() {
        // k=64 → 2 blocks of 32 with different scales
        let m: usize = 1;
        let n: usize = 1;
        let k: usize = 64;
        let bs: usize = 32;
        let packed_k = k.div_ceil(4); // 16
        let num_blocks_k = k.div_ceil(bs); // 2

        // All weights = +1
        let mut packed = vec![0u8; packed_k * n];
        for byte in packed.iter_mut() {
            *byte = 0b01_01_01_01; // four +1 values
        }
        let mut scales = vec![0.0f32; n * num_blocks_k];
        scales[0] = 1.0; // block 0
        scales[1] = 3.0; // block 1

        let act = vec![1.0f32; m * k];
        let mut out = vec![0.0f32; m * n];
        i2s_matmul_f32(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();

        // block0: 32 * 1.0 * 1.0 = 32, block1: 32 * 1.0 * 3.0 = 96 → 128
        assert_close(&out, &[128.0], 1e-4);
    }

    // ── numerical accuracy bounds ─────────────────────────────────────

    #[test]
    fn test_numerical_accuracy_f32_exact() {
        // With unit scales and small integers the result must be exact.
        let m = 4;
        let n = 4;
        let k = 8;
        let bs = 32;
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0][i % 3]).collect();
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i % 5) as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        // Must be bit-exact since all values are small integers.
        run_all_kernels(&act, &packed, &scales, m, n, k, bs, &expected, 0.0);
    }

    #[test]
    fn test_numerical_accuracy_with_scales() {
        let m = 4;
        let n = 4;
        let k = 32;
        let bs = 32;
        // Ternary pattern
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let packed_k = k.div_ceil(4);
        let num_blocks_k = k.div_ceil(bs);

        let mut packed = vec![0u8; packed_k * n];
        for col in 0..n {
            for row in 0..k {
                let val = w[row * n + col];
                let code: u8 = match val {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                let byte_idx = col * packed_k + row / 4;
                let bit_off = (row % 4) * 2;
                packed[byte_idx] |= code << bit_off;
            }
        }
        // Varying scales
        let mut scales = vec![0.0f32; n * num_blocks_k];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 0.5 + (i as f32) * 0.1;
        }

        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let mut out_f32 = vec![0.0f32; m * n];
        let mut out_blk = vec![0.0f32; m * n];

        i2s_matmul_f32(&act, &packed, &scales, &mut out_f32, m, n, k, bs).unwrap();
        i2s_matmul_blocked(&act, &packed, &scales, &mut out_blk, m, n, k, bs).unwrap();
        assert_close(&out_f32, &out_blk, 1e-4);
    }

    // ── pack_i2s helper round-trip ────────────────────────────────────

    #[test]
    fn test_pack_i2s_roundtrip() {
        let vals: [i8; 4] = [1, -1, 0, 1];
        let byte = pack_i2s(vals);
        for (i, &expected) in vals.iter().enumerate() {
            let bits = (byte >> (i * 2)) & 0x03;
            assert_eq!(decode_i2s(bits), expected, "mismatch at position {i}");
        }
    }

    #[test]
    fn test_pack_i2s_all_zero() {
        let byte = pack_i2s([0, 0, 0, 0]);
        assert_eq!(byte, 0x00);
    }

    #[test]
    fn test_pack_i2s_all_plus_one() {
        let byte = pack_i2s([1, 1, 1, 1]);
        assert_eq!(byte, 0b01_01_01_01);
    }

    #[test]
    fn test_pack_i2s_all_minus_one() {
        let byte = pack_i2s([-1, -1, -1, -1]);
        assert_eq!(byte, 0b11_11_11_11);
    }

    // ── validation / error handling ───────────────────────────────────

    #[test]
    fn test_dimension_zero_rejected() {
        let act = vec![1.0f32; 4];
        let packed = vec![0u8; 4];
        let scales = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];

        assert!(i2s_matmul_f32(&act, &packed, &scales, &mut out, 0, 2, 2, 32).is_err());
        assert!(i2s_matmul_f32(&act, &packed, &scales, &mut out, 2, 0, 2, 32).is_err());
        assert!(i2s_matmul_f32(&act, &packed, &scales, &mut out, 2, 2, 0, 32).is_err());
    }

    #[test]
    fn test_block_size_zero_rejected() {
        let act = vec![1.0f32; 4];
        let packed = vec![0u8; 2];
        let scales = vec![1.0f32; 2];
        let mut out = vec![0.0f32; 4];
        assert!(i2s_matmul_f32(&act, &packed, &scales, &mut out, 2, 2, 2, 0).is_err());
    }

    #[test]
    fn test_activation_buffer_too_small() {
        let act = vec![1.0f32; 2]; // too small for 2×4
        let packed = vec![0u8; 4];
        let scales = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        assert!(i2s_matmul_f32(&act, &packed, &scales, &mut out, 2, 2, 4, 32).is_err());
    }

    #[test]
    fn test_output_buffer_too_small() {
        let act = vec![1.0f32; 4];
        let packed = vec![0u8; 2];
        let scales = vec![1.0f32; 2];
        let mut out = vec![0.0f32; 1]; // too small for 2×2
        assert!(i2s_matmul_f32(&act, &packed, &scales, &mut out, 2, 2, 2, 32).is_err());
    }

    // ── cross-function consistency ────────────────────────────────────

    #[test]
    fn test_all_three_kernels_agree_large() {
        let m = 16;
        let n = 8;
        let k = 48;

        for &bs in &[32usize, 256] {
            let w: Vec<i8> = (0..k * n).map(|i| [1, 0, -1, 1, -1][i % 5]).collect();
            let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
            let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.03).sin()).collect();

            let mut out1 = vec![0.0f32; m * n];
            let mut out2 = vec![0.0f32; m * n];
            let mut out3 = vec![0.0f32; m * n];

            i2s_matmul_f32(&act, &packed, &scales, &mut out1, m, n, k, bs).unwrap();
            i2s_matmul_blocked(&act, &packed, &scales, &mut out2, m, n, k, bs).unwrap();
            dequantize_and_matmul(&act, &packed, &scales, &mut out3, m, n, k, bs).unwrap();

            assert_close(&out1, &out2, 1e-4);
            assert_close(&out1, &out3, 1e-4);
        }
    }
}
