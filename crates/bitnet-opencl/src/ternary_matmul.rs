//! Optimised ternary matrix multiplication for BitNet {-1, 0, +1} weights.
//!
//! BitNet models quantise weights to ternary values. This module exploits
//! that structure by replacing generic multiply-accumulate with conditional
//! add/subtract, and provides a POPCOUNT-based binary inner-product path.
//!
//! # Packing format
//!
//! Ternary weights are packed 4 per byte (2 bits each, LSB-first):
//!
//! | bits | value |
//! |------|-------|
//! | `00` |   0   |
//! | `01` |  +1   |
//! | `11` |  -1   |
//!
//! # Kernels
//!
//! - **`ternary_matmul`** — conditional add/subtract with packed int8
//! - **`ternary_popcount_matmul`** — POPCOUNT inner product for binary
//!   activation quantisation (approximate but very fast)

use std::fmt;

// ---------------------------------------------------------------------------
// Packing helpers
// ---------------------------------------------------------------------------

/// Encode a single ternary value (-1, 0, +1) into its 2-bit representation.
#[inline]
fn encode_ternary(val: i8) -> u8 {
    match val {
        1 => 0b01,
        -1 => 0b11,
        0 => 0b00,
        _ => panic!("ternary weight must be -1, 0, or +1, got {val}"),
    }
}

/// Decode a 2-bit ternary encoding back to an i8.
#[inline]
fn decode_ternary(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b11 => -1,
        _ => 0,
    }
}

/// Pack a slice of ternary i8 weights into bytes (4 weights per byte).
///
/// Packs the flat array sequentially. For matrix use, call
/// [`pack_ternary_weight_matrix`] which pads each row independently.
pub fn pack_ternary_weights(weights: &[i8]) -> Vec<u8> {
    weights
        .chunks(4)
        .map(|chunk| {
            let mut byte = 0u8;
            for (i, &w) in chunk.iter().enumerate() {
                byte |= encode_ternary(w) << (i * 2);
            }
            byte
        })
        .collect()
}

/// Pack a `[M, K]` ternary weight matrix into row-major packed format.
///
/// Each row is independently packed to `packed_k = ceil(K/4)` bytes,
/// matching the layout expected by the OpenCL kernel.
pub fn pack_ternary_weight_matrix(
    weights: &[i8],
    m: usize,
    k: usize,
) -> Vec<u8> {
    let packed_k = (k + 3) / 4;
    let mut packed = Vec::with_capacity(m * packed_k);
    for row in 0..m {
        let row_start = row * k;
        let row_weights = &weights[row_start..row_start + k];
        let mut row_packed = pack_ternary_weights(row_weights);
        row_packed.resize(packed_k, 0);
        packed.extend_from_slice(&row_packed);
    }
    packed
}

/// Unpack bytes back to ternary i8 weights.
pub fn unpack_ternary_weights(packed: &[u8], count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    for &byte in packed {
        for sub in 0..4 {
            if out.len() >= count {
                break;
            }
            out.push(decode_ternary(byte >> (sub * 2)));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Bitmask helpers (for POPCOUNT path)
// ---------------------------------------------------------------------------

/// Split ternary weights into plus/minus bitmasks (32 weights per u32 word).
pub fn build_popcount_masks(weights: &[i8]) -> (Vec<u32>, Vec<u32>) {
    let words = (weights.len() + 31) / 32;
    let mut plus = vec![0u32; words];
    let mut minus = vec![0u32; words];

    for (i, &w) in weights.iter().enumerate() {
        let word = i / 32;
        let bit = i % 32;
        match w {
            1 => plus[word] |= 1 << bit,
            -1 => minus[word] |= 1 << bit,
            _ => {}
        }
    }

    (plus, minus)
}

/// Build sign-bit mask from float activations (bit set if activation > 0).
pub fn build_activation_sign_bits(activations: &[f32]) -> Vec<u32> {
    let words = (activations.len() + 31) / 32;
    let mut bits = vec![0u32; words];

    for (i, &a) in activations.iter().enumerate() {
        if a > 0.0 {
            bits[i / 32] |= 1 << (i % 32);
        }
    }

    bits
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Errors from ternary matmul operations.
#[derive(Debug, Clone)]
pub enum TernaryMatmulError {
    /// Dimension mismatch between weight and activation matrices.
    DimensionMismatch { expected_k: usize, actual_k: usize },
    /// Weight values outside {-1, 0, +1}.
    InvalidWeight(i8),
}

impl fmt::Display for TernaryMatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                expected_k,
                actual_k,
            } => write!(
                f,
                "K dimension mismatch: weights have {expected_k}, \
                 activations have {actual_k}"
            ),
            Self::InvalidWeight(w) => {
                write!(f, "invalid ternary weight: {w}")
            }
        }
    }
}

impl std::error::Error for TernaryMatmulError {}

/// Generic (non-ternary) matmul for benchmark comparison: C = W * A.
///
/// W: `[M, K]` (f32), A: `[K, N]` (f32), C: `[M, N]`.
pub fn generic_matmul(
    w: &[f32],
    a: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += w[row * k + i] * a[i * n + col];
            }
            c[row * n + col] = acc;
        }
    }
    c
}

/// Ternary-optimised matmul (CPU reference for the OpenCL kernel).
///
/// W: `[M, K]` packed ternary, A: `[K, N]` f32.
pub fn ternary_matmul_cpu(
    packed_w: &[u8],
    activations: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let packed_k = (k + 3) / 4;
    let mut output = vec![0.0f32; m * n];

    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            let w_row_off = row * packed_k;

            for byte_idx in 0..packed_k {
                let pack = packed_w[w_row_off + byte_idx];
                let base_k = byte_idx * 4;

                for sub in 0..4 {
                    let k_idx = base_k + sub;
                    if k_idx >= k {
                        break;
                    }
                    let bits = (pack >> (sub * 2)) & 0x03;
                    match bits {
                        0b01 => acc += activations[k_idx * n + col],
                        0b11 => acc -= activations[k_idx * n + col],
                        _ => {}
                    }
                }
            }

            output[row * n + col] = acc;
        }
    }

    output
}

/// POPCOUNT-based ternary inner product (CPU reference).
pub fn popcount_matmul_cpu(
    plus_mask: &[u32],
    minus_mask: &[u32],
    act_sign_bits: &[u32],
    m: usize,
    n: usize,
    words_per_row: usize,
) -> Vec<i32> {
    let mut output = vec![0i32; m * n];

    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i32;
            let w_off = row * words_per_row;
            let a_off = col * words_per_row;

            for w in 0..words_per_row {
                let p = plus_mask[w_off + w];
                let mi = minus_mask[w_off + w];
                let a = act_sign_bits[a_off + w];

                acc += (p & a).count_ones() as i32
                    - (mi & a).count_ones() as i32;
            }

            output[row * n + col] = acc;
        }
    }

    output
}

/// Return the embedded OpenCL kernel source.
pub fn kernel_source() -> &'static str {
    include_str!("../kernels/ternary_matmul.cl")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let weights: Vec<i8> =
            vec![1, 0, -1, 1, 0, 0, -1, -1, 1];
        let packed = pack_ternary_weights(&weights);
        let unpacked =
            unpack_ternary_weights(&packed, weights.len());
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn pack_unpack_all_zeros() {
        let weights = vec![0i8; 13];
        let packed = pack_ternary_weights(&weights);
        let unpacked =
            unpack_ternary_weights(&packed, weights.len());
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn ternary_matmul_matches_generic_small() {
        let (m, k, n) = (3, 5, 4);
        let w_ternary: Vec<i8> = vec![
            1, 0, -1, 1, 0, -1, -1, 1, 0, 1, 0, 1, 1, -1, 0,
        ];
        let w_float: Vec<f32> =
            w_ternary.iter().map(|&w| w as f32).collect();
        let a: Vec<f32> = (0..k * n)
            .map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.5)
            .collect();

        let packed =
            pack_ternary_weight_matrix(&w_ternary, m, k);
        let t_out = ternary_matmul_cpu(&packed, &a, m, k, n);
        let g_out = generic_matmul(&w_float, &a, m, k, n);

        for (i, (t, g)) in t_out.iter().zip(&g_out).enumerate() {
            assert!(
                (t - g).abs() < 1e-5,
                "idx {i}: ternary={t}, generic={g}"
            );
        }
    }

    #[test]
    fn ternary_matmul_identity_like() {
        let w: Vec<i8> = vec![
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
        ];
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let packed = pack_ternary_weights(&w);
        let out = ternary_matmul_cpu(&packed, &a, 4, 4, 1);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ternary_matmul_negation() {
        let (m, k, n) = (2, 3, 1);
        let w = vec![-1i8; m * k];
        let a = vec![1.0f32, 2.0, 3.0];
        let packed = pack_ternary_weight_matrix(&w, m, k);
        let out = ternary_matmul_cpu(&packed, &a, m, k, n);
        assert!((out[0] - (-6.0)).abs() < 1e-5);
        assert!((out[1] - (-6.0)).abs() < 1e-5);
    }

    #[test]
    fn ternary_matmul_k_not_multiple_of_4() {
        let (m, k, n) = (2, 7, 2);
        let w_ternary: Vec<i8> =
            (0..m * k).map(|i| [1, 0, -1][i % 3]).collect();
        let w_float: Vec<f32> =
            w_ternary.iter().map(|&w| w as f32).collect();
        let a: Vec<f32> =
            (0..k * n).map(|i| (i as f32) * 0.5 - 2.0).collect();

        let packed =
            pack_ternary_weight_matrix(&w_ternary, m, k);
        let t_out = ternary_matmul_cpu(&packed, &a, m, k, n);
        let g_out = generic_matmul(&w_float, &a, m, k, n);

        for (t, g) in t_out.iter().zip(&g_out) {
            assert!((t - g).abs() < 1e-5);
        }
    }

    #[test]
    fn popcount_matmul_basic() {
        let w_row0: Vec<i8> = vec![1, -1, 1, 0];
        let w_row1: Vec<i8> = vec![-1, 1, 0, 1];

        let (plus0, minus0) = build_popcount_masks(&w_row0);
        let (plus1, minus1) = build_popcount_masks(&w_row1);
        let mut plus_flat = plus0;
        plus_flat.extend_from_slice(&plus1);
        let mut minus_flat = minus0;
        minus_flat.extend_from_slice(&minus1);

        // Activations: [1.0, -0.5, 0.3, 0.8] -> sign bits: {0,2,3}
        let act = vec![1.0f32, -0.5, 0.3, 0.8];
        let act_bits = build_activation_sign_bits(&act);

        let result = popcount_matmul_cpu(
            &plus_flat,
            &minus_flat,
            &act_bits,
            2,
            1,
            1,
        );

        // Row 0: plus={0,2} & sign={0,2,3} -> {0,2}=2; minus={1} & sign={0,2,3} -> {}=0 => 2
        // Row 1: plus={1,3} & sign={0,2,3} -> {3}=1; minus={0} & sign={0,2,3} -> {0}=1 => 0
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 0);
    }

    #[test]
    fn popcount_masks_roundtrip() {
        let weights: Vec<i8> = vec![
            1, 0, -1, 1, 0, 0, -1, -1, 1, 0, 1, -1, 0, 1, -1,
            0, 1, 0, -1, 1, 0, 0, -1, -1, 1, 0, 1, -1, 0, 1,
            -1, 0, 1,
        ];
        let (plus, minus) = build_popcount_masks(&weights);
        assert_eq!(plus.len(), 2);
        assert_eq!(minus.len(), 2);

        for (i, &w) in weights.iter().enumerate() {
            let word = i / 32;
            let bit = i % 32;
            let is_plus = (plus[word] >> bit) & 1 == 1;
            let is_minus = (minus[word] >> bit) & 1 == 1;
            match w {
                1 => assert!(is_plus && !is_minus),
                -1 => assert!(!is_plus && is_minus),
                0 => assert!(!is_plus && !is_minus),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn kernel_source_loads() {
        let src = kernel_source();
        assert!(src.contains("ternary_matmul"));
        assert!(src.contains("ternary_popcount_matmul"));
    }

    #[test]
    fn benchmark_ternary_vs_generic() {
        let (m, k, n) = (64, 128, 64);
        let w_ternary: Vec<i8> =
            (0..m * k).map(|i| [1, 0, -1, 0][i % 4]).collect();
        let w_float: Vec<f32> =
            w_ternary.iter().map(|&w| w as f32).collect();
        let a: Vec<f32> = (0..k * n)
            .map(|i| {
                ((i * 31 + 7) % 101) as f32 * 0.02 - 1.0
            })
            .collect();

        let packed =
            pack_ternary_weight_matrix(&w_ternary, m, k);

        let t0 = std::time::Instant::now();
        let t_out = ternary_matmul_cpu(&packed, &a, m, k, n);
        let t_ternary = t0.elapsed();

        let t1 = std::time::Instant::now();
        let g_out = generic_matmul(&w_float, &a, m, k, n);
        let t_generic = t1.elapsed();

        for (t, g) in t_out.iter().zip(&g_out) {
            assert!(
                (t - g).abs() < 1e-4,
                "ternary={t}, generic={g}"
            );
        }

        eprintln!(
            "ternary: {:?}, generic: {:?}",
            t_ternary, t_generic
        );
    }
}
