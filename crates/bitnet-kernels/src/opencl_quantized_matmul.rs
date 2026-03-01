//! Quantized matrix multiplication kernels for BitNet 1.58b ternary/binary
//! weights on Intel Arc A770 (OpenCL) and CPU reference paths.
//!
//! BitNet 1.58b uses ternary weights {-1, 0, +1}. Matrix multiplication
//! with these weights reduces to addition and subtraction — no FMA needed.
//! This module provides:
//!
//! - CPU reference implementations (unpacked, packed 2-bit, binary 1-bit)
//! - Tiled and popcount variants
//! - OpenCL kernel source strings for GPU dispatch
//!
//! # Weight Encoding
//!
//! | Encoding     | Bits | Values          |
//! |--------------|------|-----------------|
//! | Binary1Bit   |  1   | {-1, +1}        |
//! | Ternary2Bit  |  2   | {-1, 0, +1}     |
//! | I2S          |  2   | GGML I2_S compat |
//!
//! Ternary 2-bit packing: `00 = 0`, `01 = +1`, `10 = -1`, `11 = unused`.
//! Four values per byte, LSB-first.

use std::fmt;

// ── Types ───────────────────────────────────────────────────────────────────

/// Weight encoding format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightEncoding {
    /// 1-bit binary weights: {-1, +1} packed 8 per byte.
    Binary1Bit,
    /// 2-bit ternary weights: {-1, 0, +1} packed 4 per byte.
    Ternary2Bit,
    /// GGML I2_S format (2-bit ternary, different bit mapping).
    I2S,
}

impl fmt::Display for WeightEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binary1Bit => write!(f, "Binary1Bit"),
            Self::Ternary2Bit => write!(f, "Ternary2Bit"),
            Self::I2S => write!(f, "I2_S"),
        }
    }
}

/// Packed quantized weight matrix.
#[derive(Debug, Clone)]
pub struct QuantizedWeight {
    /// Packed weight bytes.
    pub data: Vec<u8>,
    /// Number of rows in the weight matrix.
    pub rows: usize,
    /// Number of columns in the weight matrix.
    pub cols: usize,
    /// Encoding format used for packing.
    pub encoding: WeightEncoding,
}

/// Accumulator type for matmul output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulatorType {
    /// 32-bit integer accumulator (exact for ternary).
    Int32,
    /// 32-bit float accumulator.
    Float32,
}

/// Configuration for quantized matrix multiplication.
#[derive(Debug, Clone)]
pub struct QuantizedMatMulConfig {
    /// Tile size along M dimension (rows of output).
    pub tile_m: usize,
    /// Tile size along N dimension (cols of output).
    pub tile_n: usize,
    /// Whether to use popcount for binary matmul.
    pub use_popcount: bool,
    /// Accumulator type.
    pub accumulator: AccumulatorType,
}

impl Default for QuantizedMatMulConfig {
    fn default() -> Self {
        Self { tile_m: 8, tile_n: 8, use_popcount: true, accumulator: AccumulatorType::Float32 }
    }
}

/// Errors specific to quantized matrix multiplication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizedMatMulError {
    /// Matrix dimensions do not match for multiplication.
    DimensionMismatch { expected_k: usize, got_k: usize },
    /// Packed data has invalid encoding or unexpected length.
    InvalidEncoding { reason: String },
    /// Accumulator overflow detected.
    Overflow,
}

impl fmt::Display for QuantizedMatMulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected_k, got_k } => {
                write!(
                    f,
                    "dimension mismatch: expected K={expected_k}, \
                     got K={got_k}"
                )
            }
            Self::InvalidEncoding { reason } => {
                write!(f, "invalid encoding: {reason}")
            }
            Self::Overflow => write!(f, "accumulator overflow"),
        }
    }
}

impl std::error::Error for QuantizedMatMulError {}

// ── Packing helpers ─────────────────────────────────────────────────────────

/// Pack ternary weights {-1, 0, +1} into 2 bits per weight.
///
/// Encoding: `00 = 0`, `01 = +1`, `10 = -1`, `11 = unused`.
/// Four values per byte, LSB-first.
pub fn pack_ternary_weights(weights: &[i8], cols: usize) -> Vec<u8> {
    assert!(
        cols > 0 && weights.len().is_multiple_of(cols),
        "weights length must be a positive multiple of cols"
    );
    let bytes_per_row = cols.div_ceil(4);
    let rows = weights.len() / cols;
    let mut packed = vec![0u8; rows * bytes_per_row];

    for row in 0..rows {
        for col in 0..cols {
            let w = weights[row * cols + col];
            let encoded: u8 = match w {
                0 => 0b00,
                1 => 0b01,
                -1 => 0b10,
                _ => panic!("ternary weight must be -1, 0, or +1, got {w}"),
            };
            let byte_idx = row * bytes_per_row + col / 4;
            let bit_pos = (col % 4) * 2;
            packed[byte_idx] |= encoded << bit_pos;
        }
    }
    packed
}

/// Unpack 2-bit packed weights back to ternary {-1, 0, +1}.
pub fn unpack_ternary_weights(packed: &[u8], rows: usize, cols: usize) -> Vec<i8> {
    let bytes_per_row = cols.div_ceil(4);
    assert_eq!(packed.len(), rows * bytes_per_row, "packed length mismatch");
    let mut weights = vec![0i8; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let byte_idx = row * bytes_per_row + col / 4;
            let bit_pos = (col % 4) * 2;
            let bits = (packed[byte_idx] >> bit_pos) & 0x03;
            weights[row * cols + col] = match bits {
                0b00 => 0,
                0b01 => 1,
                0b10 => -1,
                _ => 0, // 0b11 treated as 0 (unused)
            };
        }
    }
    weights
}

// ── CPU reference implementations ───────────────────────────────────────────

/// Reference ternary matmul: `C[m,n] = Σ_k A[m,k] * W[k,n]`
/// where W ∈ {-1, 0, +1}. Uses add/subtract instead of multiply.
///
/// Layout: `activations` is row-major `[m, k]`, `weights` is row-major
/// `[k, n]`, output is row-major `[m, n]`.
pub fn cpu_ternary_matmul(
    activations: &[f32],
    weights: &[i8],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(activations.len(), m * k);
    assert_eq!(weights.len(), k * n);
    let mut output = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let w = weights[p * n + j];
                let a = activations[i * k + p];
                match w {
                    1 => acc += a,
                    -1 => acc -= a,
                    0 => {}
                    _ => {}
                }
            }
            output[i * n + j] = acc;
        }
    }
    output
}

/// Matmul with 2-bit packed ternary weights.
pub fn cpu_packed_ternary_matmul(
    activations: &[f32],
    packed_weights: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let weights = unpack_ternary_weights(packed_weights, k, n);
    cpu_ternary_matmul(activations, &weights, m, n, k)
}

/// 1-bit binary matmul: weights are sign bits packed 8 per byte.
///
/// Bit = 0 → weight = +1, bit = 1 → weight = -1.
/// `C[i,j] = Σ_k A[i,k] * sign(W[k,j])`.
pub fn cpu_binary_matmul(
    activations: &[f32],
    weights: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let bytes_per_row = n.div_ceil(8);
    assert_eq!(weights.len(), k * bytes_per_row);
    assert_eq!(activations.len(), m * k);
    let mut output = vec![0.0f32; m * n];

    for i in 0..m {
        for p in 0..k {
            let a = activations[i * k + p];
            for j in 0..n {
                let byte_idx = p * bytes_per_row + j / 8;
                let bit_pos = j % 8;
                let bit = (weights[byte_idx] >> bit_pos) & 1;
                if bit == 0 {
                    output[i * n + j] += a;
                } else {
                    output[i * n + j] -= a;
                }
            }
        }
    }
    output
}

/// Scaled ternary matmul: output is multiplied by a per-group scale.
pub fn cpu_ternary_matmul_with_scale(
    activations: &[f32],
    weights: &[i8],
    scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut output = cpu_ternary_matmul(activations, weights, m, n, k);
    for v in &mut output {
        *v *= scale;
    }
    output
}

/// Batched ternary matmul: `batch` independent matmuls concatenated.
///
/// `activations` layout: `[batch, m, k]`, `weights` layout: `[k, n]`
/// (shared across batches), output: `[batch, m, n]`.
pub fn cpu_batched_ternary_matmul(
    activations: &[f32],
    weights: &[i8],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(activations.len(), batch * m * k);
    assert_eq!(weights.len(), k * n);
    let mut output = vec![0.0f32; batch * m * n];

    for b in 0..batch {
        let a_off = b * m * k;
        let o_off = b * m * n;
        let batch_out = cpu_ternary_matmul(&activations[a_off..a_off + m * k], weights, m, n, k);
        output[o_off..o_off + m * n].copy_from_slice(&batch_out);
    }
    output
}

/// Ternary matmul with transposed weights: `weights_t` is `[n, k]`
/// (transposed from normal `[k, n]`).
///
/// `C[i,j] = Σ_p A[i,p] * Wt[j,p]`
pub fn cpu_ternary_matmul_transposed(
    activations: &[f32],
    weights_t: &[i8],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(activations.len(), m * k);
    assert_eq!(weights_t.len(), n * k);
    let mut output = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let w = weights_t[j * k + p];
                let a = activations[i * k + p];
                match w {
                    1 => acc += a,
                    -1 => acc -= a,
                    0 => {}
                    _ => {}
                }
            }
            output[i * n + j] = acc;
        }
    }
    output
}

// ── OpenCL kernel source ────────────────────────────────────────────────────

/// OpenCL kernel sources for quantized matmul on Intel Arc GPUs.
pub const QUANTIZED_MATMUL_SRC: &str = r#"
// ─── ternary_matmul ─────────────────────────────────────────────────────
// Basic ternary matmul: weights are i8 {-1, 0, +1}.
// Global work size: (N, M)
__kernel void ternary_matmul(
    __global const float* A,   // [M, K]
    __global const char*  W,   // [K, N]  ternary weights
    __global       float* C,   // [M, N]
    const int M,
    const int N,
    const int K)
{
    const int j = get_global_id(0); // column
    const int i = get_global_id(1); // row
    if (i >= M || j >= N) return;

    float acc = 0.0f;
    for (int p = 0; p < K; ++p) {
        char w = W[p * N + j];
        float a = A[i * K + p];
        // BitNet key insight: select instead of multiply
        acc = select(acc, acc + a, (int)(w ==  1));
        acc = select(acc, acc - a, (int)(w == -1));
    }
    C[i * N + j] = acc;
}

// ─── ternary_matmul_tiled ───────────────────────────────────────────────
// Tiled version with __local memory for activation reuse.
// TILE_M and TILE_N defined via build options.
#ifndef TILE_M
#define TILE_M 8
#endif
#ifndef TILE_N
#define TILE_N 8
#endif
#ifndef TILE_K
#define TILE_K 16
#endif

__kernel void ternary_matmul_tiled(
    __global const float* A,
    __global const char*  W,
    __global       float* C,
    const int M,
    const int N,
    const int K)
{
    const int li = get_local_id(1);
    const int lj = get_local_id(0);
    const int gi = get_group_id(1) * TILE_M + li;
    const int gj = get_group_id(0) * TILE_N + lj;

    __local float tileA[TILE_M][TILE_K];
    __local char  tileW[TILE_K][TILE_N];

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE_K) {
        // Load tiles collaboratively
        if (gi < M && (t + lj) < K)
            tileA[li][lj] = A[gi * K + t + lj];
        else
            tileA[li][lj] = 0.0f;

        if ((t + li) < K && gj < N)
            tileW[li][lj] = W[(t + li) * N + gj];
        else
            tileW[li][lj] = 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Accumulate
        int kmax = min(TILE_K, K - t);
        for (int p = 0; p < kmax; ++p) {
            char w = tileW[p][lj];
            float a = tileA[li][p];
            if (w == 1) acc += a;
            else if (w == -1) acc -= a;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gi < M && gj < N)
        C[gi * N + gj] = acc;
}

// ─── packed_ternary_matmul ──────────────────────────────────────────────
// Weights packed as 2-bit ternary: 00=0, 01=+1, 10=-1.
// packed_weights layout: [K, ceil(N/4)] bytes.
__kernel void packed_ternary_matmul(
    __global const float* A,
    __global const uchar* packed_W, // [K, bytes_per_row]
    __global       float* C,
    const int M,
    const int N,
    const int K,
    const int bytes_per_row)
{
    const int j = get_global_id(0);
    const int i = get_global_id(1);
    if (i >= M || j >= N) return;

    float acc = 0.0f;
    const int byte_col = j / 4;
    const int bit_pos  = (j % 4) * 2;

    for (int p = 0; p < K; ++p) {
        uchar packed = packed_W[p * bytes_per_row + byte_col];
        uchar bits = (packed >> bit_pos) & 0x03;
        float a = A[i * K + p];
        // 00=0 (skip), 01=+1 (add), 10=-1 (sub)
        if (bits == 1) acc += a;
        else if (bits == 2) acc -= a;
    }
    C[i * N + j] = acc;
}

// ─── binary_matmul_popcount ─────────────────────────────────────────────
// 1-bit binary weights: bit=0 → +1, bit=1 → -1.
// Uses XOR+popcount for dot product.
// packed_weights: [K, ceil(N/8)] bytes.
__kernel void binary_matmul_popcount(
    __global const float* A,
    __global const uchar* packed_W,
    __global       float* C,
    const int M,
    const int N,
    const int K,
    const int bytes_per_row)
{
    const int j = get_global_id(0);
    const int i = get_global_id(1);
    if (i >= M || j >= N) return;

    float acc = 0.0f;
    const int byte_col = j / 8;
    const int bit_pos  = j % 8;

    for (int p = 0; p < K; ++p) {
        uchar packed = packed_W[p * bytes_per_row + byte_col];
        uchar bit = (packed >> bit_pos) & 1;
        float a = A[i * K + p];
        // bit=0 → +1, bit=1 → -1
        acc += (1 - 2 * (int)bit) * a;
    }
    C[i * N + j] = acc;
}
"#;

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Packing round-trip ──────────────────────────────────────────────

    #[test]
    fn pack_unpack_roundtrip_4cols() {
        let weights: Vec<i8> = vec![1, -1, 0, 1];
        let packed = pack_ternary_weights(&weights, 4);
        let unpacked = unpack_ternary_weights(&packed, 1, 4);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn pack_unpack_roundtrip_8cols() {
        let weights: Vec<i8> = vec![0, 1, -1, 0, 1, 1, -1, -1];
        let packed = pack_ternary_weights(&weights, 8);
        let unpacked = unpack_ternary_weights(&packed, 1, 8);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn pack_unpack_roundtrip_multi_row() {
        let weights: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 1, -1];
        let packed = pack_ternary_weights(&weights, 4);
        let unpacked = unpack_ternary_weights(&packed, 2, 4);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn pack_unpack_cols_not_multiple_of_4() {
        // 3 cols → 1 byte per row with padding
        let weights: Vec<i8> = vec![1, -1, 0, -1, 0, 1];
        let packed = pack_ternary_weights(&weights, 3);
        let unpacked = unpack_ternary_weights(&packed, 2, 3);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn pack_unpack_cols_5() {
        let weights: Vec<i8> = vec![1, 0, -1, 1, -1];
        let packed = pack_ternary_weights(&weights, 5);
        assert_eq!(packed.len(), 2); // ceil(5/4) = 2 bytes
        let unpacked = unpack_ternary_weights(&packed, 1, 5);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn pack_unpack_cols_1() {
        let weights: Vec<i8> = vec![1, -1, 0];
        let packed = pack_ternary_weights(&weights, 1);
        let unpacked = unpack_ternary_weights(&packed, 3, 1);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn pack_all_zeros() {
        let weights = vec![0i8; 8];
        let packed = pack_ternary_weights(&weights, 4);
        assert!(packed.iter().all(|&b| b == 0));
        let unpacked = unpack_ternary_weights(&packed, 2, 4);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn pack_all_ones() {
        let weights = vec![1i8; 8];
        let packed = pack_ternary_weights(&weights, 4);
        // Each byte: 01_01_01_01 = 0x55
        assert!(packed.iter().all(|&b| b == 0x55));
    }

    #[test]
    fn pack_all_neg_ones() {
        let weights = vec![-1i8; 8];
        let packed = pack_ternary_weights(&weights, 4);
        // Each byte: 10_10_10_10 = 0xAA
        assert!(packed.iter().all(|&b| b == 0xAA));
    }

    // ── Ternary matmul: small sizes ────────────────────────────────────

    #[test]
    fn ternary_matmul_2x2() {
        // A = [[1, 2], [3, 4]], W = [[1, -1], [0, 1]]
        // C[0,0] = 1*1 + 2*0 = 1
        // C[0,1] = 1*(-1) + 2*1 = 1
        // C[1,0] = 3*1 + 4*0 = 3
        // C[1,1] = 3*(-1) + 4*1 = 1
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let w: Vec<i8> = vec![1, -1, 0, 1];
        let c = cpu_ternary_matmul(&a, &w, 2, 2, 2);
        assert_eq!(c, vec![1.0, 1.0, 3.0, 1.0]);
    }

    #[test]
    fn ternary_matmul_4x4() {
        // Identity-like: W = I (4×4 identity in ternary)
        let a: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let w: Vec<i8> = vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        let c = cpu_ternary_matmul(&a, &w, 4, 4, 4);
        assert_eq!(c, a);
    }

    #[test]
    fn ternary_matmul_8x8_identity() {
        let n = 8;
        let a: Vec<f32> = (1..=(n * n) as u32).map(|x| x as f32).collect();
        let mut w = vec![0i8; n * n];
        for i in 0..n {
            w[i * n + i] = 1;
        }
        let c = cpu_ternary_matmul(&a, &w, n, n, n);
        assert_eq!(c, a);
    }

    // ── Special weight patterns ────────────────────────────────────────

    #[test]
    fn all_zero_weights_produce_zero_output() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = vec![0i8; 9]; // k=3, n=3
        let c = cpu_ternary_matmul(&a, &w, 2, 3, 3);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn all_one_weights_produce_row_sums() {
        // A = [[1,2,3],[4,5,6]], W = all 1s (3×2)
        // C[i,j] = sum of row i of A
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = vec![1i8; 6]; // 3×2
        let c = cpu_ternary_matmul(&a, &w, 2, 2, 3);
        assert_eq!(c, vec![6.0, 6.0, 15.0, 15.0]);
    }

    #[test]
    fn all_neg_one_weights_produce_neg_row_sums() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = vec![-1i8; 6];
        let c = cpu_ternary_matmul(&a, &w, 2, 2, 3);
        assert_eq!(c, vec![-6.0, -6.0, -15.0, -15.0]);
    }

    #[test]
    fn mixed_ternary_known_pattern() {
        // A = [[1, 2, 3]], W = [[1], [-1], [1]]
        // C = 1*1 + 2*(-1) + 3*1 = 2
        let a = vec![1.0, 2.0, 3.0];
        let w: Vec<i8> = vec![1, -1, 1];
        let c = cpu_ternary_matmul(&a, &w, 1, 1, 3);
        assert_eq!(c, vec![2.0]);
    }

    #[test]
    fn mixed_ternary_alternating() {
        // A = [[1,1,1,1]], W col = [1,-1,1,-1]
        // dot = 1 - 1 + 1 - 1 = 0
        let a = vec![1.0; 4];
        let w: Vec<i8> = vec![1, -1, 1, -1];
        let c = cpu_ternary_matmul(&a, &w, 1, 1, 4);
        assert_eq!(c, vec![0.0]);
    }

    // ── Packed matmul ──────────────────────────────────────────────────

    #[test]
    fn packed_matches_unpacked_2x2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let w: Vec<i8> = vec![1, -1, 0, 1];
        let packed = pack_ternary_weights(&w, 2);
        let c_ref = cpu_ternary_matmul(&a, &w, 2, 2, 2);
        let c_packed = cpu_packed_ternary_matmul(&a, &packed, 2, 2, 2);
        assert_eq!(c_packed, c_ref);
    }

    #[test]
    fn packed_matches_unpacked_4x4() {
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let w: Vec<i8> = vec![1, 0, -1, 0, -1, 1, 0, 1, 0, -1, 1, -1, 1, 1, -1, 0];
        let packed = pack_ternary_weights(&w, 4);
        let c_ref = cpu_ternary_matmul(&a, &w, 4, 4, 4);
        let c_packed = cpu_packed_ternary_matmul(&a, &packed, 4, 4, 4);
        assert_eq!(c_packed, c_ref);
    }

    #[test]
    fn packed_matches_unpacked_non_multiple_of_4() {
        let a = vec![1.0, 2.0, 3.0];
        let w: Vec<i8> = vec![1, -1, 0, -1, 0, 1, 0, 1, -1];
        let packed = pack_ternary_weights(&w, 3);
        let c_ref = cpu_ternary_matmul(&a, &w, 1, 3, 3);
        let c_packed = cpu_packed_ternary_matmul(&a, &packed, 1, 3, 3);
        assert_eq!(c_packed, c_ref);
    }

    // ── Binary matmul ──────────────────────────────────────────────────

    #[test]
    fn binary_matmul_all_plus_one() {
        // All bits 0 → all weights +1 → row sums
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = vec![0u8; 3]; // 3 rows × 1 byte (8 cols), all +1
        let c = cpu_binary_matmul(&a, &w, 2, 8, 3);
        // Each output = sum of activation row (k=3)
        // Row0: 1+2+3=6 repeated 8 times
        for &v in &c[..8] {
            assert_eq!(v, 6.0);
        }
    }

    #[test]
    fn binary_matmul_all_minus_one() {
        // All bits 1 → all weights -1 → negative row sums
        let a = vec![1.0, 2.0, 3.0];
        let w = vec![0xFFu8; 3]; // 3 rows × 1 byte
        let c = cpu_binary_matmul(&a, &w, 1, 8, 3);
        for &v in &c[..8] {
            assert_eq!(v, -6.0);
        }
    }

    #[test]
    fn binary_matmul_sign_only() {
        // A = [[2, 4]], W col0: bits [0, 1] → [+1, -1]
        // dot = 2*(+1) + 4*(-1) = -2
        let a = vec![2.0, 4.0];
        let w = vec![0b00u8, 0b01u8]; // row0 all +1, row1 bit0=1 → -1
        let c = cpu_binary_matmul(&a, &w, 1, 8, 2);
        // col0: 2*(+1) + 4*(-1) = -2
        assert_eq!(c[0], -2.0);
    }

    // ── Scaled matmul ──────────────────────────────────────────────────

    #[test]
    fn scaled_matmul_doubles_output() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let w: Vec<i8> = vec![1, -1, 0, 1];
        let c_base = cpu_ternary_matmul(&a, &w, 2, 2, 2);
        let c_scaled = cpu_ternary_matmul_with_scale(&a, &w, 2.0, 2, 2, 2);
        for (base, scaled) in c_base.iter().zip(c_scaled.iter()) {
            assert_eq!(*scaled, base * 2.0);
        }
    }

    #[test]
    fn scaled_matmul_zero_scale() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let w: Vec<i8> = vec![1, -1, 0, 1];
        let c = cpu_ternary_matmul_with_scale(&a, &w, 0.0, 2, 2, 2);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn scaled_matmul_negative_scale() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let w: Vec<i8> = vec![1, -1, 0, 1];
        let c_base = cpu_ternary_matmul(&a, &w, 2, 2, 2);
        let c_neg = cpu_ternary_matmul_with_scale(&a, &w, -1.0, 2, 2, 2);
        for (base, neg) in c_base.iter().zip(c_neg.iter()) {
            assert_eq!(*neg, -base);
        }
    }

    // ── Batched matmul ─────────────────────────────────────────────────

    #[test]
    fn batched_single_batch_matches_unbatched() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let w: Vec<i8> = vec![1, -1, 0, 1];
        let c_single = cpu_ternary_matmul(&a, &w, 2, 2, 2);
        let c_batched = cpu_batched_ternary_matmul(&a, &w, 1, 2, 2, 2);
        assert_eq!(c_batched, c_single);
    }

    #[test]
    fn batched_4_batches() {
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let a: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let c = cpu_batched_ternary_matmul(&a, &w, 4, 2, 2, 2);
        assert_eq!(c.len(), 4 * 2 * 2);
        // Each batch uses the same weights
        for b in 0..4 {
            let a_slice = &a[b * 4..(b + 1) * 4];
            let expected = cpu_ternary_matmul(a_slice, &w, 2, 2, 2);
            assert_eq!(&c[b * 4..(b + 1) * 4], &expected[..]);
        }
    }

    #[test]
    fn batched_16_batches() {
        let k = 4;
        let n = 2;
        let m = 1;
        let batch = 16;
        let w: Vec<i8> = vec![1, -1, 0, 1, 1, 0, -1, 1];
        let a: Vec<f32> = (0..(batch * m * k) as u32).map(|x| x as f32 * 0.1).collect();
        let c = cpu_batched_ternary_matmul(&a, &w, batch, m, n, k);
        assert_eq!(c.len(), batch * m * n);
    }

    // ── Transposed ─────────────────────────────────────────────────────

    #[test]
    fn transposed_matches_normal() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Normal weights [k=3, n=2]
        let w: Vec<i8> = vec![1, -1, 0, 1, -1, 0];
        // Transposed weights [n=2, k=3]
        let wt: Vec<i8> = vec![1, 0, -1, -1, 1, 0];
        let c_normal = cpu_ternary_matmul(&a, &w, 2, 2, 3);
        let c_trans = cpu_ternary_matmul_transposed(&a, &wt, 2, 2, 3);
        assert_eq!(c_trans, c_normal);
    }

    #[test]
    fn transposed_identity() {
        let n = 4;
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        // Identity is its own transpose
        let mut w = vec![0i8; n * n];
        for i in 0..n {
            w[i * n + i] = 1;
        }
        let c = cpu_ternary_matmul_transposed(&a, &w, 4, 4, 4);
        assert_eq!(c, a);
    }

    // ── Large matrix ───────────────────────────────────────────────────

    #[test]
    fn large_matrix_128x128() {
        let n = 128;
        let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        // Deterministic ternary weights
        let w: Vec<i8> = (0..n * n)
            .map(|i| match i % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            })
            .collect();
        let c = cpu_ternary_matmul(&a, &w, n, n, n);
        assert_eq!(c.len(), n * n);
        // Verify no NaN/Inf
        assert!(c.iter().all(|v| v.is_finite()));
    }

    // ── Quantized vs full precision ────────────────────────────────────

    #[test]
    fn ternary_matches_f32_matmul() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let w: Vec<i8> = vec![1, 0, -1, -1, 1, 0, 0, -1, 1];
        let c_ternary = cpu_ternary_matmul(&a, &w, 3, 3, 3);

        // Compute with f32 weights
        let wf: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let mut c_f32 = vec![0.0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                for p in 0..3 {
                    c_f32[i * 3 + j] += a[i * 3 + p] * wf[p * 3 + j];
                }
            }
        }
        for (t, f) in c_ternary.iter().zip(c_f32.iter()) {
            assert!((t - f).abs() < 1e-6, "mismatch: ternary={t}, f32={f}");
        }
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn edge_k1_dot_product() {
        // m=2, n=3, k=1: each output is just act * weight
        let a = vec![2.0, 5.0];
        let w: Vec<i8> = vec![1, -1, 0];
        let c = cpu_ternary_matmul(&a, &w, 2, 3, 1);
        assert_eq!(c, vec![2.0, -2.0, 0.0, 5.0, -5.0, 0.0]);
    }

    #[test]
    fn edge_m1_gemv() {
        // m=1, n=4, k=3: GEMV
        let a = vec![1.0, 2.0, 3.0];
        let w: Vec<i8> = vec![1, 0, -1, 1, -1, 1, 0, -1, 0, -1, 1, 0];
        let c = cpu_ternary_matmul(&a, &w, 1, 4, 3);
        // col0: 1*1 + 2*(-1) + 3*0 = -1
        // col1: 1*0 + 2*1 + 3*(-1) = -1
        // col2: 1*(-1) + 2*0 + 3*1 = 2
        // col3: 1*1 + 2*(-1) + 3*0 = -1
        assert_eq!(c, vec![-1.0, -1.0, 2.0, -1.0]);
    }

    #[test]
    fn edge_n1_column_sum() {
        // m=2, n=1, k=4
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let w: Vec<i8> = vec![1, -1, 1, -1];
        let c = cpu_ternary_matmul(&a, &w, 2, 1, 4);
        // row0: 1-2+3-4 = -2
        // row1: 5-6+7-8 = -2
        assert_eq!(c, vec![-2.0, -2.0]);
    }

    // ── Property-based ─────────────────────────────────────────────────

    #[test]
    fn identity_like_pattern_produces_input() {
        // W = identity → output = input
        let m = 6;
        let n = 6;
        let a: Vec<f32> = (1..=(m * n) as u32).map(|x| x as f32).collect();
        let mut w = vec![0i8; n * n];
        for i in 0..n {
            w[i * n + i] = 1;
        }
        let c = cpu_ternary_matmul(&a, &w, m, n, n);
        assert_eq!(c, a);
    }

    #[test]
    fn output_magnitude_bounded_by_k_times_max_act() {
        let m = 4;
        let n = 4;
        let k = 8;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32 - 2.0) * 3.0).collect();
        let w: Vec<i8> = (0..k * n)
            .map(|i| match i % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            })
            .collect();
        let c = cpu_ternary_matmul(&a, &w, m, n, k);
        let max_act = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let bound = k as f32 * max_act;
        for &v in &c {
            assert!(v.abs() <= bound + 1e-6, "|{v}| > bound {bound}");
        }
    }

    #[test]
    fn ternary_matmul_commutative_with_negation() {
        // Negating weights negates output
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let w: Vec<i8> = vec![1, -1, 0, 1];
        let w_neg: Vec<i8> = w.iter().map(|&v| -v).collect();
        let c = cpu_ternary_matmul(&a, &w, 2, 2, 2);
        let c_neg = cpu_ternary_matmul(&a, &w_neg, 2, 2, 2);
        for (pos, neg) in c.iter().zip(c_neg.iter()) {
            assert!((pos + neg).abs() < 1e-6);
        }
    }

    // ── Type construction ──────────────────────────────────────────────

    #[test]
    fn quantized_weight_construction() {
        let data = pack_ternary_weights(&[1, -1, 0, 1], 4);
        let qw = QuantizedWeight { data, rows: 1, cols: 4, encoding: WeightEncoding::Ternary2Bit };
        assert_eq!(qw.rows, 1);
        assert_eq!(qw.cols, 4);
        assert_eq!(qw.encoding, WeightEncoding::Ternary2Bit);
    }

    #[test]
    fn config_default() {
        let cfg = QuantizedMatMulConfig::default();
        assert_eq!(cfg.tile_m, 8);
        assert_eq!(cfg.tile_n, 8);
        assert!(cfg.use_popcount);
        assert_eq!(cfg.accumulator, AccumulatorType::Float32);
    }

    #[test]
    fn error_display() {
        let e = QuantizedMatMulError::DimensionMismatch { expected_k: 256, got_k: 128 };
        let s = format!("{e}");
        assert!(s.contains("256"));
        assert!(s.contains("128"));
    }

    #[test]
    fn weight_encoding_display() {
        assert_eq!(format!("{}", WeightEncoding::Binary1Bit), "Binary1Bit");
        assert_eq!(format!("{}", WeightEncoding::Ternary2Bit), "Ternary2Bit");
        assert_eq!(format!("{}", WeightEncoding::I2S), "I2_S");
    }

    #[test]
    fn error_overflow_variant() {
        let e = QuantizedMatMulError::Overflow;
        assert_eq!(format!("{e}"), "accumulator overflow");
    }

    #[test]
    fn error_invalid_encoding() {
        let e = QuantizedMatMulError::InvalidEncoding { reason: "bad bits".into() };
        assert!(format!("{e}").contains("bad bits"));
    }

    // ── OpenCL source ──────────────────────────────────────────────────

    #[test]
    fn opencl_source_contains_kernels() {
        assert!(QUANTIZED_MATMUL_SRC.contains("ternary_matmul"));
        assert!(QUANTIZED_MATMUL_SRC.contains("ternary_matmul_tiled"));
        assert!(QUANTIZED_MATMUL_SRC.contains("packed_ternary_matmul"));
        assert!(QUANTIZED_MATMUL_SRC.contains("binary_matmul_popcount"));
    }

    #[test]
    fn opencl_source_uses_select_not_multiply() {
        // The ternary kernel should use select or if/else, not multiply
        let basic_kernel =
            QUANTIZED_MATMUL_SRC.split("__kernel void ternary_matmul_tiled").next().unwrap();
        assert!(
            basic_kernel.contains("select")
                || basic_kernel.contains("acc +")
                || basic_kernel.contains("acc -"),
            "ternary kernel should use add/subtract, not multiply"
        );
    }

    // ── Additional edge cases ──────────────────────────────────────────

    #[test]
    fn packed_large_roundtrip() {
        let n = 128;
        let weights: Vec<i8> = (0..n * n)
            .map(|i| match i % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            })
            .collect();
        let packed = pack_ternary_weights(&weights, n);
        let unpacked = unpack_ternary_weights(&packed, n, n);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn scaled_matmul_unit_scale_matches_unscaled() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let w: Vec<i8> = vec![1, -1, 0, 1];
        let c_base = cpu_ternary_matmul(&a, &w, 2, 2, 2);
        let c_scaled = cpu_ternary_matmul_with_scale(&a, &w, 1.0, 2, 2, 2);
        assert_eq!(c_scaled, c_base);
    }

    #[test]
    fn binary_matmul_single_element() {
        // 1×1 binary: A=[3.0], W=[0x00] (bit0=0 → +1)
        let a = vec![3.0];
        let w = vec![0u8]; // 1 row, 1 byte (8 cols)
        let c = cpu_binary_matmul(&a, &w, 1, 8, 1);
        // All cols: 3.0 * (+1) = 3.0
        assert!(c[..8].iter().all(|&v| v == 3.0));
    }

    #[test]
    fn batched_zero_activations() {
        let a = vec![0.0f32; 8];
        let w: Vec<i8> = vec![1, -1, 0, 1];
        let c = cpu_batched_ternary_matmul(&a, &w, 2, 2, 2, 2);
        assert!(c.iter().all(|&v| v == 0.0));
    }
}
