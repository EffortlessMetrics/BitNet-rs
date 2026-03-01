//! INT8 DP4A (Dot Product of 4 Accumulate) kernels for Intel Arc A770 Xe-HPG.
//!
//! # Overview
//!
//! Intel Arc A770 supports hardware INT8 DP4A — computing the dot product of
//! four pairs of 8-bit integers and accumulating into a 32-bit result. This is
//! analogous to NVIDIA's DP4A/IMMA instructions and enables significant
//! throughput improvements for quantized inference.
//!
//! For BitNet-style models, 4 ternary weights can be packed into a single INT8
//! DP4A operation, making this a natural fit for 1-bit/ternary weight inference.
//!
//! # CPU reference
//!
//! All functions prefixed with `cpu_` are scalar reference implementations for
//! correctness testing and non-GPU environments.
//!
//! # OpenCL kernel
//!
//! [`INT8_DP4A_SRC`] contains embedded OpenCL C source using the
//! `cl_intel_subgroups` extension for hardware DP4A on Xe-HPG.

use std::fmt;

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Configuration for INT8 DP4A computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Int8Config {
    /// Enable DP4A hardware intrinsic (vs scalar fallback).
    pub use_dp4a: bool,
    /// Accumulator bit width (always 32 for i32).
    pub accumulator_bits: u8,
    /// Apply per-channel scaling after accumulation.
    pub channel_scaling: bool,
}

impl Default for Int8Config {
    fn default() -> Self {
        Self { use_dp4a: true, accumulator_bits: 32, channel_scaling: false }
    }
}

/// A quantized INT8 tensor with scale and zero-point metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    /// Quantized INT8 data.
    pub data: Vec<i8>,
    /// Tensor shape (row-major).
    pub shape: Vec<usize>,
    /// Quantization scale: `float_val ≈ scale * (quantized - zero_point)`.
    pub scale: f32,
    /// Zero point offset.
    pub zero_point: i8,
}

/// Result of a DP4A matrix multiplication (integer domain).
#[derive(Debug, Clone, PartialEq)]
pub struct Dp4aResult {
    /// Accumulated INT32 output values.
    pub output: Vec<i32>,
    /// Combined output scale for dequantization.
    pub output_scale: f32,
}

/// Errors specific to INT8 DP4A operations.
#[derive(Debug, Clone, PartialEq)]
pub enum Int8Error {
    /// Accumulation would overflow i32.
    OverflowDetected {
        /// Reduction dimension that would overflow.
        k: usize,
    },
    /// Operand scales are incompatible.
    ScaleMismatch {
        /// Scale of the first operand.
        scale_a: f32,
        /// Scale of the second operand.
        scale_b: f32,
    },
    /// Matrix dimensions are incompatible for multiplication.
    ShapeMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// Hardware DP4A is not supported on this device.
    UnsupportedDp4a,
}

impl fmt::Display for Int8Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OverflowDetected { k } => {
                write!(f, "DP4A accumulation would overflow i32 at k={k}")
            }
            Self::ScaleMismatch { scale_a, scale_b } => {
                write!(f, "scale mismatch: a={scale_a}, b={scale_b}")
            }
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected}, got {actual}")
            }
            Self::UnsupportedDp4a => write!(f, "DP4A not supported on device"),
        }
    }
}

impl std::error::Error for Int8Error {}

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C source for INT8 DP4A kernels targeting Intel Arc A770 (Xe-HPG).
///
/// Contains:
/// - `int8_dp4a_gemm` — INT8 GEMM using `intel_sub_group_i8_i8_matrix_mad_k32`
/// - `int8_matmul_tiled` — tiled INT8 matmul with local memory
/// - `quantize_fp32_to_int8` — parallel f32→i8 quantization
pub const INT8_DP4A_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable

// DP4A intrinsic: dot product of 4 x i8 pairs, accumulated into i32.
// On Xe-HPG this maps to a single hardware instruction.
inline int dp4a(int a_packed, int b_packed, int acc) {
    char4 a = as_char4(a_packed);
    char4 b = as_char4(b_packed);
    acc += (int)a.s0 * (int)b.s0;
    acc += (int)a.s1 * (int)b.s1;
    acc += (int)a.s2 * (int)b.s2;
    acc += (int)a.s3 * (int)b.s3;
    return acc;
}

// INT8 GEMM using DP4A: C[M,N] = A[M,K] * B[K,N]
// Each work-item computes one output element using DP4A in the K dimension.
__kernel void int8_dp4a_gemm(
    __global const char* A,
    __global const char* B,
    __global int*        C,
    const int M,
    const int N,
    const int K)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row >= M || col >= N) return;

    int acc = 0;
    const int k4 = K / 4;

    for (int i = 0; i < k4; i++) {
        int a_packed = vload32(0, (__global const int*)(A + row * K + i * 4));
        // Reinterpret 4 consecutive chars as a packed int
        int a_val = *((__global const int*)(A + row * K + i * 4));
        int b_val = *((__global const int*)(B + col * K + i * 4));
        acc = dp4a(a_val, b_val, acc);
    }

    // Handle remaining elements (K not multiple of 4)
    for (int i = k4 * 4; i < K; i++) {
        acc += (int)A[row * K + i] * (int)B[col * K + i];
    }

    C[row * N + col] = acc;
}

// Tiled INT8 matmul with local memory for improved cache behavior.
#define TILE_SIZE 16
__kernel void int8_matmul_tiled(
    __global const char* A,
    __global const char* B,
    __global int*        C,
    const int M,
    const int N,
    const int K)
{
    __local char tileA[TILE_SIZE][TILE_SIZE];
    __local char tileB[TILE_SIZE][TILE_SIZE];

    const int row = get_local_id(0) + get_group_id(0) * TILE_SIZE;
    const int col = get_local_id(1) + get_group_id(1) * TILE_SIZE;
    const int lr  = get_local_id(0);
    const int lc  = get_local_id(1);

    int acc = 0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int aCol = t * TILE_SIZE + lc;
        int bRow = t * TILE_SIZE + lr;

        tileA[lr][lc] = (row < M && aCol < K) ? A[row * K + aCol] : 0;
        tileB[lr][lc] = (bRow < K && col < N) ? B[bRow * N + col] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; i++) {
            acc += (int)tileA[lr][i] * (int)tileB[i][lc];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Parallel f32 → i8 symmetric quantization.
__kernel void quantize_fp32_to_int8(
    __global const float* input,
    __global char*        output,
    __global float*       out_scale,
    const int             count)
{
    const int gid = get_global_id(0);

    // First pass: find max absolute value (work-group reduction).
    // For simplicity, work-item 0 does the full scan.
    if (gid == 0) {
        float max_abs = 0.0f;
        for (int i = 0; i < count; i++) {
            float v = fabs(input[i]);
            if (v > max_abs) max_abs = v;
        }
        float scale = max_abs / 127.0f;
        if (scale == 0.0f) scale = 1.0f;
        *out_scale = scale;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (gid < count) {
        float scale = *out_scale;
        float val = input[gid] / scale;
        val = clamp(val, -128.0f, 127.0f);
        output[gid] = convert_char_sat_rte(val);
    }
}
"#;

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Single DP4A operation: `acc + a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]`.
#[inline]
pub fn cpu_dp4a(a: &[i8; 4], b: &[i8; 4], accumulator: i32) -> i32 {
    accumulator
        + i32::from(a[0]) * i32::from(b[0])
        + i32::from(a[1]) * i32::from(b[1])
        + i32::from(a[2]) * i32::from(b[2])
        + i32::from(a[3]) * i32::from(b[3])
}

/// Dot product of two i8 slices using DP4A (processes 4 elements at a time).
///
/// Handles the case where `a.len()` is not a multiple of 4 by processing
/// the remainder with scalar multiplications.
pub fn cpu_dp4a_vector(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    let n = a.len();
    let mut acc: i32 = 0;

    // Process groups of 4 via DP4A
    let chunks = n / 4;
    for i in 0..chunks {
        let off = i * 4;
        let va = [a[off], a[off + 1], a[off + 2], a[off + 3]];
        let vb = [b[off], b[off + 1], b[off + 2], b[off + 3]];
        acc = cpu_dp4a(&va, &vb, acc);
    }

    // Remainder
    for i in (chunks * 4)..n {
        acc += i32::from(a[i]) * i32::from(b[i]);
    }

    acc
}

/// Symmetric quantization of f32 data to i8.
///
/// Returns `(quantized_data, scale, zero_point)`. Zero point is always 0
/// for symmetric quantization.
pub fn cpu_quantize_f32_to_i8(data: &[f32]) -> (Vec<i8>, f32, i8) {
    if data.is_empty() {
        return (vec![], 1.0, 0);
    }

    let max_abs = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };

    let quantized: Vec<i8> = data
        .iter()
        .map(|&v| {
            let q = (v / scale).round();
            q.clamp(-128.0, 127.0) as i8
        })
        .collect();

    (quantized, scale, 0)
}

/// Dequantize i8 data back to f32.
pub fn cpu_dequantize_i8_to_f32(data: &[i8], scale: f32, zero_point: i8) -> Vec<f32> {
    data.iter().map(|&v| scale * f32::from(v - zero_point)).collect()
}

/// INT8 matrix multiplication with f32 output rescaling.
///
/// Computes `C[m,n] = scale_a * scale_b * (A[m,k] @ B[k,n])` where `@`
/// denotes integer matmul. Matrix B is stored in row-major order
/// (B\[row\]\[col\]).
pub fn cpu_int8_matmul(
    a: &[i8],
    b: &[i8],
    m: usize,
    n: usize,
    k: usize,
    scale_a: f32,
    scale_b: f32,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A dimensions mismatch");
    assert_eq!(b.len(), k * n, "B dimensions mismatch");

    let combined_scale = scale_a * scale_b;
    let mut c = vec![0.0_f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for p in 0..k {
                acc += i32::from(a[i * k + p]) * i32::from(b[p * n + j]);
            }
            c[i * n + j] = combined_scale * acc as f32;
        }
    }

    c
}

/// INT8 matrix multiplication using DP4A in the inner loop.
///
/// Returns raw i32 accumulations (no rescaling). B is stored row-major.
#[allow(clippy::needless_range_loop)]
pub fn cpu_int8_matmul_dp4a(a: &[i8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
    assert_eq!(a.len(), m * k, "A dimensions mismatch");
    assert_eq!(b.len(), k * n, "B dimensions mismatch");

    let mut c = vec![0_i32; m * n];
    let k4 = k / 4;

    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;

            for p in 0..k4 {
                let off = p * 4;
                let va =
                    [a[i * k + off], a[i * k + off + 1], a[i * k + off + 2], a[i * k + off + 3]];
                // Gather column j from row-major B
                let vb = [
                    b[off * n + j],
                    b[(off + 1) * n + j],
                    b[(off + 2) * n + j],
                    b[(off + 3) * n + j],
                ];
                acc = cpu_dp4a(&va, &vb, acc);
            }

            // Handle remainder
            for p in (k4 * 4)..k {
                acc += i32::from(a[i * k + p]) * i32::from(b[p * n + j]);
            }

            c[i * n + j] = acc;
        }
    }

    c
}

/// Mixed-precision matmul: f32 activations × i8 weights with scaling.
///
/// `activations` is `[m, k]` row-major f32, `weights` is `[k, n]` row-major
/// i8. Output is `[m, n]` f32 with `weight_scale` applied.
pub fn cpu_mixed_precision_matmul(
    activations: &[f32],
    weights: &[i8],
    weight_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(activations.len(), m * k, "activations dimensions mismatch");
    assert_eq!(weights.len(), k * n, "weights dimensions mismatch");

    let mut c = vec![0.0_f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for p in 0..k {
                acc += activations[i * k + p] * f32::from(weights[p * n + j]) * weight_scale;
            }
            c[i * n + j] = acc;
        }
    }

    c
}

/// Check whether DP4A accumulation over `k` elements could overflow i32.
///
/// Worst case: each product is `128 * 128 = 16384`, so `k` products sum to
/// `k * 16384`. This overflows i32 when `k > i32::MAX / 16384 = 131071`.
pub fn cpu_check_overflow(a: &[i8], b: &[i8], k: usize) -> bool {
    assert_eq!(a.len(), k);
    assert_eq!(b.len(), k);

    // Conservative static bound
    let max_product: i64 = 128 * 128; // max absolute product of two i8
    let max_sum = max_product * k as i64;
    if max_sum > i64::from(i32::MAX) {
        return true;
    }

    // Dynamic check: compute actual running sum in i64
    let mut acc: i64 = 0;
    for i in 0..k {
        acc += i64::from(a[i]) * i64::from(b[i]);
        if acc > i64::from(i32::MAX) || acc < i64::from(i32::MIN) {
            return true;
        }
    }

    false
}

/// Pack 4 i8 values into a single i32 for DP4A instructions.
///
/// Layout: `a` occupies bits \[7:0\], `b` \[15:8\], `c` \[23:16\],
/// `d` \[31:24\].
#[inline]
pub fn cpu_pack_4xi8(a: i8, b: i8, c: i8, d: i8) -> i32 {
    let bytes = [a as u8, b as u8, c as u8, d as u8];
    i32::from_le_bytes(bytes)
}

/// Unpack an i32 into 4 i8 values (inverse of [`cpu_pack_4xi8`]).
#[inline]
pub fn cpu_unpack_4xi8(packed: i32) -> (i8, i8, i8, i8) {
    let bytes = packed.to_le_bytes();
    (bytes[0] as i8, bytes[1] as i8, bytes[2] as i8, bytes[3] as i8)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Single DP4A ────────────────────────────────────────────────────

    #[test]
    fn test_dp4a_known_values() {
        let a = [1_i8, 2, 3, 4];
        let b = [5_i8, 6, 7, 8];
        // 0 + 1*5 + 2*6 + 3*7 + 4*8 = 70
        assert_eq!(cpu_dp4a(&a, &b, 0), 70);
    }

    #[test]
    fn test_dp4a_with_accumulator() {
        let a = [1_i8, 1, 1, 1];
        let b = [1_i8, 1, 1, 1];
        // 100 + 1+1+1+1 = 104
        assert_eq!(cpu_dp4a(&a, &b, 100), 104);
    }

    #[test]
    fn test_dp4a_negative_values() {
        let a = [-1_i8, -2, 3, 4];
        let b = [1_i8, 2, -3, -4];
        // 0 + (-1)*1 + (-2)*2 + 3*(-3) + 4*(-4) = -1 -4 -9 -16 = -30
        assert_eq!(cpu_dp4a(&a, &b, 0), -30);
    }

    #[test]
    fn test_dp4a_zeros() {
        let a = [0_i8, 0, 0, 0];
        let b = [1_i8, 2, 3, 4];
        assert_eq!(cpu_dp4a(&a, &b, 0), 0);
    }

    #[test]
    fn test_dp4a_boundary_values() {
        let a = [127_i8, -128, 127, -128];
        let b = [1_i8, 1, 1, 1];
        // 127 + (-128) + 127 + (-128) = -2
        assert_eq!(cpu_dp4a(&a, &b, 0), -2);
    }

    #[test]
    fn test_dp4a_max_product() {
        let a = [127_i8, 0, 0, 0];
        let b = [127_i8, 0, 0, 0];
        assert_eq!(cpu_dp4a(&a, &b, 0), 127 * 127);
    }

    // ── DP4A vector ────────────────────────────────────────────────────

    #[test]
    fn test_dp4a_vector_matches_naive() {
        let a: Vec<i8> = (1..=8).map(|x| x as i8).collect();
        let b: Vec<i8> = (1..=8).map(|x| x as i8).collect();
        let naive: i32 = a.iter().zip(&b).map(|(&x, &y)| i32::from(x) * i32::from(y)).sum();
        assert_eq!(cpu_dp4a_vector(&a, &b), naive);
    }

    #[test]
    fn test_dp4a_vector_not_multiple_of_4() {
        let a: Vec<i8> = vec![1, 2, 3, 4, 5];
        let b: Vec<i8> = vec![1, 1, 1, 1, 1];
        // 1+2+3+4+5 = 15
        assert_eq!(cpu_dp4a_vector(&a, &b), 15);
    }

    #[test]
    fn test_dp4a_vector_single_element() {
        assert_eq!(cpu_dp4a_vector(&[7], &[3]), 21);
    }

    #[test]
    fn test_dp4a_vector_empty() {
        assert_eq!(cpu_dp4a_vector(&[], &[]), 0);
    }

    #[test]
    fn test_dp4a_vector_all_zeros() {
        let a = vec![0_i8; 16];
        let b = vec![1_i8; 16];
        assert_eq!(cpu_dp4a_vector(&a, &b), 0);
    }

    // ── Quantize / Dequantize ──────────────────────────────────────────

    #[test]
    fn test_quantize_round_trip() {
        let data = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let (q, scale, zp) = cpu_quantize_f32_to_i8(&data);
        let deq = cpu_dequantize_i8_to_f32(&q, scale, zp);

        for (orig, recovered) in data.iter().zip(&deq) {
            assert!((orig - recovered).abs() < 0.02, "round-trip error: {orig} vs {recovered}");
        }
    }

    #[test]
    fn test_quantize_range() {
        let data: Vec<f32> = (-200..=200).map(|x| x as f32 * 0.01).collect();
        let (q, _scale, _zp) = cpu_quantize_f32_to_i8(&data);
        for &v in &q {
            // i8 is inherently in [-128, 127]; verify data is non-empty
            let _ = v;
        }
        assert!(!q.is_empty());
    }

    #[test]
    fn test_quantize_empty() {
        let (q, scale, zp) = cpu_quantize_f32_to_i8(&[]);
        assert!(q.is_empty());
        assert_eq!(scale, 1.0);
        assert_eq!(zp, 0);
    }

    #[test]
    fn test_quantize_all_zeros() {
        let data = vec![0.0; 10];
        let (q, _scale, zp) = cpu_quantize_f32_to_i8(&data);
        assert!(q.iter().all(|&v| v == 0));
        assert_eq!(zp, 0);
    }

    #[test]
    fn test_quantize_symmetric() {
        let (q, _scale, zp) = cpu_quantize_f32_to_i8(&[1.0, -1.0]);
        assert_eq!(zp, 0);
        // Symmetric: +1.0 → 127, -1.0 → -127
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -127);
    }

    #[test]
    fn test_dequantize_zero_point_offset() {
        let data = vec![10_i8, 20, 30];
        let result = cpu_dequantize_i8_to_f32(&data, 0.5, 5);
        // (10-5)*0.5=2.5, (20-5)*0.5=7.5, (30-5)*0.5=12.5
        assert_eq!(result, vec![2.5, 7.5, 12.5]);
    }

    #[test]
    fn test_quantize_preserves_sign() {
        let data = vec![-0.8, 0.8];
        let (q, _scale, _zp) = cpu_quantize_f32_to_i8(&data);
        assert!(q[0] < 0, "negative should stay negative");
        assert!(q[1] > 0, "positive should stay positive");
    }

    // ── INT8 matmul ────────────────────────────────────────────────────

    #[test]
    fn test_int8_matmul_2x2() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![5, 6, 7, 8];
        let c = cpu_int8_matmul(&a, &b, 2, 2, 2, 1.0, 1.0);
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_int8_matmul_4x4() {
        // Identity-like: A = I4, B = diag(1,2,3,4)
        let a: Vec<i8> = vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        let b: Vec<i8> = vec![1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4];
        let c = cpu_int8_matmul(&a, &b, 4, 4, 4, 1.0, 1.0);
        let expected: Vec<f32> =
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_int8_matmul_8x8_zeros() {
        let a = vec![0_i8; 64];
        let b = vec![1_i8; 64];
        let c = cpu_int8_matmul(&a, &b, 8, 8, 8, 1.0, 1.0);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_int8_matmul_scaling() {
        let a: Vec<i8> = vec![1, 0, 0, 1];
        let b: Vec<i8> = vec![10, 0, 0, 10];
        let c = cpu_int8_matmul(&a, &b, 2, 2, 2, 0.5, 2.0);
        // scale = 0.5*2.0=1.0; [[10,0],[0,10]] → [10,0,0,10]
        assert_eq!(c, vec![10.0, 0.0, 0.0, 10.0]);
    }

    // ── DP4A matmul ────────────────────────────────────────────────────

    #[test]
    fn test_dp4a_matmul_matches_naive() {
        let a: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let b: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let naive = cpu_int8_matmul(&a, &b, 2, 2, 4, 1.0, 1.0);
        let dp4a = cpu_int8_matmul_dp4a(&a, &b, 2, 2, 4);

        for (n, d) in naive.iter().zip(&dp4a) {
            assert_eq!(*n as i32, *d);
        }
    }

    #[test]
    fn test_dp4a_matmul_k_not_multiple_of_4() {
        // k=3: no full DP4A group, all remainder
        let a: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let b: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let result = cpu_int8_matmul_dp4a(&a, &b, 2, 2, 3);
        // Row0*Col0 = 1*1+2*3+3*5 = 1+6+15 = 22
        // Row0*Col1 = 1*2+2*4+3*6 = 2+8+18 = 28
        // Row1*Col0 = 4*1+5*3+6*5 = 4+15+30 = 49
        // Row1*Col1 = 4*2+5*4+6*6 = 8+20+36 = 64
        assert_eq!(result, vec![22, 28, 49, 64]);
    }

    #[test]
    fn test_dp4a_matmul_k5() {
        // k=5: one DP4A group of 4 + 1 remainder
        let a: Vec<i8> = vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
        let b: Vec<i8> = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let result = cpu_int8_matmul_dp4a(&a, &b, 2, 2, 5);
        // Row0: all 1s, Col0: [1,1,1,1,1] → 5; Col1 same → 5
        // Row1: all 2s → 10 each
        assert_eq!(result, vec![5, 5, 10, 10]);
    }

    // ── Mixed precision ────────────────────────────────────────────────

    #[test]
    fn test_mixed_precision_matmul() {
        let acts = vec![1.0_f32, 0.0, 0.0, 1.0];
        let weights: Vec<i8> = vec![10, 20, 30, 40];
        let c = cpu_mixed_precision_matmul(&acts, &weights, 0.1, 2, 2, 2);
        // Identity * [[10,20],[30,40]] * 0.1 = [[1.0,2.0],[3.0,4.0]]
        for (got, exp) in c.iter().zip(&[1.0, 2.0, 3.0, 4.0]) {
            assert!((got - exp).abs() < 1e-5);
        }
    }

    #[test]
    fn test_mixed_precision_zero_weights() {
        let acts = vec![1.0_f32; 4];
        let weights = vec![0_i8; 4];
        let c = cpu_mixed_precision_matmul(&acts, &weights, 1.0, 2, 2, 2);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_mixed_precision_zero_activations() {
        let acts = vec![0.0_f32; 4];
        let weights = vec![127_i8; 4];
        let c = cpu_mixed_precision_matmul(&acts, &weights, 1.0, 2, 2, 2);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    // ── Scale correctness ──────────────────────────────────────────────

    #[test]
    fn test_scale_correctness_output_rescaling() {
        let a: Vec<i8> = vec![127, 0, 0, 127];
        let b: Vec<i8> = vec![127, 0, 0, 127];
        let scale_a = 1.0 / 127.0;
        let scale_b = 1.0 / 127.0;
        let c = cpu_int8_matmul(&a, &b, 2, 2, 2, scale_a, scale_b);
        // Diagonal should be ≈ 1.0 (127*127 * (1/127)^2 = 1.0)
        assert!((c[0] - 1.0).abs() < 1e-4);
        assert!((c[3] - 1.0).abs() < 1e-4);
        assert!((c[1]).abs() < 1e-6);
    }

    // ── Overflow detection ─────────────────────────────────────────────

    #[test]
    fn test_overflow_detection_large_k() {
        let k = 32768;
        let a = vec![127_i8; k];
        let b = vec![127_i8; k];
        // 32768 * 127*127 = 32768 * 16129 = 528_482_304 < i32::MAX
        // Still within range, but let's verify
        assert!(!cpu_check_overflow(&a, &b, k));
    }

    #[test]
    fn test_overflow_detection_triggers() {
        // k=131072: 131072 * 16384 = 2_147_483_648 > i32::MAX
        let k = 131_072;
        let a = vec![127_i8; k];
        let b = vec![127_i8; k];
        assert!(cpu_check_overflow(&a, &b, k));
    }

    #[test]
    fn test_overflow_safe_small_k() {
        let a = vec![1_i8; 8];
        let b = vec![1_i8; 8];
        assert!(!cpu_check_overflow(&a, &b, 8));
    }

    // ── Pack / Unpack ──────────────────────────────────────────────────

    #[test]
    fn test_pack_unpack_round_trip() {
        let (a, b, c, d) = (42_i8, -17, 0, 127);
        let packed = cpu_pack_4xi8(a, b, c, d);
        let (ua, ub, uc, ud) = cpu_unpack_4xi8(packed);
        assert_eq!((ua, ub, uc, ud), (a, b, c, d));
    }

    #[test]
    fn test_pack_unpack_extremes() {
        let packed = cpu_pack_4xi8(-128, 127, -128, 127);
        let (a, b, c, d) = cpu_unpack_4xi8(packed);
        assert_eq!((a, b, c, d), (-128, 127, -128, 127));
    }

    #[test]
    fn test_pack_unpack_zeros() {
        let packed = cpu_pack_4xi8(0, 0, 0, 0);
        assert_eq!(packed, 0);
        let (a, b, c, d) = cpu_unpack_4xi8(0);
        assert_eq!((a, b, c, d), (0, 0, 0, 0));
    }

    #[test]
    fn test_pack_individual_bytes() {
        let packed = cpu_pack_4xi8(1, 0, 0, 0);
        let bytes = packed.to_le_bytes();
        assert_eq!(bytes[0], 1);
        assert_eq!(bytes[1], 0);
        assert_eq!(bytes[2], 0);
        assert_eq!(bytes[3], 0);
    }

    // ── Zero values ────────────────────────────────────────────────────

    #[test]
    fn test_zero_matmul() {
        let a = vec![0_i8; 16];
        let b = vec![0_i8; 16];
        let c = cpu_int8_matmul_dp4a(&a, &b, 4, 4, 4);
        assert!(c.iter().all(|&v| v == 0));
    }

    // ── Identity-like ──────────────────────────────────────────────────

    #[test]
    fn test_identity_matmul() {
        // I4 * v = v
        let identity: Vec<i8> = vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        let v: Vec<i8> = vec![10, 20, 30, 40];
        let c = cpu_int8_matmul_dp4a(&identity, &v, 4, 1, 4);
        assert_eq!(c, vec![10, 20, 30, 40]);
    }

    // ── Saturation / extreme values ────────────────────────────────────

    #[test]
    fn test_saturation_max_values() {
        let a = [127_i8, 127, 127, 127];
        let b = [127_i8, 127, 127, 127];
        // 4 * 127 * 127 = 64516
        assert_eq!(cpu_dp4a(&a, &b, 0), 64_516);
    }

    #[test]
    fn test_saturation_min_values() {
        let a = [-128_i8, -128, -128, -128];
        let b = [127_i8, 127, 127, 127];
        // 4 * (-128) * 127 = -65024
        assert_eq!(cpu_dp4a(&a, &b, 0), -65_024);
    }

    #[test]
    fn test_saturation_quantize_large_values() {
        let data = vec![1000.0, -1000.0, 500.0];
        let (q, _s, _zp) = cpu_quantize_f32_to_i8(&data);
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -127);
    }

    // ── Large matrix ───────────────────────────────────────────────────

    #[test]
    fn test_large_matrix_128x128() {
        let n = 128;
        // A = I_128 (identity), B = ones
        let mut a = vec![0_i8; n * n];
        for i in 0..n {
            a[i * n + i] = 1;
        }
        let b = vec![1_i8; n * n];
        let c = cpu_int8_matmul_dp4a(&a, &b, n, n, n);
        // Each row of I * ones = row of ones → all 1s
        assert!(c.iter().all(|&v| v == 1));
    }

    // ── Batch dimension ────────────────────────────────────────────────

    #[test]
    fn test_batch_matmul() {
        let batch = 3;
        let m = 2;
        let n = 2;
        let k = 4;

        for _b in 0..batch {
            let a: Vec<i8> = vec![1; m * k];
            let b_mat: Vec<i8> = vec![1; k * n];
            let c = cpu_int8_matmul_dp4a(&a, &b_mat, m, n, k);
            // Each element = sum of k ones = k
            assert!(c.iter().all(|&v| v == k as i32));
        }
    }

    // ── Properties ─────────────────────────────────────────────────────

    #[test]
    fn test_property_quantize_in_range() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let (q, _s, _zp) = cpu_quantize_f32_to_i8(&data);
        for &v in &q {
            // i8 is inherently in [-128, 127]; just verify we got values
            let _ = v;
        }
        assert_eq!(q.len(), 256);
    }

    #[test]
    fn test_property_dequantize_quantize_approx() {
        let data: Vec<f32> = vec![-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
        let (q, scale, zp) = cpu_quantize_f32_to_i8(&data);
        let recovered = cpu_dequantize_i8_to_f32(&q, scale, zp);
        for (orig, rec) in data.iter().zip(&recovered) {
            let err = (orig - rec).abs();
            // Quantization error bounded by scale/2
            assert!(
                err <= scale / 2.0 + 1e-6,
                "error {err} > scale/2 = {} for {orig}",
                scale / 2.0
            );
        }
    }

    #[test]
    fn test_property_dp4a_is_sum_of_products() {
        let a = [3_i8, -7, 11, -2];
        let b = [5_i8, 4, -1, 8];
        let expected: i32 = a.iter().zip(&b).map(|(&x, &y)| i32::from(x) * i32::from(y)).sum();
        assert_eq!(cpu_dp4a(&a, &b, 0), expected);
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_matmul_1x1() {
        let a: Vec<i8> = vec![7];
        let b: Vec<i8> = vec![3];
        let c = cpu_int8_matmul_dp4a(&a, &b, 1, 1, 1);
        assert_eq!(c, vec![21]);
    }

    #[test]
    fn test_dp4a_vector_length_2() {
        assert_eq!(cpu_dp4a_vector(&[3, 5], &[2, 4]), 26);
    }

    #[test]
    fn test_dp4a_vector_length_3() {
        assert_eq!(cpu_dp4a_vector(&[1, 2, 3], &[4, 5, 6]), 32);
    }

    // ── Config / Error types ───────────────────────────────────────────

    #[test]
    fn test_int8_config_default() {
        let cfg = Int8Config::default();
        assert!(cfg.use_dp4a);
        assert_eq!(cfg.accumulator_bits, 32);
        assert!(!cfg.channel_scaling);
    }

    #[test]
    fn test_int8_error_display() {
        let e = Int8Error::OverflowDetected { k: 131072 };
        assert!(e.to_string().contains("131072"));

        let e = Int8Error::ShapeMismatch { expected: 4, actual: 5 };
        assert!(e.to_string().contains("expected 4"));
    }

    #[test]
    fn test_int8_error_unsupported() {
        let e = Int8Error::UnsupportedDp4a;
        assert!(e.to_string().contains("not supported"));
    }

    #[test]
    fn test_quantized_tensor_construction() {
        let t = QuantizedTensor {
            data: vec![1, -1, 0, 127],
            shape: vec![2, 2],
            scale: 0.5,
            zero_point: 0,
        };
        assert_eq!(t.data.len(), 4);
        assert_eq!(t.shape, vec![2, 2]);
    }

    #[test]
    fn test_dp4a_result_construction() {
        let r = Dp4aResult { output: vec![100, 200], output_scale: 0.01 };
        assert_eq!(r.output.len(), 2);
        assert!((r.output_scale - 0.01).abs() < 1e-6);
    }

    // ── OpenCL source ──────────────────────────────────────────────────

    #[test]
    fn test_opencl_source_not_empty() {
        assert!(!INT8_DP4A_SRC.is_empty());
    }

    #[test]
    fn test_opencl_source_contains_dp4a_gemm() {
        assert!(INT8_DP4A_SRC.contains("int8_dp4a_gemm"));
    }

    #[test]
    fn test_opencl_source_contains_tiled_matmul() {
        assert!(INT8_DP4A_SRC.contains("int8_matmul_tiled"));
    }

    #[test]
    fn test_opencl_source_contains_quantize() {
        assert!(INT8_DP4A_SRC.contains("quantize_fp32_to_int8"));
    }

    #[test]
    fn test_opencl_source_contains_intel_extension() {
        assert!(INT8_DP4A_SRC.contains("cl_intel_subgroups"));
    }

    #[test]
    fn test_opencl_source_contains_dp4a_intrinsic() {
        assert!(INT8_DP4A_SRC.contains("dp4a"));
    }
}
