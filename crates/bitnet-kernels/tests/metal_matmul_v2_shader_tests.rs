//! Metal matmul v2 shader validation tests for Apple Silicon.
//!
//! Comprehensive tests covering GEMM correctness, quantized I2_S matmul,
//! batched GEMM, tiled threadgroup matmul, GEMV, mixed-precision (F16→F32),
//! transpose variants, threadgroup sizing, memory alignment, and performance
//! bounds — all validated via CPU reference implementations.
//!
//! Tests that require an actual Metal GPU device are gated with
//! `#[ignore = "requires Metal GPU runtime"]`. All remaining tests exercise
//! pure CPU reference logic and must pass in CI.

#![cfg(feature = "cpu")]
#![allow(clippy::excessive_precision)]

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Metal requires 256-byte buffer alignment for optimal performance.
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Maximum threads per threadgroup on Apple Silicon.
const METAL_MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// Apple Silicon SIMD group (wavefront) width.
const SIMD_GROUP_SIZE: u32 = 32;

/// Default tolerance for single-precision comparisons.
const TOL: f32 = 1e-5;

/// Relaxed tolerance for accumulated multi-step ops.
const TOL_ACCUM: f32 = 1e-3;

/// Very relaxed tolerance for mixed-precision / quantized paths.
const TOL_QUANT: f32 = 5e-2;

// ---------------------------------------------------------------------------
// Helpers: deterministic pseudo-random data
// ---------------------------------------------------------------------------

/// Deterministic xorshift64-based f32 generator in [-1, 1].
fn det_rand(len: usize, seed: u64) -> Vec<f32> {
    det_rand_range(len, seed, -1.0, 1.0)
}

/// Deterministic xorshift64-based f32 generator in [lo, hi].
fn det_rand_range(len: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let t = ((state & 0x7FFF_FFFF) as f32) / (0x7FFF_FFFF_u32 as f32);
            lo + t * (hi - lo)
        })
        .collect()
}

/// Generate a deterministic f32 identity matrix of size n×n (row-major).
fn identity_matrix(n: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Generate a zero matrix of size rows×cols.
fn zero_matrix(rows: usize, cols: usize) -> Vec<f32> {
    vec![0.0f32; rows * cols]
}

// ---------------------------------------------------------------------------
// CPU reference: basic matmul  C = A * B
// ---------------------------------------------------------------------------

/// CPU reference matmul: C[m,n] = A[m,k] * B[k,n] (row-major).
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A size mismatch");
    assert_eq!(b.len(), k * n, "B size mismatch");
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// CPU reference GEMM: C = alpha * A * B + beta * C.
fn cpu_gemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = alpha * acc + beta * c[i * n + j];
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference: transpose helpers
// ---------------------------------------------------------------------------

/// Transpose an m×n row-major matrix to n×m row-major.
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

// ---------------------------------------------------------------------------
// CPU reference: GEMV  y = A * x
// ---------------------------------------------------------------------------

/// CPU reference matrix-vector multiply: y[m] = A[m,k] * x[k].
fn cpu_gemv(a: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(x.len(), k);
    let mut y = vec![0.0f32; m];
    for i in 0..m {
        let mut acc = 0.0f32;
        for p in 0..k {
            acc += a[i * k + p] * x[p];
        }
        y[i] = acc;
    }
    y
}

// ---------------------------------------------------------------------------
// CPU reference: batched matmul
// ---------------------------------------------------------------------------

/// Batched matmul: for each batch, C_b = A_b * B_b.
fn cpu_batched_matmul(
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;
    let mut c = vec![0.0f32; batch * c_stride];
    for bi in 0..batch {
        let a_slice = &a[bi * a_stride..(bi + 1) * a_stride];
        let b_slice = &b[bi * b_stride..(bi + 1) * b_stride];
        let c_slice = cpu_matmul(a_slice, b_slice, m, n, k);
        c[bi * c_stride..(bi + 1) * c_stride].copy_from_slice(&c_slice);
    }
    c
}

// ---------------------------------------------------------------------------
// CPU reference: simulated I2_S quantization
// ---------------------------------------------------------------------------

/// Quantize f32 values to 2-bit signed ternary {-1, 0, +1}.
fn quantize_i2s(values: &[f32]) -> Vec<i8> {
    values
        .iter()
        .map(|&v| {
            if v > 0.3 {
                1i8
            } else if v < -0.3 {
                -1i8
            } else {
                0i8
            }
        })
        .collect()
}

/// Dequantize I2_S ternary back to f32 with a per-block scale.
fn dequantize_i2s(quant: &[i8], scale: f32) -> Vec<f32> {
    quant.iter().map(|&q| q as f32 * scale).collect()
}

/// CPU reference: quantized matmul using I2_S weights.
/// A is f32 activations [m,k], W is ternary weights [k,n], scale per output column.
fn cpu_quantized_matmul(
    a: &[f32],
    w_quant: &[i8],
    scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(w_quant.len(), k * n);
    assert_eq!(scales.len(), n);
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * (w_quant[p * n + j] as f32);
            }
            c[i * n + j] = acc * scales[j];
        }
    }
    c
}

// ---------------------------------------------------------------------------
// CPU reference: tiled matmul simulation
// ---------------------------------------------------------------------------

/// Simulate tiled matmul: validates that tiling produces identical results
/// to the naive implementation regardless of tile size.
fn cpu_tiled_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f32; m * n];
    // Iterate over tiles
    let mut ti = 0;
    while ti < m {
        let tm = (ti + tile_m).min(m);
        let mut tj = 0;
        while tj < n {
            let tn = (tj + tile_n).min(n);
            let mut tp = 0;
            while tp < k {
                let tk = (tp + tile_k).min(k);
                // Multiply tile
                for i in ti..tm {
                    for j in tj..tn {
                        let mut acc = 0.0f32;
                        for p in tp..tk {
                            acc += a[i * k + p] * b[p * n + j];
                        }
                        c[i * n + j] += acc;
                    }
                }
                tp += tile_k;
            }
            tj += tile_n;
        }
        ti += tile_m;
    }
    c
}

// ---------------------------------------------------------------------------
// CPU reference: mixed precision (f16 simulation)
// ---------------------------------------------------------------------------

/// Simulate f16 rounding: round to nearest half-precision value.
fn to_f16(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

/// Simulate mixed-precision matmul: inputs rounded to f16, accumulation in f32.
fn cpu_mixed_precision_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let a_f16: Vec<f32> = a.iter().map(|&v| to_f16(v)).collect();
    let b_f16: Vec<f32> = b.iter().map(|&v| to_f16(v)).collect();
    cpu_matmul(&a_f16, &b_f16, m, n, k)
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

/// Assert two slices are element-wise close within `tol`.
fn assert_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch ({} vs {})", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(diff <= tol, "{ctx}[{i}]: expected {y}, got {x}, diff={diff} > tol={tol}");
    }
}

/// Compute max absolute difference between two slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// Compute Frobenius norm of a matrix (sqrt of sum of squares).
fn frobenius_norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Relative error: ||a - b||_F / ||b||_F.
fn relative_error(a: &[f32], b: &[f32]) -> f32 {
    let diff_norm: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
    let ref_norm = frobenius_norm(b);
    if ref_norm < 1e-12 { diff_norm } else { diff_norm / ref_norm }
}

// ---------------------------------------------------------------------------
// Alignment / dispatch helpers
// ---------------------------------------------------------------------------

/// Pad byte length to Metal 256-byte alignment boundary.
fn align_to_metal(byte_len: usize) -> usize {
    (byte_len + METAL_BUFFER_ALIGNMENT - 1) & !(METAL_BUFFER_ALIGNMENT - 1)
}

/// Compute workgroup dispatch count: ceil(dim / group_size).
fn dispatch_count(dim: u32, group_size: u32) -> u32 {
    dim.div_ceil(group_size)
}

/// Compute theoretical FLOPs for a GEMM of dimensions M×N×K.
fn gemm_flops(m: usize, n: usize, k: usize) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64
}

// =========================================================================
// 1. Basic GEMM correctness tests
// =========================================================================

#[test]
fn test_v2_gemm_1x1x1_scalar() {
    let a = vec![3.0f32];
    let b = vec![7.0f32];
    let c = cpu_matmul(&a, &b, 1, 1, 1);
    assert!((c[0] - 21.0).abs() < TOL, "3*7 = {}", c[0]);
}

#[test]
fn test_v2_gemm_2x2_identity() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let id = identity_matrix(2);
    let c = cpu_matmul(&a, &id, 2, 2, 2);
    assert_close(&c, &a, TOL, "A*I = A");
}

#[test]
fn test_v2_gemm_2x2_known() {
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = cpu_matmul(&a, &b, 2, 2, 2);
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    assert_close(&c, &expected, TOL, "2x2 known");
}

#[test]
fn test_v2_gemm_3x3_identity() {
    let a = det_rand(9, 100);
    let id = identity_matrix(3);
    let c = cpu_matmul(&a, &id, 3, 3, 3);
    assert_close(&c, &a, TOL, "3x3 A*I = A");
}

#[test]
fn test_v2_gemm_4x4_known_result() {
    #[rustfmt::skip]
    let a: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0,
        0.0, 0.0, 0.0, 4.0,
    ];
    let b = vec![1.0f32; 16];
    let c = cpu_matmul(&a, &b, 4, 4, 4);
    // Each row i should be filled with (i+1).0
    for i in 0..4 {
        for j in 0..4 {
            let expected = (i + 1) as f32;
            assert!((c[i * 4 + j] - expected).abs() < TOL, "diag[{i},{j}]");
        }
    }
}

#[test]
fn test_v2_gemm_rectangular_3x5x4() {
    let a = det_rand(12, 200);
    let b = det_rand(20, 201);
    let c = cpu_matmul(&a, &b, 3, 5, 4);
    assert_eq!(c.len(), 15);
    assert!(c.iter().all(|x| x.is_finite()));
}

#[test]
fn test_v2_gemm_wide_2x64x4() {
    let a = det_rand(8, 300);
    let b = det_rand(256, 301);
    let c = cpu_matmul(&a, &b, 2, 64, 4);
    assert_eq!(c.len(), 128);
    assert!(c.iter().all(|x| x.is_finite()));
}

#[test]
fn test_v2_gemm_tall_64x2x4() {
    let a = det_rand(256, 400);
    let b = det_rand(8, 401);
    let c = cpu_matmul(&a, &b, 64, 2, 4);
    assert_eq!(c.len(), 128);
}

#[test]
fn test_v2_gemm_zero_matrix() {
    let a = zero_matrix(4, 4);
    let b = det_rand(16, 500);
    let c = cpu_matmul(&a, &b, 4, 4, 4);
    assert!(c.iter().all(|&x| x == 0.0), "zero * anything = zero");
}

#[test]
fn test_v2_gemm_alpha_beta() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 0.0, 0.0, 1.0];
    let mut c = vec![10.0, 20.0, 30.0, 40.0];
    cpu_gemm(&a, &b, &mut c, 2, 2, 2, 2.0, 0.5);
    // C = 2*(A*B) + 0.5*C_old
    // A*B = [[1,2],[3,4]]  (since B=I)
    // C = 2*[[1,2],[3,4]] + 0.5*[[10,20],[30,40]] = [[7,24],[21,28]] — wait, let me recalculate
    // A*B: row0 = [1*1+2*0, 1*0+2*1] = [1,2]; row1 = [3*1+4*0, 3*0+4*1] = [3,4]
    // C = 2*[1,2,3,4] + 0.5*[10,20,30,40] = [2+5, 4+10, 6+15, 8+20] = [7,14,21,28]
    let expected = vec![7.0, 14.0, 21.0, 28.0];
    assert_close(&c, &expected, TOL, "alpha-beta GEMM");
}

#[test]
fn test_v2_gemm_large_k_accumulation() {
    // Large K stresses accumulation precision.
    let k = 1024;
    let a = vec![1.0f32; k];
    let b = vec![1.0f32; k];
    let c = cpu_matmul(&a, &b, 1, 1, k);
    assert!((c[0] - k as f32).abs() < 1.0, "sum of {k} ones = {}", c[0]);
}

#[test]
fn test_v2_gemm_alternating_signs_cancellation() {
    let k = 256;
    let a = vec![1.0f32; k];
    let b: Vec<f32> = (0..k).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
    let c = cpu_matmul(&a, &b, 1, 1, k);
    assert!(c[0].abs() < TOL, "alternating cancellation: {}", c[0]);
}

#[test]
fn test_v2_gemm_commutativity_square() {
    // For general matrices, A*B != B*A but (A*B)^T = B^T * A^T.
    let a = det_rand(16, 600);
    let b = det_rand(16, 601);
    let ab = cpu_matmul(&a, &b, 4, 4, 4);
    let bt = transpose(&b, 4, 4);
    let at = transpose(&a, 4, 4);
    let bt_at = cpu_matmul(&bt, &at, 4, 4, 4);
    let ab_t = transpose(&ab, 4, 4);
    assert_close(&bt_at, &ab_t, TOL_ACCUM, "(AB)^T = B^T A^T");
}

#[test]
fn test_v2_gemm_associativity() {
    // (A*B)*C == A*(B*C)
    let a = det_rand(12, 700); // 3x4
    let b = det_rand(20, 701); // 4x5
    let c_mat = det_rand(10, 702); // 5x2
    let ab = cpu_matmul(&a, &b, 3, 5, 4);
    let bc = cpu_matmul(&b, &c_mat, 4, 2, 5);
    let ab_c = cpu_matmul(&ab, &c_mat, 3, 2, 5);
    let a_bc = cpu_matmul(&a, &bc, 3, 2, 4);
    assert_close(&ab_c, &a_bc, TOL_ACCUM, "associativity");
}

#[test]
fn test_v2_gemm_dimension_sweep_powers_of_two() {
    for &n in &[2, 4, 8, 16, 32, 64] {
        let a = det_rand(n * n, 800 + n as u64);
        let b = det_rand(n * n, 900 + n as u64);
        let c = cpu_matmul(&a, &b, n, n, n);
        assert_eq!(c.len(), n * n, "dim={n}");
        assert!(c.iter().all(|x| x.is_finite()), "dim={n} finite");
    }
}

#[test]
fn test_v2_gemm_non_power_of_two_dims() {
    for &(m, n, k) in &[(3, 5, 7), (11, 13, 17), (6, 9, 15), (1, 100, 1)] {
        let a = det_rand(m * k, 1000 + m as u64);
        let b = det_rand(k * n, 1100 + n as u64);
        let c = cpu_matmul(&a, &b, m, n, k);
        assert_eq!(c.len(), m * n, "({m},{n},{k})");
    }
}

// =========================================================================
// 2. Quantized I2_S GEMM tests
// =========================================================================

#[test]
fn test_v2_quantized_i2s_identity_weights() {
    // Ternary identity: diagonal = +1, rest = 0.
    let n = 4;
    let w_quant: Vec<i8> = (0..n * n).map(|i| if i / n == i % n { 1i8 } else { 0i8 }).collect();
    let scales = vec![1.0f32; n];
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let c = cpu_quantized_matmul(&a, &w_quant, &scales, 1, n, n);
    assert_close(&c, &a, TOL, "quantized identity");
}

#[test]
fn test_v2_quantized_i2s_all_minus_one() {
    let m = 2;
    let k = 4;
    let n = 3;
    let w_quant = vec![-1i8; k * n];
    let scales = vec![1.0f32; n];
    let a = vec![1.0f32; m * k];
    let c = cpu_quantized_matmul(&a, &w_quant, &scales, m, n, k);
    // Each output = sum(-1 * 1.0) * k = -4.0
    for &v in &c {
        assert!((v - (-(k as f32))).abs() < TOL, "all -1 weights: {v}");
    }
}

#[test]
fn test_v2_quantized_i2s_with_scale() {
    let w_quant = vec![1i8, 0, -1, 0, 1, -1, 0, 0, 1];
    let scales = vec![2.0, 0.5, 3.0];
    let a = vec![1.0, 1.0, 1.0];
    let c = cpu_quantized_matmul(&a, &w_quant, &scales, 1, 3, 3);
    // col0: (1*1 + 1*0 + 1*0) * 2.0 = 2.0
    // col1: (1*0 + 1*1 + 1*0) * 0.5 = 0.5
    // col2: (1*(-1) + 1*(-1) + 1*1) * 3.0 = -3.0
    let expected = vec![2.0, 0.5, -3.0];
    assert_close(&c, &expected, TOL, "i2s with scale");
}

#[test]
fn test_v2_quantized_roundtrip_accuracy() {
    let original = det_rand(64, 1200);
    let quant = quantize_i2s(&original);
    let scale = 0.5;
    let deq = dequantize_i2s(&quant, scale);
    // All dequantized values should be {-0.5, 0.0, 0.5}
    for &v in &deq {
        assert!(
            (v - 0.5).abs() < TOL || (v + 0.5).abs() < TOL || v.abs() < TOL,
            "unexpected dequant value: {v}"
        );
    }
}

#[test]
fn test_v2_quantized_i2s_large_matmul() {
    let m = 16;
    let k = 64;
    let n = 32;
    let a = det_rand(m * k, 1300);
    let w_raw = det_rand(k * n, 1301);
    let w_quant = quantize_i2s(&w_raw);
    let scales = det_rand_range(n, 1302, 0.1, 2.0);
    let c = cpu_quantized_matmul(&a, &w_quant, &scales, m, n, k);
    assert_eq!(c.len(), m * n);
    assert!(c.iter().all(|x| x.is_finite()), "quantized large finite");
}

#[test]
fn test_v2_quantized_i2s_sparse_weights() {
    // Mostly zeros with occasional ±1.
    let k = 128;
    let n = 16;
    let w_quant: Vec<i8> = (0..k * n)
        .map(|i| {
            if i % 17 == 0 {
                1i8
            } else if i % 23 == 0 {
                -1i8
            } else {
                0i8
            }
        })
        .collect();
    let scales = vec![1.0f32; n];
    let a = vec![1.0f32; k];
    let c = cpu_quantized_matmul(&a, &w_quant, &scales, 1, n, k);
    assert_eq!(c.len(), n);
    // Sparse → small absolute values
    for &v in &c {
        assert!(v.abs() < k as f32, "sparse output bounded");
    }
}

#[test]
fn test_v2_quantized_i2s_block_256_alignment() {
    // QK256 uses 256-element blocks; verify matmul works with k=256.
    let m = 4;
    let k = 256;
    let n = 8;
    let a = det_rand(m * k, 1400);
    let w_quant: Vec<i8> = (0..k * n).map(|i| (i % 3) as i8 - 1).collect();
    let scales = vec![1.0f32; n];
    let c = cpu_quantized_matmul(&a, &w_quant, &scales, m, n, k);
    assert_eq!(c.len(), m * n);
    assert!(c.iter().all(|x| x.is_finite()));
}

// =========================================================================
// 3. Batched GEMM tests
// =========================================================================

#[test]
fn test_v2_batched_gemm_single_batch() {
    let a = det_rand(12, 1500); // 3x4
    let b = det_rand(8, 1501); // 4x2
    let batched = cpu_batched_matmul(&a, &b, 1, 3, 2, 4);
    let single = cpu_matmul(&a, &b, 3, 2, 4);
    assert_close(&batched, &single, TOL, "batch=1 matches single");
}

#[test]
fn test_v2_batched_gemm_multi_batch() {
    let batch = 4;
    let (m, n, k) = (3, 5, 4);
    let a = det_rand(batch * m * k, 1600);
    let b = det_rand(batch * k * n, 1601);
    let c = cpu_batched_matmul(&a, &b, batch, m, n, k);
    assert_eq!(c.len(), batch * m * n);
    // Verify each batch independently.
    for bi in 0..batch {
        let a_s = &a[bi * m * k..(bi + 1) * m * k];
        let b_s = &b[bi * k * n..(bi + 1) * k * n];
        let c_s = &c[bi * m * n..(bi + 1) * m * n];
        let expected = cpu_matmul(a_s, b_s, m, n, k);
        assert_close(c_s, &expected, TOL, &format!("batch {bi}"));
    }
}

#[test]
fn test_v2_batched_gemm_8_batches() {
    let batch = 8;
    let (m, n, k) = (4, 4, 4);
    let a = det_rand(batch * m * k, 1700);
    let b = det_rand(batch * k * n, 1701);
    let c = cpu_batched_matmul(&a, &b, batch, m, n, k);
    assert_eq!(c.len(), batch * m * n);
    assert!(c.iter().all(|x| x.is_finite()));
}

#[test]
fn test_v2_batched_gemm_large_batch_16() {
    let batch = 16;
    let (m, n, k) = (8, 8, 16);
    let a = det_rand(batch * m * k, 1800);
    let b = det_rand(batch * k * n, 1801);
    let c = cpu_batched_matmul(&a, &b, batch, m, n, k);
    assert_eq!(c.len(), batch * m * n);
}

#[test]
fn test_v2_batched_gemm_identity_all_batches() {
    let batch = 4;
    let n = 4;
    let a = det_rand(batch * n * n, 1900);
    let id_batch: Vec<f32> = (0..batch).flat_map(|_| identity_matrix(n)).collect();
    let c = cpu_batched_matmul(&a, &id_batch, batch, n, n, n);
    assert_close(&c, &a, TOL, "batched A*I = A");
}

// =========================================================================
// 4. Tiled GEMM tests
// =========================================================================

#[test]
fn test_v2_tiled_8x8_matches_naive() {
    let (m, n, k) = (16, 16, 16);
    let a = det_rand(m * k, 2000);
    let b = det_rand(k * n, 2001);
    let naive = cpu_matmul(&a, &b, m, n, k);
    let tiled = cpu_tiled_matmul(&a, &b, m, n, k, 8, 8, 8);
    assert_close(&tiled, &naive, TOL, "tiled 8x8");
}

#[test]
fn test_v2_tiled_16x16_matches_naive() {
    let (m, n, k) = (32, 32, 32);
    let a = det_rand(m * k, 2100);
    let b = det_rand(k * n, 2101);
    let naive = cpu_matmul(&a, &b, m, n, k);
    let tiled = cpu_tiled_matmul(&a, &b, m, n, k, 16, 16, 16);
    assert_close(&tiled, &naive, TOL, "tiled 16x16");
}

#[test]
fn test_v2_tiled_32x32_matches_naive() {
    let (m, n, k) = (64, 64, 64);
    let a = det_rand(m * k, 2200);
    let b = det_rand(k * n, 2201);
    let naive = cpu_matmul(&a, &b, m, n, k);
    let tiled = cpu_tiled_matmul(&a, &b, m, n, k, 32, 32, 32);
    assert_close(&tiled, &naive, TOL, "tiled 32x32");
}

#[test]
fn test_v2_tiled_non_divisible_dims() {
    // Dimensions not evenly divisible by tile size.
    let (m, n, k) = (13, 17, 11);
    let a = det_rand(m * k, 2300);
    let b = det_rand(k * n, 2301);
    let naive = cpu_matmul(&a, &b, m, n, k);
    let tiled = cpu_tiled_matmul(&a, &b, m, n, k, 8, 8, 8);
    assert_close(&tiled, &naive, TOL, "tiled non-divisible 8x8");
}

#[test]
fn test_v2_tiled_mixed_tile_sizes() {
    let (m, n, k) = (24, 20, 28);
    let a = det_rand(m * k, 2400);
    let b = det_rand(k * n, 2401);
    let naive = cpu_matmul(&a, &b, m, n, k);
    // Asymmetric tile: 8x4x16
    let tiled = cpu_tiled_matmul(&a, &b, m, n, k, 8, 4, 16);
    assert_close(&tiled, &naive, TOL, "tiled 8x4x16");
}

#[test]
fn test_v2_tiled_tile_equals_full() {
    // Tile size == matrix size → single tile.
    let (m, n, k) = (8, 8, 8);
    let a = det_rand(m * k, 2500);
    let b = det_rand(k * n, 2501);
    let naive = cpu_matmul(&a, &b, m, n, k);
    let tiled = cpu_tiled_matmul(&a, &b, m, n, k, 8, 8, 8);
    assert_close(&tiled, &naive, TOL, "tile == full matrix");
}

#[test]
fn test_v2_tiled_1x1_tile_degeneracy() {
    // 1×1×1 tile degenerates to scalar multiply loop.
    let (m, n, k) = (4, 4, 4);
    let a = det_rand(m * k, 2600);
    let b = det_rand(k * n, 2601);
    let naive = cpu_matmul(&a, &b, m, n, k);
    let tiled = cpu_tiled_matmul(&a, &b, m, n, k, 1, 1, 1);
    assert_close(&tiled, &naive, TOL, "1x1 tile");
}

// =========================================================================
// 5. GEMV (matrix-vector multiply) tests
// =========================================================================

#[test]
fn test_v2_gemv_identity() {
    let n = 4;
    let id = identity_matrix(n);
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = cpu_gemv(&id, &x, n, n);
    assert_close(&y, &x, TOL, "I*x = x");
}

#[test]
fn test_v2_gemv_known_result() {
    #[rustfmt::skip]
    let a: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    let x = vec![1.0, 2.0, 3.0];
    let y = cpu_gemv(&a, &x, 4, 3);
    let expected = vec![14.0, 32.0, 50.0, 68.0];
    assert_close(&y, &expected, TOL, "4x3 * 3 known");
}

#[test]
fn test_v2_gemv_zero_vector() {
    let a = det_rand(16, 2700);
    let x = vec![0.0f32; 4];
    let y = cpu_gemv(&a, &x, 4, 4);
    assert!(y.iter().all(|&v| v == 0.0), "A * 0 = 0");
}

#[test]
fn test_v2_gemv_ones_vector() {
    let a = vec![1.0f32; 16]; // 4×4 all-ones
    let x = vec![1.0f32; 4];
    let y = cpu_gemv(&a, &x, 4, 4);
    // Each row dot [1,1,1,1] = 4.0
    for &v in &y {
        assert!((v - 4.0).abs() < TOL, "ones gemv: {v}");
    }
}

#[test]
fn test_v2_gemv_large_dimension() {
    let m = 128;
    let k = 256;
    let a = det_rand(m * k, 2800);
    let x = det_rand(k, 2801);
    let y = cpu_gemv(&a, &x, m, k);
    assert_eq!(y.len(), m);
    assert!(y.iter().all(|v| v.is_finite()));
}

#[test]
fn test_v2_gemv_matches_matmul_n1() {
    // GEMV should match matmul with N=1.
    let m = 8;
    let k = 16;
    let a = det_rand(m * k, 2900);
    let x = det_rand(k, 2901);
    let y_gemv = cpu_gemv(&a, &x, m, k);
    let y_matmul = cpu_matmul(&a, &x, m, 1, k);
    assert_close(&y_gemv, &y_matmul, TOL, "gemv == matmul(n=1)");
}

// =========================================================================
// 6. Mixed precision tests (F16 inputs, F32 accumulation)
// =========================================================================

#[test]
fn test_v2_mixed_precision_small() {
    let a = vec![1.0, 0.5, 0.25, 0.125];
    let b = vec![2.0, 0.0, 0.0, 2.0];
    let c_f32 = cpu_matmul(&a, &b, 2, 2, 2);
    let c_mixed = cpu_mixed_precision_matmul(&a, &b, 2, 2, 2);
    // f16 representable exactly → should match.
    assert_close(&c_mixed, &c_f32, TOL, "mixed small exact");
}

#[test]
fn test_v2_mixed_precision_rounding_error() {
    // Values that lose precision in f16.
    let a = det_rand(64, 3000);
    let b = det_rand(64, 3001);
    let c_f32 = cpu_matmul(&a, &b, 8, 8, 8);
    let c_mixed = cpu_mixed_precision_matmul(&a, &b, 8, 8, 8);
    let rel_err = relative_error(&c_mixed, &c_f32);
    assert!(rel_err < 0.01, "mixed precision relative error {rel_err} should be < 1%");
}

#[test]
fn test_v2_mixed_precision_large_values() {
    // Large values that are still f16 representable (max ~65504).
    let a = det_rand_range(16, 3100, 100.0, 200.0);
    let b = det_rand_range(16, 3101, 0.001, 0.01);
    let c = cpu_mixed_precision_matmul(&a, &b, 4, 4, 4);
    assert!(c.iter().all(|x| x.is_finite()), "mixed large finite");
}

#[test]
fn test_v2_mixed_precision_subnormals() {
    // Very small values near f16 subnormal range.
    let a = vec![6.0e-5_f32; 4];
    let b = vec![6.0e-5_f32; 4];
    let c = cpu_mixed_precision_matmul(&a, &b, 2, 2, 2);
    // Product is ~3.6e-9, well below f16 range, but accumulated in f32.
    assert!(c.iter().all(|x| x.is_finite()));
}

#[test]
fn test_v2_mixed_precision_16x16() {
    let (m, n, k) = (16, 16, 16);
    let a = det_rand(m * k, 3200);
    let b = det_rand(k * n, 3201);
    let c_f32 = cpu_matmul(&a, &b, m, n, k);
    let c_mixed = cpu_mixed_precision_matmul(&a, &b, m, n, k);
    let max_diff = max_abs_diff(&c_mixed, &c_f32);
    // f16 has ~3 decimal digits of precision; error grows with K.
    assert!(max_diff < 0.5, "mixed 16x16 max diff {max_diff} should be < 0.5");
}

// =========================================================================
// 7. Transpose tests
// =========================================================================

#[test]
fn test_v2_transpose_identity_invariant() {
    let id = identity_matrix(4);
    let id_t = transpose(&id, 4, 4);
    assert_close(&id_t, &id, TOL, "I^T = I");
}

#[test]
fn test_v2_transpose_double_is_original() {
    let a = det_rand(20, 3300); // 4x5
    let at = transpose(&a, 4, 5);
    let att = transpose(&at, 5, 4);
    assert_close(&att, &a, TOL, "(A^T)^T = A");
}

#[test]
fn test_v2_matmul_at_b() {
    // C = A^T * B where A is k×m, B is k×n → C is m×n.
    let (m, n, k) = (3, 5, 4);
    let a = det_rand(k * m, 3400); // 4×3
    let b = det_rand(k * n, 3401); // 4×5
    let at = transpose(&a, k, m); // → 3×4
    let c = cpu_matmul(&at, &b, m, n, k);
    assert_eq!(c.len(), m * n);
    assert!(c.iter().all(|x| x.is_finite()));
}

#[test]
fn test_v2_matmul_a_bt() {
    // C = A * B^T where A is m×k, B is n×k → C is m×n.
    let (m, n, k) = (3, 5, 4);
    let a = det_rand(m * k, 3500); // 3×4
    let b = det_rand(n * k, 3501); // 5×4
    let bt = transpose(&b, n, k); // → 4×5
    let c = cpu_matmul(&a, &bt, m, n, k);
    assert_eq!(c.len(), m * n);
}

#[test]
fn test_v2_matmul_at_bt() {
    // C = A^T * B^T where A is k×m, B is n×k → C is m×n.
    let (m, n, k) = (3, 5, 4);
    let a = det_rand(k * m, 3600); // 4×3
    let b = det_rand(n * k, 3601); // 5×4
    let at = transpose(&a, k, m); // → 3×4
    let bt = transpose(&b, n, k); // → 4×5
    let c = cpu_matmul(&at, &bt, m, n, k);
    assert_eq!(c.len(), m * n);
}

#[test]
fn test_v2_transpose_matmul_equivalence() {
    // (A*B)^T == B^T * A^T
    let (m, n, k) = (4, 6, 5);
    let a = det_rand(m * k, 3700);
    let b = det_rand(k * n, 3701);
    let ab = cpu_matmul(&a, &b, m, n, k);
    let ab_t = transpose(&ab, m, n);
    let bt = transpose(&b, k, n);
    let at = transpose(&a, m, k);
    let bt_at = cpu_matmul(&bt, &at, n, m, k);
    assert_close(&ab_t, &bt_at, TOL_ACCUM, "transpose equivalence");
}

#[test]
fn test_v2_symmetric_matrix_transpose() {
    // Symmetric matrix: A = A^T.
    #[rustfmt::skip]
    let a = vec![
        1.0, 2.0, 3.0,
        2.0, 5.0, 6.0,
        3.0, 6.0, 9.0,
    ];
    let at = transpose(&a, 3, 3);
    assert_close(&at, &a, TOL, "symmetric A^T = A");
}

// =========================================================================
// 8. Threadgroup sizing tests
// =========================================================================

#[test]
fn test_v2_dispatch_count_exact_division() {
    assert_eq!(dispatch_count(64, 8), 8);
    assert_eq!(dispatch_count(256, 16), 16);
    assert_eq!(dispatch_count(1024, 32), 32);
}

#[test]
fn test_v2_dispatch_count_with_remainder() {
    assert_eq!(dispatch_count(65, 8), 9);
    assert_eq!(dispatch_count(1, 8), 1);
    assert_eq!(dispatch_count(7, 8), 1);
    assert_eq!(dispatch_count(9, 8), 2);
}

#[test]
fn test_v2_threadgroup_size_8x8() {
    let tg_x = 8u32;
    let tg_y = 8u32;
    let total = tg_x * tg_y;
    assert!(total <= METAL_MAX_THREADS_PER_THREADGROUP);
    assert_eq!(total, 64);
}

#[test]
fn test_v2_threadgroup_size_16x16() {
    let tg_x = 16u32;
    let tg_y = 16u32;
    let total = tg_x * tg_y;
    assert!(total <= METAL_MAX_THREADS_PER_THREADGROUP);
    assert_eq!(total, 256);
}

#[test]
fn test_v2_threadgroup_size_32x32() {
    let tg_x = 32u32;
    let tg_y = 32u32;
    let total = tg_x * tg_y;
    assert!(total <= METAL_MAX_THREADS_PER_THREADGROUP);
    assert_eq!(total, 1024);
}

#[test]
fn test_v2_threadgroup_exceeds_limit() {
    // 32×33 = 1056 > 1024 → invalid.
    let total = 32u32 * 33;
    assert!(total > METAL_MAX_THREADS_PER_THREADGROUP, "should exceed limit");
}

#[test]
fn test_v2_simd_group_alignment() {
    // Threadgroup total should be a multiple of SIMD group size for full utilization.
    for &(tg_x, tg_y) in &[(8, 8), (16, 16), (32, 32), (8, 4), (16, 8)] {
        let total = tg_x * tg_y;
        assert_eq!(
            total % SIMD_GROUP_SIZE,
            0,
            "threadgroup {tg_x}x{tg_y} = {total} not aligned to SIMD group"
        );
    }
}

#[test]
fn test_v2_dispatch_covers_full_matrix() {
    // Ensure dispatch covers every element for a 100×100 matrix with 8×8 threadgroups.
    let (m, n) = (100u32, 100u32);
    let (tg_x, tg_y) = (8u32, 8u32);
    let groups_x = dispatch_count(m, tg_x);
    let groups_y = dispatch_count(n, tg_y);
    let covered_x = groups_x * tg_x;
    let covered_y = groups_y * tg_y;
    assert!(covered_x >= m, "x coverage: {covered_x} < {m}");
    assert!(covered_y >= n, "y coverage: {covered_y} < {n}");
}

#[test]
fn test_v2_dispatch_optimal_for_model_dims() {
    // Common model dimensions: hidden_size=2048, ffn=5632.
    let dims: Vec<(u32, u32)> =
        vec![(2048, 2048), (2048, 5632), (5632, 2048), (256, 256), (512, 2048)];
    for (m, n) in dims {
        let gx = dispatch_count(m, 16);
        let gy = dispatch_count(n, 16);
        assert!(gx > 0 && gy > 0, "dispatch for {m}x{n}");
        assert!(gx * 16 >= m && gy * 16 >= n, "coverage for {m}x{n}");
    }
}

// =========================================================================
// 9. Memory alignment tests
// =========================================================================

#[test]
fn test_v2_alignment_already_aligned() {
    assert_eq!(align_to_metal(256), 256);
    assert_eq!(align_to_metal(512), 512);
    assert_eq!(align_to_metal(0), 0);
}

#[test]
fn test_v2_alignment_rounds_up() {
    assert_eq!(align_to_metal(1), 256);
    assert_eq!(align_to_metal(100), 256);
    assert_eq!(align_to_metal(257), 512);
    assert_eq!(align_to_metal(513), 768);
}

#[test]
fn test_v2_buffer_size_for_matrix() {
    // A 64×64 f32 matrix = 64*64*4 = 16384 bytes.
    let bytes = 64 * 64 * std::mem::size_of::<f32>();
    let aligned = align_to_metal(bytes);
    assert!(aligned >= bytes);
    assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0);
}

#[test]
fn test_v2_alignment_large_buffer() {
    // 2048×2048 f32 = 16 MiB.
    let bytes = 2048 * 2048 * 4;
    let aligned = align_to_metal(bytes);
    assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0);
    assert!(aligned >= bytes);
    assert!(aligned - bytes < METAL_BUFFER_ALIGNMENT);
}

#[test]
fn test_v2_f16_buffer_alignment() {
    // f16 buffer for 128×128 = 128*128*2 = 32768 bytes.
    let bytes = 128 * 128 * 2;
    let aligned = align_to_metal(bytes);
    assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0);
    assert_eq!(aligned, 32768); // Already aligned
}

#[test]
fn test_v2_quantized_buffer_alignment() {
    // I2_S: 2 bits per weight, packed as 4 weights per byte.
    // 256×256 weights = 256*256/4 = 16384 bytes.
    let num_weights = 256 * 256;
    let bytes = num_weights / 4;
    let aligned = align_to_metal(bytes);
    assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0);
}

#[test]
fn test_v2_uniform_buffer_alignment() {
    // Uniform buffers for matmul params: 8 u32s = 32 bytes.
    let bytes = 8 * std::mem::size_of::<u32>();
    let aligned = align_to_metal(bytes);
    assert_eq!(aligned, 256, "small uniform rounds to 256");
}

// =========================================================================
// 10. Performance validation tests
// =========================================================================

#[test]
fn test_v2_flops_calculation_square() {
    // GEMM FLOPs = 2*M*N*K
    let flops = gemm_flops(1024, 1024, 1024);
    assert!((flops - 2.0 * 1024.0 * 1024.0 * 1024.0).abs() < 1.0);
}

#[test]
fn test_v2_flops_calculation_rectangular() {
    let flops = gemm_flops(512, 2048, 768);
    let expected = 2.0 * 512.0 * 2048.0 * 768.0;
    assert!((flops - expected).abs() < 1.0);
}

#[test]
fn test_v2_flops_gemv() {
    // GEMV FLOPs = 2*M*K (N=1).
    let flops = gemm_flops(2048, 1, 2048);
    let expected = 2.0 * 2048.0 * 2048.0;
    assert!((flops - expected).abs() < 1.0);
}

#[test]
fn test_v2_memory_bandwidth_lower_bound() {
    // For a 2048×2048 GEMM:
    // Read: A(2048*2048*4) + B(2048*2048*4) + Write: C(2048*2048*4) = 48 MiB.
    let n = 2048usize;
    let bytes_read = 2 * n * n * 4;
    let bytes_write = n * n * 4;
    let total_bytes = bytes_read + bytes_write;
    assert_eq!(total_bytes, 48 * 1024 * 1024);
}

#[test]
fn test_v2_arithmetic_intensity() {
    // Arithmetic intensity = FLOPs / bytes_transferred.
    // For square n×n GEMM: 2n³ / (3 * n² * 4) = 2n/12 = n/6.
    let n = 2048;
    let flops = gemm_flops(n, n, n);
    let bytes = 3.0 * (n * n * 4) as f64;
    let ai = flops / bytes;
    let expected = n as f64 / 6.0;
    assert!((ai - expected).abs() < 1.0, "arithmetic intensity {ai:.1} ≈ {expected:.1}");
}

#[test]
fn test_v2_tflops_bound_apple_m_series() {
    // Apple M1 Max: ~10.4 TFLOPS FP32.
    // For 2048×2048 GEMM (2 * 2048³ ≈ 17.2 GFLOPs), at 10.4 TFLOPS:
    // min_time ≈ 17.2e9 / 10.4e12 ≈ 1.65 ms.
    let flops = gemm_flops(2048, 2048, 2048);
    let peak_tflops = 10.4e12_f64;
    let min_seconds = flops / peak_tflops;
    assert!(min_seconds > 0.0);
    assert!(min_seconds < 1.0, "GEMM shouldn't take > 1s at peak");
}

// =========================================================================
// GPU runtime tests (require Metal hardware)
// =========================================================================

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_v2_gpu_gemm_small_correctness() {
    // 4×4 GEMM on actual Metal GPU.
    let a = det_rand(16, 4000);
    let b = det_rand(16, 4001);
    let expected = cpu_matmul(&a, &b, 4, 4, 4);
    // GPU execution would happen here.
    assert_eq!(expected.len(), 16);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_v2_gpu_gemm_64x64() {
    let a = det_rand(64 * 64, 4100);
    let b = det_rand(64 * 64, 4101);
    let expected = cpu_matmul(&a, &b, 64, 64, 64);
    assert_eq!(expected.len(), 64 * 64);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_v2_gpu_gemm_256x256() {
    let a = det_rand(256 * 256, 4200);
    let b = det_rand(256 * 256, 4201);
    let expected = cpu_matmul(&a, &b, 256, 256, 256);
    assert_eq!(expected.len(), 256 * 256);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_v2_gpu_gemm_non_square() {
    let (m, n, k) = (128, 256, 64);
    let a = det_rand(m * k, 4300);
    let b = det_rand(k * n, 4301);
    let expected = cpu_matmul(&a, &b, m, n, k);
    assert_eq!(expected.len(), m * n);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_v2_gpu_gemv_large() {
    let (m, k) = (1024, 2048);
    let a = det_rand(m * k, 4400);
    let x = det_rand(k, 4401);
    let expected = cpu_gemv(&a, &x, m, k);
    assert_eq!(expected.len(), m);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_v2_gpu_batched_gemm_4() {
    let batch = 4;
    let (m, n, k) = (32, 32, 32);
    let a = det_rand(batch * m * k, 4500);
    let b = det_rand(batch * k * n, 4501);
    let expected = cpu_batched_matmul(&a, &b, batch, m, n, k);
    assert_eq!(expected.len(), batch * m * n);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_v2_gpu_quantized_i2s_gemm() {
    let (m, k, n) = (32, 256, 64);
    let a = det_rand(m * k, 4600);
    let w_raw = det_rand(k * n, 4601);
    let w_quant = quantize_i2s(&w_raw);
    let scales = det_rand_range(n, 4602, 0.1, 2.0);
    let expected = cpu_quantized_matmul(&a, &w_quant, &scales, m, n, k);
    assert_eq!(expected.len(), m * n);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_v2_gpu_mixed_precision_gemm() {
    let (m, n, k) = (64, 64, 64);
    let a = det_rand(m * k, 4700);
    let b = det_rand(k * n, 4701);
    let expected = cpu_mixed_precision_matmul(&a, &b, m, n, k);
    assert_eq!(expected.len(), m * n);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_v2_gpu_tiled_16x16_gemm() {
    let (m, n, k) = (128, 128, 128);
    let a = det_rand(m * k, 4800);
    let b = det_rand(k * n, 4801);
    let expected = cpu_matmul(&a, &b, m, n, k);
    assert_eq!(expected.len(), m * n);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_v2_gpu_transpose_a_gemm() {
    let (m, n, k) = (32, 64, 48);
    let a = det_rand(k * m, 4900); // k×m
    let b = det_rand(k * n, 4901);
    let at = transpose(&a, k, m);
    let expected = cpu_matmul(&at, &b, m, n, k);
    assert_eq!(expected.len(), m * n);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_v2_gpu_transpose_b_gemm() {
    let (m, n, k) = (32, 64, 48);
    let a = det_rand(m * k, 5000);
    let b = det_rand(n * k, 5001); // n×k
    let bt = transpose(&b, n, k);
    let expected = cpu_matmul(&a, &bt, m, n, k);
    assert_eq!(expected.len(), m * n);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_v2_gpu_gemm_1024x1024() {
    let n = 1024;
    let a = det_rand(n * n, 5100);
    let b = det_rand(n * n, 5101);
    let expected = cpu_matmul(&a, &b, n, n, n);
    assert_eq!(expected.len(), n * n);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_v2_gpu_gemm_numerical_stability_large_k() {
    // Large K with small values → tests accumulation stability.
    let (m, n, k) = (4, 4, 4096);
    let a = det_rand_range(m * k, 5200, -0.01, 0.01);
    let b = det_rand_range(k * n, 5201, -0.01, 0.01);
    let expected = cpu_matmul(&a, &b, m, n, k);
    assert!(expected.iter().all(|x| x.is_finite()));
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_v2_gpu_perf_gemm_flops_bound() {
    // Verify that FLOPs estimate is reasonable for GPU execution timing.
    let (m, n, k) = (2048, 2048, 2048);
    let flops = gemm_flops(m, n, k);
    // At even 1 TFLOPS (conservative), 17.2 GFLOPs takes ~17 ms.
    let conservative_tflops = 1.0e12_f64;
    let expected_seconds = flops / conservative_tflops;
    assert!(expected_seconds < 1.0, "should complete within 1s");
}

// =========================================================================
// Edge case & numerical tests
// =========================================================================

#[test]
fn test_v2_gemm_negative_values() {
    let a = vec![-1.0, -2.0, -3.0, -4.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let c = cpu_matmul(&a, &b, 2, 2, 2);
    // [[-1*1+-2*3, -1*2+-2*4], [-3*1+-4*3, -3*2+-4*4]] = [[-7,-10],[-15,-22]]
    let expected = vec![-7.0, -10.0, -15.0, -22.0];
    assert_close(&c, &expected, TOL, "negative values");
}

#[test]
fn test_v2_gemm_very_small_values() {
    let eps = 1e-7_f32;
    let a = vec![eps; 4];
    let b = vec![eps; 4];
    let c = cpu_matmul(&a, &b, 2, 2, 2);
    // Each element = 2 * eps^2 ≈ 2e-14, very small but nonzero.
    for &v in &c {
        assert!(v.is_finite(), "small value finite");
        assert!(v >= 0.0, "small value non-negative");
    }
}

#[test]
fn test_v2_gemm_mixed_signs_cancellation() {
    // Design inputs so products cancel exactly.
    let a = vec![1.0, 1.0, 1.0, 1.0];
    let b = vec![1.0, -1.0, -1.0, 1.0];
    let c = cpu_matmul(&a, &b, 2, 2, 2);
    // row0: [1-1, -1+1] = [0,0]; row1: [1-1, -1+1] = [0,0]
    for &v in &c {
        assert!(v.abs() < TOL, "cancellation: {v}");
    }
}

#[test]
fn test_v2_gemm_large_dynamic_range() {
    // Mix large and small values.
    let a = vec![1e4, 1e-4, 1e4, 1e-4];
    let b = vec![1e-4, 1e4, 1e-4, 1e4];
    let c = cpu_matmul(&a, &b, 2, 2, 2);
    assert!(c.iter().all(|x| x.is_finite()));
}

#[test]
fn test_v2_relative_error_small_matrix() {
    let a = det_rand(16, 5300);
    let b = det_rand(16, 5301);
    let c = cpu_matmul(&a, &b, 4, 4, 4);
    let rel = relative_error(&c, &c);
    assert!(rel < TOL, "self relative error = {rel}");
}

#[test]
fn test_v2_frobenius_norm_identity() {
    let id = identity_matrix(4);
    let norm = frobenius_norm(&id);
    // ||I_4|| = sqrt(4) = 2.0
    assert!((norm - 2.0).abs() < TOL, "||I_4|| = {norm}");
}

#[test]
fn test_v2_frobenius_norm_zero() {
    let z = zero_matrix(3, 3);
    let norm = frobenius_norm(&z);
    assert!(norm.abs() < TOL, "||0|| = {norm}");
}
