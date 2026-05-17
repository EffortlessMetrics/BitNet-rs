#![cfg(feature = "cpu")]
#![allow(clippy::needless_range_loop)]
//! AVX2 vs scalar parity tests for matrix operations and cache-friendly
//! matmul kernels.
//!
//! Each test exercises a kernel with AVX2 tiling / micro-kernel dispatch
//! and compares the output against a naïve triple-loop (or equivalent)
//! scalar reference within defined tolerances.
//!
//! Dimensions are chosen to exercise both the SIMD body and scalar tails.
//!
//! All tests are pure-math — no model files required.

use bitnet_kernels::cpu::cache_matmul;
use bitnet_kernels::cpu::matrix_ops::{self, MatmulConfig};

#[path = "common/avx2_parity.rs"]
mod avx2_parity;
use avx2_parity::{assert_vec_parity, pseudo_rand};

// ── Tolerance constants ────────────────────────────────────────────────
//
// FMA vs multiply-then-add can differ by up to ~0.5 ULP per op; with
// thousands of accumulations the absolute error can reach ~1e-4 for
// matmul on moderate matrices.  Element-wise ops are tighter.

/// Absolute tolerance for matmul / reduction-heavy kernels.
const MATMUL_ABS_TOL: f32 = 1e-4;
/// Relative tolerance for matmul / reduction-heavy kernels.
const MATMUL_REL_TOL: f32 = 1e-3;

/// Absolute tolerance for element-wise kernels (add, sub, scale).
const ELEM_ABS_TOL: f32 = 1e-6;
/// Relative tolerance for element-wise kernels.
const ELEM_REL_TOL: f32 = 1e-5;

// ── Naïve reference implementations ───────────────────────────────────

fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// A[m×k] × Bᵀ[n×k] = C[m×n]  (b_t stored row-major as [n, k]).
fn naive_matmul_transposed(a: &[f32], b_t: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b_t[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
}

fn naive_matvec(a: &[f32], x: &[f32], y: &mut [f32], m: usize, k: usize) {
    for i in 0..m {
        let mut sum = 0.0f32;
        for p in 0..k {
            sum += a[i * k + p] * x[p];
        }
        y[i] = sum;
    }
}

fn naive_outer_product(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = a[i] * b[j];
        }
    }
}

/// cache_matmul microkernel reference: a_panel is row-major [mr, K],
/// b_panel is column-major [K, nr] (stride = K per column).
fn naive_microkernel(a_panel: &[f32], b_panel: &[f32], c_block: &mut [f32], mr: usize, nr: usize) {
    let k_len = a_panel.len() / mr;
    for i in 0..mr {
        for j in 0..nr {
            let mut sum = 0.0f32;
            for p in 0..k_len {
                // b_panel is column-major: column j starts at j*k_len
                sum += a_panel[i * k_len + p] * b_panel[j * k_len + p];
            }
            c_block[i * nr + j] = sum;
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// matrix_ops tests
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_matmul_small_parity() {
    let (m, n, k) = (4, 4, 4);
    let a = pseudo_rand(m * k, 42);
    let b = pseudo_rand(k * n, 123);
    let cfg = MatmulConfig::default();

    let mut c_simd = vec![0.0f32; m * n];
    matrix_ops::simd_matmul(&a, &b, &mut c_simd, m, n, k, &cfg).unwrap();

    let mut c_ref = vec![0.0f32; m * n];
    naive_matmul(&a, &b, &mut c_ref, m, n, k);

    assert_vec_parity(&c_simd, &c_ref, MATMUL_ABS_TOL, MATMUL_REL_TOL, "matmul_small");
}

#[test]
fn test_matmul_nonsquare_parity() {
    let (m, n, k) = (3, 7, 5);
    let a = pseudo_rand(m * k, 7);
    let b = pseudo_rand(k * n, 19);
    let cfg = MatmulConfig::default();

    let mut c_simd = vec![0.0f32; m * n];
    matrix_ops::simd_matmul(&a, &b, &mut c_simd, m, n, k, &cfg).unwrap();

    let mut c_ref = vec![0.0f32; m * n];
    naive_matmul(&a, &b, &mut c_ref, m, n, k);

    assert_vec_parity(&c_simd, &c_ref, MATMUL_ABS_TOL, MATMUL_REL_TOL, "matmul_nonsquare");
}

#[test]
fn test_matmul_large_parity() {
    let (m, n, k) = (16, 16, 32);
    let a = pseudo_rand(m * k, 314);
    let b = pseudo_rand(k * n, 271);
    let cfg = MatmulConfig::default();

    let mut c_simd = vec![0.0f32; m * n];
    matrix_ops::simd_matmul(&a, &b, &mut c_simd, m, n, k, &cfg).unwrap();

    let mut c_ref = vec![0.0f32; m * n];
    naive_matmul(&a, &b, &mut c_ref, m, n, k);

    assert_vec_parity(&c_simd, &c_ref, MATMUL_ABS_TOL, MATMUL_REL_TOL, "matmul_large");
}

#[test]
fn test_matmul_transposed_parity() {
    let (m, n, k) = (4, 6, 8);
    let a = pseudo_rand(m * k, 55);
    // b_t is [n, k] row-major
    let b_t = pseudo_rand(n * k, 77);
    let cfg = MatmulConfig::default();

    let mut c_simd = vec![0.0f32; m * n];
    matrix_ops::simd_matmul_transposed(&a, &b_t, &mut c_simd, m, n, k, &cfg).unwrap();

    let mut c_ref = vec![0.0f32; m * n];
    naive_matmul_transposed(&a, &b_t, &mut c_ref, m, n, k);

    assert_vec_parity(&c_simd, &c_ref, MATMUL_ABS_TOL, MATMUL_REL_TOL, "matmul_transposed");
}

#[test]
fn test_matvec_parity() {
    let (m, k) = (8, 16);
    let a = pseudo_rand(m * k, 99);
    let x = pseudo_rand(k, 101);

    let mut y_simd = vec![0.0f32; m];
    matrix_ops::simd_matvec(&a, &x, &mut y_simd, m, k).unwrap();

    let mut y_ref = vec![0.0f32; m];
    naive_matvec(&a, &x, &mut y_ref, m, k);

    assert_vec_parity(&y_simd, &y_ref, MATMUL_ABS_TOL, MATMUL_REL_TOL, "matvec");
}

#[test]
fn test_outer_product_parity() {
    let (m, n) = (8, 12);
    let a = pseudo_rand(m, 200);
    let b = pseudo_rand(n, 201);

    let mut c_simd = vec![0.0f32; m * n];
    matrix_ops::outer_product(&a, &b, &mut c_simd, m, n).unwrap();

    let mut c_ref = vec![0.0f32; m * n];
    naive_outer_product(&a, &b, &mut c_ref, m, n);

    assert_vec_parity(&c_simd, &c_ref, ELEM_ABS_TOL, ELEM_REL_TOL, "outer_product");
}

#[test]
fn test_matrix_add_parity() {
    let (m, n) = (8, 16);
    let a = pseudo_rand(m * n, 300);
    let b = pseudo_rand(m * n, 301);

    let mut c_simd = vec![0.0f32; m * n];
    matrix_ops::matrix_add(&a, &b, &mut c_simd, m, n).unwrap();

    let c_ref: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();

    assert_vec_parity(&c_simd, &c_ref, ELEM_ABS_TOL, ELEM_REL_TOL, "matrix_add");
}

#[test]
fn test_matrix_sub_parity() {
    let (m, n) = (8, 16);
    let a = pseudo_rand(m * n, 400);
    let b = pseudo_rand(m * n, 401);

    let mut c_simd = vec![0.0f32; m * n];
    matrix_ops::matrix_sub(&a, &b, &mut c_simd, m, n).unwrap();

    let c_ref: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect();

    assert_vec_parity(&c_simd, &c_ref, ELEM_ABS_TOL, ELEM_REL_TOL, "matrix_sub");
}

#[test]
fn test_matrix_scale_parity() {
    let (m, n) = (8, 16);
    let alpha = 2.5f32;
    let a = pseudo_rand(m * n, 500);

    let mut c_simd = vec![0.0f32; m * n];
    matrix_ops::matrix_scale(&a, &mut c_simd, m, n, alpha).unwrap();

    let c_ref: Vec<f32> = a.iter().map(|&x| x * alpha).collect();

    assert_vec_parity(&c_simd, &c_ref, ELEM_ABS_TOL, ELEM_REL_TOL, "matrix_scale");
}

// ══════════════════════════════════════════════════════════════════════
// cache_matmul tests
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_cache_matmul_microkernel_parity() {
    let (mr, nr, k) = (4, 8, 8);
    // a_panel: row-major [mr, k]
    let a_panel = pseudo_rand(mr * k, 600);
    // b_panel: column-major [k, nr] stored as nr columns of k elements
    let b_panel = pseudo_rand(nr * k, 601);

    let mut c_simd = vec![0.0f32; mr * nr];
    cache_matmul::matmul_avx2_microkernel(&a_panel, &b_panel, &mut c_simd, mr, nr).unwrap();

    let mut c_ref = vec![0.0f32; mr * nr];
    naive_microkernel(&a_panel, &b_panel, &mut c_ref, mr, nr);

    assert_vec_parity(&c_simd, &c_ref, MATMUL_ABS_TOL, MATMUL_REL_TOL, "cache_matmul_microkernel");
}
