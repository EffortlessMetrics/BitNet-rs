//! SIMD-optimized matrix operations for CPU inference.
//!
//! Provides tiled matrix multiplication (with optional AVX2 acceleration),
//! Strassen's algorithm for large matrices, packed-layout matmul for cache
//! efficiency, quantized GEMV (INT2/INT4 × FP32), and common element-wise
//! matrix utilities (add, subtract, scale, transpose, outer product).
//!
//! All matrices are **row-major** `f32` slices unless otherwise noted.
#![allow(unsafe_op_in_unsafe_fn)]

use bitnet_common::{BitNetError, KernelError, Result};

// ── Helpers ────────────────────────────────────────────────────────────

fn invalid_args(reason: impl Into<String>) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.into() })
}

// ── Configuration ──────────────────────────────────────────────────────

fn checked_product(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| invalid_args(format!("{label} overflow: {lhs} * {rhs}")))
}

/// Tile-sizes and runtime knobs for [`simd_matmul`].
#[derive(Debug, Clone, Copy)]
pub struct MatmulConfig {
    /// Tile height (rows of A per micro-kernel invocation).
    pub tile_m: usize,
    /// Tile width (columns of B per micro-kernel invocation).
    pub tile_n: usize,
    /// Tile depth (shared-dimension stride).
    pub tile_k: usize,
    /// If `true` **and** the target is x86_64 with AVX2, use SIMD inner
    /// loops; otherwise fall back to scalar code.
    pub use_avx2: bool,
}

impl MatmulConfig {
    /// Sensible defaults tuned for L1-resident 4×8×8 micro-tiles.
    pub const DEFAULT: Self = Self { tile_m: 4, tile_n: 8, tile_k: 8, use_avx2: true };

    pub fn new(tile_m: usize, tile_n: usize, tile_k: usize, use_avx2: bool) -> Self {
        Self { tile_m, tile_n, tile_k, use_avx2 }
    }

    fn validate(&self) -> Result<()> {
        if self.tile_m == 0 || self.tile_n == 0 || self.tile_k == 0 {
            return Err(invalid_args("tile dimensions must be > 0"));
        }
        Ok(())
    }
}

impl Default for MatmulConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ── Runtime AVX2 detection ─────────────────────────────────────────────

#[inline]
fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ── Scalar micro-kernel ────────────────────────────────────────────────

/// Tile parameters for a micro-kernel invocation.
struct TileArgs {
    i0: usize,
    j0: usize,
    p0: usize,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
}

/// Scalar accumulate: `C[i,j] += sum_p A[i,p]*B[p,j]` for a single
/// tile.
#[inline]
fn scalar_tile_accum(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    t: &TileArgs,
) {
    let i_end = (t.i0 + t.tile_m).min(m);
    let j_end = (t.j0 + t.tile_n).min(n);
    let p_end = (t.p0 + t.tile_k).min(k);
    for i in t.i0..i_end {
        for j in t.j0..j_end {
            let mut acc = 0.0f32;
            for p in t.p0..p_end {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] += acc;
        }
    }
}

// ── AVX2 micro-kernel ──────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn avx2_tile_accum(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    t: &TileArgs,
) {
    use std::arch::x86_64::*;

    let i_end = (t.i0 + t.tile_m).min(m);
    let p_end = (t.p0 + t.tile_k).min(k);

    for i in t.i0..i_end {
        // Process columns in chunks of 8 using AVX2.
        let mut j = t.j0;
        let j_end = (t.j0 + t.tile_n).min(n);
        while j + 8 <= j_end {
            let mut acc = _mm256_loadu_ps(c.as_ptr().add(i * n + j));
            for p in t.p0..p_end {
                let a_val = _mm256_set1_ps(a[i * k + p]);
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(p * n + j));
                acc = _mm256_fmadd_ps(a_val, b_vec, acc);
            }
            _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), acc);
            j += 8;
        }
        // Scalar tail for remaining columns.
        while j < j_end {
            let mut acc = 0.0f32;
            for p in t.p0..p_end {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] += acc;
            j += 1;
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────────

/// Tiled matrix multiplication: `C[m×n] = A[m×k] · B[k×n]`.
///
/// Uses AVX2+FMA when `cfg.use_avx2` is `true` and the CPU supports it.
pub fn simd_matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    cfg: &MatmulConfig,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(invalid_args("dimensions must be > 0"));
    }
    cfg.validate()?;
    let a_required = checked_product(m, k, "A dimensions")?;
    let b_required = checked_product(k, n, "B dimensions")?;
    let c_required = checked_product(m, n, "C dimensions")?;
    if a.len() < a_required {
        return Err(invalid_args(format!("A too small: need {}, got {}", a_required, a.len())));
    }
    if b.len() < b_required {
        return Err(invalid_args(format!("B too small: need {}, got {}", b_required, b.len())));
    }
    if c.len() < c_required {
        return Err(invalid_args(format!("C too small: need {}, got {}", c_required, c.len())));
    }
    c[..c_required].fill(0.0);

    let use_simd = cfg.use_avx2 && avx2_available();

    for i0 in (0..m).step_by(cfg.tile_m) {
        for j0 in (0..n).step_by(cfg.tile_n) {
            for p0 in (0..k).step_by(cfg.tile_k) {
                let t = TileArgs {
                    i0,
                    j0,
                    p0,
                    tile_m: cfg.tile_m,
                    tile_n: cfg.tile_n,
                    tile_k: cfg.tile_k,
                };
                if use_simd {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        avx2_tile_accum(a, b, c, m, n, k, &t);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        scalar_tile_accum(a, b, c, m, n, k, &t);
                    }
                } else {
                    scalar_tile_accum(a, b, c, m, n, k, &t);
                }
            }
        }
    }
    Ok(())
}

/// Matrix multiplication with B transposed: `C[m×n] = A[m×k] · Bᵀ[n×k]`.
///
/// `b_t` is stored row-major as `[n, k]` (each row is a column of the
/// logical B).
pub fn simd_matmul_transposed(
    a: &[f32],
    b_t: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    cfg: &MatmulConfig,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(invalid_args("dimensions must be > 0"));
    }
    cfg.validate()?;
    let a_required = checked_product(m, k, "A dimensions")?;
    let b_t_required = checked_product(n, k, "B^T dimensions")?;
    let c_required = checked_product(m, n, "C dimensions")?;
    if a.len() < a_required {
        return Err(invalid_args(format!("A too small: need {}, got {}", a_required, a.len())));
    }
    if b_t.len() < b_t_required {
        return Err(invalid_args(format!(
            "B^T too small: need {}, got {}",
            b_t_required,
            b_t.len()
        )));
    }
    if c.len() < c_required {
        return Err(invalid_args(format!("C too small: need {}, got {}", c_required, c.len())));
    }
    c[..c_required].fill(0.0);

    let use_simd = cfg.use_avx2 && avx2_available();

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            let a_row = &a[i * k..];
            let b_row = &b_t[j * k..];
            if use_simd {
                acc += simd_dot(a_row, b_row, k);
            } else {
                for p in 0..k {
                    acc += a_row[p] * b_row[p];
                }
            }
            c[i * n + j] = acc;
        }
    }
    Ok(())
}

/// SIMD dot product (AVX2 when available, scalar fallback).
#[inline]
fn simd_dot(a: &[f32], b: &[f32], len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            return unsafe { avx2_dot(a, b, len) };
        }
    }
    scalar_dot(a, b, len)
}

#[inline]
fn scalar_dot(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..len {
        acc += a[i] * b[i];
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn avx2_dot(a: &[f32], b: &[f32], len: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    // Horizontal sum of 8 floats.
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut s = _mm_cvtss_f32(result);
    // Scalar tail.
    while i < len {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// Matrix-vector multiply: `y[m] = A[m×k] · x[k]`.
pub fn simd_matvec(a: &[f32], x: &[f32], y: &mut [f32], m: usize, k: usize) -> Result<()> {
    if m == 0 || k == 0 {
        return Err(invalid_args("dimensions must be > 0"));
    }
    if a.len() < m * k {
        return Err(invalid_args(format!("A too small: need {}, got {}", m * k, a.len())));
    }
    if x.len() < k {
        return Err(invalid_args(format!("x too small: need {k}, got {}", x.len())));
    }
    if y.len() < m {
        return Err(invalid_args(format!("y too small: need {m}, got {}", y.len())));
    }
    for i in 0..m {
        y[i] = simd_dot(&a[i * k..], x, k);
    }
    Ok(())
}

/// Batched matrix-vector multiply: `Y[b][m] = A[b][m×k] · X[b][k]`.
///
/// `a` is `[batch, m, k]`, `x` is `[batch, k]`, `y` is `[batch, m]`.
pub fn simd_batch_matvec(
    a: &[f32],
    x: &[f32],
    y: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
) -> Result<()> {
    if batch == 0 || m == 0 || k == 0 {
        return Err(invalid_args("dimensions must be > 0"));
    }
    if a.len() < batch * m * k {
        return Err(invalid_args(format!("A too small: need {}, got {}", batch * m * k, a.len())));
    }
    if x.len() < batch * k {
        return Err(invalid_args(format!("x too small: need {}, got {}", batch * k, x.len())));
    }
    if y.len() < batch * m {
        return Err(invalid_args(format!("y too small: need {}, got {}", batch * m, y.len())));
    }
    for b in 0..batch {
        let a_off = b * m * k;
        let x_off = b * k;
        let y_off = b * m;
        for i in 0..m {
            y[y_off + i] = simd_dot(&a[a_off + i * k..], &x[x_off..], k);
        }
    }
    Ok(())
}

/// Outer product: `C[m×n] = a[m] ⊗ b[n]`  (i.e. `C[i,j] = a[i] * b[j]`).
pub fn outer_product(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize) -> Result<()> {
    if m == 0 || n == 0 {
        return Err(invalid_args("dimensions must be > 0"));
    }
    if a.len() < m {
        return Err(invalid_args(format!("a too small: need {m}, got {}", a.len())));
    }
    if b.len() < n {
        return Err(invalid_args(format!("b too small: need {n}, got {}", b.len())));
    }
    if c.len() < m * n {
        return Err(invalid_args(format!("C too small: need {}, got {}", m * n, c.len())));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            unsafe {
                avx2_outer_product(a, b, c, m, n);
            }
            return Ok(());
        }
    }

    for i in 0..m {
        let ai = a[i];
        for j in 0..n {
            c[i * n + j] = ai * b[j];
        }
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_outer_product(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize) {
    use std::arch::x86_64::*;

    for i in 0..m {
        let ai = _mm256_set1_ps(a[i]);
        let mut j = 0usize;
        while j + 8 <= n {
            let bv = _mm256_loadu_ps(b.as_ptr().add(j));
            let cv = _mm256_mul_ps(ai, bv);
            _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), cv);
            j += 8;
        }
        while j < n {
            c[i * n + j] = a[i] * b[j];
            j += 1;
        }
    }
}

/// Element-wise matrix addition: `C = A + B` (both `m×n`).
pub fn matrix_add(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize) -> Result<()> {
    let len = m * n;
    if len == 0 {
        return Err(invalid_args("dimensions must be > 0"));
    }
    if a.len() < len || b.len() < len || c.len() < len {
        return Err(invalid_args(format!(
            "buffers too small for {m}×{n}: a={}, b={}, c={}",
            a.len(),
            b.len(),
            c.len()
        )));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            unsafe { avx2_elementwise_binop(a, b, c, len, BinOp::Add) };
            return Ok(());
        }
    }

    for i in 0..len {
        c[i] = a[i] + b[i];
    }
    Ok(())
}

/// Element-wise matrix subtraction: `C = A - B` (both `m×n`).
pub fn matrix_sub(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize) -> Result<()> {
    let len = m * n;
    if len == 0 {
        return Err(invalid_args("dimensions must be > 0"));
    }
    if a.len() < len || b.len() < len || c.len() < len {
        return Err(invalid_args(format!(
            "buffers too small for {m}×{n}: a={}, b={}, c={}",
            a.len(),
            b.len(),
            c.len()
        )));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            unsafe { avx2_elementwise_binop(a, b, c, len, BinOp::Sub) };
            return Ok(());
        }
    }

    for i in 0..len {
        c[i] = a[i] - b[i];
    }
    Ok(())
}

#[allow(dead_code)]
enum BinOp {
    Add,
    Sub,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_elementwise_binop(a: &[f32], b: &[f32], c: &mut [f32], len: usize, op: BinOp) {
    use std::arch::x86_64::*;

    let mut i = 0usize;
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vc = match op {
            BinOp::Add => _mm256_add_ps(va, vb),
            BinOp::Sub => _mm256_sub_ps(va, vb),
        };
        _mm256_storeu_ps(c.as_mut_ptr().add(i), vc);
        i += 8;
    }
    while i < len {
        c[i] = match op {
            BinOp::Add => a[i] + b[i],
            BinOp::Sub => a[i] - b[i],
        };
        i += 1;
    }
}

/// Scale every element: `C = alpha * A` (both `m×n`).
pub fn matrix_scale(a: &[f32], c: &mut [f32], m: usize, n: usize, alpha: f32) -> Result<()> {
    let len = m * n;
    if len == 0 {
        return Err(invalid_args("dimensions must be > 0"));
    }
    if a.len() < len || c.len() < len {
        return Err(invalid_args(format!(
            "buffers too small for {m}×{n}: a={}, c={}",
            a.len(),
            c.len()
        )));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            unsafe { avx2_scale(a, c, len, alpha) };
            return Ok(());
        }
    }

    for i in 0..len {
        c[i] = alpha * a[i];
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_scale(a: &[f32], c: &mut [f32], len: usize, alpha: f32) {
    use std::arch::x86_64::*;

    let valpha = _mm256_set1_ps(alpha);
    let mut i = 0usize;
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        _mm256_storeu_ps(c.as_mut_ptr().add(i), _mm256_mul_ps(valpha, va));
        i += 8;
    }
    while i < len {
        c[i] = alpha * a[i];
        i += 1;
    }
}

/// Row-major 2-D matrix transpose: `B[n×m] = Aᵀ[m×n]`.
///
/// When `out` is `Some`, the result is written there.  When `None`, a new
/// buffer is allocated and returned.
pub fn matrix_transpose(
    a: &[f32],
    m: usize,
    n: usize,
    out: Option<&mut [f32]>,
) -> Result<Vec<f32>> {
    let len = m * n;
    if len == 0 {
        return Err(invalid_args("dimensions must be > 0"));
    }
    if a.len() < len {
        return Err(invalid_args(format!("A too small: need {len}, got {}", a.len())));
    }

    let mut owned = Vec::new();
    let has_out = out.is_some();

    let dst: &mut [f32] = match out {
        Some(buf) => {
            if buf.len() < len {
                return Err(invalid_args(format!(
                    "output too small: need {len}, got {}",
                    buf.len()
                )));
            }
            buf
        }
        None => {
            owned = vec![0.0f32; len];
            &mut owned
        }
    };

    // 4×4 cache-friendly blocking.
    const BLK: usize = 4;
    for ib in (0..m).step_by(BLK) {
        for jb in (0..n).step_by(BLK) {
            let ie = (ib + BLK).min(m);
            let je = (jb + BLK).min(n);
            for i in ib..ie {
                for j in jb..je {
                    dst[j * m + i] = a[i * n + j];
                }
            }
        }
    }

    if has_out { Ok(Vec::new()) } else { Ok(owned) }
}

// ── Strassen ───────────────────────────────────────────────────────────

/// Strassen matrix multiplication for large square matrices.
///
/// Falls back to tiled matmul when `n ≤ threshold`.  Only supports
/// square `n×n` matrices.
pub fn strassen_matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    threshold: usize,
) -> Result<()> {
    if n == 0 {
        return Err(invalid_args("dimension must be > 0"));
    }
    if a.len() < n * n || b.len() < n * n || c.len() < n * n {
        return Err(invalid_args(format!(
            "buffers too small for {n}×{n}: a={}, b={}, c={}",
            a.len(),
            b.len(),
            c.len()
        )));
    }
    c[..n * n].fill(0.0);
    strassen_recurse(a, b, c, n, threshold);
    Ok(())
}

fn strassen_recurse(a: &[f32], b: &[f32], c: &mut [f32], n: usize, threshold: usize) {
    if n <= threshold || n <= 1 || !n.is_multiple_of(2) {
        // Base case: naive matmul.
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..n {
                    s += a[i * n + p] * b[p * n + j];
                }
                c[i * n + j] = s;
            }
        }
        return;
    }

    let h = n / 2;
    let sz = h * h;

    // Extract quadrants.
    let (a11, a12, a21, a22) = split_quadrants(a, n, h);
    let (b11, b12, b21, b22) = split_quadrants(b, n, h);

    // Seven Strassen products.
    let mut m1 = vec![0.0f32; sz];
    let mut m2 = vec![0.0f32; sz];
    let mut m3 = vec![0.0f32; sz];
    let mut m4 = vec![0.0f32; sz];
    let mut m5 = vec![0.0f32; sz];
    let mut m6 = vec![0.0f32; sz];
    let mut m7 = vec![0.0f32; sz];
    let mut t1 = vec![0.0f32; sz];
    let mut t2 = vec![0.0f32; sz];

    // M1 = (A11 + A22) * (B11 + B22)
    add_sub(&a11, &a22, &mut t1, sz, true);
    add_sub(&b11, &b22, &mut t2, sz, true);
    strassen_recurse(&t1, &t2, &mut m1, h, threshold);

    // M2 = (A21 + A22) * B11
    add_sub(&a21, &a22, &mut t1, sz, true);
    strassen_recurse(&t1, &b11, &mut m2, h, threshold);

    // M3 = A11 * (B12 - B22)
    add_sub(&b12, &b22, &mut t1, sz, false);
    strassen_recurse(&a11, &t1, &mut m3, h, threshold);

    // M4 = A22 * (B21 - B11)
    add_sub(&b21, &b11, &mut t1, sz, false);
    strassen_recurse(&a22, &t1, &mut m4, h, threshold);

    // M5 = (A11 + A12) * B22
    add_sub(&a11, &a12, &mut t1, sz, true);
    strassen_recurse(&t1, &b22, &mut m5, h, threshold);

    // M6 = (A21 - A11) * (B11 + B12)
    add_sub(&a21, &a11, &mut t1, sz, false);
    add_sub(&b11, &b12, &mut t2, sz, true);
    strassen_recurse(&t1, &t2, &mut m6, h, threshold);

    // M7 = (A12 - A22) * (B21 + B22)
    add_sub(&a12, &a22, &mut t1, sz, false);
    add_sub(&b21, &b22, &mut t2, sz, true);
    strassen_recurse(&t1, &t2, &mut m7, h, threshold);

    // Assemble C quadrants.
    // C11 = M1 + M4 - M5 + M7
    // C12 = M3 + M5
    // C21 = M2 + M4
    // C22 = M1 - M2 + M3 + M6
    let mut c11 = vec![0.0f32; sz];
    let mut c12 = vec![0.0f32; sz];
    let mut c21 = vec![0.0f32; sz];
    let mut c22 = vec![0.0f32; sz];

    for i in 0..sz {
        c11[i] = m1[i] + m4[i] - m5[i] + m7[i];
        c12[i] = m3[i] + m5[i];
        c21[i] = m2[i] + m4[i];
        c22[i] = m1[i] - m2[i] + m3[i] + m6[i];
    }

    merge_quadrants(c, n, h, &c11, &c12, &c21, &c22);
}

fn split_quadrants(m: &[f32], n: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let sz = h * h;
    let mut q11 = vec![0.0f32; sz];
    let mut q12 = vec![0.0f32; sz];
    let mut q21 = vec![0.0f32; sz];
    let mut q22 = vec![0.0f32; sz];
    for i in 0..h {
        for j in 0..h {
            q11[i * h + j] = m[i * n + j];
            q12[i * h + j] = m[i * n + (j + h)];
            q21[i * h + j] = m[(i + h) * n + j];
            q22[i * h + j] = m[(i + h) * n + (j + h)];
        }
    }
    (q11, q12, q21, q22)
}

fn merge_quadrants(
    c: &mut [f32],
    n: usize,
    h: usize,
    c11: &[f32],
    c12: &[f32],
    c21: &[f32],
    c22: &[f32],
) {
    for i in 0..h {
        for j in 0..h {
            c[i * n + j] = c11[i * h + j];
            c[i * n + (j + h)] = c12[i * h + j];
            c[(i + h) * n + j] = c21[i * h + j];
            c[(i + h) * n + (j + h)] = c22[i * h + j];
        }
    }
}

/// Element-wise add (when `is_add`) or subtract.
fn add_sub(a: &[f32], b: &[f32], c: &mut [f32], len: usize, is_add: bool) {
    if is_add {
        for i in 0..len {
            c[i] = a[i] + b[i];
        }
    } else {
        for i in 0..len {
            c[i] = a[i] - b[i];
        }
    }
}

// ── Packed layout matmul ───────────────────────────────────────────────

/// Packed-layout matrix multiplication for cache efficiency.
///
/// The caller pre-packs B into column-panel order (`[k, n]` → panels of
/// width `panel_w`).  This avoids TLB thrashing on large `n`.
///
/// `b_packed` layout: for each column panel `jp` of width `pw`:
///   `b_packed[jp * k * pw + p * pw + jj]` = `B[p, jp*pw + jj]`.
pub fn packed_matmul(
    a: &[f32],
    b_packed: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    panel_w: usize,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 || panel_w == 0 {
        return Err(invalid_args("dimensions and panel_w must be > 0"));
    }
    if a.len() < m * k {
        return Err(invalid_args(format!("A too small: need {}, got {}", m * k, a.len())));
    }
    let num_panels = n.div_ceil(panel_w);
    let b_need = num_panels * k * panel_w;
    if b_packed.len() < b_need {
        return Err(invalid_args(format!(
            "B_packed too small: need {b_need}, got {}",
            b_packed.len()
        )));
    }
    if c.len() < m * n {
        return Err(invalid_args(format!("C too small: need {}, got {}", m * n, c.len())));
    }
    c[..m * n].fill(0.0);

    for jp in 0..num_panels {
        let j0 = jp * panel_w;
        let pw = panel_w.min(n - j0);
        let panel_base = jp * k * panel_w;
        for i in 0..m {
            let a_row = &a[i * k..];
            for (p, &ap) in a_row.iter().take(k).enumerate() {
                let bp_base = panel_base + p * panel_w;
                for jj in 0..pw {
                    c[i * n + j0 + jj] += ap * b_packed[bp_base + jj];
                }
            }
        }
    }
    Ok(())
}

/// Pack matrix B `[k, n]` into column-panel layout for [`packed_matmul`].
pub fn pack_b_col_panels(b: &[f32], k: usize, n: usize, panel_w: usize) -> Result<Vec<f32>> {
    if k == 0 || n == 0 || panel_w == 0 {
        return Err(invalid_args("dimensions and panel_w must be > 0"));
    }
    if b.len() < k * n {
        return Err(invalid_args(format!("B too small: need {}, got {}", k * n, b.len())));
    }
    let num_panels = n.div_ceil(panel_w);
    let mut packed = vec![0.0f32; num_panels * k * panel_w];
    for jp in 0..num_panels {
        let j0 = jp * panel_w;
        let pw = panel_w.min(n - j0);
        for p in 0..k {
            for jj in 0..pw {
                packed[jp * k * panel_w + p * panel_w + jj] = b[p * n + j0 + jj];
            }
        }
    }
    Ok(packed)
}

// ── Quantized GEMV ─────────────────────────────────────────────────────

/// Quantized matrix-vector multiply (INT2 weights × FP32 activations).
///
/// `weights_packed`: column-major I2_S packed, `ceil(k/4) * n` bytes.
/// Each byte contains 4 ternary values ({-1, 0, +1}), 2 bits each,
/// LSB-first.
///
/// `scales`: one `f32` per block of `block_size` elements along `k`,
/// per output column → `n * ceil(k / block_size)` entries.
#[allow(clippy::needless_range_loop)]
pub fn gemv_quantized(
    x: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    if m == 0 || k == 0 || block_size == 0 {
        return Err(invalid_args("dimensions and block_size must be > 0"));
    }
    let packed_k = k.div_ceil(4);
    let num_blocks = k.div_ceil(block_size);

    if x.len() < k {
        return Err(invalid_args(format!("x too small: need {k}, got {}", x.len())));
    }
    if weights_packed.len() < packed_k * m {
        return Err(invalid_args(format!(
            "weights too small: need {}, got {}",
            packed_k * m,
            weights_packed.len()
        )));
    }
    if scales.len() < m * num_blocks {
        return Err(invalid_args(format!(
            "scales too small: need {}, got {}",
            m * num_blocks,
            scales.len()
        )));
    }
    if y.len() < m {
        return Err(invalid_args(format!("y too small: need {m}, got {}", y.len())));
    }

    for col in 0..m {
        let mut acc = 0.0f32;
        for blk in 0..num_blocks {
            let blk_start = blk * block_size;
            let blk_end = (blk_start + block_size).min(k);
            let scale = scales[col * num_blocks + blk];
            let mut blk_acc = 0.0f32;
            for idx in blk_start..blk_end {
                let byte_idx = idx / 4;
                let bit_off = (idx % 4) * 2;
                let byte = weights_packed[col * packed_k + byte_idx];
                let w = decode_i2s((byte >> bit_off) & 0x03);
                blk_acc += (w as f32) * x[idx];
            }
            acc += scale * blk_acc;
        }
        y[col] = acc;
    }
    Ok(())
}

/// Quantized matrix-vector multiply (INT4 weights × FP32 activations).
///
/// `weights_packed`: column-major INT4 packed, `ceil(k/2) * n` bytes.
/// Each byte contains 2 signed 4-bit values (low nibble first,
/// sign-magnitude with range [-7, 7]).
///
/// `scales`: one `f32` per block of `block_size` elements along `k`,
/// per output column.
#[allow(clippy::needless_range_loop)]
pub fn gemv_quantized_int4(
    x: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    if m == 0 || k == 0 || block_size == 0 {
        return Err(invalid_args("dimensions and block_size must be > 0"));
    }
    let packed_k = k.div_ceil(2);
    let num_blocks = k.div_ceil(block_size);

    if x.len() < k {
        return Err(invalid_args(format!("x too small: need {k}, got {}", x.len())));
    }
    if weights_packed.len() < packed_k * m {
        return Err(invalid_args(format!(
            "weights too small: need {}, got {}",
            packed_k * m,
            weights_packed.len()
        )));
    }
    if scales.len() < m * num_blocks {
        return Err(invalid_args(format!(
            "scales too small: need {}, got {}",
            m * num_blocks,
            scales.len()
        )));
    }
    if y.len() < m {
        return Err(invalid_args(format!("y too small: need {m}, got {}", y.len())));
    }

    for col in 0..m {
        let mut acc = 0.0f32;
        for blk in 0..num_blocks {
            let blk_start = blk * block_size;
            let blk_end = (blk_start + block_size).min(k);
            let scale = scales[col * num_blocks + blk];
            let mut blk_acc = 0.0f32;
            for idx in blk_start..blk_end {
                let byte_idx = idx / 2;
                let byte = weights_packed[col * packed_k + byte_idx];
                let nibble = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let w = decode_int4(nibble);
                blk_acc += (w as f32) * x[idx];
            }
            acc += scale * blk_acc;
        }
        y[col] = acc;
    }
    Ok(())
}

/// Decode a 4-bit signed integer (sign-magnitude, 4-bit 2's complement).
#[inline(always)]
fn decode_i2s(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b11 => -1,
        _ => 0,
    }
}

/// Decode a 4-bit signed nibble as signed integer in [-8, 7].
#[inline(always)]
fn decode_int4(nibble: u8) -> i8 {
    let n = nibble & 0x0F;
    if n >= 8 { (n as i8) - 16 } else { n as i8 }
}

// ── Pack helpers for I2_S ──────────────────────────────────────────────

/// Pack four ternary values ({-1, 0, +1}) into one byte, LSB-first.
pub fn pack_i2s_values(vals: [i8; 4]) -> u8 {
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

/// Pack two INT4 values into one byte (low nibble first).
pub fn pack_int4_values(lo: i8, hi: i8) -> u8 {
    let encode = |v: i8| -> u8 { if v < 0 { (v + 16) as u8 } else { v as u8 } };
    (encode(lo) & 0x0F) | ((encode(hi) & 0x0F) << 4)
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive reference matmul for verification.
    fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (diff={})",
                (x - y).abs()
            );
        }
    }

    // ── MatmulConfig ───────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let c = MatmulConfig::default();
        assert_eq!(c.tile_m, 4);
        assert_eq!(c.tile_n, 8);
        assert_eq!(c.tile_k, 8);
        assert!(c.use_avx2);
    }

    #[test]
    fn test_config_new() {
        let c = MatmulConfig::new(2, 3, 4, false);
        assert_eq!(c.tile_m, 2);
        assert_eq!(c.tile_n, 3);
        assert_eq!(c.tile_k, 4);
        assert!(!c.use_avx2);
    }

    #[test]
    fn test_config_zero_tile_rejected() {
        let c = MatmulConfig::new(0, 4, 4, false);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_const_default() {
        let c = MatmulConfig::DEFAULT;
        assert_eq!(c.tile_m, 4);
        assert!(c.use_avx2);
    }

    // ── simd_matmul ────────────────────────────────────────────

    #[test]
    fn test_matmul_identity() {
        let n = 4;
        let mut eye = vec![0.0f32; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let a: Vec<f32> = (0..n * n).map(|i| i as f32 + 1.0).collect();
        let mut c = vec![0.0f32; n * n];
        simd_matmul(&a, &eye, &mut c, n, n, n, &MatmulConfig::default()).unwrap();
        assert_close(&c, &a, 1e-5);
    }

    #[test]
    fn test_matmul_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
        let mut c = vec![0.0f32; 4];
        simd_matmul(&a, &b, &mut c, 2, 2, 2, &MatmulConfig::default()).unwrap();
        assert_close(&c, &[19.0, 22.0, 43.0, 50.0], 1e-5);
    }

    #[test]
    fn test_matmul_rect() {
        let (m, n, k) = (3, 5, 4);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.2).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul(&a, &b, &mut c, m, n, k, &MatmulConfig::default()).unwrap();
        assert_close(&c, &expected, 1e-4);
    }

    #[test]
    fn test_matmul_large() {
        let (m, n, k) = (33, 17, 25);
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.07).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.11).cos()).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul(&a, &b, &mut c, m, n, k, &MatmulConfig::default()).unwrap();
        assert_close(&c, &expected, 1e-3);
    }

    #[test]
    fn test_matmul_scalar_fallback() {
        let (m, n, k) = (5, 6, 7);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.5).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        let cfg = MatmulConfig::new(4, 4, 4, false);
        simd_matmul(&a, &b, &mut c, m, n, k, &cfg).unwrap();
        assert_close(&c, &expected, 1e-3);
    }

    #[test]
    fn test_matmul_small_tiles() {
        let (m, n, k) = (10, 10, 10);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.02).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        let cfg = MatmulConfig::new(2, 2, 2, true);
        simd_matmul(&a, &b, &mut c, m, n, k, &cfg).unwrap();
        assert_close(&c, &expected, 1e-3);
    }

    #[test]
    fn test_matmul_1x1() {
        let mut c = [0.0f32];
        simd_matmul(&[3.0], &[7.0], &mut c, 1, 1, 1, &MatmulConfig::default()).unwrap();
        assert_close(&c, &[21.0], 1e-6);
    }

    #[test]
    fn test_matmul_zero_dim_rejected() {
        let mut c = [0.0f32; 4];
        assert!(
            simd_matmul(&[1.0; 4], &[1.0; 4], &mut c, 0, 2, 2, &MatmulConfig::default()).is_err()
        );
    }

    #[test]
    fn test_matmul_a_too_small() {
        let mut c = [0.0f32; 4];
        assert!(
            simd_matmul(&[1.0; 3], &[1.0; 4], &mut c, 2, 2, 2, &MatmulConfig::default()).is_err()
        );
    }

    #[test]
    fn test_matmul_b_too_small() {
        let mut c = [0.0f32; 4];
        assert!(
            simd_matmul(&[1.0; 4], &[1.0; 3], &mut c, 2, 2, 2, &MatmulConfig::default()).is_err()
        );
    }

    #[test]
    fn test_matmul_c_too_small() {
        let mut c = [0.0f32; 3];
        assert!(
            simd_matmul(&[1.0; 4], &[1.0; 4], &mut c, 2, 2, 2, &MatmulConfig::default()).is_err()
        );
    }

    #[test]
    fn test_matmul_dimension_overflow_rejected() {
        let mut c = [0.0f32; 1];
        assert!(
            simd_matmul(&[1.0], &[1.0], &mut c, usize::MAX, 2, 2, &MatmulConfig::default())
                .is_err()
        );
    }

    // ── simd_matmul_transposed ─────────────────────────────────

    #[test]
    fn test_matmul_transposed_identity() {
        let n = 4;
        let mut eye = vec![0.0f32; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let a: Vec<f32> = (0..n * n).map(|i| i as f32 + 1.0).collect();
        let mut c = vec![0.0f32; n * n];
        simd_matmul_transposed(&a, &eye, &mut c, n, n, n, &MatmulConfig::default()).unwrap();
        assert_close(&c, &a, 1e-5);
    }

    #[test]
    fn test_matmul_transposed_2x2() {
        // A = [[1,2],[3,4]], B^T = [[5,7],[6,8]] → B = [[5,6],[7,8]]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b_t = [5.0, 7.0, 6.0, 8.0]; // rows of B^T
        let mut c = vec![0.0f32; 4];
        simd_matmul_transposed(&a, &b_t, &mut c, 2, 2, 2, &MatmulConfig::default()).unwrap();
        assert_close(&c, &[19.0, 22.0, 43.0, 50.0], 1e-5);
    }

    #[test]
    fn test_matmul_transposed_rect() {
        let (m, n, k) = (4, 6, 5);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.2).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        // Transpose B → B^T[n, k]
        let mut b_t = vec![0.0f32; n * k];
        for p in 0..k {
            for j in 0..n {
                b_t[j * k + p] = b[p * n + j];
            }
        }
        let mut c = vec![0.0f32; m * n];
        simd_matmul_transposed(&a, &b_t, &mut c, m, n, k, &MatmulConfig::default()).unwrap();
        assert_close(&c, &expected, 1e-4);
    }

    #[test]
    fn test_matmul_transposed_zero_dim_rejected() {
        let mut c = [0.0f32; 4];
        assert!(
            simd_matmul_transposed(&[1.0; 4], &[1.0; 4], &mut c, 0, 2, 2, &MatmulConfig::default())
                .is_err()
        );
    }

    #[test]
    fn test_matmul_transposed_b_too_small() {
        let mut c = [0.0f32; 4];
        assert!(
            simd_matmul_transposed(&[1.0; 4], &[1.0; 3], &mut c, 2, 2, 2, &MatmulConfig::default())
                .is_err()
        );
    }

    #[test]
    fn test_matmul_transposed_dimension_overflow_rejected() {
        let mut c = [0.0f32; 1];
        assert!(
            simd_matmul_transposed(
                &[1.0],
                &[1.0],
                &mut c,
                usize::MAX,
                2,
                2,
                &MatmulConfig::default()
            )
            .is_err()
        );
    }

    // ── simd_matvec ────────────────────────────────────────────

    #[test]
    fn test_matvec_identity() {
        let n = 4;
        let mut eye = vec![0.0f32; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let x: Vec<f32> = (1..=4).map(|i| i as f32).collect();
        let mut y = vec![0.0f32; n];
        simd_matvec(&eye, &x, &mut y, n, n).unwrap();
        assert_close(&y, &x, 1e-6);
    }

    #[test]
    fn test_matvec_2x3() {
        // A = [[1,2,3],[4,5,6]], x = [1,1,1] → y = [6, 15]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 1.0, 1.0];
        let mut y = [0.0f32; 2];
        simd_matvec(&a, &x, &mut y, 2, 3).unwrap();
        assert_close(&y, &[6.0, 15.0], 1e-5);
    }

    #[test]
    fn test_matvec_large() {
        let (m, k) = (64, 128);
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.03).cos()).collect();
        let mut y = vec![0.0f32; m];
        simd_matvec(&a, &x, &mut y, m, k).unwrap();
        // Verify against naive.
        let mut expected = vec![0.0f32; m];
        for i in 0..m {
            for p in 0..k {
                expected[i] += a[i * k + p] * x[p];
            }
        }
        assert_close(&y, &expected, 1e-3);
    }

    #[test]
    fn test_matvec_zero_dim_rejected() {
        let mut y = [0.0f32; 2];
        assert!(simd_matvec(&[1.0; 4], &[1.0; 2], &mut y, 0, 2).is_err());
    }

    #[test]
    fn test_matvec_x_too_small() {
        let mut y = [0.0f32; 2];
        assert!(simd_matvec(&[1.0; 4], &[1.0; 1], &mut y, 2, 2).is_err());
    }

    #[test]
    fn test_matvec_y_too_small() {
        let mut y = [0.0f32; 1];
        assert!(simd_matvec(&[1.0; 4], &[1.0; 2], &mut y, 2, 2).is_err());
    }

    // ── simd_batch_matvec ──────────────────────────────────────

    #[test]
    fn test_batch_matvec_single_batch() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [0.0f32; 2];
        simd_batch_matvec(&a, &x, &mut y, 1, 2, 2).unwrap();
        assert_close(&y, &[3.0, 7.0], 1e-5);
    }

    #[test]
    fn test_batch_matvec_two_batches() {
        // batch0: A=[[1,2],[3,4]], x=[1,0] → [1,3]
        // batch1: A=[[5,6],[7,8]], x=[0,1] → [6,8]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = [1.0, 0.0, 0.0, 1.0];
        let mut y = [0.0f32; 4];
        simd_batch_matvec(&a, &x, &mut y, 2, 2, 2).unwrap();
        assert_close(&y, &[1.0, 3.0, 6.0, 8.0], 1e-5);
    }

    #[test]
    fn test_batch_matvec_zero_dim_rejected() {
        let mut y = [0.0f32; 4];
        assert!(simd_batch_matvec(&[1.0; 8], &[1.0; 4], &mut y, 0, 2, 2).is_err());
    }

    #[test]
    fn test_batch_matvec_a_too_small() {
        let mut y = [0.0f32; 4];
        assert!(simd_batch_matvec(&[1.0; 7], &[1.0; 4], &mut y, 2, 2, 2).is_err());
    }

    // ── outer_product ──────────────────────────────────────────

    #[test]
    fn test_outer_product_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0];
        // [[4,5],[8,10],[12,15]]
        let mut c = vec![0.0f32; 6];
        outer_product(&a, &b, &mut c, 3, 2).unwrap();
        assert_close(&c, &[4.0, 5.0, 8.0, 10.0, 12.0, 15.0], 1e-5);
    }

    #[test]
    fn test_outer_product_1x1() {
        let mut c = [0.0f32];
        outer_product(&[3.0], &[7.0], &mut c, 1, 1).unwrap();
        assert_close(&c, &[21.0], 1e-6);
    }

    #[test]
    fn test_outer_product_large() {
        let m = 16;
        let n = 20;
        let a: Vec<f32> = (0..m).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let mut c = vec![0.0f32; m * n];
        outer_product(&a, &b, &mut c, m, n).unwrap();
        for i in 0..m {
            for j in 0..n {
                assert!((c[i * n + j] - a[i] * b[j]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_outer_product_zero_dim_rejected() {
        let mut c = [0.0f32; 4];
        assert!(outer_product(&[1.0], &[1.0], &mut c, 0, 1).is_err());
    }

    #[test]
    fn test_outer_product_c_too_small() {
        let mut c = [0.0f32; 1];
        assert!(outer_product(&[1.0, 2.0], &[3.0, 4.0], &mut c, 2, 2).is_err());
    }

    // ── matrix_add ─────────────────────────────────────────────

    #[test]
    fn test_add_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        let mut c = [0.0f32; 4];
        matrix_add(&a, &b, &mut c, 2, 2).unwrap();
        assert_close(&c, &[11.0, 22.0, 33.0, 44.0], 1e-6);
    }

    #[test]
    fn test_add_large() {
        let n = 64;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
        let mut c = vec![0.0f32; n];
        matrix_add(&a, &b, &mut c, 1, n).unwrap();
        assert_close(&c, &vec![0.0f32; n], 1e-6);
    }

    #[test]
    fn test_add_zero_dim_rejected() {
        let mut c = [0.0f32; 4];
        assert!(matrix_add(&[1.0; 4], &[1.0; 4], &mut c, 0, 4).is_err());
    }

    #[test]
    fn test_add_buf_too_small() {
        let mut c = [0.0f32; 3];
        assert!(matrix_add(&[1.0; 4], &[1.0; 4], &mut c, 2, 2).is_err());
    }

    // ── matrix_sub ─────────────────────────────────────────────

    #[test]
    fn test_sub_basic() {
        let a = [10.0, 20.0, 30.0, 40.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut c = [0.0f32; 4];
        matrix_sub(&a, &b, &mut c, 2, 2).unwrap();
        assert_close(&c, &[9.0, 18.0, 27.0, 36.0], 1e-6);
    }

    #[test]
    fn test_sub_self_is_zero() {
        let a = [5.0, 3.0, 7.0, 1.0];
        let mut c = [0.0f32; 4];
        matrix_sub(&a, &a, &mut c, 2, 2).unwrap();
        assert_close(&c, &[0.0; 4], 1e-6);
    }

    #[test]
    fn test_sub_zero_dim_rejected() {
        let mut c = [0.0f32; 4];
        assert!(matrix_sub(&[1.0; 4], &[1.0; 4], &mut c, 0, 4).is_err());
    }

    // ── matrix_scale ───────────────────────────────────────────

    #[test]
    fn test_scale_by_two() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let mut c = [0.0f32; 4];
        matrix_scale(&a, &mut c, 2, 2, 2.0).unwrap();
        assert_close(&c, &[2.0, 4.0, 6.0, 8.0], 1e-6);
    }

    #[test]
    fn test_scale_by_zero() {
        let a = [5.0, 3.0, 7.0, 1.0];
        let mut c = [0.0f32; 4];
        matrix_scale(&a, &mut c, 2, 2, 0.0).unwrap();
        assert_close(&c, &[0.0; 4], 1e-6);
    }

    #[test]
    fn test_scale_by_negative() {
        let a = [1.0, -2.0];
        let mut c = [0.0f32; 2];
        matrix_scale(&a, &mut c, 1, 2, -3.0).unwrap();
        assert_close(&c, &[-3.0, 6.0], 1e-6);
    }

    #[test]
    fn test_scale_large() {
        let n = 64;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut c = vec![0.0f32; n];
        matrix_scale(&a, &mut c, 1, n, 0.5).unwrap();
        let expected: Vec<f32> = a.iter().map(|x| x * 0.5).collect();
        assert_close(&c, &expected, 1e-6);
    }

    #[test]
    fn test_scale_zero_dim_rejected() {
        let mut c = [0.0f32; 4];
        assert!(matrix_scale(&[1.0; 4], &mut c, 0, 4, 1.0).is_err());
    }

    // ── matrix_transpose ───────────────────────────────────────

    #[test]
    fn test_transpose_2x3() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let result = matrix_transpose(&a, 2, 3, None).unwrap();
        // 3×2: [[1,4],[2,5],[3,6]]
        assert_close(&result, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-6);
    }

    #[test]
    fn test_transpose_square() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let result = matrix_transpose(&a, 2, 2, None).unwrap();
        assert_close(&result, &[1.0, 3.0, 2.0, 4.0], 1e-6);
    }

    #[test]
    fn test_transpose_1x1() {
        let result = matrix_transpose(&[42.0], 1, 1, None).unwrap();
        assert_close(&result, &[42.0], 1e-6);
    }

    #[test]
    fn test_transpose_out_of_place() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let mut out = vec![0.0f32; 6];
        matrix_transpose(&a, 2, 3, Some(&mut out)).unwrap();
        assert_close(&out, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-6);
    }

    #[test]
    fn test_transpose_involution() {
        // (A^T)^T = A
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let at = matrix_transpose(&a, 2, 3, None).unwrap();
        let att = matrix_transpose(&at, 3, 2, None).unwrap();
        assert_close(&att, &a, 1e-6);
    }

    #[test]
    fn test_transpose_zero_dim_rejected() {
        assert!(matrix_transpose(&[1.0; 4], 0, 4, None).is_err());
    }

    #[test]
    fn test_transpose_out_too_small() {
        let mut out = vec![0.0f32; 3];
        assert!(matrix_transpose(&[1.0; 6], 2, 3, Some(&mut out)).is_err());
    }

    // ── strassen_matmul ────────────────────────────────────────

    #[test]
    fn test_strassen_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        strassen_matmul(&a, &b, &mut c, 2, 1).unwrap();
        assert_close(&c, &[19.0, 22.0, 43.0, 50.0], 1e-4);
    }

    #[test]
    fn test_strassen_4x4() {
        let n = 4;
        let a: Vec<f32> = (0..n * n).map(|i| i as f32 + 1.0).collect();
        let b: Vec<f32> = (0..n * n).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let expected = naive_matmul(&a, &b, n, n, n);
        let mut c = vec![0.0f32; n * n];
        strassen_matmul(&a, &b, &mut c, n, 2).unwrap();
        assert_close(&c, &expected, 1e-3);
    }

    #[test]
    fn test_strassen_8x8() {
        let n = 8;
        let a: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.1).sin()).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.07).cos()).collect();
        let expected = naive_matmul(&a, &b, n, n, n);
        let mut c = vec![0.0f32; n * n];
        strassen_matmul(&a, &b, &mut c, n, 4).unwrap();
        assert_close(&c, &expected, 1e-3);
    }

    #[test]
    fn test_strassen_identity() {
        let n = 4;
        let mut eye = vec![0.0f32; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let a: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
        let mut c = vec![0.0f32; n * n];
        strassen_matmul(&a, &eye, &mut c, n, 2).unwrap();
        assert_close(&c, &a, 1e-4);
    }

    #[test]
    fn test_strassen_odd_falls_back() {
        // n=3 is odd → immediate fallback to naive.
        let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        let mut c = [0.0f32; 9];
        strassen_matmul(&a, &b, &mut c, 3, 2).unwrap();
        assert_close(&c, &b, 1e-5);
    }

    #[test]
    fn test_strassen_zero_dim_rejected() {
        let mut c = [0.0f32; 4];
        assert!(strassen_matmul(&[1.0; 4], &[1.0; 4], &mut c, 0, 2).is_err());
    }

    #[test]
    fn test_strassen_buf_too_small() {
        let mut c = [0.0f32; 3];
        assert!(strassen_matmul(&[1.0; 4], &[1.0; 4], &mut c, 2, 1).is_err());
    }

    // ── packed_matmul ──────────────────────────────────────────

    #[test]
    fn test_pack_b_roundtrip() {
        let (k, n) = (4, 6);
        let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let packed = pack_b_col_panels(&b, k, n, 4).unwrap();
        // Verify: we can reconstruct B from packed.
        let mut b_rec = vec![0.0f32; k * n];
        let panel_w = 4;
        let num_panels = n.div_ceil(panel_w);
        for jp in 0..num_panels {
            let j0 = jp * panel_w;
            let pw = panel_w.min(n - j0);
            for p in 0..k {
                for jj in 0..pw {
                    b_rec[p * n + j0 + jj] = packed[jp * k * panel_w + p * panel_w + jj];
                }
            }
        }
        assert_close(&b_rec, &b, 1e-6);
    }

    #[test]
    fn test_packed_matmul_basic() {
        let (m, n, k) = (3, 5, 4);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.2).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let packed = pack_b_col_panels(&b, k, n, 4).unwrap();
        let mut c = vec![0.0f32; m * n];
        packed_matmul(&a, &packed, &mut c, m, n, k, 4).unwrap();
        assert_close(&c, &expected, 1e-4);
    }

    #[test]
    fn test_packed_matmul_panel_w_1() {
        let (m, n, k) = (2, 3, 2);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 + 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 + 1.0).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let packed = pack_b_col_panels(&b, k, n, 1).unwrap();
        let mut c = vec![0.0f32; m * n];
        packed_matmul(&a, &packed, &mut c, m, n, k, 1).unwrap();
        assert_close(&c, &expected, 1e-4);
    }

    #[test]
    fn test_packed_matmul_panel_w_larger_than_n() {
        let (m, n, k) = (2, 3, 4);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.3).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let packed = pack_b_col_panels(&b, k, n, 8).unwrap();
        let mut c = vec![0.0f32; m * n];
        packed_matmul(&a, &packed, &mut c, m, n, k, 8).unwrap();
        assert_close(&c, &expected, 1e-4);
    }

    #[test]
    fn test_packed_matmul_zero_dim_rejected() {
        let mut c = [0.0f32; 4];
        assert!(packed_matmul(&[1.0; 4], &[1.0; 8], &mut c, 0, 2, 2, 4).is_err());
    }

    #[test]
    fn test_packed_matmul_zero_panel_rejected() {
        let mut c = [0.0f32; 4];
        assert!(packed_matmul(&[1.0; 4], &[1.0; 8], &mut c, 2, 2, 2, 0).is_err());
    }

    // ── gemv_quantized (INT2) ──────────────────────────────────

    #[test]
    fn test_gemv_quantized_all_ones() {
        // 2 outputs, k=4, all weights = +1, scale = 1.0
        let k = 4;
        let m = 2;
        let block_size = 4;
        let x = [1.0, 2.0, 3.0, 4.0];
        // Pack: all +1 → 0b01_01_01_01 = 0x55
        let packed = vec![0x55u8; m]; // one byte per column (k=4, packed_k=1)
        let scales = vec![1.0f32; m]; // one block each
        let mut y = vec![0.0f32; m];
        gemv_quantized(&x, &packed, &scales, &mut y, m, k, block_size).unwrap();
        // Each output = 1*1 + 1*2 + 1*3 + 1*4 = 10
        assert_close(&y, &[10.0, 10.0], 1e-5);
    }

    #[test]
    fn test_gemv_quantized_mixed() {
        let k = 4;
        let m = 1;
        let block_size = 4;
        let x = [1.0, 2.0, 3.0, 4.0];
        // Weights: [+1, -1, 0, +1]
        let packed = vec![pack_i2s_values([1, -1, 0, 1])];
        let scales = vec![2.0f32];
        let mut y = vec![0.0f32; 1];
        gemv_quantized(&x, &packed, &scales, &mut y, m, k, block_size).unwrap();
        // dot = 1*1 + (-1)*2 + 0*3 + 1*4 = 3; scaled = 2*3 = 6
        assert_close(&y, &[6.0], 1e-5);
    }

    #[test]
    fn test_gemv_quantized_two_blocks() {
        let k = 8;
        let m = 1;
        let block_size = 4;
        let x = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // All weights = +1 across two blocks.
        let packed = vec![0x55, 0x55]; // packed_k = 2
        let scales = vec![1.0, 2.0]; // block0 scale=1, block1 scale=2
        let mut y = vec![0.0f32; 1];
        gemv_quantized(&x, &packed, &scales, &mut y, m, k, block_size).unwrap();
        // block0: 4*1=4, block1: 4*2=8 → 12
        assert_close(&y, &[12.0], 1e-5);
    }

    #[test]
    fn test_gemv_quantized_zero_dim_rejected() {
        let mut y = [0.0f32; 1];
        assert!(gemv_quantized(&[1.0], &[0x55], &[1.0], &mut y, 0, 4, 4).is_err());
    }

    #[test]
    fn test_gemv_quantized_zero_block_rejected() {
        let mut y = [0.0f32; 1];
        assert!(gemv_quantized(&[1.0], &[0x55], &[1.0], &mut y, 1, 4, 0).is_err());
    }

    #[test]
    fn test_gemv_quantized_x_too_small() {
        let mut y = [0.0f32; 1];
        assert!(gemv_quantized(&[1.0], &[0x55], &[1.0], &mut y, 1, 4, 4).is_err());
    }

    // ── gemv_quantized_int4 ────────────────────────────────────

    #[test]
    fn test_gemv_int4_basic() {
        let k = 2;
        let m = 1;
        let block_size = 2;
        let x = [1.0, 2.0];
        // Weights: [3, -2] → packed: lo=3 (0x03), hi=-2 (0x0E) → byte 0xE3
        let packed = vec![pack_int4_values(3, -2)];
        let scales = vec![1.0f32];
        let mut y = vec![0.0f32; 1];
        gemv_quantized_int4(&x, &packed, &scales, &mut y, m, k, block_size).unwrap();
        // 3*1 + (-2)*2 = -1
        assert_close(&y, &[-1.0], 1e-5);
    }

    #[test]
    fn test_gemv_int4_two_outputs() {
        let k = 2;
        let m = 2;
        let block_size = 2;
        let x = [1.0, 1.0];
        // col0: [1, 1], col1: [2, -3]
        let packed = vec![pack_int4_values(1, 1), pack_int4_values(2, -3)];
        let scales = vec![1.0, 1.0];
        let mut y = vec![0.0f32; 2];
        gemv_quantized_int4(&x, &packed, &scales, &mut y, m, k, block_size).unwrap();
        // col0: 1+1=2, col1: 2+(-3)=-1
        assert_close(&y, &[2.0, -1.0], 1e-5);
    }

    #[test]
    fn test_gemv_int4_zero_dim_rejected() {
        let mut y = [0.0f32; 1];
        assert!(gemv_quantized_int4(&[1.0], &[0x00], &[1.0], &mut y, 0, 2, 2).is_err());
    }

    // ── Pack helpers ───────────────────────────────────────────

    #[test]
    fn test_pack_i2s_roundtrip() {
        let vals = [1i8, -1, 0, 1];
        let byte = pack_i2s_values(vals);
        for (i, &v) in vals.iter().enumerate() {
            let bits = (byte >> (i * 2)) & 0x03;
            assert_eq!(decode_i2s(bits), v, "mismatch at position {i}");
        }
    }

    #[test]
    fn test_pack_int4_roundtrip() {
        for lo in -8i8..8 {
            for hi in -8i8..8 {
                let byte = pack_int4_values(lo, hi);
                let lo_dec = decode_int4(byte & 0x0F);
                let hi_dec = decode_int4(byte >> 4);
                assert_eq!(lo_dec, lo, "lo mismatch: {lo}");
                assert_eq!(hi_dec, hi, "hi mismatch: {hi}");
            }
        }
    }

    // ── Cross-function properties ──────────────────────────────

    #[test]
    fn test_matmul_matches_matvec_single_col() {
        // Matmul with n=1 should match matvec.
        let (m, k) = (5, 8);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let x: Vec<f32> = (0..k).map(|i| i as f32 * 0.3).collect();
        let mut y_mv = vec![0.0f32; m];
        simd_matvec(&a, &x, &mut y_mv, m, k).unwrap();
        let mut y_mm = vec![0.0f32; m];
        simd_matmul(&a, &x, &mut y_mm, m, 1, k, &MatmulConfig::default()).unwrap();
        assert_close(&y_mv, &y_mm, 1e-4);
    }

    #[test]
    fn test_add_sub_inverse() {
        let n = 16;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.7).collect();
        let mut sum = vec![0.0f32; n];
        matrix_add(&a, &b, &mut sum, 1, n).unwrap();
        let mut diff = vec![0.0f32; n];
        matrix_sub(&sum, &b, &mut diff, 1, n).unwrap();
        assert_close(&diff, &a, 1e-5);
    }

    #[test]
    fn test_scale_then_add_is_linear() {
        // alpha*(A+B) = alpha*A + alpha*B
        let n = 12;
        let alpha = 2.5f32;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let mut a_plus_b = vec![0.0f32; n];
        matrix_add(&a, &b, &mut a_plus_b, 1, n).unwrap();
        let mut lhs = vec![0.0f32; n];
        matrix_scale(&a_plus_b, &mut lhs, 1, n, alpha).unwrap();
        let mut sa = vec![0.0f32; n];
        let mut sb = vec![0.0f32; n];
        matrix_scale(&a, &mut sa, 1, n, alpha).unwrap();
        matrix_scale(&b, &mut sb, 1, n, alpha).unwrap();
        let mut rhs = vec![0.0f32; n];
        matrix_add(&sa, &sb, &mut rhs, 1, n).unwrap();
        assert_close(&lhs, &rhs, 1e-4);
    }

    #[test]
    fn test_outer_product_rank1() {
        // rank-1 outer product: (a ⊗ b) * x = a * (b · x)
        let m = 3;
        let n = 4;
        let a: Vec<f32> = (0..m).map(|i| i as f32 + 1.0).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let mut ab = vec![0.0f32; m * n];
        outer_product(&a, &b, &mut ab, m, n).unwrap();
        let mut y1 = vec![0.0f32; m];
        simd_matvec(&ab, &x, &mut y1, m, n).unwrap();
        let bdotx: f32 = b.iter().zip(x.iter()).map(|(bi, xi)| bi * xi).sum();
        let y2: Vec<f32> = a.iter().map(|ai| ai * bdotx).collect();
        assert_close(&y1, &y2, 1e-4);
    }

    #[test]
    fn test_strassen_vs_simd_matmul() {
        let n = 8;
        let a: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.13).sin()).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.17).cos()).collect();
        let mut c1 = vec![0.0f32; n * n];
        simd_matmul(&a, &b, &mut c1, n, n, n, &MatmulConfig::default()).unwrap();
        let mut c2 = vec![0.0f32; n * n];
        strassen_matmul(&a, &b, &mut c2, n, 4).unwrap();
        assert_close(&c1, &c2, 1e-3);
    }

    #[test]
    fn test_packed_vs_naive() {
        let (m, n, k) = (7, 11, 9);
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.1).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.2).cos()).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let packed = pack_b_col_panels(&b, k, n, 4).unwrap();
        let mut c = vec![0.0f32; m * n];
        packed_matmul(&a, &packed, &mut c, m, n, k, 4).unwrap();
        assert_close(&c, &expected, 1e-3);
    }

    #[test]
    fn test_transposed_vs_regular() {
        let (m, n, k) = (6, 8, 5);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.2).collect();
        let mut c1 = vec![0.0f32; m * n];
        simd_matmul(&a, &b, &mut c1, m, n, k, &MatmulConfig::default()).unwrap();
        // Transpose B.
        let mut b_t = vec![0.0f32; n * k];
        for p in 0..k {
            for j in 0..n {
                b_t[j * k + p] = b[p * n + j];
            }
        }
        let mut c2 = vec![0.0f32; m * n];
        simd_matmul_transposed(&a, &b_t, &mut c2, m, n, k, &MatmulConfig::default()).unwrap();
        assert_close(&c1, &c2, 1e-4);
    }
}
