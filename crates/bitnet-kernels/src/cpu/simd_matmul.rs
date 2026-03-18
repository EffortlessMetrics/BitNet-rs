//! CPU SIMD-optimized general-purpose matrix multiplication kernels.
//!
//! Provides f32 GEMM, I2_S quantized matmul (fused dequant + GEMM), and
//! batched matrix multiplication with runtime AVX2 detection and scalar
//! fallback.  The public entry points auto-dispatch to the fastest
//! available code path at runtime via
//! `std::arch::is_x86_feature_detected!("avx2")` on x86_64 or fall back
//! to portable scalar loops on other architectures.
//!
//! # Layout conventions
//!
//! * All matrices are **row-major** unless the corresponding
//!   `transpose_*` flag in [`SimdMatmulConfig`] is set.
//! * I2_S packed weights use the same column-major packing layout as
//!   [`super::quantized_matmul`]: each output column stores
//!   `ceil(k/4)` bytes, 4 ternary values per byte (2 bits each,
//!   LSB-first).
#![allow(unsafe_op_in_unsafe_fn)]

use bitnet_common::{BitNetError, KernelError, Result};
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

// ── Configuration ──────────────────────────────────────────────────────

/// Parameters for a single GEMM invocation.
///
/// `C = alpha * op(A) * op(B) + beta * C`
///
/// where `op(X) = X` when the corresponding transpose flag is false,
/// and `op(X) = X^T` otherwise.
#[derive(Debug, Clone)]
pub struct SimdMatmulConfig {
    /// Rows of `op(A)` and `C`.
    pub m: usize,
    /// Columns of `op(B)` and `C`.
    pub n: usize,
    /// Inner (shared) dimension.
    pub k: usize,
    /// Scalar multiplier applied to the product `op(A) * op(B)`.
    pub alpha: f32,
    /// Scalar multiplier applied to the existing contents of `C`
    /// before accumulation.
    pub beta: f32,
    /// When `true`, `A` is stored as `k × m` (transposed).
    pub transpose_a: bool,
    /// When `true`, `B` is stored as `n × k` (transposed).
    pub transpose_b: bool,
}

impl SimdMatmulConfig {
    /// Create a minimal config for `C = A * B` (no scaling, no
    /// transposition).
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k, alpha: 1.0, beta: 0.0, transpose_a: false, transpose_b: false }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Decode a 2-bit I2_S code to its signed value.
#[inline(always)]
fn decode_i2s(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b11 => -1,
        _ => 0, // 0b00 → 0, 0b10 (unused) → 0
    }
}

/// Read element `(row, col)` from a possibly-transposed matrix.
#[inline(always)]
fn elem(data: &[f32], row: usize, col: usize, ld: usize, transposed: bool) -> f32 {
    if transposed { data[col * ld + row] } else { data[row * ld + col] }
}

fn validate_f32_args(a: &[f32], b: &[f32], c: &[f32], cfg: &SimdMatmulConfig) -> Result<()> {
    let (m, n, k) = (cfg.m, cfg.n, cfg.k);
    if m == 0 || n == 0 || k == 0 {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("dimensions must be > 0: m={m}, n={n}, k={k}"),
        }));
    }
    let a_need = if cfg.transpose_a { k * m } else { m * k };
    let b_need = if cfg.transpose_b { n * k } else { k * n };
    if a.len() < a_need {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("A too small: need {a_need}, got {}", a.len()),
        }));
    }
    if b.len() < b_need {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("B too small: need {b_need}, got {}", b.len()),
        }));
    }
    if c.len() < m * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("C too small: need {}, got {}", m * n, c.len()),
        }));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_i2s_args(
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
            reason: format!("activations too small: need {}, got {}", m * k, activations.len()),
        }));
    }
    if weights_packed.len() < packed_k * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "weights_packed too small: need {}, got {}",
                packed_k * n,
                weights_packed.len()
            ),
        }));
    }
    if scales.len() < n * num_blocks_k {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("scales too small: need {}, got {}", n * num_blocks_k, scales.len()),
        }));
    }
    if out.len() < m * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("output too small: need {}, got {}", m * n, out.len()),
        }));
    }
    Ok(())
}

// ── Runtime dispatch ───────────────────────────────────────────────────

/// Returns `true` when AVX2 is available at runtime.
#[inline]
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ── Public API ─────────────────────────────────────────────────────────

/// General-purpose f32 GEMM with optional transposition and scaling.
///
/// `C = alpha * op(A) * op(B) + beta * C`
///
/// Dispatches to an AVX2 inner loop when available; otherwise uses a
/// portable scalar implementation.
pub fn simd_matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], cfg: &SimdMatmulConfig) -> Result<()> {
    validate_f32_args(a, b, c, cfg)?;

    let SimdMatmulConfig { m, n, k, alpha, beta, transpose_a, transpose_b } = *cfg;

    let ld_a = if transpose_a { m } else { k };
    let ld_b = if transpose_b { k } else { n };

    // Apply beta to existing C.
    if beta == 0.0 {
        c[..m * n].fill(0.0);
    } else if (beta - 1.0).abs() > f32::EPSILON {
        for v in c[..m * n].iter_mut() {
            *v *= beta;
        }
    }

    if has_avx2() {
        #[cfg(target_arch = "x86_64")]
        // Safety: guarded by runtime AVX2 check above.
        unsafe {
            gemm_avx2(a, b, c, m, n, k, ld_a, ld_b, alpha, transpose_a, transpose_b);
        }
        #[cfg(not(target_arch = "x86_64"))]
        gemm_scalar(a, b, c, m, n, k, ld_a, ld_b, alpha, transpose_a, transpose_b);
    } else {
        gemm_scalar(a, b, c, m, n, k, ld_a, ld_b, alpha, transpose_a, transpose_b);
    }
    Ok(())
}

/// I2_S quantized matrix multiplication (fused dequant + GEMM).
///
/// `out[m×n] = activations[m×k] · dequant(weights_packed[k×n], scales)`
///
/// Weights are stored in 2-bit I2_S packed format (4 values/byte),
/// column-major within each output column.  `scales` has one entry
/// per block of `block_size` elements along `k` per column:
/// `n * ceil(k / block_size)` entries total.
///
/// Dispatches to an AVX2 path when available on x86_64.
#[allow(clippy::too_many_arguments)]
pub fn simd_matmul_i2s(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    validate_i2s_args(activations, weights_packed, scales, out, m, n, k, block_size)?;

    if has_avx2() {
        #[cfg(target_arch = "x86_64")]
        // Safety: guarded by runtime AVX2 check above.
        unsafe {
            i2s_gemm_avx2(activations, weights_packed, scales, out, m, n, k, block_size);
        }
        #[cfg(not(target_arch = "x86_64"))]
        i2s_gemm_scalar(activations, weights_packed, scales, out, m, n, k, block_size);
    } else {
        i2s_gemm_scalar(activations, weights_packed, scales, out, m, n, k, block_size);
    }
    Ok(())
}

/// Batched matrix multiplication.
///
/// Runs `simd_matmul_f32` independently for each `(A_i, B_i) → C_i`
/// triple. All batches share the same [`SimdMatmulConfig`].
pub fn simd_matmul_batched(
    a_batch: &[&[f32]],
    b_batch: &[&[f32]],
    c_batch: &mut [&mut [f32]],
    cfg: &SimdMatmulConfig,
) -> Result<()> {
    if a_batch.len() != b_batch.len() || a_batch.len() != c_batch.len() {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "batch size mismatch: A={}, B={}, C={}",
                a_batch.len(),
                b_batch.len(),
                c_batch.len()
            ),
        }));
    }
    if a_batch.is_empty() {
        return Ok(());
    }
    for (i, c) in c_batch.iter_mut().enumerate() {
        simd_matmul_f32(a_batch[i], b_batch[i], c, cfg)?;
    }
    Ok(())
}

// ── Scalar implementation ──────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn gemm_scalar(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    ld_a: usize,
    ld_b: usize,
    alpha: f32,
    ta: bool,
    tb: bool,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += elem(a, i, l, ld_a, ta) * elem(b, l, j, ld_b, tb);
            }
            c[i * n + j] += alpha * acc;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn i2s_gemm_scalar(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) {
    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);

    out[..m * n].fill(0.0);

    for row in 0..m {
        let a_row = &activations[row * k..(row + 1) * k];
        for col in 0..n {
            let mut acc = 0.0f32;
            for blk in 0..num_blocks_k {
                let blk_start = blk * block_size;
                let blk_end = (blk_start + block_size).min(k);
                let scale = scales[col * num_blocks_k + blk];

                for (rel, &a_val) in a_row[blk_start..blk_end].iter().enumerate() {
                    let idx = blk_start + rel;
                    let byte_idx = col * packed_k + idx / 4;
                    let bit_off = (idx % 4) * 2;
                    let bits = (weights_packed[byte_idx] >> bit_off) & 0x03;
                    let w = decode_i2s(bits) as f32 * scale;
                    acc += a_val * w;
                }
            }
            out[row * n + col] = acc;
        }
    }
}

// ── AVX2 fast paths (x86_64 only) ─────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_avx2(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    ld_a: usize,
    ld_b: usize,
    alpha: f32,
    ta: bool,
    tb: bool,
) {
    // For non-transposed, contiguous case we vectorise the k-loop.
    if !ta && !tb {
        for i in 0..m {
            for j in 0..n {
                let mut acc = _mm256_setzero_ps();
                let mut l = 0usize;
                while l + 8 <= k {
                    let av = _mm256_loadu_ps(a.as_ptr().add(i * ld_a + l));
                    // Gather b[l..l+8][j] with stride ld_b.
                    let bv = _mm256_set_ps(
                        *b.get_unchecked((l + 7) * ld_b + j),
                        *b.get_unchecked((l + 6) * ld_b + j),
                        *b.get_unchecked((l + 5) * ld_b + j),
                        *b.get_unchecked((l + 4) * ld_b + j),
                        *b.get_unchecked((l + 3) * ld_b + j),
                        *b.get_unchecked((l + 2) * ld_b + j),
                        *b.get_unchecked((l + 1) * ld_b + j),
                        *b.get_unchecked(l * ld_b + j),
                    );
                    acc = _mm256_fmadd_ps(av, bv, acc);
                    l += 8;
                }
                // Horizontal sum of acc.
                let mut sum = hsum_avx2(acc);
                // Scalar tail.
                for l2 in l..k {
                    sum += a[i * ld_a + l2] * b[l2 * ld_b + j];
                }
                *c.get_unchecked_mut(i * n + j) += alpha * sum;
            }
        }
    } else {
        // Transposed cases: fall back to scalar elem() accessors.
        gemm_scalar(a, b, c, m, n, k, ld_a, ld_b, alpha, ta, tb);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn i2s_gemm_avx2(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) {
    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);

    out[..m * n].fill(0.0);

    // Temporary buffer for dequantised block weights (max 256).
    let mut w_buf = [0i8; 256];

    for col in 0..n {
        for blk in 0..num_blocks_k {
            let blk_start = blk * block_size;
            let blk_end = (blk_start + block_size).min(k);
            let blk_len = blk_end - blk_start;
            let scale = *scales.get_unchecked(col * num_blocks_k + blk);

            // Dequant block into w_buf.
            for idx in blk_start..blk_end {
                let byte_idx = col * packed_k + idx / 4;
                let bit_off = (idx % 4) * 2;
                let bits = (*weights_packed.get_unchecked(byte_idx) >> bit_off) & 0x03;
                *w_buf.get_unchecked_mut(idx - blk_start) = decode_i2s(bits);
            }

            let scale_v = _mm256_set1_ps(scale);

            for row in 0..m {
                let a_base = row * k + blk_start;
                let mut acc = _mm256_setzero_ps();
                let mut r = 0usize;

                // Process 8 elements at a time.
                while r + 8 <= blk_len {
                    let av = _mm256_loadu_ps(activations.as_ptr().add(a_base + r));
                    // Convert i8 weights to f32 via epi32.
                    let wi = _mm256_set_epi32(
                        *w_buf.get_unchecked(r + 7) as i32,
                        *w_buf.get_unchecked(r + 6) as i32,
                        *w_buf.get_unchecked(r + 5) as i32,
                        *w_buf.get_unchecked(r + 4) as i32,
                        *w_buf.get_unchecked(r + 3) as i32,
                        *w_buf.get_unchecked(r + 2) as i32,
                        *w_buf.get_unchecked(r + 1) as i32,
                        *w_buf.get_unchecked(r) as i32,
                    );
                    let wf = _mm256_cvtepi32_ps(wi);
                    acc = _mm256_fmadd_ps(av, wf, acc);
                    r += 8;
                }

                // Multiply accumulated dot by scale.
                acc = _mm256_mul_ps(acc, scale_v);
                let mut sum = hsum_avx2(acc);

                // Scalar tail.
                for r2 in r..blk_len {
                    sum += activations[a_base + r2] * w_buf[r2] as f32 * scale;
                }

                *out.get_unchecked_mut(row * n + col) += sum;
            }
        }
    }
}

/// Horizontal sum of an `__m256` register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

// ── Tile configuration ─────────────────────────────────────────────────

/// Cache-tile parameters for [`simd_matmul_f32_tiled`].
///
/// Tiling partitions the GEMM into sub-blocks that fit in L1/L2 cache,
/// dramatically reducing cache misses for large matrices.  The default
/// tile sizes (64x64x64) target a 32 KiB L1d / 256 KiB L2 hierarchy
/// common on modern x86 desktop and server parts.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Tile rows (M dimension).  Must be > 0.
    pub tile_m: usize,
    /// Tile columns (N dimension).  Must be > 0.
    pub tile_n: usize,
    /// Tile depth (K dimension).  Must be > 0.
    pub tile_k: usize,
}

impl TileConfig {
    /// Sensible defaults for a 32 KiB L1d cache.
    pub const DEFAULT: Self = Self { tile_m: 64, tile_n: 64, tile_k: 64 };

    /// Smaller tiles for systems with a 16 KiB L1d cache.
    pub const SMALL: Self = Self { tile_m: 32, tile_n: 32, tile_k: 32 };

    /// Larger tiles for systems with a 64+ KiB L1d cache (e.g. Zen 4).
    pub const LARGE: Self = Self { tile_m: 128, tile_n: 128, tile_k: 128 };

    /// Create a custom tile configuration.
    pub fn new(tile_m: usize, tile_n: usize, tile_k: usize) -> Self {
        Self { tile_m, tile_n, tile_k }
    }
}

impl Default for TileConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ── Tiled f32 GEMM ────────────────────────────────────────────────────

/// Cache-tiled f32 GEMM with AVX2/AVX-512 dispatch.
///
/// Semantics are identical to [`simd_matmul_f32`] --- this variant
/// additionally partitions the work into cache-friendly tiles controlled
/// by [`TileConfig`].  For small matrices the tiling overhead is
/// negligible; for large ones it can yield 2-4x speedup from reduced
/// L2/L3 misses.
///
/// `C = alpha * op(A) * op(B) + beta * C`
pub fn simd_matmul_f32_tiled(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    cfg: &SimdMatmulConfig,
    tiles: &TileConfig,
) -> Result<()> {
    validate_f32_args(a, b, c, cfg)?;

    if tiles.tile_m == 0 || tiles.tile_n == 0 || tiles.tile_k == 0 {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: "tile dimensions must be > 0".into(),
        }));
    }

    let SimdMatmulConfig { m, n, k, alpha, beta, transpose_a, transpose_b } = *cfg;

    let ld_a = if transpose_a { m } else { k };
    let ld_b = if transpose_b { k } else { n };

    // Apply beta to existing C.
    if beta == 0.0 {
        c[..m * n].fill(0.0);
    } else if (beta - 1.0).abs() > f32::EPSILON {
        for v in c[..m * n].iter_mut() {
            *v *= beta;
        }
    }

    // Tile over K (outer) -> M -> N so that the B-tile stays hot in cache
    // across all M-rows that read it.
    for kk in (0..k).step_by(tiles.tile_k) {
        let k_end = (kk + tiles.tile_k).min(k);
        for ii in (0..m).step_by(tiles.tile_m) {
            let i_end = (ii + tiles.tile_m).min(m);
            for jj in (0..n).step_by(tiles.tile_n) {
                let j_end = (jj + tiles.tile_n).min(n);

                #[cfg(target_arch = "x86_64")]
                {
                    if has_avx512f() {
                        // Safety: guarded by runtime AVX-512F check.
                        unsafe {
                            gemm_tile_avx512(
                                a,
                                b,
                                c,
                                ii,
                                i_end,
                                jj,
                                j_end,
                                kk,
                                k_end,
                                n,
                                ld_a,
                                ld_b,
                                alpha,
                                transpose_a,
                                transpose_b,
                            );
                        }
                        continue;
                    }
                    if has_avx2() {
                        // Safety: guarded by runtime AVX2 check.
                        unsafe {
                            gemm_tile_avx2(
                                a,
                                b,
                                c,
                                ii,
                                i_end,
                                jj,
                                j_end,
                                kk,
                                k_end,
                                n,
                                ld_a,
                                ld_b,
                                alpha,
                                transpose_a,
                                transpose_b,
                            );
                        }
                        continue;
                    }
                }
                gemm_tile_scalar(
                    a,
                    b,
                    c,
                    ii,
                    i_end,
                    jj,
                    j_end,
                    kk,
                    k_end,
                    n,
                    ld_a,
                    ld_b,
                    alpha,
                    transpose_a,
                    transpose_b,
                );
            }
        }
    }
    Ok(())
}

// ── Runtime AVX-512F detection ─────────────────────────────────────────

/// Returns `true` when AVX-512F is available at runtime.
#[inline]
#[cfg(target_arch = "x86_64")]
fn has_avx512f() -> bool {
    is_x86_feature_detected!("avx512f")
}

// ── Scalar tile kernel ─────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn gemm_tile_scalar(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    k0: usize,
    k1: usize,
    n: usize,
    ld_a: usize,
    ld_b: usize,
    alpha: f32,
    ta: bool,
    tb: bool,
) {
    for i in i0..i1 {
        for j in j0..j1 {
            let mut acc = 0.0f32;
            for l in k0..k1 {
                acc += elem(a, i, l, ld_a, ta) * elem(b, l, j, ld_b, tb);
            }
            c[i * n + j] += alpha * acc;
        }
    }
}

// ── AVX2 tile kernel ───────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_tile_avx2(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    k0: usize,
    k1: usize,
    n: usize,
    ld_a: usize,
    ld_b: usize,
    alpha: f32,
    ta: bool,
    tb: bool,
) {
    // Fast path: both A and B non-transposed.
    if !ta && !tb {
        for i in i0..i1 {
            for j in j0..j1 {
                let mut acc = _mm256_setzero_ps();
                let mut l = k0;
                while l + 8 <= k1 {
                    let av = _mm256_loadu_ps(a.as_ptr().add(i * ld_a + l));
                    let bv = _mm256_set_ps(
                        *b.get_unchecked((l + 7) * ld_b + j),
                        *b.get_unchecked((l + 6) * ld_b + j),
                        *b.get_unchecked((l + 5) * ld_b + j),
                        *b.get_unchecked((l + 4) * ld_b + j),
                        *b.get_unchecked((l + 3) * ld_b + j),
                        *b.get_unchecked((l + 2) * ld_b + j),
                        *b.get_unchecked((l + 1) * ld_b + j),
                        *b.get_unchecked(l * ld_b + j),
                    );
                    acc = _mm256_fmadd_ps(av, bv, acc);
                    l += 8;
                }
                let mut sum = hsum_avx2(acc);
                for l2 in l..k1 {
                    sum += a[i * ld_a + l2] * b[l2 * ld_b + j];
                }
                *c.get_unchecked_mut(i * n + j) += alpha * sum;
            }
        }
    } else if ta && !tb {
        // A transposed (k*m stored), B normal.
        for i in i0..i1 {
            for j in j0..j1 {
                let mut acc = _mm256_setzero_ps();
                let mut l = k0;
                while l + 8 <= k1 {
                    let av = _mm256_set_ps(
                        *a.get_unchecked((l + 7) * ld_a + i),
                        *a.get_unchecked((l + 6) * ld_a + i),
                        *a.get_unchecked((l + 5) * ld_a + i),
                        *a.get_unchecked((l + 4) * ld_a + i),
                        *a.get_unchecked((l + 3) * ld_a + i),
                        *a.get_unchecked((l + 2) * ld_a + i),
                        *a.get_unchecked((l + 1) * ld_a + i),
                        *a.get_unchecked(l * ld_a + i),
                    );
                    let bv = _mm256_set_ps(
                        *b.get_unchecked((l + 7) * ld_b + j),
                        *b.get_unchecked((l + 6) * ld_b + j),
                        *b.get_unchecked((l + 5) * ld_b + j),
                        *b.get_unchecked((l + 4) * ld_b + j),
                        *b.get_unchecked((l + 3) * ld_b + j),
                        *b.get_unchecked((l + 2) * ld_b + j),
                        *b.get_unchecked((l + 1) * ld_b + j),
                        *b.get_unchecked(l * ld_b + j),
                    );
                    acc = _mm256_fmadd_ps(av, bv, acc);
                    l += 8;
                }
                let mut sum = hsum_avx2(acc);
                for l2 in l..k1 {
                    sum += a[l2 * ld_a + i] * b[l2 * ld_b + j];
                }
                *c.get_unchecked_mut(i * n + j) += alpha * sum;
            }
        }
    } else if !ta && tb {
        // A normal, B transposed (n*k stored) -> both rows contiguous.
        for i in i0..i1 {
            for j in j0..j1 {
                let mut acc = _mm256_setzero_ps();
                let mut l = k0;
                while l + 8 <= k1 {
                    let av = _mm256_loadu_ps(a.as_ptr().add(i * ld_a + l));
                    let bv = _mm256_loadu_ps(b.as_ptr().add(j * ld_b + l));
                    acc = _mm256_fmadd_ps(av, bv, acc);
                    l += 8;
                }
                let mut sum = hsum_avx2(acc);
                for l2 in l..k1 {
                    sum += a[i * ld_a + l2] * b[j * ld_b + l2];
                }
                *c.get_unchecked_mut(i * n + j) += alpha * sum;
            }
        }
    } else {
        // Both transposed: fall back to scalar for this tile.
        gemm_tile_scalar(a, b, c, i0, i1, j0, j1, k0, k1, n, ld_a, ld_b, alpha, ta, tb);
    }
}

// ── AVX-512 tile kernel ────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_tile_avx512(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    k0: usize,
    k1: usize,
    n: usize,
    ld_a: usize,
    ld_b: usize,
    alpha: f32,
    ta: bool,
    tb: bool,
) {
    if !ta && !tb {
        for i in i0..i1 {
            for j in j0..j1 {
                let mut acc = _mm512_setzero_ps();
                let mut l = k0;
                // 16-wide loop
                while l + 16 <= k1 {
                    let av = _mm512_loadu_ps(a.as_ptr().add(i * ld_a + l));
                    let bv = _mm512_set_ps(
                        *b.get_unchecked((l + 15) * ld_b + j),
                        *b.get_unchecked((l + 14) * ld_b + j),
                        *b.get_unchecked((l + 13) * ld_b + j),
                        *b.get_unchecked((l + 12) * ld_b + j),
                        *b.get_unchecked((l + 11) * ld_b + j),
                        *b.get_unchecked((l + 10) * ld_b + j),
                        *b.get_unchecked((l + 9) * ld_b + j),
                        *b.get_unchecked((l + 8) * ld_b + j),
                        *b.get_unchecked((l + 7) * ld_b + j),
                        *b.get_unchecked((l + 6) * ld_b + j),
                        *b.get_unchecked((l + 5) * ld_b + j),
                        *b.get_unchecked((l + 4) * ld_b + j),
                        *b.get_unchecked((l + 3) * ld_b + j),
                        *b.get_unchecked((l + 2) * ld_b + j),
                        *b.get_unchecked((l + 1) * ld_b + j),
                        *b.get_unchecked(l * ld_b + j),
                    );
                    acc = _mm512_fmadd_ps(av, bv, acc);
                    l += 16;
                }
                let mut sum = _mm512_reduce_add_ps(acc);
                // Scalar tail
                for l2 in l..k1 {
                    sum += a[i * ld_a + l2] * b[l2 * ld_b + j];
                }
                *c.get_unchecked_mut(i * n + j) += alpha * sum;
            }
        }
    } else if !ta && tb {
        // A normal, B transposed -> both rows contiguous.
        for i in i0..i1 {
            for j in j0..j1 {
                let mut acc = _mm512_setzero_ps();
                let mut l = k0;
                while l + 16 <= k1 {
                    let av = _mm512_loadu_ps(a.as_ptr().add(i * ld_a + l));
                    let bv = _mm512_loadu_ps(b.as_ptr().add(j * ld_b + l));
                    acc = _mm512_fmadd_ps(av, bv, acc);
                    l += 16;
                }
                let mut sum = _mm512_reduce_add_ps(acc);
                for l2 in l..k1 {
                    sum += a[i * ld_a + l2] * b[j * ld_b + l2];
                }
                *c.get_unchecked_mut(i * n + j) += alpha * sum;
            }
        }
    } else {
        // Other transpose combos: delegate to AVX2 tile kernel.
        gemm_tile_avx2(a, b, c, i0, i1, j0, j1, k0, k1, n, ld_a, ld_b, alpha, ta, tb);
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol {tol})");
        }
    }

    fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for l in 0..k {
                    s += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    /// Pack ternary weight matrix (row-major k×n) into I2_S bytes with
    /// column-major packing and uniform scale = 1.0.
    fn pack_weights(w: &[i8], k: usize, n: usize, block_size: usize) -> (Vec<u8>, Vec<f32>) {
        let packed_k = k.div_ceil(4);
        let num_blocks_k = k.div_ceil(block_size);
        let mut packed = vec![0u8; packed_k * n];
        for col in 0..n {
            for row in 0..k {
                let code: u8 = match w[row * n + col] {
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

    // ── f32 GEMM tests ────────────────────────────────────────────────

    #[test]
    fn test_f32_identity_2x2() {
        let cfg = SimdMatmulConfig::new(2, 2, 2);
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 4];
        simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
        assert_close(&c, &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_f32_identity_4x4() {
        let cfg = SimdMatmulConfig::new(4, 4, 4);
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut b = [0.0f32; 16];
        for i in 0..4 {
            b[i * 4 + i] = 1.0;
        }
        let mut c = [0.0f32; 16];
        simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
        assert_close(&c, &a, 1e-6);
    }

    #[test]
    fn test_f32_zero_matrix() {
        let cfg = SimdMatmulConfig::new(3, 3, 3);
        let a = [1.0f32; 9];
        let b = [0.0f32; 9];
        let mut c = [0.0f32; 9];
        simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
        assert_close(&c, &vec![0.0; 9], 1e-6);
    }

    #[test]
    fn test_f32_rectangular_m_gt_n() {
        let (m, n, k) = (4, 2, 3);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_f32_rectangular_n_gt_m() {
        let (m, n, k) = (2, 5, 4);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
        assert_close(&c, &expected, 1e-4);
    }

    #[test]
    fn test_f32_alpha_scaling() {
        let mut cfg = SimdMatmulConfig::new(2, 2, 2);
        cfg.alpha = 2.0;
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [3.0, 0.0, 0.0, 5.0];
        let mut c = [0.0f32; 4];
        simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
        assert_close(&c, &[6.0, 0.0, 0.0, 10.0], 1e-6);
    }

    #[test]
    fn test_f32_beta_accumulate() {
        let mut cfg = SimdMatmulConfig::new(2, 2, 2);
        cfg.beta = 1.0;
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let mut c = [10.0f32; 4];
        simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
        // C = 1.0*I*I + 1.0*[10,10,10,10] = [11,10,10,11]
        assert_close(&c, &[11.0, 10.0, 10.0, 11.0], 1e-6);
    }

    #[test]
    fn test_f32_transpose_a() {
        // A stored as k×m (transposed): [[1,3],[2,4]]
        // → logical A = [[1,2],[3,4]]
        let mut cfg = SimdMatmulConfig::new(2, 2, 2);
        cfg.transpose_a = true;
        let a_t = [1.0, 3.0, 2.0, 4.0]; // k=2, m=2 stored
        let b = [1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 4];
        simd_matmul_f32(&a_t, &b, &mut c, &cfg).unwrap();
        assert_close(&c, &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_f32_transpose_b() {
        let mut cfg = SimdMatmulConfig::new(2, 2, 2);
        cfg.transpose_b = true;
        let a = [1.0, 2.0, 3.0, 4.0];
        // B stored as n×k: [[1,0],[0,1]] → logical B = [[1,0],[0,1]]
        let b_t = [1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 4];
        simd_matmul_f32(&a, &b_t, &mut c, &cfg).unwrap();
        assert_close(&c, &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_f32_1x1() {
        let cfg = SimdMatmulConfig::new(1, 1, 1);
        let mut c = [0.0f32; 1];
        simd_matmul_f32(&[3.0], &[4.0], &mut c, &cfg).unwrap();
        assert_close(&c, &[12.0], 1e-6);
    }

    #[test]
    fn test_f32_large_k_avx_tail() {
        // k=17 exercises the AVX2 8-wide loop plus a scalar tail.
        let (m, n, k) = (2, 3, 17);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
        assert_close(&c, &expected, 1e-3);
    }

    #[test]
    fn test_f32_dimension_zero_rejected() {
        let cfg = SimdMatmulConfig::new(0, 2, 2);
        let mut c = [0.0f32; 4];
        assert!(simd_matmul_f32(&[1.0; 4], &[1.0; 4], &mut c, &cfg).is_err());
    }

    #[test]
    fn test_f32_buffer_too_small() {
        let cfg = SimdMatmulConfig::new(2, 2, 2);
        let mut c = [0.0f32; 1]; // too small
        assert!(simd_matmul_f32(&[1.0; 4], &[1.0; 4], &mut c, &cfg).is_err());
    }

    // ── I2_S quantized matmul tests ───────────────────────────────────

    #[test]
    fn test_i2s_identity_2x2() {
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_weights(&w, 2, 2, 32);
        let act = [3.0f32, -2.0, 5.0, 7.0];
        let expected = naive_matmul(&act, &[1.0, 0.0, 0.0, 1.0], 2, 2, 2);
        let mut out = [0.0f32; 4];
        simd_matmul_i2s(&act, &packed, &scales, &mut out, 2, 2, 2, 32).unwrap();
        assert_close(&out, &expected, 1e-6);
    }

    #[test]
    fn test_i2s_all_ones() {
        let (m, n, k, bs) = (4, 4, 4, 32);
        let w = vec![1i8; k * n];
        let (packed, scales) = pack_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        simd_matmul_i2s(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn test_i2s_all_neg_ones() {
        let (m, n, k, bs) = (3, 3, 4, 32);
        let w = vec![-1i8; k * n];
        let (packed, scales) = pack_weights(&w, k, n, bs);
        let act = vec![1.0f32; m * k];
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        simd_matmul_i2s(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn test_i2s_zero_weights() {
        let (m, n, k, bs) = (4, 4, 8, 32);
        let w = vec![0i8; k * n];
        let (packed, scales) = pack_weights(&w, k, n, bs);
        let act = vec![42.0f32; m * k];
        let mut out = vec![0.0f32; m * n];
        simd_matmul_i2s(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &vec![0.0f32; m * n], 1e-6);
    }

    #[test]
    fn test_i2s_non_aligned_k() {
        let (m, n, k, bs) = (3, 2, 5, 32);
        let w: Vec<i8> = vec![1, 0, -1, 1, 0, 1, -1, 0, 1, -1];
        let (packed, scales) = pack_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32 + 0.5).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        simd_matmul_i2s(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn test_i2s_block256() {
        let (m, n, k, bs) = (2, 2, 2, 256);
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_weights(&w, k, n, bs);
        let act = [3.0f32, -2.0, 5.0, 7.0];
        let expected = naive_matmul(&act, &[1.0, 0.0, 0.0, 1.0], 2, 2, 2);
        let mut out = [0.0f32; 4];
        simd_matmul_i2s(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 1e-6);
    }

    #[test]
    fn test_i2s_1x1() {
        let w = vec![1i8];
        let (packed, scales) = pack_weights(&w, 1, 1, 32);
        let mut out = [0.0f32; 1];
        simd_matmul_i2s(&[7.5], &packed, &scales, &mut out, 1, 1, 1, 32).unwrap();
        assert_close(&out, &[7.5], 1e-6);
    }

    #[test]
    fn test_i2s_large_mixed() {
        let (m, n, k, bs) = (16, 8, 48, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, 0, -1, 1, -1][i % 5]).collect();
        let (packed, scales) = pack_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.03).sin()).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        simd_matmul_i2s(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn test_i2s_dimension_zero_rejected() {
        let (packed, scales) = pack_weights(&[1], 1, 1, 32);
        let mut out = [0.0f32; 1];
        assert!(simd_matmul_i2s(&[1.0], &packed, &scales, &mut out, 0, 1, 1, 32).is_err());
    }

    #[test]
    fn test_i2s_block_size_zero_rejected() {
        let mut out = [0.0f32; 1];
        assert!(simd_matmul_i2s(&[1.0], &[0], &[1.0], &mut out, 1, 1, 1, 0).is_err());
    }

    // ── Batched matmul tests ──────────────────────────────────────────

    #[test]
    fn test_batched_identity() {
        let cfg = SimdMatmulConfig::new(2, 2, 2);
        let eye = [1.0, 0.0, 0.0, 1.0];
        let a1 = [1.0f32, 2.0, 3.0, 4.0];
        let a2 = [5.0f32, 6.0, 7.0, 8.0];
        let mut c1 = [0.0f32; 4];
        let mut c2 = [0.0f32; 4];

        {
            let a_batch: Vec<&[f32]> = vec![&a1, &a2];
            let b_batch: Vec<&[f32]> = vec![&eye, &eye];
            let mut c_batch: Vec<&mut [f32]> = vec![&mut c1, &mut c2];
            simd_matmul_batched(&a_batch, &b_batch, &mut c_batch, &cfg).unwrap();
        }

        assert_close(&c1, &a1, 1e-6);
        assert_close(&c2, &a2, 1e-6);
    }

    #[test]
    fn test_batched_size_mismatch() {
        let cfg = SimdMatmulConfig::new(2, 2, 2);
        let a = [1.0f32; 4];
        let b = [1.0f32; 4];
        let mut c = [0.0f32; 4];
        let a_batch: Vec<&[f32]> = vec![&a, &a];
        let b_batch: Vec<&[f32]> = vec![&b];
        let mut c_batch: Vec<&mut [f32]> = vec![&mut c];
        assert!(simd_matmul_batched(&a_batch, &b_batch, &mut c_batch, &cfg).is_err());
    }

    #[test]
    fn test_batched_empty() {
        let cfg = SimdMatmulConfig::new(2, 2, 2);
        let a_batch: Vec<&[f32]> = vec![];
        let b_batch: Vec<&[f32]> = vec![];
        let mut c_batch: Vec<&mut [f32]> = vec![];
        simd_matmul_batched(&a_batch, &b_batch, &mut c_batch, &cfg).unwrap();
    }

    // ── Numerical precision ───────────────────────────────────────────

    #[test]
    fn test_f32_exact_small_integers() {
        let cfg = SimdMatmulConfig::new(2, 2, 2);
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
        // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8]
        assert_close(&c, &[19.0, 22.0, 43.0, 50.0], 0.0);
    }

    #[test]
    fn test_f32_negative_values() {
        let cfg = SimdMatmulConfig::new(2, 2, 2);
        let a = [-1.0, 2.0, 3.0, -4.0];
        let b = [1.0, -2.0, -3.0, 4.0];
        let mut c = [0.0f32; 4];
        simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
        // [-1+2*(-3), -1*(-2)+2*4, 3+(-4)*(-3), 3*(-2)+(-4)*4]
        assert_close(&c, &[-7.0, 10.0, 15.0, -22.0], 0.0);
    }

    #[test]
    fn test_i2s_numerical_accuracy_exact() {
        // Unit scales + small integers → exact result.
        let (m, n, k, bs) = (4, 4, 8, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0][i % 3]).collect();
        let (packed, scales) = pack_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i % 5) as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        simd_matmul_i2s(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 0.0);
    }
}

#[cfg(test)]
mod tiled_tests {
    use super::*;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol {tol})");
        }
    }

    fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for l in 0..k {
                    s += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    #[test]
    fn test_tiled_basic_correctness_vs_naive() {
        let (m, n, k) = (8, 8, 8);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let tiles = TileConfig::DEFAULT;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.2).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c, &cfg, &tiles).unwrap();
        assert_close(&c, &expected, 1e-4);
    }

    #[test]
    fn test_tiled_matches_untiled() {
        let (m, n, k) = (16, 16, 32);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let tiles = TileConfig::new(4, 4, 8);
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.37).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.23).cos()).collect();
        let mut c_tiled = vec![0.0f32; m * n];
        let mut c_untiled = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c_tiled, &cfg, &tiles).unwrap();
        simd_matmul_f32(&a, &b, &mut c_untiled, &cfg).unwrap();
        assert_close(&c_tiled, &c_untiled, 1e-4);
    }

    #[test]
    fn test_tiled_identity_matrix() {
        let n = 16;
        let cfg = SimdMatmulConfig::new(n, n, n);
        let tiles = TileConfig::new(4, 4, 4);
        let a: Vec<f32> = (0..n * n).map(|i| (i as f32) + 1.0).collect();
        let mut eye = vec![0.0f32; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let mut c = vec![0.0f32; n * n];
        simd_matmul_f32_tiled(&a, &eye, &mut c, &cfg, &tiles).unwrap();
        assert_close(&c, &a, 1e-5);
    }

    #[test]
    fn test_tiled_transpose_a() {
        let (m, n, k) = (4, 4, 8);
        let mut cfg = SimdMatmulConfig::new(m, n, k);
        cfg.transpose_a = true;
        let tiles = TileConfig::new(2, 2, 4);
        let a: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.2).collect();
        let mut c_tiled = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c_tiled, &cfg, &tiles).unwrap();
        simd_matmul_f32(&a, &b, &mut c_ref, &cfg).unwrap();
        assert_close(&c_tiled, &c_ref, 1e-4);
    }

    #[test]
    fn test_tiled_transpose_b() {
        let (m, n, k) = (4, 4, 8);
        let mut cfg = SimdMatmulConfig::new(m, n, k);
        cfg.transpose_b = true;
        let tiles = TileConfig::new(2, 2, 4);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.2).collect();
        let mut c_tiled = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c_tiled, &cfg, &tiles).unwrap();
        simd_matmul_f32(&a, &b, &mut c_ref, &cfg).unwrap();
        assert_close(&c_tiled, &c_ref, 1e-4);
    }

    #[test]
    fn test_tiled_transpose_both() {
        let (m, n, k) = (4, 4, 8);
        let mut cfg = SimdMatmulConfig::new(m, n, k);
        cfg.transpose_a = true;
        cfg.transpose_b = true;
        let tiles = TileConfig::new(2, 2, 4);
        let a: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.2).collect();
        let mut c_tiled = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c_tiled, &cfg, &tiles).unwrap();
        simd_matmul_f32(&a, &b, &mut c_ref, &cfg).unwrap();
        assert_close(&c_tiled, &c_ref, 1e-4);
    }

    #[test]
    fn test_tiled_tall_skinny() {
        let (m, n, k) = (64, 4, 16);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let tiles = TileConfig::new(8, 2, 8);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c, &cfg, &tiles).unwrap();
        assert_close(&c, &expected, 1e-2);
    }

    #[test]
    fn test_tiled_wide_short() {
        let (m, n, k) = (4, 64, 16);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let tiles = TileConfig::new(2, 16, 8);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c, &cfg, &tiles).unwrap();
        assert_close(&c, &expected, 1e-2);
    }

    #[test]
    fn test_tiled_256x256() {
        let (m, n, k) = (256, 256, 256);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let tiles = TileConfig::DEFAULT;
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.002).cos()).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c, &cfg, &tiles).unwrap();
        let max_err =
            c.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 0.5, "max error {max_err} exceeds 0.5");
    }

    #[test]
    fn test_tiled_512x512() {
        let (m, n, k) = (512, 512, 512);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let tiles = TileConfig::LARGE;
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.0001).sin() * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.0002).cos() * 0.01).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c, &cfg, &tiles).unwrap();
        let max_err =
            c.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "max error {max_err} exceeds 1e-4");
    }

    #[test]
    fn test_tiled_numerical_precision_vs_naive() {
        let (m, n, k) = (32, 32, 64);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let tiles = TileConfig::new(8, 8, 16);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0).recip()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0).recip()).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c, &cfg, &tiles).unwrap();
        assert_close(&c, &expected, 1e-4);
    }

    #[test]
    fn test_tiled_alpha_beta() {
        let (m, n, k) = (8, 8, 8);
        let mut cfg = SimdMatmulConfig::new(m, n, k);
        cfg.alpha = 2.5;
        cfg.beta = 0.5;
        let tiles = TileConfig::new(4, 4, 4);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let mut c_tiled = vec![1.0f32; m * n];
        let mut c_ref = vec![1.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c_tiled, &cfg, &tiles).unwrap();
        simd_matmul_f32(&a, &b, &mut c_ref, &cfg).unwrap();
        assert_close(&c_tiled, &c_ref, 1e-3);
    }

    #[test]
    fn test_tiled_not_divisible_by_tile() {
        let (m, n, k) = (7, 7, 7);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let tiles = TileConfig::new(4, 4, 4);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c, &cfg, &tiles).unwrap();
        assert_close(&c, &expected, 1e-3);
    }

    #[test]
    fn test_tiled_tile_larger_than_matrix() {
        let (m, n, k) = (3, 3, 3);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let tiles = TileConfig::new(64, 64, 64);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c, &cfg, &tiles).unwrap();
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_tiled_single_element_tile() {
        let (m, n, k) = (4, 4, 4);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let tiles = TileConfig::new(1, 1, 1);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c, &cfg, &tiles).unwrap();
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_tiled_1x1() {
        let cfg = SimdMatmulConfig::new(1, 1, 1);
        let tiles = TileConfig::DEFAULT;
        let mut c = [0.0f32; 1];
        simd_matmul_f32_tiled(&[5.0], &[3.0], &mut c, &cfg, &tiles).unwrap();
        assert_close(&c, &[15.0], 1e-6);
    }

    #[test]
    fn test_tile_config_small_preset() {
        let (m, n, k) = (16, 16, 16);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let tiles = TileConfig::SMALL;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c, &cfg, &tiles).unwrap();
        assert_close(&c, &expected, 1e-2);
    }

    #[test]
    fn test_tile_config_large_preset() {
        let (m, n, k) = (16, 16, 16);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let tiles = TileConfig::LARGE;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c, &cfg, &tiles).unwrap();
        assert_close(&c, &expected, 1e-2);
    }

    #[test]
    fn test_tiled_identity_property() {
        for n in [1, 3, 7, 16, 33] {
            let cfg = SimdMatmulConfig::new(n, n, n);
            let tiles = TileConfig::new(4, 4, 4);
            let a: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.1 + 0.5).collect();
            let mut eye = vec![0.0f32; n * n];
            for i in 0..n {
                eye[i * n + i] = 1.0;
            }
            let mut c = vec![0.0f32; n * n];
            simd_matmul_f32_tiled(&a, &eye, &mut c, &cfg, &tiles).unwrap();
            assert_close(&c, &a, 1e-4);
        }
    }

    #[test]
    fn test_tiled_transpose_property() {
        // (A*B)^T = B^T * A^T
        let (m, n, k) = (8, 6, 10);
        let tiles = TileConfig::new(4, 4, 4);
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.13).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.17).cos()).collect();

        let cfg_ab = SimdMatmulConfig::new(m, n, k);
        let mut c_ab = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c_ab, &cfg_ab, &tiles).unwrap();

        let mut c_ab_t = vec![0.0f32; n * m];
        for i in 0..m {
            for j in 0..n {
                c_ab_t[j * m + i] = c_ab[i * n + j];
            }
        }

        let mut cfg_bt_at = SimdMatmulConfig::new(n, m, k);
        cfg_bt_at.transpose_a = true;
        cfg_bt_at.transpose_b = true;
        let mut c_bt_at = vec![0.0f32; n * m];
        simd_matmul_f32_tiled(&b, &a, &mut c_bt_at, &cfg_bt_at, &tiles).unwrap();

        assert_close(&c_bt_at, &c_ab_t, 1e-3);
    }

    #[test]
    fn test_tiled_distributive_property() {
        // A * (B + C) = A*B + A*C
        let (m, n, k) = (8, 8, 8);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let tiles = TileConfig::new(4, 4, 4);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.2).collect();
        let cv: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();

        let b_plus_c: Vec<f32> = b.iter().zip(cv.iter()).map(|(&x, &y)| x + y).collect();

        let mut lhs = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b_plus_c, &mut lhs, &cfg, &tiles).unwrap();

        let mut ab = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut ab, &cfg, &tiles).unwrap();
        let mut ac = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &cv, &mut ac, &cfg, &tiles).unwrap();
        let rhs: Vec<f32> = ab.iter().zip(ac.iter()).map(|(&x, &y)| x + y).collect();

        assert_close(&lhs, &rhs, 1e-2);
    }

    #[test]
    fn test_tiled_zero_tile_rejected() {
        let cfg = SimdMatmulConfig::new(2, 2, 2);
        let tiles = TileConfig::new(0, 4, 4);
        let mut c = [0.0f32; 4];
        assert!(simd_matmul_f32_tiled(&[1.0; 4], &[1.0; 4], &mut c, &cfg, &tiles).is_err());
    }

    #[test]
    fn test_tiled_zero_dim_rejected() {
        let cfg = SimdMatmulConfig::new(0, 2, 2);
        let tiles = TileConfig::DEFAULT;
        let mut c = [0.0f32; 4];
        assert!(simd_matmul_f32_tiled(&[1.0; 4], &[1.0; 4], &mut c, &cfg, &tiles).is_err());
    }

    #[test]
    fn test_tiled_asymmetric_tiles() {
        let (m, n, k) = (20, 12, 24);
        let cfg = SimdMatmulConfig::new(m, n, k);
        let tiles = TileConfig::new(3, 5, 7);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32_tiled(&a, &b, &mut c, &cfg, &tiles).unwrap();
        assert_close(&c, &expected, 1e-2);
    }
}
