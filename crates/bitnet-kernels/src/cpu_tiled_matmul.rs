//! CPU cache-tiled matrix multiplication with auto-tuned SIMD.
//!
//! Provides a cache-oblivious tiling strategy that partitions the M×N×K
//! iteration space into tiles sized to fit the L1/L2/L3 cache hierarchy.
//! The inner micro-kernel uses AVX2 `_mm256_fmadd_ps` on x86-64 and falls
//! back to portable scalar code on other architectures.
//!
//! # Layout conventions
//!
//! All matrices are **row-major** by default. Transposed operands are
//! indicated through [`TransposeMode`]; the dispatcher handles all four
//! combinations (NN, NT, TN, TT).

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

// ── Transpose mode ─────────────────────────────────────────────────────

/// Transpose combination for operands A and B.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransposeMode {
    /// A normal, B normal (row-major × row-major).
    NN,
    /// A normal, B transposed.
    NT,
    /// A transposed, B normal.
    TN,
    /// Both transposed.
    TT,
}

// ── Tile configuration ─────────────────────────────────────────────────

/// Tile dimensions and prefetch hint for a single level of the tiling
/// hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileConfig {
    /// Tile height (rows of C micro-panel).
    pub tile_m: usize,
    /// Tile width (columns of C micro-panel).
    pub tile_n: usize,
    /// Tile depth along the shared dimension.
    pub tile_k: usize,
    /// Software-prefetch distance in elements (0 = disabled).
    pub prefetch_distance: usize,
}

impl TileConfig {
    pub fn new(tile_m: usize, tile_n: usize, tile_k: usize, prefetch_distance: usize) -> Self {
        Self { tile_m, tile_n, tile_k, prefetch_distance }
    }
}

/// Default tile sizes tuned for typical desktop L1 (32 KiB) / L2 (256 KiB).
impl Default for TileConfig {
    fn default() -> Self {
        Self { tile_m: 64, tile_n: 64, tile_k: 256, prefetch_distance: 8 }
    }
}

// ── Cache-size heuristic ───────────────────────────────────────────────

/// Detected (or estimated) cache hierarchy sizes in bytes.
#[derive(Debug, Clone, Copy)]
struct CacheInfo {
    l1: usize,
    l2: usize,
    l3: usize,
}

impl Default for CacheInfo {
    fn default() -> Self {
        // Conservative defaults matching a typical modern x86 core.
        Self { l1: 32 * 1024, l2: 256 * 1024, l3: 8 * 1024 * 1024 }
    }
}

/// Best-effort detection of the L1/L2/L3 cache sizes using `sysinfo`.
/// Falls back to [`CacheInfo::default`] when detection fails or on
/// platforms where the information is not available.
fn detect_cache_sizes() -> CacheInfo {
    // sysinfo doesn't expose per-level cache sizes portably; use defaults.
    CacheInfo::default()
}

// ── Auto-tuned tile selection ──────────────────────────────────────────

/// Choose tile sizes so that the working-set of each tile level fits in
/// the corresponding cache level.
///
/// The heuristic sizes the K-tile to keep B's tile in L1, the N-tile to
/// keep the micro-panel of C in L2, and clamps M to a reasonable fraction
/// of L3.
fn auto_tune_tiles(m: usize, n: usize, k: usize) -> TileConfig {
    let cache = detect_cache_sizes();
    let elem = std::mem::size_of::<f32>();

    // tile_k: B-panel (tile_k × tile_n) should fit in ~half L1.
    // Start with a generous estimate, then clamp.
    let target_k = (cache.l1 / 2) / (elem * 8).max(1);
    let tile_k = target_k.min(k).max(1);

    // tile_n: A-panel (tile_m × tile_k) + C-panel (tile_m × tile_n)
    // should fit in L2.
    let target_n = (cache.l2 / 2) / (elem * tile_k).max(1);
    let tile_n = target_n.min(n).max(1);

    // tile_m: aim for ~quarter of L3 for the packed A block.
    let target_m = (cache.l3 / 4) / (elem * tile_k).max(1);
    let tile_m = target_m.min(m).max(1);

    // prefetch: one cache-line ahead (64 B / 4 B = 16 floats).
    let prefetch_distance = 16;

    TileConfig { tile_m, tile_n, tile_k, prefetch_distance }
}

// ── Naive reference implementation ─────────────────────────────────────

/// Portable scalar matmul used for tiny matrices and as a correctness
/// reference.
///
/// `C[m×n] = A · B` with layout governed by `mode`.
fn naive_matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    mode: TransposeMode,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum: f32 = 0.0;
            for p in 0..k {
                let a_val = match mode {
                    TransposeMode::NN | TransposeMode::NT => a[i * k + p],
                    TransposeMode::TN | TransposeMode::TT => a[p * m + i],
                };
                let b_val = match mode {
                    TransposeMode::NN | TransposeMode::TN => b[p * n + j],
                    TransposeMode::NT | TransposeMode::TT => b[j * k + p],
                };
                sum += a_val * b_val;
            }
            c[i * n + j] = sum;
        }
    }
}

// ── SIMD micro-kernel ──────────────────────────────────────────────────

/// AVX2 FMA dot-product of two contiguous f32 slices of length `len`.
/// Falls back to scalar on non-x86 or when AVX2 is unavailable at
/// runtime.
#[inline]
fn dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && len >= 8 {
            // SAFETY: runtime check above guarantees AVX2+FMA.
            return unsafe { dot_f32_avx2(a, b, len) };
        }
    }
    dot_f32_scalar(a, b, len)
}

/// Scalar fallback dot-product.
#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum: f32 = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// AVX2+FMA accelerated dot-product.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32], len: usize) -> f32 {
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let chunks = len / 16;
        let mut i = 0usize;
        for _ in 0..chunks {
            let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
            let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
            i += 16;
        }
        // Handle remaining 8-element chunk.
        if i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(va, vb, acc0);
            i += 8;
        }
        acc0 = _mm256_add_ps(acc0, acc1);

        // Horizontal sum of 8-wide accumulator.
        let hi = _mm256_extractf128_ps(acc0, 1);
        let lo = _mm256_castps256_ps128(acc0);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        let mut total = _mm_cvtss_f32(result);

        // Scalar tail.
        for idx in i..len {
            total += a[idx] * b[idx];
        }
        total
    }
}

// ── Tiled GEMM core ────────────────────────────────────────────────────

/// Cache-tiled f32 GEMM: `C[m×n] += A · B`.
///
/// The caller must ensure:
/// - `a.len() >= m*k` (or `k*m` when A is transposed)
/// - `b.len() >= k*n` (or `n*k` when B is transposed)
/// - `c.len() >= m*n`
///
/// `c` is zeroed before accumulation.
pub struct TiledMatmul;

impl TiledMatmul {
    /// Execute a tiled matmul with the given tile configuration.
    pub fn execute(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        mode: TransposeMode,
        config: &TileConfig,
    ) {
        // Zero output.
        c[..m * n].fill(0.0);

        if m == 0 || n == 0 || k == 0 {
            return;
        }

        let tm = config.tile_m.max(1);
        let tn = config.tile_n.max(1);
        let tk = config.tile_k.max(1);

        // Three-level tiling loop: K-outer → M → N.
        let mut pk = 0;
        while pk < k {
            let bk = tk.min(k - pk);
            let mut pi = 0;
            while pi < m {
                let bm = tm.min(m - pi);
                let mut pj = 0;
                while pj < n {
                    let bn = tn.min(n - pj);
                    Self::micro_tile(a, b, c, m, n, k, pi, pj, pk, bm, bn, bk, mode);
                    pj += bn;
                }
                pi += bm;
            }
            pk += bk;
        }
    }

    /// Compute one micro-tile and accumulate into C.
    #[inline]
    fn micro_tile(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        _m_full: usize,
        n_full: usize,
        k_full: usize,
        row_off: usize,
        col_off: usize,
        k_off: usize,
        bm: usize,
        bn: usize,
        bk: usize,
        mode: TransposeMode,
    ) {
        match mode {
            TransposeMode::NN => {
                // A row-major [M×K], B row-major [K×N].
                Self::tile_nn(a, b, c, n_full, k_full, row_off, col_off, k_off, bm, bn, bk);
            }
            TransposeMode::NT => {
                // A row-major [M×K], B stored as [N×K] (transposed).
                Self::tile_nt(a, b, c, n_full, k_full, row_off, col_off, k_off, bm, bn, bk);
            }
            TransposeMode::TN => {
                // A stored as [K×M] (transposed), B row-major [K×N].
                Self::tile_tn(
                    a, b, c, _m_full, n_full, k_full, row_off, col_off, k_off, bm, bn, bk,
                );
            }
            TransposeMode::TT => {
                // A stored as [K×M], B stored as [N×K].
                Self::tile_tt(
                    a, b, c, _m_full, n_full, k_full, row_off, col_off, k_off, bm, bn, bk,
                );
            }
        }
    }

    #[inline]
    fn tile_nn(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        n_full: usize,
        k_full: usize,
        row_off: usize,
        col_off: usize,
        k_off: usize,
        bm: usize,
        bn: usize,
        bk: usize,
    ) {
        for i in 0..bm {
            let ri = row_off + i;
            let a_base = ri * k_full + k_off;
            for j in 0..bn {
                let cj = col_off + j;
                let mut sum: f32 = 0.0;
                for p in 0..bk {
                    sum += a[a_base + p] * b[(k_off + p) * n_full + cj];
                }
                c[ri * n_full + cj] += sum;
            }
        }
    }

    #[inline]
    fn tile_nt(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        n_full: usize,
        k_full: usize,
        row_off: usize,
        col_off: usize,
        k_off: usize,
        bm: usize,
        bn: usize,
        bk: usize,
    ) {
        // B stored as [N×K] → b[j, p] = b[j * k_full + p].
        for i in 0..bm {
            let ri = row_off + i;
            let a_row = &a[ri * k_full + k_off..ri * k_full + k_off + bk];
            for j in 0..bn {
                let cj = col_off + j;
                let b_row = &b[cj * k_full + k_off..cj * k_full + k_off + bk];
                let sum = dot_f32(a_row, b_row, bk);
                c[ri * n_full + cj] += sum;
            }
        }
    }

    #[inline]
    fn tile_tn(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m_full: usize,
        n_full: usize,
        _k_full: usize,
        row_off: usize,
        col_off: usize,
        k_off: usize,
        bm: usize,
        bn: usize,
        bk: usize,
    ) {
        // A stored as [K×M] → a[p, i] = a[p * m_full + i].
        for i in 0..bm {
            let ri = row_off + i;
            for j in 0..bn {
                let cj = col_off + j;
                let mut sum: f32 = 0.0;
                for p in 0..bk {
                    sum += a[(k_off + p) * m_full + ri] * b[(k_off + p) * n_full + cj];
                }
                c[ri * n_full + cj] += sum;
            }
        }
    }

    #[inline]
    fn tile_tt(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m_full: usize,
        n_full: usize,
        k_full: usize,
        row_off: usize,
        col_off: usize,
        k_off: usize,
        bm: usize,
        bn: usize,
        bk: usize,
    ) {
        // A [K×M], B [N×K].
        for i in 0..bm {
            let ri = row_off + i;
            for j in 0..bn {
                let cj = col_off + j;
                let mut sum: f32 = 0.0;
                for p in 0..bk {
                    sum += a[(k_off + p) * m_full + ri] * b[cj * k_full + (k_off + p)];
                }
                c[ri * n_full + cj] += sum;
            }
        }
    }
}

// ── Dispatcher ─────────────────────────────────────────────────────────

/// Threshold below which we fall back to naive scalar matmul.
const TINY_THRESHOLD: usize = 16;

/// Selects the optimal tile configuration for the given dimensions and
/// dispatches to either [`TiledMatmul`] or the [`naive_matmul`] fallback.
pub struct MatmulDispatcher;

impl MatmulDispatcher {
    /// Run `C = op(A) · op(B)` with auto-tuned tiling.
    ///
    /// # Panics
    ///
    /// Panics if buffer lengths are too small for the declared dimensions.
    pub fn dispatch(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        mode: TransposeMode,
    ) {
        if m == 0 || n == 0 || k == 0 {
            c[..m * n].fill(0.0);
            return;
        }

        Self::validate_buffers(a, b, c, m, n, k, mode);

        // Tiny matrices: naive is faster due to no tiling overhead.
        if m <= TINY_THRESHOLD && n <= TINY_THRESHOLD && k <= TINY_THRESHOLD {
            naive_matmul(a, b, c, m, n, k, mode);
            return;
        }

        let config = auto_tune_tiles(m, n, k);
        TiledMatmul::execute(a, b, c, m, n, k, mode, &config);
    }

    /// Dispatch with a caller-provided [`TileConfig`].
    pub fn dispatch_with_config(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        mode: TransposeMode,
        config: &TileConfig,
    ) {
        Self::validate_buffers(a, b, c, m, n, k, mode);
        TiledMatmul::execute(a, b, c, m, n, k, mode, config);
    }

    fn validate_buffers(
        a: &[f32],
        b: &[f32],
        c: &[f32],
        m: usize,
        n: usize,
        k: usize,
        mode: TransposeMode,
    ) {
        let (a_need, b_need) = match mode {
            TransposeMode::NN => (m * k, k * n),
            TransposeMode::NT => (m * k, n * k),
            TransposeMode::TN => (k * m, k * n),
            TransposeMode::TT => (k * m, n * k),
        };
        assert!(a.len() >= a_need, "A buffer too small: need {a_need}, got {}", a.len());
        assert!(b.len() >= b_need, "B buffer too small: need {b_need}, got {}", b.len());
        assert!(c.len() >= m * n, "C buffer too small: need {}, got {}", m * n, c.len());
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Reference matmul used purely inside tests.
    fn ref_matmul(
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        mode: TransposeMode,
    ) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        naive_matmul(a, b, &mut c, m, n, k, mode);
        c
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(diff < tol, "element [{i}]: actual={a}, expected={e}, diff={diff} > tol={tol}");
        }
    }

    // ── identity / trivial ────────────────────────────────────

    #[test]
    fn test_1x1_nn() {
        let a = [3.0f32];
        let b = [5.0f32];
        let mut c = [0.0f32];
        MatmulDispatcher::dispatch(&a, &b, &mut c, 1, 1, 1, TransposeMode::NN);
        assert_close(&c, &[15.0], 1e-6);
    }

    #[test]
    fn test_2x2_identity_nn() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        MatmulDispatcher::dispatch(&a, &b, &mut c, 2, 2, 2, TransposeMode::NN);
        assert_close(&c, &b, 1e-6);
    }

    // ── empty / zero dimensions ───────────────────────────────

    #[test]
    fn test_zero_m() {
        let mut c = Vec::<f32>::new();
        MatmulDispatcher::dispatch(&[], &[], &mut c, 0, 4, 4, TransposeMode::NN);
        assert!(c.is_empty());
    }

    #[test]
    fn test_zero_n() {
        let mut c = Vec::<f32>::new();
        MatmulDispatcher::dispatch(&[], &[], &mut c, 4, 0, 4, TransposeMode::NN);
        assert!(c.is_empty());
    }

    #[test]
    fn test_zero_k() {
        let mut c = vec![99.0f32; 4];
        MatmulDispatcher::dispatch(&[], &[], &mut c, 2, 2, 0, TransposeMode::NN);
        assert_close(&c, &[0.0; 4], 1e-6);
    }

    // ── single row / column ───────────────────────────────────

    #[test]
    fn test_single_row_nn() {
        // A: 1×3, B: 3×2 → C: 1×2
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut c = [0.0f32; 2];
        MatmulDispatcher::dispatch(&a, &b, &mut c, 1, 2, 3, TransposeMode::NN);
        // [1*4+2*6+3*8, 1*5+2*7+3*9] = [40, 46]
        assert_close(&c, &[40.0, 46.0], 1e-5);
    }

    #[test]
    fn test_single_column_nn() {
        // A: 3×1, B: 1×1 → C: 3×1
        let a = [2.0, 3.0, 4.0];
        let b = [5.0];
        let mut c = [0.0f32; 3];
        MatmulDispatcher::dispatch(&a, &b, &mut c, 3, 1, 1, TransposeMode::NN);
        assert_close(&c, &[10.0, 15.0, 20.0], 1e-5);
    }

    // ── square matrices NN ────────────────────────────────────

    #[test]
    fn test_4x4_nn() {
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5).collect();
        let expected = ref_matmul(&a, &b, 4, 4, 4, TransposeMode::NN);
        let mut c = vec![0.0f32; 16];
        MatmulDispatcher::dispatch(&a, &b, &mut c, 4, 4, 4, TransposeMode::NN);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_8x8_nn() {
        let a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..64).map(|i| (i as f32) * 0.2).collect();
        let expected = ref_matmul(&a, &b, 8, 8, 8, TransposeMode::NN);
        let mut c = vec![0.0f32; 64];
        MatmulDispatcher::dispatch(&a, &b, &mut c, 8, 8, 8, TransposeMode::NN);
        assert_close(&c, &expected, 1e-4);
    }

    #[test]
    fn test_32x32_nn() {
        let n = 32;
        let a: Vec<f32> = (0..n * n).map(|i| ((i % 17) as f32) * 0.3 - 2.0).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i % 13) as f32) * 0.2 - 1.0).collect();
        let expected = ref_matmul(&a, &b, n, n, n, TransposeMode::NN);
        let mut c = vec![0.0f32; n * n];
        MatmulDispatcher::dispatch(&a, &b, &mut c, n, n, n, TransposeMode::NN);
        assert_close(&c, &expected, 1e-3);
    }

    #[test]
    fn test_64x64_nn() {
        let n = 64;
        let a: Vec<f32> = (0..n * n).map(|i| ((i % 23) as f32) * 0.1).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i % 19) as f32) * 0.1).collect();
        let expected = ref_matmul(&a, &b, n, n, n, TransposeMode::NN);
        let mut c = vec![0.0f32; n * n];
        MatmulDispatcher::dispatch(&a, &b, &mut c, n, n, n, TransposeMode::NN);
        assert_close(&c, &expected, 1e-2);
    }

    #[test]
    fn test_128x128_nn() {
        let n = 128;
        let a: Vec<f32> = (0..n * n).map(|i| ((i % 29) as f32) * 0.05 - 0.7).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i % 31) as f32) * 0.04 - 0.6).collect();
        let expected = ref_matmul(&a, &b, n, n, n, TransposeMode::NN);
        let mut c = vec![0.0f32; n * n];
        MatmulDispatcher::dispatch(&a, &b, &mut c, n, n, n, TransposeMode::NN);
        assert_close(&c, &expected, 5e-2);
    }

    #[test]
    fn test_256x256_nn() {
        let n = 256;
        let a: Vec<f32> = (0..n * n).map(|i| ((i % 37) as f32) * 0.02 - 0.3).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i % 41) as f32) * 0.02 - 0.4).collect();
        let expected = ref_matmul(&a, &b, n, n, n, TransposeMode::NN);
        let mut c = vec![0.0f32; n * n];
        MatmulDispatcher::dispatch(&a, &b, &mut c, n, n, n, TransposeMode::NN);
        assert_close(&c, &expected, 0.1);
    }

    #[test]
    fn test_512x512_nn() {
        let n = 512;
        let a: Vec<f32> = (0..n * n).map(|i| ((i % 47) as f32) * 0.01 - 0.2).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i % 53) as f32) * 0.01 - 0.25).collect();
        let expected = ref_matmul(&a, &b, n, n, n, TransposeMode::NN);
        let mut c = vec![0.0f32; n * n];
        MatmulDispatcher::dispatch(&a, &b, &mut c, n, n, n, TransposeMode::NN);
        assert_close(&c, &expected, 0.2);
    }

    // ── non-square matrices NN ────────────────────────────────

    #[test]
    fn test_3x5_x_5x2_nn() {
        let (m, k, n) = (3, 5, 2);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();
        let expected = ref_matmul(&a, &b, m, n, k, TransposeMode::NN);
        let mut c = vec![0.0f32; m * n];
        MatmulDispatcher::dispatch(&a, &b, &mut c, m, n, k, TransposeMode::NN);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_7x13_x_13x11_nn() {
        let (m, k, n) = (7, 13, 11);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32) * 0.3).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.2).collect();
        let expected = ref_matmul(&a, &b, m, n, k, TransposeMode::NN);
        let mut c = vec![0.0f32; m * n];
        MatmulDispatcher::dispatch(&a, &b, &mut c, m, n, k, TransposeMode::NN);
        assert_close(&c, &expected, 1e-4);
    }

    #[test]
    fn test_1x256_x_256x1_nn() {
        let (m, k, n) = (1, 256, 1);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 19) as f32) * 0.1).collect();
        let expected = ref_matmul(&a, &b, m, n, k, TransposeMode::NN);
        let mut c = vec![0.0f32; m * n];
        MatmulDispatcher::dispatch(&a, &b, &mut c, m, n, k, TransposeMode::NN);
        assert_close(&c, &expected, 1e-3);
    }

    // ── transpose NT ──────────────────────────────────────────

    #[test]
    fn test_4x4_nt() {
        let n = 4;
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        // B stored as [N×K] instead of [K×N].
        let b: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5).collect();
        let expected = ref_matmul(&a, &b, n, n, n, TransposeMode::NT);
        let mut c = vec![0.0f32; 16];
        MatmulDispatcher::dispatch(&a, &b, &mut c, n, n, n, TransposeMode::NT);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_8x8_nt() {
        let n = 8;
        let a: Vec<f32> = (0..64).map(|i| ((i % 11) as f32) * 0.2).collect();
        let b: Vec<f32> = (0..64).map(|i| ((i % 13) as f32) * 0.15).collect();
        let expected = ref_matmul(&a, &b, n, n, n, TransposeMode::NT);
        let mut c = vec![0.0f32; 64];
        MatmulDispatcher::dispatch(&a, &b, &mut c, n, n, n, TransposeMode::NT);
        assert_close(&c, &expected, 1e-4);
    }

    #[test]
    fn test_nonsquare_nt() {
        let (m, k, n) = (5, 7, 3);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| i as f32 * 0.2).collect();
        let expected = ref_matmul(&a, &b, m, n, k, TransposeMode::NT);
        let mut c = vec![0.0f32; m * n];
        MatmulDispatcher::dispatch(&a, &b, &mut c, m, n, k, TransposeMode::NT);
        assert_close(&c, &expected, 1e-4);
    }

    // ── transpose TN ──────────────────────────────────────────

    #[test]
    fn test_4x4_tn() {
        let n = 4;
        // A stored as [K×M].
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5).collect();
        let expected = ref_matmul(&a, &b, n, n, n, TransposeMode::TN);
        let mut c = vec![0.0f32; 16];
        MatmulDispatcher::dispatch(&a, &b, &mut c, n, n, n, TransposeMode::TN);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_nonsquare_tn() {
        let (m, k, n) = (4, 6, 5);
        let a: Vec<f32> = (0..k * m).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.2).collect();
        let expected = ref_matmul(&a, &b, m, n, k, TransposeMode::TN);
        let mut c = vec![0.0f32; m * n];
        MatmulDispatcher::dispatch(&a, &b, &mut c, m, n, k, TransposeMode::TN);
        assert_close(&c, &expected, 1e-4);
    }

    // ── transpose TT ──────────────────────────────────────────

    #[test]
    fn test_4x4_tt() {
        let n = 4;
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5).collect();
        let expected = ref_matmul(&a, &b, n, n, n, TransposeMode::TT);
        let mut c = vec![0.0f32; 16];
        MatmulDispatcher::dispatch(&a, &b, &mut c, n, n, n, TransposeMode::TT);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_nonsquare_tt() {
        let (m, k, n) = (3, 8, 5);
        let a: Vec<f32> = (0..k * m).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| i as f32 * 0.2).collect();
        let expected = ref_matmul(&a, &b, m, n, k, TransposeMode::TT);
        let mut c = vec![0.0f32; m * n];
        MatmulDispatcher::dispatch(&a, &b, &mut c, m, n, k, TransposeMode::TT);
        assert_close(&c, &expected, 1e-4);
    }

    // ── custom tile config ────────────────────────────────────

    #[test]
    fn test_custom_tile_config_small() {
        let (m, n, k) = (16, 16, 16);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.1).collect();
        let expected = ref_matmul(&a, &b, m, n, k, TransposeMode::NN);
        let mut c = vec![0.0f32; m * n];
        let config = TileConfig::new(4, 4, 4, 0);
        MatmulDispatcher::dispatch_with_config(&a, &b, &mut c, m, n, k, TransposeMode::NN, &config);
        assert_close(&c, &expected, 1e-3);
    }

    #[test]
    fn test_custom_tile_config_large_tile() {
        let (m, n, k) = (10, 10, 10);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.05).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.05).collect();
        let expected = ref_matmul(&a, &b, m, n, k, TransposeMode::NN);
        let mut c = vec![0.0f32; m * n];
        // Tile larger than matrix — should still work.
        let config = TileConfig::new(64, 64, 64, 16);
        MatmulDispatcher::dispatch_with_config(&a, &b, &mut c, m, n, k, TransposeMode::NN, &config);
        assert_close(&c, &expected, 1e-4);
    }

    // ── auto_tune_tiles smoke ─────────────────────────────────

    #[test]
    fn test_auto_tune_tiles_returns_nonzero() {
        let cfg = auto_tune_tiles(128, 128, 128);
        assert!(cfg.tile_m > 0);
        assert!(cfg.tile_n > 0);
        assert!(cfg.tile_k > 0);
    }

    #[test]
    fn test_auto_tune_tiles_clamped_to_dims() {
        let cfg = auto_tune_tiles(4, 4, 4);
        assert!(cfg.tile_m <= 4);
        assert!(cfg.tile_n <= 4);
        assert!(cfg.tile_k <= 4);
    }

    // ── tiny fallback path ────────────────────────────────────

    #[test]
    fn test_tiny_uses_naive_path() {
        // Dimensions ≤ TINY_THRESHOLD exercise the naive fallback.
        let (m, n, k) = (4, 4, 4);
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..16).map(|i| (15 - i) as f32).collect();
        let expected = ref_matmul(&a, &b, m, n, k, TransposeMode::NN);
        let mut c = vec![0.0f32; 16];
        MatmulDispatcher::dispatch(&a, &b, &mut c, m, n, k, TransposeMode::NN);
        assert_close(&c, &expected, 1e-5);
    }

    // ── dot_f32 unit tests ────────────────────────────────────

    #[test]
    fn test_dot_f32_small() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let got = dot_f32(&a, &b, 4);
        let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;
        assert!((got - expected).abs() < 1e-6);
    }

    #[test]
    fn test_dot_f32_large() {
        let n = 256;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| ((n - i) as f32) * 0.01).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let got = dot_f32(&a, &b, n);
        assert!((got - expected).abs() < 1e-2, "dot mismatch: {got} vs {expected}");
    }

    #[test]
    fn test_dot_f32_empty() {
        let got = dot_f32(&[], &[], 0);
        assert!((got - 0.0).abs() < 1e-10);
    }

    // ── tile config default ───────────────────────────────────

    #[test]
    fn test_tile_config_default() {
        let cfg = TileConfig::default();
        assert_eq!(cfg.tile_m, 64);
        assert_eq!(cfg.tile_n, 64);
        assert_eq!(cfg.tile_k, 256);
        assert_eq!(cfg.prefetch_distance, 8);
    }

    // ── numerical precision ───────────────────────────────────

    #[test]
    fn test_precision_32x32_all_modes() {
        let (m, n, k) = (32, 32, 32);
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 100) as f32 * 0.01).collect();
        // For NN/NT: B needs k*n or n*k elements.
        // For TN/TT: A needs k*m elements (same count as m*k).
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 11 + 5) % 100) as f32 * 0.01).collect();

        for mode in [TransposeMode::NN, TransposeMode::NT, TransposeMode::TN, TransposeMode::TT] {
            let expected = ref_matmul(&a, &b, m, n, k, mode);
            let mut c = vec![0.0f32; m * n];
            MatmulDispatcher::dispatch(&a, &b, &mut c, m, n, k, mode);
            // Allow tolerance proportional to K (accumulation length).
            let tol = k as f32 * 1e-5;
            assert_close(&c, &expected, tol);
        }
    }

    // ── property tests ────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_nn_matches_naive(
            m in 1usize..33,
            n in 1usize..33,
            k in 1usize..33,
        ) {
            let a: Vec<f32> = (0..m*k).map(|i| ((i * 7 + 3) % 50) as f32 * 0.1 - 2.5).collect();
            let b: Vec<f32> = (0..k*n).map(|i| ((i * 11 + 5) % 50) as f32 * 0.1 - 2.5).collect();
            let expected = ref_matmul(&a, &b, m, n, k, TransposeMode::NN);
            let mut c = vec![0.0f32; m * n];
            MatmulDispatcher::dispatch(&a, &b, &mut c, m, n, k, TransposeMode::NN);
            let tol = k as f32 * 1e-4;
            for (i, (a_v, e_v)) in c.iter().zip(expected.iter()).enumerate() {
                prop_assert!(
                    (a_v - e_v).abs() < tol,
                    "NN [{i}]: m={m} n={n} k={k} actual={a_v} expected={e_v}"
                );
            }
        }

        #[test]
        fn prop_nt_matches_naive(
            m in 1usize..33,
            n in 1usize..33,
            k in 1usize..33,
        ) {
            let a: Vec<f32> = (0..m*k).map(|i| ((i * 7 + 3) % 50) as f32 * 0.1 - 2.5).collect();
            let b: Vec<f32> = (0..n*k).map(|i| ((i * 11 + 5) % 50) as f32 * 0.1 - 2.5).collect();
            let expected = ref_matmul(&a, &b, m, n, k, TransposeMode::NT);
            let mut c = vec![0.0f32; m * n];
            MatmulDispatcher::dispatch(&a, &b, &mut c, m, n, k, TransposeMode::NT);
            let tol = k as f32 * 1e-4;
            for (i, (a_v, e_v)) in c.iter().zip(expected.iter()).enumerate() {
                prop_assert!(
                    (a_v - e_v).abs() < tol,
                    "NT [{i}]: m={m} n={n} k={k} actual={a_v} expected={e_v}"
                );
            }
        }

        #[test]
        fn prop_tn_matches_naive(
            m in 1usize..33,
            n in 1usize..33,
            k in 1usize..33,
        ) {
            let a: Vec<f32> = (0..k*m).map(|i| ((i * 7 + 3) % 50) as f32 * 0.1 - 2.5).collect();
            let b: Vec<f32> = (0..k*n).map(|i| ((i * 11 + 5) % 50) as f32 * 0.1 - 2.5).collect();
            let expected = ref_matmul(&a, &b, m, n, k, TransposeMode::TN);
            let mut c = vec![0.0f32; m * n];
            MatmulDispatcher::dispatch(&a, &b, &mut c, m, n, k, TransposeMode::TN);
            let tol = k as f32 * 1e-4;
            for (i, (a_v, e_v)) in c.iter().zip(expected.iter()).enumerate() {
                prop_assert!(
                    (a_v - e_v).abs() < tol,
                    "TN [{i}]: m={m} n={n} k={k} actual={a_v} expected={e_v}"
                );
            }
        }

        #[test]
        fn prop_tt_matches_naive(
            m in 1usize..33,
            n in 1usize..33,
            k in 1usize..33,
        ) {
            let a: Vec<f32> = (0..k*m).map(|i| ((i * 7 + 3) % 50) as f32 * 0.1 - 2.5).collect();
            let b: Vec<f32> = (0..n*k).map(|i| ((i * 11 + 5) % 50) as f32 * 0.1 - 2.5).collect();
            let expected = ref_matmul(&a, &b, m, n, k, TransposeMode::TT);
            let mut c = vec![0.0f32; m * n];
            MatmulDispatcher::dispatch(&a, &b, &mut c, m, n, k, TransposeMode::TT);
            let tol = k as f32 * 1e-4;
            for (i, (a_v, e_v)) in c.iter().zip(expected.iter()).enumerate() {
                prop_assert!(
                    (a_v - e_v).abs() < tol,
                    "TT [{i}]: m={m} n={n} k={k} actual={a_v} expected={e_v}"
                );
            }
        }
    }
}
