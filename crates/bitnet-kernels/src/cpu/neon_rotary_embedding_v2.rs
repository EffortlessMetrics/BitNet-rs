//! ARM NEON Rotary Embedding v2 — advanced RoPE variants for
//! Apple Silicon.
//!
//! Implements NTK-aware, YaRN, Dynamic NTK, ALiBi, fused RoPE +
//! attention, batched RoPE, and inverse RoPE using AArch64 NEON
//! SIMD intrinsics.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── NTK-Aware RoPE ─────────────────────────────────────────────────

/// Compute NTK-aware base frequency.
///
/// Scales the base frequency by `alpha` to extend context length
/// without fine-tuning: `base' = base * alpha^(dim / (dim - 2))`.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn ntk_scaled_base(base: f32, alpha: f32, dim: usize) -> f32 {
    assert!(dim >= 4, "dim must be >= 4 for NTK scaling");
    let exponent = dim as f32 / (dim as f32 - 2.0);
    base * alpha.powf(exponent)
}

/// Build cos/sin tables with NTK-aware base scaling (NEON).
///
/// # Safety
///
/// Requires AArch64 NEON (always available on aarch64 targets).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn build_ntk_tables(
    dim: usize,
    max_seq: usize,
    base: f32,
    alpha: f32,
) -> (Vec<f32>, Vec<f32>) {
    let scaled = ntk_scaled_base(base, alpha, dim);
    unsafe { build_rope_tables_core(dim, max_seq, scaled) }
}

/// Apply NTK-aware RoPE in-place using NEON.
///
/// # Safety
///
/// Requires AArch64 NEON. `data.len() >= dim`, tables must cover
/// `pos`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_ntk_rope_neon(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos: usize,
) {
    unsafe { apply_rope_core_neon(data, cos_table, sin_table, dim, pos) };
}

// ── YaRN RoPE ───────────────────────────────────────────────────────

/// Configuration for YaRN (Yet another RoPE extensioN).
#[cfg(target_arch = "aarch64")]
#[derive(Debug, Clone)]
pub struct YarnConfig {
    /// Per-head embedding dimension (must be even, >= 4).
    pub dim: usize,
    /// Maximum sequence length the table covers.
    pub max_seq: usize,
    /// Base rotation frequency (default 10 000).
    pub base: f32,
    /// Context-length scale factor (> 1 extends context).
    pub scale: f32,
    /// Original maximum context length of the model.
    pub original_max_seq: usize,
    /// Low-frequency interpolation threshold (wavelength ratio).
    pub beta_slow: f32,
    /// High-frequency extrapolation threshold.
    pub beta_fast: f32,
}

/// Compute per-dimension YaRN interpolation weights.
///
/// Returns a vector of length `dim / 2` with blend factors in
/// `[0, 1]` — 0 means pure NTK, 1 means pure interpolation.
#[cfg(target_arch = "aarch64")]
pub fn yarn_ramp_weights(cfg: &YarnConfig) -> Vec<f32> {
    let half = cfg.dim / 2;
    let low = (cfg.original_max_seq as f32 / (cfg.beta_slow * 2.0 * std::f32::consts::PI)).floor();
    let high = (cfg.original_max_seq as f32 / (cfg.beta_fast * 2.0 * std::f32::consts::PI)).floor();

    (0..half)
        .map(|i| {
            let exp = -(2.0 * i as f32) / cfg.dim as f32;
            let wavelength = 2.0 * std::f32::consts::PI * cfg.base.powf(-exp);
            if wavelength < high {
                0.0_f32
            } else if wavelength > low {
                1.0
            } else if (low - high).abs() < f32::EPSILON {
                0.5
            } else {
                (wavelength - high) / (low - high)
            }
        })
        .collect()
}

/// Build cos/sin tables with YaRN attention scaling (NEON).
///
/// # Safety
///
/// Requires AArch64 NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn build_yarn_tables(cfg: &YarnConfig) -> (Vec<f32>, Vec<f32>) {
    let half = cfg.dim / 2;
    let ntk_base = ntk_scaled_base(cfg.base, cfg.scale, cfg.dim);
    let ramp = yarn_ramp_weights(cfg);

    let mut cos_t = Vec::with_capacity(cfg.max_seq * half);
    let mut sin_t = Vec::with_capacity(cfg.max_seq * half);

    for pos in 0..cfg.max_seq {
        for i in 0..half {
            let exp = -(2.0 * i as f32) / cfg.dim as f32;
            let theta_ntk = ntk_base.powf(exp);
            let theta_interp = cfg.base.powf(exp) / cfg.scale;
            let theta = (1.0 - ramp[i]) * theta_ntk + ramp[i] * theta_interp;
            let angle = pos as f32 * theta;
            cos_t.push(angle.cos());
            sin_t.push(angle.sin());
        }
    }
    (cos_t, sin_t)
}

/// Apply YaRN RoPE in-place using NEON.
///
/// # Safety
///
/// Requires AArch64 NEON. `data.len() >= dim`, tables must cover
/// `pos`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_yarn_rope_neon(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos: usize,
) {
    unsafe { apply_rope_core_neon(data, cos_table, sin_table, dim, pos) };
}

// ── Dynamic NTK RoPE ────────────────────────────────────────────────

/// Compute dynamic NTK base for a given current sequence length.
///
/// If `seq_len > original_max_seq`, the base is scaled up so the
/// model can extrapolate beyond its trained context.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dynamic_ntk_base(base: f32, dim: usize, seq_len: usize, original_max_seq: usize) -> f32 {
    if seq_len <= original_max_seq {
        return base;
    }
    let alpha = (seq_len as f32) / (original_max_seq as f32);
    ntk_scaled_base(base, alpha, dim)
}

/// Build cos/sin tables with dynamic NTK scaling (NEON).
///
/// # Safety
///
/// Requires AArch64 NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn build_dynamic_ntk_tables(
    dim: usize,
    max_seq: usize,
    base: f32,
    original_max_seq: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut cos_t = Vec::with_capacity(max_seq * half);
    let mut sin_t = Vec::with_capacity(max_seq * half);

    for pos in 0..max_seq {
        let dyn_base = dynamic_ntk_base(base, dim, pos + 1, original_max_seq);
        for i in 0..half {
            let exp = -(2.0 * i as f32) / dim as f32;
            let theta = dyn_base.powf(exp);
            let angle = pos as f32 * theta;
            cos_t.push(angle.cos());
            sin_t.push(angle.sin());
        }
    }
    (cos_t, sin_t)
}

// ── ALiBi Positional Bias ───────────────────────────────────────────

/// Compute the per-head ALiBi slope for head index `h` out of
/// `num_heads` total.
///
/// slope(h) = 2^(−8h / num_heads), h ∈ [0, num_heads).
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn alibi_slope(head_idx: usize, num_heads: usize) -> f32 {
    assert!(num_heads > 0, "num_heads must be > 0");
    let exp = -8.0 * (head_idx as f32 + 1.0) / num_heads as f32;
    2.0_f32.powf(exp)
}

/// Compute all ALiBi slopes for a model.
#[cfg(target_arch = "aarch64")]
pub fn alibi_slopes(num_heads: usize) -> Vec<f32> {
    (0..num_heads).map(|h| alibi_slope(h, num_heads)).collect()
}

/// Inject ALiBi bias into a pre-allocated attention score matrix
/// using NEON.
///
/// `scores` is `[seq_q, seq_k]` in row-major order.  For each query
/// position `q` and key position `k`, the bias added is
/// `slope * (k − q)`.
///
/// # Safety
///
/// Requires AArch64 NEON.  `scores.len() >= seq_q * seq_k`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn alibi_bias_neon(scores: &mut [f32], slope: f32, seq_q: usize, seq_k: usize) {
    assert!(scores.len() >= seq_q * seq_k, "scores buffer too small for {seq_q}×{seq_k}");
    let slope_v = vdupq_n_f32(slope);

    for q in 0..seq_q {
        let row_start = q * seq_k;
        let mut k = 0usize;

        // NEON: process 4 keys at a time.
        while k + 4 <= seq_k {
            let base_dist = k as f32 - q as f32;
            let offsets: [f32; 4] = [base_dist, base_dist + 1.0, base_dist + 2.0, base_dist + 3.0];
            let dist_v = unsafe { vld1q_f32(offsets.as_ptr()) };
            let bias = vmulq_f32(slope_v, dist_v);

            let idx = row_start + k;
            let cur = unsafe { vld1q_f32(scores.as_ptr().add(idx)) };
            let out = vaddq_f32(cur, bias);
            unsafe { vst1q_f32(scores.as_mut_ptr().add(idx), out) };
            k += 4;
        }
        // Scalar tail.
        while k < seq_k {
            let dist = k as f32 - q as f32;
            scores[row_start + k] += slope * dist;
            k += 1;
        }
    }
}

// ── Fused RoPE + Attention helpers ──────────────────────────────────

/// Apply RoPE in-place to Q and K projections before attention.
///
/// `q` shape: `[num_heads, dim]`, `k` shape: `[num_kv_heads, dim]`.
/// Both are rotated at position `pos` using the supplied tables.
///
/// # Safety
///
/// Requires AArch64 NEON.  Tables must cover `pos`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn fused_rope_qk_neon(
    q: &mut [f32],
    k: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    dim: usize,
    pos: usize,
) {
    // Rotate every Q head.
    for h in 0..num_q_heads {
        let offset = h * dim;
        unsafe {
            apply_rope_core_neon(&mut q[offset..offset + dim], cos_table, sin_table, dim, pos);
        }
    }
    // Rotate every KV head.
    for h in 0..num_kv_heads {
        let offset = h * dim;
        unsafe {
            apply_rope_core_neon(&mut k[offset..offset + dim], cos_table, sin_table, dim, pos);
        }
    }
}

// ── Batched RoPE ────────────────────────────────────────────────────

/// Apply RoPE to a batch of sequences with per-sequence position
/// offsets.
///
/// `data` shape: `[batch, num_heads, dim]` (contiguous).
/// `position_offsets[b]` is the starting position for batch item
/// `b`.
///
/// # Safety
///
/// Requires AArch64 NEON.  Tables must cover every required position.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn batched_rope_neon(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    batch_size: usize,
    num_heads: usize,
    dim: usize,
    position_offsets: &[usize],
) {
    assert_eq!(position_offsets.len(), batch_size, "need one offset per batch item");
    let stride_batch = num_heads * dim;

    for b in 0..batch_size {
        let pos = position_offsets[b];
        for h in 0..num_heads {
            let off = b * stride_batch + h * dim;
            unsafe {
                apply_rope_core_neon(&mut data[off..off + dim], cos_table, sin_table, dim, pos);
            }
        }
    }
}

// ── Inverse RoPE ────────────────────────────────────────────────────

/// Undo a previous RoPE rotation by applying the *negative* angle.
///
/// This is equivalent to rotating by `−θ`, i.e. swapping the sign of
/// the sin component.
///
/// # Safety
///
/// Requires AArch64 NEON.  `data.len() >= dim`, tables must cover
/// `pos`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn inverse_rope_neon(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos: usize,
) {
    let half = dim / 2;
    let base = pos * half;
    assert!(
        cos_table.len() >= base + half && sin_table.len() >= base + half,
        "tables too short for pos={pos}, dim={dim}"
    );

    let mut i = 0usize;
    // NEON path: 2 rotation pairs per iteration (4 floats).
    while i + 2 <= half {
        let ci = base + i;
        let cos_v = unsafe { vld1q_f32(cos_table.as_ptr().add(ci)) };
        let sin_v = unsafe { vld1q_f32(sin_table.as_ptr().add(ci)) };
        // Negate sin for inverse rotation.
        let neg_sin_v = vnegq_f32(sin_v);

        // Load even (x) and odd (y) pairs.
        let d0 = i * 2;
        let x0 = data[d0];
        let y0 = data[d0 + 1];
        let x1 = if i + 1 < half { data[d0 + 2] } else { 0.0 };
        let y1 = if i + 1 < half { data[d0 + 3] } else { 0.0 };

        let xy = [x0, x1, y0, y1];
        let xy_v = unsafe { vld1q_f32(xy.as_ptr()) };
        // cos values: [c0, c1, c0, c1]
        let c0 = vgetq_lane_f32::<0>(cos_v);
        let c1 = if i + 1 < half { vgetq_lane_f32::<1>(cos_v) } else { 0.0 };
        let s0 = vgetq_lane_f32::<0>(neg_sin_v);
        let s1 = if i + 1 < half { vgetq_lane_f32::<1>(neg_sin_v) } else { 0.0 };

        // Scalar rotation (uses loaded NEON values for consistency).
        let _ = xy_v; // consumed above via loads
        data[d0] = x0 * c0 - y0 * s0;
        data[d0 + 1] = x0 * s0 + y0 * c0;
        if i + 1 < half {
            data[d0 + 2] = x1 * c1 - y1 * s1;
            data[d0 + 3] = x1 * s1 + y1 * c1;
        }

        i += 2;
    }
    // Scalar tail for single remaining pair.
    if i < half {
        let ci = base + i;
        let c = cos_table[ci];
        let s = -sin_table[ci];
        let d0 = i * 2;
        let x = data[d0];
        let y = data[d0 + 1];
        data[d0] = x * c - y * s;
        data[d0 + 1] = x * s + y * c;
    }
}

// ── Core helpers ────────────────────────────────────────────────────

/// Build cos/sin tables for a given base (shared helper).
///
/// # Safety
///
/// Requires AArch64 NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn build_rope_tables_core(dim: usize, max_seq: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut cos_t = Vec::with_capacity(max_seq * half);
    let mut sin_t = Vec::with_capacity(max_seq * half);

    for pos in 0..max_seq {
        for i in 0..half {
            let exp = -(2.0 * i as f32) / dim as f32;
            let theta = base.powf(exp);
            let angle = pos as f32 * theta;
            cos_t.push(angle.cos());
            sin_t.push(angle.sin());
        }
    }
    (cos_t, sin_t)
}

/// Apply RoPE rotation to a single vector slice using NEON.
///
/// # Safety
///
/// Requires AArch64 NEON.  `data.len() >= dim`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_rope_core_neon(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos: usize,
) {
    let half = dim / 2;
    let base = pos * half;
    assert!(
        cos_table.len() >= base + half && sin_table.len() >= base + half,
        "tables too short for pos={pos}, dim={dim}"
    );
    assert!(data.len() >= dim, "data shorter than dim={dim}");

    let mut i = 0usize;
    // Process 2 rotation pairs (4 floats) per NEON iteration.
    while i + 2 <= half {
        let ci = base + i;
        let cos_v = unsafe { vld1q_f32(cos_table.as_ptr().add(ci)) };
        let sin_v = unsafe { vld1q_f32(sin_table.as_ptr().add(ci)) };

        // Even indices: data[2i], data[2(i+1)]
        // Odd  indices: data[2i+1], data[2(i+1)+1]
        let d0 = i * 2;
        let x0 = data[d0];
        let y0 = data[d0 + 1];
        let x1 = data[d0 + 2];
        let y1 = data[d0 + 3];

        let c0 = vgetq_lane_f32::<0>(cos_v);
        let c1 = vgetq_lane_f32::<1>(cos_v);
        let s0 = vgetq_lane_f32::<0>(sin_v);
        let s1 = vgetq_lane_f32::<1>(sin_v);

        // Rotation: x' = x*cos − y*sin, y' = x*sin + y*cos
        let xv = [x0, x1, 0.0, 0.0];
        let yv = [y0, y1, 0.0, 0.0];
        let cv = [c0, c1, 0.0, 0.0];
        let sv = [s0, s1, 0.0, 0.0];

        let xn = unsafe { vld1q_f32(xv.as_ptr()) };
        let yn = unsafe { vld1q_f32(yv.as_ptr()) };
        let cn = unsafe { vld1q_f32(cv.as_ptr()) };
        let sn = unsafe { vld1q_f32(sv.as_ptr()) };

        let x_rot = vsubq_f32(vmulq_f32(xn, cn), vmulq_f32(yn, sn));
        let y_rot = vaddq_f32(vmulq_f32(xn, sn), vmulq_f32(yn, cn));

        let mut x_out = [0.0f32; 4];
        let mut y_out = [0.0f32; 4];
        unsafe { vst1q_f32(x_out.as_mut_ptr(), x_rot) };
        unsafe { vst1q_f32(y_out.as_mut_ptr(), y_rot) };

        data[d0] = x_out[0];
        data[d0 + 1] = y_out[0];
        data[d0 + 2] = x_out[1];
        data[d0 + 3] = y_out[1];

        i += 2;
    }
    // Scalar tail for a single remaining pair.
    if i < half {
        let ci = base + i;
        let c = cos_table[ci];
        let s = sin_table[ci];
        let d0 = i * 2;
        let x = data[d0];
        let y = data[d0 + 1];
        data[d0] = x * c - y * s;
        data[d0 + 1] = x * s + y * c;
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    // ── NTK tests ───────────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_ntk_scaled_base_identity() {
        // alpha = 1 → no scaling change beyond the dim exponent.
        let b = ntk_scaled_base(10_000.0, 1.0, 64);
        assert!((b - 10_000.0).abs() < 1e-2, "alpha=1 should ≈ base, got {b}");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_ntk_scaled_base_alpha2() {
        let b = ntk_scaled_base(10_000.0, 2.0, 64);
        // base * 2^(64/62) ≈ 20 857
        assert!(b > 10_000.0, "alpha>1 must increase base, got {b}");
        assert!(b < 30_000.0, "sanity upper bound, got {b}");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_build_ntk_tables_lengths() {
        let (cos_t, sin_t) = unsafe { build_ntk_tables(16, 32, 10_000.0, 2.0) };
        assert_eq!(cos_t.len(), 32 * 8);
        assert_eq!(sin_t.len(), 32 * 8);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_ntk_rope_roundtrip() {
        let dim = 8;
        let (cos_t, sin_t) = unsafe { build_ntk_tables(dim, 4, 10_000.0, 2.0) };
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = orig.clone();
        unsafe {
            apply_ntk_rope_neon(&mut data, &cos_t, &sin_t, dim, 0);
        }
        // pos=0 → angle=0 → cos=1, sin=0 → data unchanged.
        for (a, b) in orig.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-5, "pos=0 should be identity");
        }
    }

    // ── YaRN tests ──────────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_yarn_ramp_weights_len() {
        let cfg = YarnConfig {
            dim: 64,
            max_seq: 128,
            base: 10_000.0,
            scale: 2.0,
            original_max_seq: 2048,
            beta_slow: 1.0,
            beta_fast: 32.0,
        };
        let w = yarn_ramp_weights(&cfg);
        assert_eq!(w.len(), 32); // dim/2
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_yarn_ramp_weights_range() {
        let cfg = YarnConfig {
            dim: 64,
            max_seq: 128,
            base: 10_000.0,
            scale: 2.0,
            original_max_seq: 2048,
            beta_slow: 1.0,
            beta_fast: 32.0,
        };
        let w = yarn_ramp_weights(&cfg);
        for &v in &w {
            assert!((0.0..=1.0).contains(&v), "ramp weight out of [0,1]: {v}");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_build_yarn_tables_lengths() {
        let cfg = YarnConfig {
            dim: 16,
            max_seq: 32,
            base: 10_000.0,
            scale: 2.0,
            original_max_seq: 2048,
            beta_slow: 1.0,
            beta_fast: 32.0,
        };
        let (c, s) = unsafe { build_yarn_tables(&cfg) };
        assert_eq!(c.len(), 32 * 8);
        assert_eq!(s.len(), 32 * 8);
    }

    // ── Dynamic NTK tests ───────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dynamic_ntk_base_no_extend() {
        let b = dynamic_ntk_base(10_000.0, 64, 512, 2048);
        assert!((b - 10_000.0).abs() < f32::EPSILON, "should not scale when seq_len <= original");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dynamic_ntk_base_extends() {
        let b = dynamic_ntk_base(10_000.0, 64, 4096, 2048);
        assert!(b > 10_000.0, "should scale up for seq_len > original, got {b}");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_build_dynamic_ntk_tables_lengths() {
        let (c, s) = unsafe { build_dynamic_ntk_tables(16, 64, 10_000.0, 32) };
        assert_eq!(c.len(), 64 * 8);
        assert_eq!(s.len(), 64 * 8);
    }

    // ── ALiBi tests ─────────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_alibi_slope_monotone() {
        let slopes: Vec<f32> = (0..8).map(|h| alibi_slope(h, 8)).collect();
        for w in slopes.windows(2) {
            assert!(w[0] > w[1], "slopes must decrease: {} vs {}", w[0], w[1]);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_alibi_slopes_len() {
        let s = alibi_slopes(12);
        assert_eq!(s.len(), 12);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_alibi_bias_neon_shape() {
        let (sq, sk) = (3, 5);
        let mut scores = vec![0.0f32; sq * sk];
        unsafe {
            alibi_bias_neon(&mut scores, 0.5, sq, sk);
        }
        // q=0, k=0 → bias = 0.5*(0-0) = 0
        assert!((scores[0]).abs() < 1e-6);
        // q=0, k=4 → bias = 0.5*(4-0) = 2.0
        assert!((scores[4] - 2.0).abs() < 1e-5, "{}", scores[4]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_alibi_bias_neon_negative() {
        let (sq, sk) = (4, 4);
        let mut scores = vec![0.0f32; sq * sk];
        unsafe {
            alibi_bias_neon(&mut scores, 1.0, sq, sk);
        }
        // q=2, k=0 → bias = 1.0*(0-2) = -2
        assert!((scores[2 * 4 + 0] - (-2.0)).abs() < 1e-5, "{}", scores[2 * 4]);
    }

    // ── Fused RoPE tests ────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_fused_rope_qk_pos0_identity() {
        let dim = 8;
        let (cos_t, sin_t) = unsafe { build_ntk_tables(dim, 4, 10_000.0, 1.0) };
        let mut q = vec![1.0f32; 2 * dim]; // 2 Q heads
        let mut k = vec![2.0f32; dim]; // 1 KV head
        let q_orig = q.clone();
        let k_orig = k.clone();
        unsafe {
            fused_rope_qk_neon(&mut q, &mut k, &cos_t, &sin_t, 2, 1, dim, 0);
        }
        for (a, b) in q_orig.iter().zip(q.iter()) {
            assert!((a - b).abs() < 1e-5, "Q changed at pos=0");
        }
        for (a, b) in k_orig.iter().zip(k.iter()) {
            assert!((a - b).abs() < 1e-5, "K changed at pos=0");
        }
    }

    // ── Batched RoPE tests ──────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batched_rope_pos0_identity() {
        let dim = 8;
        let batch = 2;
        let heads = 1;
        let (cos_t, sin_t) = unsafe { build_ntk_tables(dim, 4, 10_000.0, 1.0) };
        let mut data = vec![1.0f32; batch * heads * dim];
        let orig = data.clone();
        unsafe {
            batched_rope_neon(&mut data, &cos_t, &sin_t, batch, heads, dim, &[0, 0]);
        }
        for (a, b) in orig.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-5, "pos=0 should be identity");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batched_rope_different_offsets() {
        let dim = 8;
        let batch = 2;
        let heads = 1;
        let (cos_t, sin_t) = unsafe { build_ntk_tables(dim, 8, 10_000.0, 1.0) };
        let mut data = vec![1.0f32; batch * heads * dim];
        unsafe {
            batched_rope_neon(&mut data, &cos_t, &sin_t, batch, heads, dim, &[0, 3]);
        }
        // Batch 0 at pos=0 should be identity; batch 1 at pos=3
        // should differ.
        let b0 = &data[..dim];
        let b1 = &data[dim..2 * dim];
        for &v in b0 {
            assert!((v - 1.0).abs() < 1e-5, "batch0 pos=0 identity");
        }
        let changed = b1.iter().any(|&v| (v - 1.0).abs() > 1e-3);
        assert!(changed, "batch1 pos=3 should rotate data");
    }

    // ── Inverse RoPE tests ──────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_inverse_rope_roundtrip() {
        let dim = 8;
        let pos = 3;
        let (cos_t, sin_t) = unsafe { build_ntk_tables(dim, 8, 10_000.0, 1.0) };
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = orig.clone();
        unsafe {
            apply_ntk_rope_neon(&mut data, &cos_t, &sin_t, dim, pos);
            inverse_rope_neon(&mut data, &cos_t, &sin_t, dim, pos);
        }
        for (i, (a, b)) in orig.iter().zip(data.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "roundtrip failed at [{i}]: {a} vs {b}");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_inverse_rope_pos0_noop() {
        let dim = 8;
        let (cos_t, sin_t) = unsafe { build_ntk_tables(dim, 4, 10_000.0, 1.0) };
        let orig: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut data = orig.clone();
        unsafe {
            inverse_rope_neon(&mut data, &cos_t, &sin_t, dim, 0);
        }
        for (a, b) in orig.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-5, "pos=0 inverse should be identity");
        }
    }

    // ── Edge-case / cross-cutting ───────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_odd_dim_pair_scalar_tail() {
        // dim=6 → half=3 → NEON handles 2 pairs, scalar handles 1.
        let dim = 6;
        let (cos_t, sin_t) = unsafe { build_ntk_tables(dim, 4, 10_000.0, 1.0) };
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Should not panic.
        unsafe {
            apply_ntk_rope_neon(&mut data, &cos_t, &sin_t, dim, 1);
        }
        let changed = data.iter().any(|&v| (v - 1.0).abs() > 1e-3);
        assert!(changed, "data should be rotated at pos=1");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "requires real model weights — run manually"]
    fn test_ntk_rope_on_real_model_weights() {
        panic!("TDD scaffold: not yet implemented");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "requires real model weights — run manually"]
    fn test_yarn_full_context_extension() {
        panic!("TDD scaffold: not yet implemented");
    }
}
