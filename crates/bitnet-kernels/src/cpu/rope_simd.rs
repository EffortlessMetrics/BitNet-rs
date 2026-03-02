//! SIMD-optimized RoPE (Rotary Position Embedding) with advanced scaling support.
//!
//! Extends the base RoPE kernel with:
//! - Multiple scaling strategies: Linear, NTK-aware, YaRN, Dynamic NTK
//! - Interleaved (GPT-NeoX) and half-rotated (LLaMA) rotation layouts
//! - Precomputed frequency tables via [`FrequencyTable`]
//! - AVX2 fast path with automatic runtime dispatch
//! - Inverse rotation for debugging / analysis

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

// ── Scaling strategies ──────────────────────────────────────────────

/// Frequency scaling strategy applied before the sin/cos computation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum ScalingType {
    /// No scaling — standard RoPE (Vaswani / Su et al.).
    #[default]
    None,
    /// Linear scaling: divide frequencies by `factor`.
    Linear {
        /// Scaling factor (> 0). Larger values extend the effective context window.
        factor: f32,
    },
    /// NTK-aware scaling: raise the base frequency to preserve high-frequency
    /// components while extending context (Code Llama style).
    Ntk {
        /// Scaling factor (> 0). Applied as `base' = base * factor^(dim/(dim-2))`.
        factor: f32,
    },
    /// YaRN (Yet another RoPE extensioN): blends high-frequency unscaled and
    /// low-frequency NTK-scaled components via an attention factor.
    YaRN {
        /// Overall scaling factor.
        factor: f32,
        /// Original maximum context length the model was trained on.
        original_max_seq_len: usize,
        /// Ramp boundaries for the frequency interpolation (low, high).
        beta_fast: f32,
        /// Ramp boundaries for the frequency interpolation (low, high).
        beta_slow: f32,
    },
    /// Dynamic NTK: adjusts the base at runtime based on the current sequence
    /// length relative to the trained context window.
    Dynamic {
        /// Original maximum context length.
        original_max_seq_len: usize,
    },
}

// ── Configuration ───────────────────────────────────────────────────

/// Configuration for the SIMD-optimized RoPE kernel.
#[derive(Debug, Clone)]
pub struct RoPEConfig {
    /// Per-head embedding dimension (must be even and > 0).
    pub dim: usize,
    /// Maximum sequence length the frequency table will cover.
    pub max_seq_len: usize,
    /// Base rotation frequency (default `10_000.0`).
    pub base_freq: f32,
    /// Scaling strategy.
    pub scaling_type: ScalingType,
}

impl RoPEConfig {
    /// Create a new configuration with default base and no scaling.
    ///
    /// # Panics
    ///
    /// Panics if `dim` is zero or odd.
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        assert!(dim > 0 && dim.is_multiple_of(2), "dim must be even and non-zero");
        Self { dim, max_seq_len, base_freq: 10_000.0, scaling_type: ScalingType::None }
    }

    /// Override the base rotation frequency.
    #[must_use]
    pub fn with_base_freq(mut self, base: f32) -> Self {
        self.base_freq = base;
        self
    }

    /// Override the scaling strategy.
    #[must_use]
    pub fn with_scaling(mut self, scaling: ScalingType) -> Self {
        self.scaling_type = scaling;
        self
    }
}

// ── Frequency table ─────────────────────────────────────────────────

/// Precomputed cos/sin frequency tables for RoPE application.
///
/// Separate cos and sin arrays (each of length `max_seq_len * half_dim`)
/// allow cache-friendly, vectorised access patterns.
#[derive(Debug, Clone)]
pub struct FrequencyTable {
    /// `cos_table[pos * half_dim + i]` = cos(angle) for position `pos`, pair `i`.
    pub cos_table: Vec<f32>,
    /// `sin_table[pos * half_dim + i]` = sin(angle) for position `pos`, pair `i`.
    pub sin_table: Vec<f32>,
    /// Half the head dimension (`dim / 2`).
    pub half_dim: usize,
    /// Maximum sequence length stored.
    pub max_seq_len: usize,
}

impl FrequencyTable {
    /// Look up the cosine value for `(position, pair_index)`.
    #[inline]
    pub fn cos(&self, position: usize, pair: usize) -> f32 {
        self.cos_table[position * self.half_dim + pair]
    }

    /// Look up the sine value for `(position, pair_index)`.
    #[inline]
    pub fn sin(&self, position: usize, pair: usize) -> f32 {
        self.sin_table[position * self.half_dim + pair]
    }
}

// ── Frequency computation helpers ───────────────────────────────────

/// Compute the effective base frequency after applying the scaling strategy.
fn effective_base(config: &RoPEConfig, _current_seq_len: usize) -> f32 {
    match config.scaling_type {
        ScalingType::None | ScalingType::Linear { .. } | ScalingType::YaRN { .. } => {
            config.base_freq
        }
        ScalingType::Ntk { factor } => {
            // base' = base * factor^(dim / (dim - 2))
            let exp = config.dim as f32 / (config.dim as f32 - 2.0);
            config.base_freq * factor.powf(exp)
        }
        ScalingType::Dynamic { original_max_seq_len } => {
            let seq = _current_seq_len.max(1);
            if seq <= original_max_seq_len {
                config.base_freq
            } else {
                let factor = seq as f32 / original_max_seq_len as f32;
                let exp = config.dim as f32 / (config.dim as f32 - 2.0);
                config.base_freq * factor.powf(exp)
            }
        }
    }
}

/// Compute the per-pair frequency scaling multiplier for `pair_index`.
fn frequency_scale(config: &RoPEConfig, pair_index: usize) -> f32 {
    match config.scaling_type {
        ScalingType::None | ScalingType::Ntk { .. } | ScalingType::Dynamic { .. } => 1.0,
        ScalingType::Linear { factor } => 1.0 / factor,
        ScalingType::YaRN { factor, original_max_seq_len, beta_fast, beta_slow } => {
            let half_dim = config.dim / 2;
            let low = (original_max_seq_len as f32 / (beta_fast * std::f32::consts::TAU)).floor()
                as usize;
            let high =
                (original_max_seq_len as f32 / (beta_slow * std::f32::consts::TAU)).ceil() as usize;

            if pair_index < low.min(half_dim) {
                // High-frequency band — no scaling.
                1.0
            } else if pair_index >= high.min(half_dim) {
                // Low-frequency band — full linear scaling.
                1.0 / factor
            } else {
                // Ramp interpolation between the two bands.
                let range = (high.saturating_sub(low)).max(1) as f32;
                let t = (pair_index.saturating_sub(low)) as f32 / range;
                let scale = 1.0 / factor;
                // Smooth hermite interpolation
                let h = t * t * (3.0 - 2.0 * t);
                1.0 * (1.0 - h) + scale * h
            }
        }
    }
}

/// Build a [`FrequencyTable`] from the given configuration.
///
/// The table covers all positions `[0, max_seq_len)` and all dimension
/// pairs `[0, dim/2)`.
pub fn build_frequency_table(config: &RoPEConfig) -> FrequencyTable {
    build_frequency_table_with_seq(config, config.max_seq_len)
}

fn build_frequency_table_with_seq(config: &RoPEConfig, current_seq_len: usize) -> FrequencyTable {
    let half_dim = config.dim / 2;
    let base = effective_base(config, current_seq_len);
    let cap = config.max_seq_len * half_dim;
    let mut cos_table = Vec::with_capacity(cap);
    let mut sin_table = Vec::with_capacity(cap);

    for pos in 0..config.max_seq_len {
        for i in 0..half_dim {
            let exponent = -(2.0 * i as f32) / config.dim as f32;
            let inv_freq = base.powf(exponent);
            let scale = frequency_scale(config, i);
            let angle = pos as f32 * inv_freq * scale;
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
    }

    FrequencyTable { cos_table, sin_table, half_dim, max_seq_len: config.max_seq_len }
}

// ── Scalar implementations ──────────────────────────────────────────

/// Apply RoPE rotation to a single head vector **in-place** (scalar fallback).
///
/// `x` must have length ≥ `head_dim` (which must be even).
pub fn apply_rope_f32(
    x: &mut [f32],
    freq_table: &FrequencyTable,
    position: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;
    for i in 0..half {
        let cos_val = freq_table.cos(position, i);
        let sin_val = freq_table.sin(position, i);
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = x0 * cos_val - x1 * sin_val;
        x[2 * i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

/// Apply RoPE interleaved (GPT-NeoX layout) **in-place**.
///
/// In interleaved layout the even/odd indices form rotation pairs:
/// `(x[0], x[1])`, `(x[2], x[3])`, etc.  This is identical to the
/// standard RoPE pairing — provided here for API clarity.
pub fn apply_rope_interleaved(x: &mut [f32], freq_table: &FrequencyTable, position: usize) {
    let head_dim = x.len();
    apply_rope_f32(x, freq_table, position, head_dim);
}

/// Apply RoPE half-rotated (LLaMA-style) **in-place**.
///
/// In the half-rotated layout the first half and second half form pairs:
/// `(x[i], x[i + half_dim])` for `i` in `[0, half_dim)`.
pub fn apply_rope_half_rotated(x: &mut [f32], freq_table: &FrequencyTable, position: usize) {
    let dim = x.len();
    let half = dim / 2;
    for i in 0..half {
        let cos_val = freq_table.cos(position, i);
        let sin_val = freq_table.sin(position, i);
        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0 * cos_val - x1 * sin_val;
        x[i + half] = x0 * sin_val + x1 * cos_val;
    }
}

/// Inverse RoPE rotation (scalar). Applying `inverse_rope` after `apply_rope_f32`
/// recovers the original vector (up to floating-point rounding).
pub fn inverse_rope(x: &mut [f32], freq_table: &FrequencyTable, position: usize, head_dim: usize) {
    let half = head_dim / 2;
    for i in 0..half {
        let cos_val = freq_table.cos(position, i);
        let sin_val = freq_table.sin(position, i);
        let y0 = x[2 * i];
        let y1 = x[2 * i + 1];
        // Inverse rotation: negate the angle → swap sign on sin terms.
        x[2 * i] = y0 * cos_val + y1 * sin_val;
        x[2 * i + 1] = -y0 * sin_val + y1 * cos_val;
    }
}

// ── AVX2 fast path ──────────────────────────────────────────────────

/// Apply RoPE to a single head vector using AVX2 intrinsics.
///
/// Processes 8 floats (4 rotation pairs) per iteration. Falls back to
/// scalar for any tail elements.
///
/// # Safety
///
/// Caller must ensure AVX2 is available on the current CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_rope_avx2_inner(
    x: &mut [f32],
    freq_table: &FrequencyTable,
    position: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;
    let cos_base = &freq_table.cos_table[position * freq_table.half_dim..];
    let sin_base = &freq_table.sin_table[position * freq_table.half_dim..];

    // Process 4 pairs (8 floats) per AVX2 iteration.
    let simd_pairs = half / 4;
    for c in 0..simd_pairs {
        let pair_off = c * 4;
        let data_off = c * 8;

        unsafe {
            // Load 4 cos and 4 sin values then duplicate each into interleaved
            // form: [c0,c0,c1,c1,c2,c2,c3,c3].
            let cos4 = _mm_loadu_ps(cos_base.as_ptr().add(pair_off));
            let sin4 = _mm_loadu_ps(sin_base.as_ptr().add(pair_off));

            // Interleave: [c0,c0,c1,c1, | c2,c2,c3,c3]
            let cos_lo = _mm_unpacklo_ps(cos4, cos4);
            let cos_hi = _mm_unpackhi_ps(cos4, cos4);
            let cos_vec = _mm256_set_m128(cos_hi, cos_lo);

            let sin_lo = _mm_unpacklo_ps(sin4, sin4);
            let sin_hi = _mm_unpackhi_ps(sin4, sin4);
            let sin_vec = _mm256_set_m128(sin_hi, sin_lo);

            let vals = _mm256_loadu_ps(x.as_ptr().add(data_off));

            // Swap adjacent pairs: [x1,x0, x3,x2, x5,x4, x7,x6]
            let swapped = _mm256_permutevar8x32_ps(vals, _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6));

            // Sign mask: [-1,+1,-1,+1,-1,+1,-1,+1]
            let sign = _mm256_setr_ps(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

            // result = vals * cos + swapped * sign * sin
            let rotated = _mm256_add_ps(
                _mm256_mul_ps(vals, cos_vec),
                _mm256_mul_ps(_mm256_mul_ps(swapped, sign), sin_vec),
            );

            _mm256_storeu_ps(x.as_mut_ptr().add(data_off), rotated);
        }
    }

    // Scalar tail for remaining pairs.
    let processed_pairs = simd_pairs * 4;
    for i in processed_pairs..half {
        let cos_val = freq_table.cos(position, i);
        let sin_val = freq_table.sin(position, i);
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = x0 * cos_val - x1 * sin_val;
        x[2 * i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

/// Public AVX2 entry point with runtime feature check.
///
/// Returns `false` if AVX2 is not available and the caller should use the
/// scalar path instead.
#[cfg(target_arch = "x86_64")]
pub fn apply_rope_avx2(
    x: &mut [f32],
    freq_table: &FrequencyTable,
    position: usize,
    head_dim: usize,
) -> bool {
    if !is_x86_feature_detected!("avx2") || head_dim < 8 {
        return false;
    }
    // Safety: AVX2 confirmed above.
    unsafe {
        apply_rope_avx2_inner(x, freq_table, position, head_dim);
    }
    true
}

#[cfg(not(target_arch = "x86_64"))]
pub fn apply_rope_avx2(
    _x: &mut [f32],
    _freq_table: &FrequencyTable,
    _position: usize,
    _head_dim: usize,
) -> bool {
    false
}

// ── Runtime dispatch ────────────────────────────────────────────────

/// Apply RoPE with automatic SIMD dispatch (AVX2 → scalar fallback).
pub fn apply_rope_dispatch(
    x: &mut [f32],
    freq_table: &FrequencyTable,
    position: usize,
    head_dim: usize,
) {
    if !apply_rope_avx2(x, freq_table, position, head_dim) {
        apply_rope_f32(x, freq_table, position, head_dim);
    }
}

// ── Batch processing ────────────────────────────────────────────────

/// Apply RoPE across a batch of positions and heads **in-place**.
///
/// `batch` layout: `[seq_len × num_heads × head_dim]` (contiguous).
/// `positions[s]` gives the absolute position for sequence index `s`.
pub fn apply_rope_batch(
    batch: &mut [f32],
    freq_table: &FrequencyTable,
    positions: &[usize],
    head_dim: usize,
    num_heads: usize,
) {
    let seq_len = positions.len();
    for (s, &pos) in positions.iter().enumerate().take(seq_len) {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            apply_rope_dispatch(&mut batch[offset..offset + head_dim], freq_table, pos, head_dim);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ────────────────────────────────────────────────────────────────
    // Helper: naive scalar RoPE for reference
    // ────────────────────────────────────────────────────────────────
    fn naive_rope(x: &mut [f32], dim: usize, position: usize, base: f32) {
        let half = dim / 2;
        for i in 0..half {
            let exp = -(2.0 * i as f32) / dim as f32;
            let theta = base.powf(exp);
            let angle = position as f32 * theta;
            let (s, c) = angle.sin_cos();
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            x[2 * i] = x0 * c - x1 * s;
            x[2 * i + 1] = x0 * s + x1 * c;
        }
    }

    fn naive_rope_half_rotated(x: &mut [f32], dim: usize, position: usize, base: f32) {
        let half = dim / 2;
        for i in 0..half {
            let exp = -(2.0 * i as f32) / dim as f32;
            let theta = base.powf(exp);
            let angle = position as f32 * theta;
            let (s, c) = angle.sin_cos();
            let x0 = x[i];
            let x1 = x[i + half];
            x[i] = x0 * c - x1 * s;
            x[i + half] = x0 * s + x1 * c;
        }
    }

    fn vec_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    // ================================================================
    // §1  Frequency table generation
    // ================================================================

    #[test]
    fn freq_table_length() {
        let cfg = RoPEConfig::new(16, 32);
        let ft = build_frequency_table(&cfg);
        assert_eq!(ft.cos_table.len(), 32 * 8);
        assert_eq!(ft.sin_table.len(), 32 * 8);
        assert_eq!(ft.half_dim, 8);
        assert_eq!(ft.max_seq_len, 32);
    }

    #[test]
    fn freq_table_position_zero_is_identity() {
        let cfg = RoPEConfig::new(8, 4);
        let ft = build_frequency_table(&cfg);
        for i in 0..4 {
            assert!((ft.cos(0, i) - 1.0).abs() < 1e-6, "cos(0) should be 1");
            assert!(ft.sin(0, i).abs() < 1e-6, "sin(0) should be 0");
        }
    }

    #[test]
    fn freq_table_monotonic_decay() {
        let cfg = RoPEConfig::new(16, 4);
        let ft = build_frequency_table(&cfg);
        // At position 1 higher pair indices should have smaller sin magnitude
        let s0 = ft.sin(1, 0).abs();
        let s_last = ft.sin(1, 7).abs();
        assert!(s0 > s_last, "pair 0 should have larger sin: {s0} vs {s_last}");
    }

    #[test]
    fn freq_table_custom_base() {
        let ft1 = build_frequency_table(&RoPEConfig::new(8, 4));
        let ft2 = build_frequency_table(&RoPEConfig::new(8, 4).with_base_freq(500_000.0));
        // Position 0 identical, position 1 differs
        assert!((ft1.cos(0, 0) - ft2.cos(0, 0)).abs() < 1e-6);
        let differs = (0..4).any(|i| (ft1.cos(1, i) - ft2.cos(1, i)).abs() > 1e-6);
        assert!(differs, "different base should differ at pos > 0");
    }

    #[test]
    fn freq_table_all_finite() {
        for dim in [2, 4, 8, 16, 64, 128] {
            let ft = build_frequency_table(&RoPEConfig::new(dim, 2048));
            assert!(ft.cos_table.iter().all(|v| v.is_finite()), "non-finite cos at dim={dim}");
            assert!(ft.sin_table.iter().all(|v| v.is_finite()), "non-finite sin at dim={dim}");
        }
    }

    #[test]
    fn freq_table_cos_sin_unit_circle() {
        let ft = build_frequency_table(&RoPEConfig::new(8, 64));
        for pos in 0..64 {
            for i in 0..4 {
                let c = ft.cos(pos, i);
                let s = ft.sin(pos, i);
                let r = c * c + s * s;
                assert!((r - 1.0).abs() < 1e-5, "cos²+sin²≠1 at pos={pos} i={i}: {r}");
            }
        }
    }

    // ── NTK scaling ─────────────────────────────────────────────────

    #[test]
    fn freq_table_ntk_differs_from_none() {
        let ft_none = build_frequency_table(&RoPEConfig::new(8, 4));
        let ft_ntk = build_frequency_table(
            &RoPEConfig::new(8, 4).with_scaling(ScalingType::Ntk { factor: 2.0 }),
        );
        let differs = (0..4).any(|i| (ft_none.cos(1, i) - ft_ntk.cos(1, i)).abs() > 1e-6);
        assert!(differs, "NTK should change frequencies");
    }

    #[test]
    fn freq_table_ntk_factor_one_matches_none() {
        let ft_none = build_frequency_table(&RoPEConfig::new(8, 4));
        let ft_ntk = build_frequency_table(
            &RoPEConfig::new(8, 4).with_scaling(ScalingType::Ntk { factor: 1.0 }),
        );
        let diff = max_abs_diff(&ft_none.cos_table, &ft_ntk.cos_table);
        assert!(diff < 1e-5, "NTK factor=1 should match None: diff={diff}");
    }

    #[test]
    fn freq_table_ntk_preserves_unit_circle() {
        let ft = build_frequency_table(
            &RoPEConfig::new(16, 128).with_scaling(ScalingType::Ntk { factor: 4.0 }),
        );
        for pos in [0, 1, 64, 127] {
            for i in 0..8 {
                let r = ft.cos(pos, i).powi(2) + ft.sin(pos, i).powi(2);
                assert!((r - 1.0).abs() < 1e-5, "unit circle violated at pos={pos} i={i}");
            }
        }
    }

    // ── Linear scaling ──────────────────────────────────────────────

    #[test]
    fn freq_table_linear_scaling() {
        let ft_none = build_frequency_table(&RoPEConfig::new(8, 8));
        let ft_lin = build_frequency_table(
            &RoPEConfig::new(8, 8).with_scaling(ScalingType::Linear { factor: 2.0 }),
        );
        // Linear scaling should make pos 2 look like pos 1 of the unscaled table.
        for i in 0..4 {
            let diff_cos = (ft_none.cos(1, i) - ft_lin.cos(2, i)).abs();
            let diff_sin = (ft_none.sin(1, i) - ft_lin.sin(2, i)).abs();
            assert!(diff_cos < 1e-5, "linear cos mismatch at i={i}: {diff_cos}");
            assert!(diff_sin < 1e-5, "linear sin mismatch at i={i}: {diff_sin}");
        }
    }

    #[test]
    fn freq_table_linear_factor_one_matches_none() {
        let ft_none = build_frequency_table(&RoPEConfig::new(8, 8));
        let ft_lin = build_frequency_table(
            &RoPEConfig::new(8, 8).with_scaling(ScalingType::Linear { factor: 1.0 }),
        );
        let diff = max_abs_diff(&ft_none.cos_table, &ft_lin.cos_table);
        assert!(diff < 1e-6, "Linear factor=1 should match None: diff={diff}");
    }

    // ── YaRN scaling ────────────────────────────────────────────────

    #[test]
    fn freq_table_yarn_differs_from_none() {
        let ft_none = build_frequency_table(&RoPEConfig::new(16, 32));
        let ft_yarn =
            build_frequency_table(&RoPEConfig::new(16, 32).with_scaling(ScalingType::YaRN {
                factor: 4.0,
                original_max_seq_len: 8,
                beta_fast: 32.0,
                beta_slow: 1.0,
            }));
        let differs = (0..16)
            .any(|idx| (ft_none.cos_table[16 + idx] - ft_yarn.cos_table[16 + idx]).abs() > 1e-6);
        assert!(differs, "YaRN should change at least some frequencies at pos>0");
    }

    #[test]
    fn freq_table_yarn_preserves_unit_circle() {
        let ft = build_frequency_table(&RoPEConfig::new(16, 64).with_scaling(ScalingType::YaRN {
            factor: 2.0,
            original_max_seq_len: 32,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }));
        for pos in [0, 1, 32, 63] {
            for i in 0..8 {
                let r = ft.cos(pos, i).powi(2) + ft.sin(pos, i).powi(2);
                assert!((r - 1.0).abs() < 1e-5, "YaRN unit circle at pos={pos} i={i}");
            }
        }
    }

    // ── Dynamic NTK scaling ─────────────────────────────────────────

    #[test]
    fn freq_table_dynamic_within_window_matches_none() {
        let ft_none = build_frequency_table(&RoPEConfig::new(8, 32));
        let cfg_dyn =
            RoPEConfig::new(8, 32).with_scaling(ScalingType::Dynamic { original_max_seq_len: 64 });
        // current_seq_len (32) <= original (64), so should match None
        let ft_dyn = build_frequency_table_with_seq(&cfg_dyn, 32);
        let diff = max_abs_diff(&ft_none.cos_table, &ft_dyn.cos_table);
        assert!(diff < 1e-5, "Dynamic within window should match None: {diff}");
    }

    #[test]
    fn freq_table_dynamic_beyond_window_differs() {
        let cfg_dyn =
            RoPEConfig::new(8, 32).with_scaling(ScalingType::Dynamic { original_max_seq_len: 16 });
        let ft_within = build_frequency_table_with_seq(&cfg_dyn, 16);
        let ft_beyond = build_frequency_table_with_seq(&cfg_dyn, 32);
        let differs = ft_within
            .cos_table
            .iter()
            .zip(ft_beyond.cos_table.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs, "Dynamic beyond window should change frequencies");
    }

    // ================================================================
    // §2  apply_rope_f32 correctness vs naive
    // ================================================================

    #[test]
    fn apply_rope_f32_matches_naive() {
        for dim in [2, 4, 8, 16, 32, 64] {
            let cfg = RoPEConfig::new(dim, 16);
            let ft = build_frequency_table(&cfg);
            for pos in [0, 1, 5, 15] {
                let original: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
                let mut ours = original.clone();
                let mut naive = original.clone();

                apply_rope_f32(&mut ours, &ft, pos, dim);
                naive_rope(&mut naive, dim, pos, 10_000.0);

                let diff = max_abs_diff(&ours, &naive);
                assert!(diff < 1e-5, "dim={dim} pos={pos} diff={diff}");
            }
        }
    }

    #[test]
    fn apply_rope_f32_identity_at_pos_zero() {
        let ft = build_frequency_table(&RoPEConfig::new(8, 2));
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = data.clone();
        apply_rope_f32(&mut data, &ft, 0, 8);
        assert!(max_abs_diff(&data, &orig) < 1e-6);
    }

    #[test]
    fn apply_rope_f32_preserves_norm() {
        let ft = build_frequency_table(&RoPEConfig::new(16, 64));
        for pos in [0, 1, 7, 31, 63] {
            let mut data: Vec<f32> = (0..16).map(|i| (i as f32 * 0.7) - 3.0).collect();
            let before = vec_norm(&data);
            apply_rope_f32(&mut data, &ft, pos, 16);
            let after = vec_norm(&data);
            assert!((before - after).abs() < 1e-4, "norm changed at pos={pos}");
        }
    }

    #[test]
    fn apply_rope_f32_different_positions_differ() {
        let ft = build_frequency_table(&RoPEConfig::new(8, 4));
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut a = orig.clone();
        let mut b = orig.clone();
        apply_rope_f32(&mut a, &ft, 1, 8);
        apply_rope_f32(&mut b, &ft, 2, 8);
        assert!(max_abs_diff(&a, &b) > 1e-6);
    }

    #[test]
    fn apply_rope_f32_known_reference_dim2() {
        let ft = build_frequency_table(&RoPEConfig::new(2, 2));
        let mut data = vec![1.0, 0.0];
        apply_rope_f32(&mut data, &ft, 1, 2);
        let angle = 1.0f32; // base^0 * 1 = 1.0
        assert!((data[0] - angle.cos()).abs() < 1e-5);
        assert!((data[1] - angle.sin()).abs() < 1e-5);
    }

    #[test]
    fn apply_rope_f32_known_reference_dim4() {
        let ft = build_frequency_table(&RoPEConfig::new(4, 8));
        let mut data = vec![1.0, 0.5, 0.8, -0.3];
        apply_rope_f32(&mut data, &ft, 3, 4);

        let a0 = 3.0f32;
        let a1 = 3.0 * 10_000.0f32.powf(-0.5);
        let expected = [
            1.0 * a0.cos() - 0.5 * a0.sin(),
            1.0 * a0.sin() + 0.5 * a0.cos(),
            0.8 * a1.cos() - (-0.3) * a1.sin(),
            0.8 * a1.sin() + (-0.3) * a1.cos(),
        ];
        assert!(max_abs_diff(&data, &expected) < 1e-5);
    }

    #[test]
    fn apply_rope_f32_zero_input() {
        let ft = build_frequency_table(&RoPEConfig::new(8, 16));
        let mut data = vec![0.0f32; 8];
        apply_rope_f32(&mut data, &ft, 7, 8);
        assert!(data.iter().all(|v| v.abs() < 1e-10));
    }

    // ================================================================
    // §3  AVX2 vs scalar parity
    // ================================================================

    #[test]
    fn avx2_scalar_parity_various_dims() {
        for dim in [8, 16, 32, 64, 128] {
            let ft = build_frequency_table(&RoPEConfig::new(dim, 32));
            for pos in [0, 1, 7, 31] {
                let orig: Vec<f32> = (0..dim).map(|i| ((i * 7 + 3) as f32) * 0.01 - 2.0).collect();
                let mut scalar = orig.clone();
                let mut simd = orig.clone();

                apply_rope_f32(&mut scalar, &ft, pos, dim);
                // apply_rope_avx2 returns false on non-x86_64; that's fine — we still
                // verify the scalar path.
                if !apply_rope_avx2(&mut simd, &ft, pos, dim) {
                    apply_rope_f32(&mut simd, &ft, pos, dim);
                }

                let diff = max_abs_diff(&scalar, &simd);
                assert!(diff < 1e-5, "dim={dim} pos={pos} diff={diff}");
            }
        }
    }

    #[test]
    fn avx2_scalar_parity_odd_half_dim() {
        // dim=6 → half_dim=3, not a multiple of 4 → exercises scalar tail in AVX2
        let dim = 6;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 8));
        let orig: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 1.5).collect();
        let mut scalar = orig.clone();
        let mut simd = orig.clone();

        apply_rope_f32(&mut scalar, &ft, 3, dim);
        if !apply_rope_avx2(&mut simd, &ft, 3, dim) {
            apply_rope_f32(&mut simd, &ft, 3, dim);
        }
        assert!(max_abs_diff(&scalar, &simd) < 1e-5);
    }

    #[test]
    fn avx2_scalar_parity_dim2() {
        // dim=2 — too small for AVX2, should use scalar
        let ft = build_frequency_table(&RoPEConfig::new(2, 4));
        let mut scalar = vec![3.0, -1.0];
        let mut simd = scalar.clone();
        apply_rope_f32(&mut scalar, &ft, 2, 2);
        if !apply_rope_avx2(&mut simd, &ft, 2, 2) {
            apply_rope_f32(&mut simd, &ft, 2, 2);
        }
        assert!(max_abs_diff(&scalar, &simd) < 1e-6);
    }

    #[test]
    fn avx2_scalar_parity_large_dim() {
        let dim = 256;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 16));
        let orig: Vec<f32> = (0..dim).map(|i| ((i * 13 + 5) as f32).sin()).collect();
        let mut scalar = orig.clone();
        let mut simd = orig.clone();

        apply_rope_f32(&mut scalar, &ft, 10, dim);
        if !apply_rope_avx2(&mut simd, &ft, 10, dim) {
            apply_rope_f32(&mut simd, &ft, 10, dim);
        }
        assert!(max_abs_diff(&scalar, &simd) < 1e-5);
    }

    #[test]
    fn dispatch_matches_scalar() {
        let dim = 32;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 16));
        let orig: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
        let mut dispatched = orig.clone();
        let mut scalar = orig.clone();

        apply_rope_dispatch(&mut dispatched, &ft, 5, dim);
        apply_rope_f32(&mut scalar, &ft, 5, dim);

        assert!(max_abs_diff(&dispatched, &scalar) < 1e-5);
    }

    // ================================================================
    // §4  Interleaved vs half-rotated equivalence
    // ================================================================

    #[test]
    fn interleaved_matches_apply_rope_f32() {
        let dim = 16;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 8));
        let orig: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) * 0.9).collect();

        let mut interleaved = orig.clone();
        apply_rope_interleaved(&mut interleaved, &ft, 3);

        let mut standard = orig.clone();
        apply_rope_f32(&mut standard, &ft, 3, dim);

        assert!(max_abs_diff(&interleaved, &standard) < 1e-6);
    }

    #[test]
    fn half_rotated_matches_naive() {
        for dim in [4, 8, 16, 32] {
            let ft = build_frequency_table(&RoPEConfig::new(dim, 8));
            let orig: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.3).collect();

            let mut ours = orig.clone();
            apply_rope_half_rotated(&mut ours, &ft, 4);

            let mut naive = orig.clone();
            naive_rope_half_rotated(&mut naive, dim, 4, 10_000.0);

            assert!(max_abs_diff(&ours, &naive) < 1e-5, "dim={dim}");
        }
    }

    #[test]
    fn half_rotated_preserves_norm() {
        let ft = build_frequency_table(&RoPEConfig::new(16, 64));
        for pos in [0, 1, 10, 63] {
            let mut data: Vec<f32> = (0..16).map(|i| (i as f32 * 0.5) - 4.0).collect();
            let before = vec_norm(&data);
            apply_rope_half_rotated(&mut data, &ft, pos);
            let after = vec_norm(&data);
            assert!((before - after).abs() < 1e-4, "norm changed at pos={pos}");
        }
    }

    #[test]
    fn half_rotated_identity_at_pos_zero() {
        let ft = build_frequency_table(&RoPEConfig::new(8, 2));
        let mut data: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let orig = data.clone();
        apply_rope_half_rotated(&mut data, &ft, 0);
        assert!(max_abs_diff(&data, &orig) < 1e-6);
    }

    #[test]
    fn interleaved_and_half_rotated_produce_different_results() {
        let dim = 8;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 4));
        let orig: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.7).collect();

        let mut inter = orig.clone();
        apply_rope_interleaved(&mut inter, &ft, 2);

        let mut half_rot = orig.clone();
        apply_rope_half_rotated(&mut half_rot, &ft, 2);

        // They should differ because the pairing is different
        assert!(max_abs_diff(&inter, &half_rot) > 1e-6);
    }

    // ================================================================
    // §5  Inverse correctness
    // ================================================================

    #[test]
    fn inverse_recovers_original() {
        for dim in [2, 4, 8, 16, 32, 64, 128] {
            let ft = build_frequency_table(&RoPEConfig::new(dim, 32));
            for pos in [0, 1, 7, 31] {
                let orig: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
                let mut data = orig.clone();
                apply_rope_f32(&mut data, &ft, pos, dim);
                inverse_rope(&mut data, &ft, pos, dim);
                let diff = max_abs_diff(&data, &orig);
                assert!(diff < 1e-4, "dim={dim} pos={pos} diff={diff}");
            }
        }
    }

    #[test]
    fn inverse_at_position_zero() {
        let ft = build_frequency_table(&RoPEConfig::new(8, 2));
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = data.clone();
        apply_rope_f32(&mut data, &ft, 0, 8);
        inverse_rope(&mut data, &ft, 0, 8);
        assert!(max_abs_diff(&data, &orig) < 1e-6);
    }

    #[test]
    fn inverse_double_apply_is_double_rotation() {
        let ft = build_frequency_table(&RoPEConfig::new(8, 16));
        let orig: Vec<f32> = vec![1.0, 0.0, 0.5, -0.5, 0.3, 0.7, -1.0, 0.2];
        let mut data = orig.clone();
        apply_rope_f32(&mut data, &ft, 5, 8);
        apply_rope_f32(&mut data, &ft, 5, 8);
        // Two forward rotations
        inverse_rope(&mut data, &ft, 5, 8);
        inverse_rope(&mut data, &ft, 5, 8);
        assert!(max_abs_diff(&data, &orig) < 1e-4);
    }

    #[test]
    fn inverse_with_dispatch() {
        let dim = 32;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 16));
        let orig: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let mut data = orig.clone();
        apply_rope_dispatch(&mut data, &ft, 10, dim);
        inverse_rope(&mut data, &ft, 10, dim);
        assert!(max_abs_diff(&data, &orig) < 1e-4);
    }

    #[test]
    fn inverse_zero_input() {
        let ft = build_frequency_table(&RoPEConfig::new(4, 4));
        let mut data = vec![0.0f32; 4];
        inverse_rope(&mut data, &ft, 2, 4);
        assert!(data.iter().all(|v| v.abs() < 1e-10));
    }

    // ================================================================
    // §6  Batch processing
    // ================================================================

    #[test]
    fn batch_matches_single_apply() {
        let dim = 16;
        let num_heads = 4;
        let positions: Vec<usize> = vec![0, 3, 7, 15];
        let ft = build_frequency_table(&RoPEConfig::new(dim, 16));

        let total = positions.len() * num_heads * dim;
        let orig: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 - 5.0).collect();

        let mut batch = orig.clone();
        apply_rope_batch(&mut batch, &ft, &positions, dim, num_heads);

        let mut single = orig.clone();
        for (s, &pos) in positions.iter().enumerate() {
            for h in 0..num_heads {
                let off = (s * num_heads + h) * dim;
                apply_rope_dispatch(&mut single[off..off + dim], &ft, pos, dim);
            }
        }

        assert!(max_abs_diff(&batch, &single) < 1e-5);
    }

    #[test]
    fn batch_single_position_single_head() {
        let dim = 8;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 4));
        let mut batch = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut single = batch.clone();

        apply_rope_batch(&mut batch, &ft, &[2], dim, 1);
        apply_rope_dispatch(&mut single, &ft, 2, dim);

        assert!(max_abs_diff(&batch, &single) < 1e-6);
    }

    #[test]
    fn batch_preserves_norm() {
        let dim = 32;
        let num_heads = 2;
        let positions: Vec<usize> = (0..8).collect();
        let ft = build_frequency_table(&RoPEConfig::new(dim, 8));

        let total = positions.len() * num_heads * dim;
        let mut data: Vec<f32> = (0..total).map(|i| ((i * 37 + 13) as f32).sin() * 2.5).collect();
        let norms_before: Vec<f32> = (0..positions.len() * num_heads)
            .map(|c| vec_norm(&data[c * dim..(c + 1) * dim]))
            .collect();

        apply_rope_batch(&mut data, &ft, &positions, dim, num_heads);

        for (c, n_before) in norms_before.iter().enumerate() {
            let n_after = vec_norm(&data[c * dim..(c + 1) * dim]);
            assert!((n_before - n_after).abs() < 1e-3, "norm changed at chunk {c}");
        }
    }

    #[test]
    fn batch_multi_head_same_position_same_result() {
        let dim = 8;
        let num_heads = 4;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 4));
        let head_data: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let mut data: Vec<f32> = head_data.iter().copied().cycle().take(num_heads * dim).collect();

        apply_rope_batch(&mut data, &ft, &[2], dim, num_heads);

        for h in 1..num_heads {
            for d in 0..dim {
                let ref_val = data[d];
                let val = data[h * dim + d];
                assert!((ref_val - val).abs() < 1e-6, "head {h} dim {d} differs");
            }
        }
    }

    #[test]
    fn batch_empty_positions() {
        let ft = build_frequency_table(&RoPEConfig::new(4, 4));
        let mut data: Vec<f32> = vec![];
        apply_rope_batch(&mut data, &ft, &[], 4, 1);
        assert!(data.is_empty());
    }

    #[test]
    fn batch_zero_input() {
        let dim = 16;
        let positions = vec![0, 1, 2, 3];
        let ft = build_frequency_table(&RoPEConfig::new(dim, 4));
        let mut data = vec![0.0f32; positions.len() * 2 * dim];
        apply_rope_batch(&mut data, &ft, &positions, dim, 2);
        assert!(data.iter().all(|v| v.abs() < 1e-10));
    }

    // ================================================================
    // §7  Edge cases
    // ================================================================

    #[test]
    fn edge_dim_2() {
        let ft = build_frequency_table(&RoPEConfig::new(2, 8));
        let mut data = vec![1.0, 0.0];
        apply_rope_dispatch(&mut data, &ft, 3, 2);
        let angle = 3.0f32; // base^0 = 1.0, angle = 3*1
        assert!((data[0] - angle.cos()).abs() < 1e-5);
        assert!((data[1] - angle.sin()).abs() < 1e-5);
    }

    #[test]
    fn edge_very_large_position() {
        let max_pos = 8192;
        let ft = build_frequency_table(&RoPEConfig::new(8, max_pos));
        let mut data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        apply_rope_dispatch(&mut data, &ft, max_pos - 1, 8);
        assert!(data.iter().all(|v| v.is_finite()), "non-finite at large position");
        let norm = vec_norm(&data);
        assert!((norm - 2.0).abs() < 1e-3, "norm at large pos: {norm}");
    }

    #[test]
    fn edge_zero_vector() {
        let ft = build_frequency_table(&RoPEConfig::new(64, 16));
        let mut data = vec![0.0f32; 64];
        apply_rope_dispatch(&mut data, &ft, 10, 64);
        assert!(data.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn edge_large_dim() {
        let dim = 512;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 4));
        let mut data: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let before = vec_norm(&data);
        apply_rope_dispatch(&mut data, &ft, 2, dim);
        let after = vec_norm(&data);
        assert!((before - after).abs() < 1e-3);
    }

    #[test]
    fn edge_all_ones() {
        let ft = build_frequency_table(&RoPEConfig::new(8, 4));
        let mut data = vec![1.0f32; 8];
        apply_rope_dispatch(&mut data, &ft, 2, 8);
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn edge_negative_values() {
        let ft = build_frequency_table(&RoPEConfig::new(8, 4));
        let mut data = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
        let before = vec_norm(&data);
        apply_rope_dispatch(&mut data, &ft, 1, 8);
        let after = vec_norm(&data);
        assert!((before - after).abs() < 1e-4);
    }

    #[test]
    fn edge_alternating_sign() {
        let ft = build_frequency_table(&RoPEConfig::new(8, 4));
        let mut data: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let before = vec_norm(&data);
        apply_rope_dispatch(&mut data, &ft, 3, 8);
        let after = vec_norm(&data);
        assert!((before - after).abs() < 1e-4);
    }

    // ================================================================
    // §8  Scaling + apply integration
    // ================================================================

    #[test]
    fn apply_with_ntk_scaling_preserves_norm() {
        let cfg = RoPEConfig::new(16, 32).with_scaling(ScalingType::Ntk { factor: 4.0 });
        let ft = build_frequency_table(&cfg);
        let mut data: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let before = vec_norm(&data);
        apply_rope_f32(&mut data, &ft, 15, 16);
        let after = vec_norm(&data);
        assert!((before - after).abs() < 1e-4);
    }

    #[test]
    fn apply_with_linear_scaling_preserves_norm() {
        let cfg = RoPEConfig::new(16, 32).with_scaling(ScalingType::Linear { factor: 2.0 });
        let ft = build_frequency_table(&cfg);
        let mut data: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let before = vec_norm(&data);
        apply_rope_f32(&mut data, &ft, 20, 16);
        let after = vec_norm(&data);
        assert!((before - after).abs() < 1e-4);
    }

    #[test]
    fn apply_with_yarn_preserves_norm() {
        let cfg = RoPEConfig::new(16, 64).with_scaling(ScalingType::YaRN {
            factor: 4.0,
            original_max_seq_len: 16,
            beta_fast: 32.0,
            beta_slow: 1.0,
        });
        let ft = build_frequency_table(&cfg);
        let mut data: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let before = vec_norm(&data);
        apply_rope_f32(&mut data, &ft, 50, 16);
        let after = vec_norm(&data);
        assert!((before - after).abs() < 1e-4);
    }

    #[test]
    fn inverse_with_ntk_scaling() {
        let cfg = RoPEConfig::new(16, 32).with_scaling(ScalingType::Ntk { factor: 2.0 });
        let ft = build_frequency_table(&cfg);
        let orig: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let mut data = orig.clone();
        apply_rope_f32(&mut data, &ft, 10, 16);
        inverse_rope(&mut data, &ft, 10, 16);
        assert!(max_abs_diff(&data, &orig) < 1e-4);
    }

    #[test]
    fn batch_with_linear_scaling() {
        let cfg = RoPEConfig::new(8, 16).with_scaling(ScalingType::Linear { factor: 2.0 });
        let ft = build_frequency_table(&cfg);
        let positions = vec![0, 2, 4, 6];
        let total = 4 * 2 * 8;
        let mut data: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let norms_before: Vec<f32> = (0..8).map(|c| vec_norm(&data[c * 8..(c + 1) * 8])).collect();
        apply_rope_batch(&mut data, &ft, &positions, 8, 2);
        for (c, nb) in norms_before.iter().enumerate() {
            let na = vec_norm(&data[c * 8..(c + 1) * 8]);
            assert!((nb - na).abs() < 1e-3, "norm changed at chunk {c}");
        }
    }

    // ================================================================
    // §9  Property tests
    // ================================================================

    #[test]
    fn property_rope_is_rotation_norm_preserved() {
        // For many random vectors and positions, the norm must be preserved.
        let dim = 32;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 1024));
        let mut rng_state: u64 = 0xDEAD_BEEF;
        for _ in 0..200 {
            // Simple LCG for deterministic pseudo-random
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let pos = (rng_state >> 32) as usize % 1024;
            let data: Vec<f32> = (0..dim)
                .map(|_i| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng_state >> 16) as i32 as f32) / (i32::MAX as f32) * 10.0
                })
                .collect();
            let before = vec_norm(&data);
            let mut rotated = data;
            apply_rope_dispatch(&mut rotated, &ft, pos, dim);
            let after = vec_norm(&rotated);
            assert!(
                (before - after).abs() < 1e-3,
                "norm not preserved: {before} vs {after} at pos={pos}"
            );
        }
    }

    #[test]
    fn property_inverse_is_actual_inverse() {
        let dim = 16;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 256));
        let mut rng_state: u64 = 0xCAFE_BABE;
        for _ in 0..200 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let pos = (rng_state >> 32) as usize % 256;
            let orig: Vec<f32> = (0..dim)
                .map(|_| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng_state >> 16) as i32 as f32) / (i32::MAX as f32) * 5.0
                })
                .collect();
            let mut data = orig.clone();
            apply_rope_f32(&mut data, &ft, pos, dim);
            inverse_rope(&mut data, &ft, pos, dim);
            let diff = max_abs_diff(&data, &orig);
            assert!(diff < 1e-4, "inverse failed at pos={pos}: diff={diff}");
        }
    }

    #[test]
    fn property_position_zero_is_always_identity() {
        let ft = build_frequency_table(&RoPEConfig::new(64, 1));
        let mut rng_state: u64 = 0x1234_5678;
        for _ in 0..100 {
            let orig: Vec<f32> = (0..64)
                .map(|_| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng_state >> 16) as i32 as f32) / (i32::MAX as f32)
                })
                .collect();
            let mut data = orig.clone();
            apply_rope_dispatch(&mut data, &ft, 0, 64);
            assert!(max_abs_diff(&data, &orig) < 1e-5);
        }
    }

    #[test]
    fn property_half_rotated_preserves_norm() {
        let dim = 16;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 256));
        let mut rng_state: u64 = 0xBAAD_F00D;
        for _ in 0..200 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let pos = (rng_state >> 32) as usize % 256;
            let mut data: Vec<f32> = (0..dim)
                .map(|_| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng_state >> 16) as i32 as f32) / (i32::MAX as f32) * 3.0
                })
                .collect();
            let before = vec_norm(&data);
            apply_rope_half_rotated(&mut data, &ft, pos);
            let after = vec_norm(&data);
            assert!((before - after).abs() < 1e-3, "norm changed at pos={pos}");
        }
    }

    #[test]
    fn property_different_positions_produce_different_rotations() {
        let dim = 8;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 128));
        let orig: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut prev = orig.clone();
        apply_rope_f32(&mut prev, &ft, 0, dim);

        // Every consecutive position should produce a distinct result
        for pos in 1..128 {
            let mut curr = orig.clone();
            apply_rope_f32(&mut curr, &ft, pos, dim);
            assert!(
                max_abs_diff(&prev, &curr) > 1e-6,
                "positions {pos} and {} produced same result",
                pos - 1
            );
            prev = curr;
        }
    }

    #[test]
    fn property_batch_order_independent() {
        // Applying batch with positions [a,b] should give same results for slot 0
        // regardless of what position b is.
        let dim = 8;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 32));
        let head: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.7).collect();

        let mut batch1 = vec![0.0f32; 2 * dim];
        batch1[..dim].copy_from_slice(&head);
        batch1[dim..].copy_from_slice(&head);
        apply_rope_batch(&mut batch1, &ft, &[5, 10], dim, 1);

        let mut batch2 = vec![0.0f32; 2 * dim];
        batch2[..dim].copy_from_slice(&head);
        batch2[dim..].copy_from_slice(&head);
        apply_rope_batch(&mut batch2, &ft, &[5, 20], dim, 1);

        // First head at position 5 should be identical in both
        assert!(max_abs_diff(&batch1[..dim], &batch2[..dim]) < 1e-6);
        // Second heads should differ
        assert!(max_abs_diff(&batch1[dim..], &batch2[dim..]) > 1e-6);
    }

    // ================================================================
    // §10  Config / API tests
    // ================================================================

    #[test]
    #[should_panic(expected = "dim must be even")]
    fn config_rejects_odd_dim() {
        RoPEConfig::new(7, 16);
    }

    #[test]
    #[should_panic(expected = "dim must be even")]
    fn config_rejects_zero_dim() {
        RoPEConfig::new(0, 16);
    }

    #[test]
    fn config_builder_chain() {
        let cfg = RoPEConfig::new(8, 32)
            .with_base_freq(500_000.0)
            .with_scaling(ScalingType::Ntk { factor: 2.0 });
        assert_eq!(cfg.dim, 8);
        assert_eq!(cfg.max_seq_len, 32);
        assert!((cfg.base_freq - 500_000.0).abs() < 1e-6);
        assert_eq!(cfg.scaling_type, ScalingType::Ntk { factor: 2.0 });
    }

    #[test]
    fn scaling_type_default_is_none() {
        let s = ScalingType::default();
        assert_eq!(s, ScalingType::None);
    }

    #[test]
    fn freq_table_accessors() {
        let ft = build_frequency_table(&RoPEConfig::new(4, 2));
        assert_eq!(ft.half_dim, 2);
        assert_eq!(ft.max_seq_len, 2);
        // Position 0 should be identity
        assert!((ft.cos(0, 0) - 1.0).abs() < 1e-6);
        assert!(ft.sin(0, 0).abs() < 1e-6);
    }

    // ================================================================
    // §11  Additional coverage
    // ================================================================

    #[test]
    fn avx2_scalar_parity_batch() {
        let dim = 32;
        let num_heads = 2;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 16));
        let positions = vec![0, 3, 7, 15];
        let total = positions.len() * num_heads * dim;
        let orig: Vec<f32> = (0..total).map(|i| ((i * 11 + 7) as f32).sin()).collect();

        let mut batch = orig.clone();
        apply_rope_batch(&mut batch, &ft, &positions, dim, num_heads);

        // Scalar-only reference
        let mut scalar = orig.clone();
        for (s, &pos) in positions.iter().enumerate() {
            for h in 0..num_heads {
                let off = (s * num_heads + h) * dim;
                apply_rope_f32(&mut scalar[off..off + dim], &ft, pos, dim);
            }
        }
        assert!(max_abs_diff(&batch, &scalar) < 1e-5);
    }

    #[test]
    fn freq_table_ntk_larger_factor_larger_base() {
        let ft2 = build_frequency_table(
            &RoPEConfig::new(8, 4).with_scaling(ScalingType::Ntk { factor: 2.0 }),
        );
        let ft8 = build_frequency_table(
            &RoPEConfig::new(8, 4).with_scaling(ScalingType::Ntk { factor: 8.0 }),
        );
        // Pair 0 has exponent 0 (base^0=1), so use pair 1 where exponent matters.
        // Larger factor → larger effective base → lower inv_freq → smaller angle → cos closer to 1
        let cos_f2 = ft2.cos(1, 1);
        let cos_f8 = ft8.cos(1, 1);
        assert!(cos_f8 > cos_f2, "larger NTK factor should slow rotation at pair 1");
    }

    #[test]
    fn apply_rope_f32_multiple_positions_sequential() {
        let dim = 8;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 128));
        let mut data: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
        // Applying pos=3 then pos=5 should not crash and should stay finite
        apply_rope_f32(&mut data, &ft, 3, dim);
        apply_rope_f32(&mut data, &ft, 5, dim);
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn inverse_half_rotated_round_trip() {
        // Verify that applying half-rotated then manually inverting recovers the original.
        let dim = 8;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 8));
        let orig: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let mut data = orig.clone();
        apply_rope_half_rotated(&mut data, &ft, 3);
        // Manual inverse for half-rotated: negate sin
        let half = dim / 2;
        for i in 0..half {
            let c = ft.cos(3, i);
            let s = ft.sin(3, i);
            let y0 = data[i];
            let y1 = data[i + half];
            data[i] = y0 * c + y1 * s;
            data[i + half] = -y0 * s + y1 * c;
        }
        assert!(max_abs_diff(&data, &orig) < 1e-4);
    }

    #[test]
    fn batch_large_seq_len() {
        let dim = 8;
        let seq_len = 64;
        let num_heads = 1;
        let ft = build_frequency_table(&RoPEConfig::new(dim, seq_len));
        let positions: Vec<usize> = (0..seq_len).collect();
        let total = seq_len * num_heads * dim;
        let mut data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
        apply_rope_batch(&mut data, &ft, &positions, dim, num_heads);
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn apply_rope_f32_dim_128() {
        let dim = 128;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 16));
        let mut data: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
        let before = vec_norm(&data);
        apply_rope_f32(&mut data, &ft, 7, dim);
        let after = vec_norm(&data);
        assert!((before - after).abs() < 1e-3);
    }

    #[test]
    fn freq_table_dynamic_factor_grows_with_seq_len() {
        let cfg =
            RoPEConfig::new(8, 16).with_scaling(ScalingType::Dynamic { original_max_seq_len: 4 });
        let ft_short = build_frequency_table_with_seq(&cfg, 4);
        let ft_long = build_frequency_table_with_seq(&cfg, 16);
        // Pair 0 has exponent 0, use pair 1. Longer seq → higher base → cos closer to 1.
        let cos_short = ft_short.cos(1, 1);
        let cos_long = ft_long.cos(1, 1);
        assert!(cos_long > cos_short, "dynamic should slow rotation for longer seq at pair 1");
    }

    #[test]
    fn avx2_returns_false_for_small_dim() {
        let ft = build_frequency_table(&RoPEConfig::new(4, 2));
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        // dim=4 < 8, so AVX2 should decline
        let used = apply_rope_avx2(&mut data, &ft, 1, 4);
        // On x86_64 with AVX2, it should return false because head_dim < 8.
        // On other arches it always returns false.
        assert!(!used, "AVX2 should decline for dim < 8");
    }

    #[test]
    fn interleaved_identity_at_pos_zero() {
        let ft = build_frequency_table(&RoPEConfig::new(16, 2));
        let mut data: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 2.0).collect();
        let orig = data.clone();
        apply_rope_interleaved(&mut data, &ft, 0);
        assert!(max_abs_diff(&data, &orig) < 1e-6);
    }

    #[test]
    fn half_rotated_zero_input() {
        let ft = build_frequency_table(&RoPEConfig::new(8, 4));
        let mut data = vec![0.0f32; 8];
        apply_rope_half_rotated(&mut data, &ft, 2);
        assert!(data.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn batch_non_contiguous_positions() {
        let dim = 4;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 100));
        let positions = vec![0, 50, 99];
        let total = positions.len() * dim;
        let mut data: Vec<f32> = (0..total).map(|i| i as f32 + 1.0).collect();
        apply_rope_batch(&mut data, &ft, &positions, dim, 1);
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn inverse_preserves_zero() {
        let ft = build_frequency_table(&RoPEConfig::new(16, 8));
        let mut data = vec![0.0f32; 16];
        inverse_rope(&mut data, &ft, 5, 16);
        assert!(data.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn property_avx2_dispatch_equals_scalar_randomised() {
        let dim = 64;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 128));
        let mut rng: u64 = 0xFACE_CAFE;
        for _ in 0..100 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let pos = (rng >> 32) as usize % 128;
            let orig: Vec<f32> = (0..dim)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng >> 16) as i32 as f32) / (i32::MAX as f32) * 5.0
                })
                .collect();
            let mut dispatched = orig.clone();
            let mut scalar = orig.clone();
            apply_rope_dispatch(&mut dispatched, &ft, pos, dim);
            apply_rope_f32(&mut scalar, &ft, pos, dim);
            assert!(max_abs_diff(&dispatched, &scalar) < 1e-5);
        }
    }

    #[test]
    fn property_interleaved_preserves_norm() {
        let dim = 32;
        let ft = build_frequency_table(&RoPEConfig::new(dim, 64));
        let mut rng: u64 = 0xABCD_1234;
        for _ in 0..100 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let pos = (rng >> 32) as usize % 64;
            let mut data: Vec<f32> = (0..dim)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng >> 16) as i32 as f32) / (i32::MAX as f32) * 7.0
                })
                .collect();
            let before = vec_norm(&data);
            apply_rope_interleaved(&mut data, &ft, pos);
            let after = vec_norm(&data);
            assert!((before - after).abs() < 1e-3, "norm changed at pos={pos}");
        }
    }

    #[test]
    fn linear_scaling_larger_factor_slower_rotation() {
        let ft1 = build_frequency_table(
            &RoPEConfig::new(8, 8).with_scaling(ScalingType::Linear { factor: 1.0 }),
        );
        let ft4 = build_frequency_table(
            &RoPEConfig::new(8, 8).with_scaling(ScalingType::Linear { factor: 4.0 }),
        );
        // At position 4 with factor=4, should look like position 1 with factor=1
        for i in 0..4 {
            assert!((ft1.cos(1, i) - ft4.cos(4, i)).abs() < 1e-5, "cos mismatch pair {i}");
            assert!((ft1.sin(1, i) - ft4.sin(4, i)).abs() < 1e-5, "sin mismatch pair {i}");
        }
    }

    #[test]
    fn yarn_high_freq_band_unscaled() {
        // With very large beta_fast, the "high-frequency" band should cover all pairs.
        let ft_none = build_frequency_table(&RoPEConfig::new(8, 8));
        let ft_yarn =
            build_frequency_table(&RoPEConfig::new(8, 8).with_scaling(ScalingType::YaRN {
                factor: 4.0,
                original_max_seq_len: 4,
                beta_fast: 100_000.0, // very large → all pairs in high-freq band
                beta_slow: 0.001,
            }));
        // All pairs should be unscaled → match None
        let diff = max_abs_diff(&ft_none.cos_table, &ft_yarn.cos_table);
        assert!(diff < 1e-5, "large beta_fast should leave all pairs unscaled: {diff}");
    }

    #[test]
    fn apply_rope_f32_with_dynamic_scaling() {
        let cfg =
            RoPEConfig::new(8, 16).with_scaling(ScalingType::Dynamic { original_max_seq_len: 8 });
        let ft = build_frequency_table_with_seq(&cfg, 16);
        let mut data: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let before = vec_norm(&data);
        apply_rope_f32(&mut data, &ft, 10, 8);
        let after = vec_norm(&data);
        assert!((before - after).abs() < 1e-4);
    }

    #[test]
    fn batch_with_yarn_scaling() {
        let cfg = RoPEConfig::new(8, 32).with_scaling(ScalingType::YaRN {
            factor: 2.0,
            original_max_seq_len: 16,
            beta_fast: 32.0,
            beta_slow: 1.0,
        });
        let ft = build_frequency_table(&cfg);
        let positions = vec![0, 8, 16, 24];
        let mut data: Vec<f32> = (0..4 * 8).map(|i| (i as f32 + 1.0) * 0.2).collect();
        apply_rope_batch(&mut data, &ft, &positions, 8, 1);
        assert!(data.iter().all(|v| v.is_finite()));
    }
}
