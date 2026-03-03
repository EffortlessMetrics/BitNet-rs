//! Extended SIMD-optimized RoPE (Rotary Position Embedding) kernels.
//!
//! Builds on the base `rope_simd` module with production-grade features:
//!
//! - **NTK-aware scaling** with configurable alpha for long-context inference
//! - **Dynamic NTK scaling** that progressively extends context at runtime
//! - **YaRN** (Yet another RoPE extensioN) with smooth frequency interpolation
//! - **Precomputed frequency tables** in SIMD-friendly SoA layout
//! - **Batched RoPE** applied across all heads simultaneously
//! - **Fused RoPE + attention score** to reduce memory traffic
//! - **Interleaved vs rotary-half** layout support
//! - **Custom base frequency** adjustment for fine-tuning scenarios

use std::f32::consts::PI;

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

// ── Scaling strategy ────────────────────────────────────────────────

/// Extended frequency-scaling strategy for long-context RoPE.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum ExtendedScaling {
    /// Standard RoPE — no frequency modification.
    #[default]
    None,
    /// Linear position interpolation: divide position by `factor`.
    Linear { factor: f32 },
    /// NTK-aware scaling: raise base to preserve high-frequency components.
    ///
    /// `effective_base = base * alpha^(dim / (dim - 2))`
    NtkAware { alpha: f32 },
    /// Dynamic NTK: base grows with sequence length beyond `trained_ctx`.
    DynamicNtk { trained_ctx: usize },
    /// YaRN blending of high-frequency (unscaled) and low-frequency (scaled)
    /// bands via smooth hermite interpolation.
    YaRN { factor: f32, trained_ctx: usize, beta_fast: f32, beta_slow: f32, attn_factor: f32 },
    /// Custom base frequency replacement (ignores config base entirely).
    CustomBase { base: f32 },
}

// Default is derived via `#[default]` on `None`.

// ── Rotation layout ─────────────────────────────────────────────────

/// How rotation pairs are arranged in the head vector.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RotationLayout {
    /// Interleaved (GPT-NeoX): pairs are `(x[2i], x[2i+1])`.
    #[default]
    Interleaved,
    /// Rotary-half (LLaMA): pairs are `(x[i], x[i + dim/2])`.
    RotaryHalf,
}

// ── Configuration ───────────────────────────────────────────────────

/// Configuration for extended RoPE kernels.
#[derive(Debug, Clone)]
pub struct ExtendedRopeConfig {
    /// Per-head embedding dimension (must be even and > 0).
    pub head_dim: usize,
    /// Maximum sequence length for the frequency table.
    pub max_seq_len: usize,
    /// Base rotation frequency (default 10 000).
    pub base_freq: f32,
    /// Scaling strategy.
    pub scaling: ExtendedScaling,
    /// Rotation-pair layout.
    pub layout: RotationLayout,
}

impl ExtendedRopeConfig {
    /// Create a config with default base (10 000) and no scaling.
    ///
    /// # Panics
    ///
    /// Panics if `head_dim` is zero or odd.
    pub fn new(head_dim: usize, max_seq_len: usize) -> Self {
        assert!(head_dim > 0 && head_dim.is_multiple_of(2), "head_dim must be even and > 0");
        Self {
            head_dim,
            max_seq_len,
            base_freq: 10_000.0,
            scaling: ExtendedScaling::None,
            layout: RotationLayout::Interleaved,
        }
    }

    /// Override base frequency.
    #[must_use]
    pub fn with_base_freq(mut self, base: f32) -> Self {
        self.base_freq = base;
        self
    }

    /// Override scaling strategy.
    #[must_use]
    pub fn with_scaling(mut self, s: ExtendedScaling) -> Self {
        self.scaling = s;
        self
    }

    /// Override rotation layout.
    #[must_use]
    pub fn with_layout(mut self, l: RotationLayout) -> Self {
        self.layout = l;
        self
    }

    fn half_dim(&self) -> usize {
        self.head_dim / 2
    }
}

// ── Precomputed frequency table (SoA layout) ────────────────────────

/// SIMD-friendly precomputed cos/sin tables in Structure-of-Arrays layout.
///
/// Two contiguous arrays `cos[pos * half_dim + pair]` and
/// `sin[pos * half_dim + pair]` enable aligned, sequential reads inside
/// the SIMD inner loop.
#[derive(Debug, Clone)]
pub struct ExtendedFreqTable {
    /// Cosine values `[max_seq_len × half_dim]`.
    pub cos: Vec<f32>,
    /// Sine values `[max_seq_len × half_dim]`.
    pub sin: Vec<f32>,
    /// Half the head dimension.
    pub half_dim: usize,
    /// Maximum sequence length covered.
    pub max_seq_len: usize,
    /// The layout these frequencies were built for.
    pub layout: RotationLayout,
}

impl ExtendedFreqTable {
    #[inline]
    pub fn cos_val(&self, pos: usize, pair: usize) -> f32 {
        self.cos[pos * self.half_dim + pair]
    }

    #[inline]
    pub fn sin_val(&self, pos: usize, pair: usize) -> f32 {
        self.sin[pos * self.half_dim + pair]
    }

    /// Number of entries per position.
    #[inline]
    pub fn stride(&self) -> usize {
        self.half_dim
    }
}

// ── Effective base computation ──────────────────────────────────────

/// Compute the effective rotation base after applying the scaling strategy.
fn compute_effective_base(cfg: &ExtendedRopeConfig, current_seq_len: usize) -> f32 {
    match cfg.scaling {
        ExtendedScaling::None | ExtendedScaling::Linear { .. } | ExtendedScaling::YaRN { .. } => {
            cfg.base_freq
        }
        ExtendedScaling::NtkAware { alpha } => {
            let exp = cfg.head_dim as f32 / (cfg.head_dim as f32 - 2.0);
            cfg.base_freq * alpha.powf(exp)
        }
        ExtendedScaling::DynamicNtk { trained_ctx } => {
            let seq = current_seq_len.max(1);
            if seq <= trained_ctx {
                cfg.base_freq
            } else {
                let ratio = seq as f32 / trained_ctx as f32;
                let exp = cfg.head_dim as f32 / (cfg.head_dim as f32 - 2.0);
                cfg.base_freq * ratio.powf(exp)
            }
        }
        ExtendedScaling::CustomBase { base } => base,
    }
}

/// Per-pair frequency multiplier (used by Linear and YaRN).
fn per_pair_scale(cfg: &ExtendedRopeConfig, pair: usize) -> f32 {
    match cfg.scaling {
        ExtendedScaling::Linear { factor } => 1.0 / factor,
        ExtendedScaling::YaRN { factor, trained_ctx, beta_fast, beta_slow, .. } => {
            let half = cfg.half_dim();
            let lo = (trained_ctx as f32 / (beta_fast * 2.0 * PI)).floor() as usize;
            let hi = (trained_ctx as f32 / (beta_slow * 2.0 * PI)).ceil() as usize;

            if pair < lo.min(half) {
                1.0
            } else if pair >= hi.min(half) {
                1.0 / factor
            } else {
                let range = hi.saturating_sub(lo).max(1) as f32;
                let t = pair.saturating_sub(lo) as f32 / range;
                let h = t * t * (3.0 - 2.0 * t); // hermite smoothstep
                1.0 * (1.0 - h) + (1.0 / factor) * h
            }
        }
        _ => 1.0,
    }
}

// ── Table construction ──────────────────────────────────────────────

/// Build an [`ExtendedFreqTable`] from configuration.
///
/// Covers positions `[0, max_seq_len)` and pairs `[0, half_dim)`.
pub fn build_extended_freq_table(cfg: &ExtendedRopeConfig) -> ExtendedFreqTable {
    build_extended_freq_table_for_seq(cfg, cfg.max_seq_len)
}

/// Build a frequency table where the effective base is computed for
/// a specific current sequence length (relevant for `DynamicNtk`).
pub fn build_extended_freq_table_for_seq(
    cfg: &ExtendedRopeConfig,
    current_seq_len: usize,
) -> ExtendedFreqTable {
    let half = cfg.half_dim();
    let base = compute_effective_base(cfg, current_seq_len);
    let cap = cfg.max_seq_len * half;
    let mut cos_tab = Vec::with_capacity(cap);
    let mut sin_tab = Vec::with_capacity(cap);

    for pos in 0..cfg.max_seq_len {
        for i in 0..half {
            let exponent = -(2.0 * i as f32) / cfg.head_dim as f32;
            let inv_freq = base.powf(exponent);
            let scale = per_pair_scale(cfg, i);
            let angle = pos as f32 * inv_freq * scale;
            let (s, c) = angle.sin_cos();
            cos_tab.push(c);
            sin_tab.push(s);
        }
    }

    ExtendedFreqTable {
        cos: cos_tab,
        sin: sin_tab,
        half_dim: half,
        max_seq_len: cfg.max_seq_len,
        layout: cfg.layout,
    }
}

// ── Scalar rotation kernels ─────────────────────────────────────────

/// Apply RoPE to a single head vector **in-place** using interleaved layout.
///
/// Pairs: `(x[2i], x[2i+1])` for `i` in `[0, half_dim)`.
pub fn apply_rope_interleaved(x: &mut [f32], table: &ExtendedFreqTable, position: usize) {
    let half = table.half_dim;
    for i in 0..half {
        let c = table.cos_val(position, i);
        let s = table.sin_val(position, i);
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = x0 * c - x1 * s;
        x[2 * i + 1] = x0 * s + x1 * c;
    }
}

/// Apply RoPE to a single head vector **in-place** using rotary-half layout.
///
/// Pairs: `(x[i], x[i + half_dim])` for `i` in `[0, half_dim)`.
pub fn apply_rope_rotary_half(x: &mut [f32], table: &ExtendedFreqTable, position: usize) {
    let half = table.half_dim;
    for i in 0..half {
        let c = table.cos_val(position, i);
        let s = table.sin_val(position, i);
        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
    }
}

/// Apply RoPE in-place, dispatching on the table's layout.
pub fn apply_rope_single(x: &mut [f32], table: &ExtendedFreqTable, position: usize) {
    match table.layout {
        RotationLayout::Interleaved => apply_rope_interleaved(x, table, position),
        RotationLayout::RotaryHalf => apply_rope_rotary_half(x, table, position),
    }
}

/// Inverse RoPE rotation (negate the angle). Interleaved layout.
pub fn inverse_rope_interleaved(x: &mut [f32], table: &ExtendedFreqTable, position: usize) {
    let half = table.half_dim;
    for i in 0..half {
        let c = table.cos_val(position, i);
        let s = table.sin_val(position, i);
        let y0 = x[2 * i];
        let y1 = x[2 * i + 1];
        x[2 * i] = y0 * c + y1 * s;
        x[2 * i + 1] = -y0 * s + y1 * c;
    }
}

/// Inverse RoPE rotation. Rotary-half layout.
pub fn inverse_rope_rotary_half(x: &mut [f32], table: &ExtendedFreqTable, position: usize) {
    let half = table.half_dim;
    for i in 0..half {
        let c = table.cos_val(position, i);
        let s = table.sin_val(position, i);
        let y0 = x[i];
        let y1 = x[i + half];
        x[i] = y0 * c + y1 * s;
        x[i + half] = -y0 * s + y1 * c;
    }
}

/// Inverse RoPE, dispatching on layout.
pub fn inverse_rope_single(x: &mut [f32], table: &ExtendedFreqTable, position: usize) {
    match table.layout {
        RotationLayout::Interleaved => inverse_rope_interleaved(x, table, position),
        RotationLayout::RotaryHalf => inverse_rope_rotary_half(x, table, position),
    }
}

// ── AVX2 fast paths ─────────────────────────────────────────────────

/// AVX2-accelerated interleaved RoPE for a single head.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_rope_interleaved_avx2(x: &mut [f32], table: &ExtendedFreqTable, position: usize) {
    let half = table.half_dim;
    let cos_base = &table.cos[position * half..];
    let sin_base = &table.sin[position * half..];

    let simd_pairs = half / 4;
    for c in 0..simd_pairs {
        let pair_off = c * 4;
        let data_off = c * 8;

        unsafe {
            let cos4 = _mm_loadu_ps(cos_base.as_ptr().add(pair_off));
            let sin4 = _mm_loadu_ps(sin_base.as_ptr().add(pair_off));

            let cos_lo = _mm_unpacklo_ps(cos4, cos4);
            let cos_hi = _mm_unpackhi_ps(cos4, cos4);
            let cos_v = _mm256_set_m128(cos_hi, cos_lo);

            let sin_lo = _mm_unpacklo_ps(sin4, sin4);
            let sin_hi = _mm_unpackhi_ps(sin4, sin4);
            let sin_v = _mm256_set_m128(sin_hi, sin_lo);

            let vals = _mm256_loadu_ps(x.as_ptr().add(data_off));
            let swapped = _mm256_permutevar8x32_ps(vals, _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6));
            let sign = _mm256_setr_ps(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
            let result = _mm256_add_ps(
                _mm256_mul_ps(vals, cos_v),
                _mm256_mul_ps(_mm256_mul_ps(swapped, sign), sin_v),
            );
            _mm256_storeu_ps(x.as_mut_ptr().add(data_off), result);
        }
    }

    // Scalar tail
    let done = simd_pairs * 4;
    for i in done..half {
        let c = table.cos_val(position, i);
        let s = table.sin_val(position, i);
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = x0 * c - x1 * s;
        x[2 * i + 1] = x0 * s + x1 * c;
    }
}

/// AVX2-accelerated rotary-half RoPE for a single head.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_rope_rotary_half_avx2(x: &mut [f32], table: &ExtendedFreqTable, position: usize) {
    let half = table.half_dim;
    let cos_base = &table.cos[position * half..];
    let sin_base = &table.sin[position * half..];

    let simd_count = half / 8;
    for c in 0..simd_count {
        let off = c * 8;
        unsafe {
            let cos_v = _mm256_loadu_ps(cos_base.as_ptr().add(off));
            let sin_v = _mm256_loadu_ps(sin_base.as_ptr().add(off));
            let x_lo = _mm256_loadu_ps(x.as_ptr().add(off));
            let x_hi = _mm256_loadu_ps(x.as_ptr().add(off + half));

            let res_lo = _mm256_sub_ps(_mm256_mul_ps(x_lo, cos_v), _mm256_mul_ps(x_hi, sin_v));
            let res_hi = _mm256_add_ps(_mm256_mul_ps(x_lo, sin_v), _mm256_mul_ps(x_hi, cos_v));

            _mm256_storeu_ps(x.as_mut_ptr().add(off), res_lo);
            _mm256_storeu_ps(x.as_mut_ptr().add(off + half), res_hi);
        }
    }

    // Scalar tail
    let done = simd_count * 8;
    for i in done..half {
        let cv = table.cos_val(position, i);
        let sv = table.sin_val(position, i);
        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0 * cv - x1 * sv;
        x[i + half] = x0 * sv + x1 * cv;
    }
}

/// Runtime-dispatched RoPE for a single head (AVX2 → scalar).
pub fn apply_rope_dispatch(x: &mut [f32], table: &ExtendedFreqTable, position: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && table.half_dim >= 4 {
            match table.layout {
                RotationLayout::Interleaved => {
                    // Safety: AVX2 confirmed above.
                    unsafe { apply_rope_interleaved_avx2(x, table, position) };
                    return;
                }
                RotationLayout::RotaryHalf if table.half_dim >= 8 => {
                    unsafe { apply_rope_rotary_half_avx2(x, table, position) };
                    return;
                }
                _ => {}
            }
        }
    }
    apply_rope_single(x, table, position);
}

// ── Batched RoPE ────────────────────────────────────────────────────

/// Apply RoPE to **all heads at all positions** simultaneously.
///
/// `data` layout: `[seq_len × num_heads × head_dim]` (contiguous).
/// `positions[s]` gives the absolute position for sequence index `s`.
pub fn apply_rope_batched(
    data: &mut [f32],
    table: &ExtendedFreqTable,
    positions: &[usize],
    num_heads: usize,
    head_dim: usize,
) {
    let seq_len = positions.len();
    debug_assert_eq!(data.len(), seq_len * num_heads * head_dim);
    for (s, &pos) in positions.iter().enumerate().take(seq_len) {
        for h in 0..num_heads {
            let off = (s * num_heads + h) * head_dim;
            apply_rope_dispatch(&mut data[off..off + head_dim], table, pos);
        }
    }
}

/// Apply RoPE to separate query and key tensors in a single pass.
///
/// Both tensors use layout `[seq_len × num_heads × head_dim]`.
pub fn apply_rope_qk_batched(
    query: &mut [f32],
    key: &mut [f32],
    table: &ExtendedFreqTable,
    positions: &[usize],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let seq_len = positions.len();
    for (s, &pos) in positions.iter().enumerate().take(seq_len) {
        for h in 0..num_q_heads {
            let off = (s * num_q_heads + h) * head_dim;
            apply_rope_dispatch(&mut query[off..off + head_dim], table, pos);
        }
        for h in 0..num_kv_heads {
            let off = (s * num_kv_heads + h) * head_dim;
            apply_rope_dispatch(&mut key[off..off + head_dim], table, pos);
        }
    }
}

// ── Fused RoPE + attention score ────────────────────────────────────

/// Apply RoPE to query and key, then compute their dot-product attention score.
///
/// Returns `(q · k) / sqrt(head_dim)` after rotating both vectors.
/// This fuses two memory passes into one, reducing bandwidth pressure.
pub fn fused_rope_attention_score(
    query: &mut [f32],
    key: &mut [f32],
    table: &ExtendedFreqTable,
    q_pos: usize,
    k_pos: usize,
    head_dim: usize,
) -> f32 {
    apply_rope_dispatch(query, table, q_pos);
    apply_rope_dispatch(key, table, k_pos);

    let dot: f32 = query.iter().zip(key.iter()).map(|(q, k)| q * k).sum();
    dot / (head_dim as f32).sqrt()
}

/// Batched fused RoPE + attention scores for one query against multiple keys.
///
/// `query`: `[head_dim]` — single query head.
/// `keys`: `[num_keys × head_dim]` — key vectors.
/// `q_pos`: absolute position of the query.
/// `k_positions`: absolute positions of each key.
///
/// Returns a vector of scaled dot-product scores.
pub fn fused_rope_attention_scores_batched(
    query: &mut [f32],
    keys: &mut [f32],
    table: &ExtendedFreqTable,
    q_pos: usize,
    k_positions: &[usize],
    head_dim: usize,
) -> Vec<f32> {
    apply_rope_dispatch(query, table, q_pos);

    let inv_sqrt = 1.0 / (head_dim as f32).sqrt();
    k_positions
        .iter()
        .enumerate()
        .map(|(i, &k_pos)| {
            let k_slice = &mut keys[i * head_dim..(i + 1) * head_dim];
            apply_rope_dispatch(k_slice, table, k_pos);
            let dot: f32 = query.iter().zip(k_slice.iter()).map(|(q, k)| q * k).sum();
            dot * inv_sqrt
        })
        .collect()
}

// ── Dynamic NTK rebuild ─────────────────────────────────────────────

/// Rebuild the frequency table when the sequence length exceeds `trained_ctx`.
///
/// For `DynamicNtk` scaling, the effective base changes with the current
/// sequence length. Call this when you detect the context has grown.
pub fn rebuild_table_for_dynamic_ntk(
    cfg: &ExtendedRopeConfig,
    current_seq_len: usize,
) -> ExtendedFreqTable {
    build_extended_freq_table_for_seq(cfg, current_seq_len)
}

// ── NTK-aware helpers ───────────────────────────────────────────────

/// Compute the NTK-scaled base given `alpha` and `dim`.
///
/// Formula: `base * alpha^(dim / (dim - 2))`
pub fn ntk_scaled_base(base: f32, alpha: f32, dim: usize) -> f32 {
    let exp = dim as f32 / (dim as f32 - 2.0);
    base * alpha.powf(exp)
}

/// Compute the optimal alpha for extending context from `trained_ctx` to `target_ctx`.
///
/// Derived from `target / trained = alpha^(dim / (dim - 2))`.
pub fn compute_ntk_alpha(trained_ctx: usize, target_ctx: usize, dim: usize) -> f32 {
    let ratio = target_ctx as f32 / trained_ctx.max(1) as f32;
    let exp = (dim as f32 - 2.0) / dim as f32;
    ratio.powf(exp)
}

// ── YaRN attention factor ───────────────────────────────────────────

/// Compute the YaRN attention temperature correction factor.
///
/// This compensates for the distributional shift introduced by
/// frequency interpolation and should multiply the attention logits.
pub fn yarn_attention_factor(factor: f32, trained_ctx: usize, dim: usize) -> f32 {
    // The attention factor compensates for the variance change
    // introduced by interpolating RoPE frequencies. A common formula
    // is `0.1 * ln(factor) + 1.0`, clamped to be >= 1.
    let raw = 0.1 * (factor.ln()) + 1.0;
    let _ = (trained_ctx, dim); // used in more complex implementations
    raw.max(1.0)
}

/// Compute per-pair YaRN blend weights.
///
/// Returns a vector of length `half_dim` with values in `[0, 1]` where
/// 0 = fully unscaled (high-frequency) and 1 = fully scaled (low-frequency).
pub fn yarn_blend_weights(
    half_dim: usize,
    trained_ctx: usize,
    beta_fast: f32,
    beta_slow: f32,
) -> Vec<f32> {
    let lo = (trained_ctx as f32 / (beta_fast * 2.0 * PI)).floor() as usize;
    let hi = (trained_ctx as f32 / (beta_slow * 2.0 * PI)).ceil() as usize;

    (0..half_dim)
        .map(|i| {
            if i < lo.min(half_dim) {
                0.0
            } else if i >= hi.min(half_dim) {
                1.0
            } else {
                let range = hi.saturating_sub(lo).max(1) as f32;
                let t = i.saturating_sub(lo) as f32 / range;
                t * t * (3.0 - 2.0 * t)
            }
        })
        .collect()
}

// ── Frequency table queries ─────────────────────────────────────────

/// Extract the raw inverse frequencies from a table (for debugging).
///
/// Returns `half_dim` values representing the angle-per-position for each pair.
pub fn extract_inv_frequencies(cfg: &ExtendedRopeConfig) -> Vec<f32> {
    let base = compute_effective_base(cfg, cfg.max_seq_len);
    let half = cfg.half_dim();
    (0..half)
        .map(|i| {
            let exponent = -(2.0 * i as f32) / cfg.head_dim as f32;
            base.powf(exponent) * per_pair_scale(cfg, i)
        })
        .collect()
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ─────────────────────────────────────────────────────

    fn naive_rope_interleaved(x: &mut [f32], dim: usize, pos: usize, base: f32) {
        let half = dim / 2;
        for i in 0..half {
            let exp = -(2.0 * i as f32) / dim as f32;
            let theta = base.powf(exp);
            let angle = pos as f32 * theta;
            let (s, c) = angle.sin_cos();
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            x[2 * i] = x0 * c - x1 * s;
            x[2 * i + 1] = x0 * s + x1 * c;
        }
    }

    fn naive_rope_rotary_half(x: &mut [f32], dim: usize, pos: usize, base: f32) {
        let half = dim / 2;
        for i in 0..half {
            let exp = -(2.0 * i as f32) / dim as f32;
            let theta = base.powf(exp);
            let angle = pos as f32 * theta;
            let (s, c) = angle.sin_cos();
            let x0 = x[i];
            let x1 = x[i + half];
            x[i] = x0 * c - x1 * s;
            x[i + half] = x0 * s + x1 * c;
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    fn vec_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn vec_norm(a: &[f32]) -> f32 {
        a.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let d = vec_dot(a, b);
        let na = vec_norm(a);
        let nb = vec_norm(b);
        if na < 1e-12 || nb < 1e-12 {
            return 0.0;
        }
        d / (na * nb)
    }

    fn make_data(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect()
    }

    // ================================================================
    // §1  Config & construction
    // ================================================================

    #[test]
    fn config_defaults() {
        let cfg = ExtendedRopeConfig::new(64, 512);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.max_seq_len, 512);
        assert!((cfg.base_freq - 10_000.0).abs() < 1e-6);
        assert_eq!(cfg.scaling, ExtendedScaling::None);
        assert_eq!(cfg.layout, RotationLayout::Interleaved);
    }

    #[test]
    fn config_builder_chain() {
        let cfg = ExtendedRopeConfig::new(32, 128)
            .with_base_freq(500_000.0)
            .with_scaling(ExtendedScaling::NtkAware { alpha: 2.0 })
            .with_layout(RotationLayout::RotaryHalf);
        assert!((cfg.base_freq - 500_000.0).abs() < 1e-6);
        assert_eq!(cfg.scaling, ExtendedScaling::NtkAware { alpha: 2.0 });
        assert_eq!(cfg.layout, RotationLayout::RotaryHalf);
    }

    #[test]
    #[should_panic(expected = "head_dim must be even")]
    fn config_rejects_odd_dim() {
        ExtendedRopeConfig::new(7, 32);
    }

    #[test]
    #[should_panic(expected = "head_dim must be even")]
    fn config_rejects_zero_dim() {
        ExtendedRopeConfig::new(0, 32);
    }

    // ================================================================
    // §2  Frequency table
    // ================================================================

    #[test]
    fn freq_table_dimensions() {
        let cfg = ExtendedRopeConfig::new(16, 64);
        let ft = build_extended_freq_table(&cfg);
        assert_eq!(ft.half_dim, 8);
        assert_eq!(ft.max_seq_len, 64);
        assert_eq!(ft.cos.len(), 64 * 8);
        assert_eq!(ft.sin.len(), 64 * 8);
    }

    #[test]
    fn freq_table_pos_zero_is_identity() {
        let cfg = ExtendedRopeConfig::new(8, 4);
        let ft = build_extended_freq_table(&cfg);
        for i in 0..4 {
            assert!((ft.cos_val(0, i) - 1.0).abs() < 1e-6);
            assert!(ft.sin_val(0, i).abs() < 1e-6);
        }
    }

    #[test]
    fn freq_table_sin_cos_bounded() {
        let cfg = ExtendedRopeConfig::new(32, 128);
        let ft = build_extended_freq_table(&cfg);
        for v in &ft.cos {
            assert!(v.abs() <= 1.0 + 1e-6, "cos out of [-1,1]: {v}");
        }
        for v in &ft.sin {
            assert!(v.abs() <= 1.0 + 1e-6, "sin out of [-1,1]: {v}");
        }
    }

    #[test]
    fn freq_table_pythagorean_identity() {
        let cfg = ExtendedRopeConfig::new(16, 64);
        let ft = build_extended_freq_table(&cfg);
        for pos in 0..64 {
            for i in 0..8 {
                let c = ft.cos_val(pos, i);
                let s = ft.sin_val(pos, i);
                assert!((c * c + s * s - 1.0).abs() < 1e-5, "pos={pos} i={i}");
            }
        }
    }

    #[test]
    fn freq_table_monotonic_freq_decay() {
        let cfg = ExtendedRopeConfig::new(16, 8);
        let ft = build_extended_freq_table(&cfg);
        // At pos=1, higher pairs have smaller |sin| (slower rotation)
        let s0 = ft.sin_val(1, 0).abs();
        let s_last = ft.sin_val(1, 7).abs();
        assert!(s0 > s_last, "pair 0 should rotate faster: {s0} vs {s_last}");
    }

    #[test]
    fn freq_table_stride() {
        let cfg = ExtendedRopeConfig::new(32, 16);
        let ft = build_extended_freq_table(&cfg);
        assert_eq!(ft.stride(), 16);
    }

    #[test]
    fn freq_table_custom_base() {
        let ft1 = build_extended_freq_table(&ExtendedRopeConfig::new(8, 4));
        let ft2 =
            build_extended_freq_table(&ExtendedRopeConfig::new(8, 4).with_base_freq(500_000.0));
        assert!((ft1.cos_val(0, 0) - ft2.cos_val(0, 0)).abs() < 1e-6);
        // pair > 0 has base-dependent inv_freq → different sin values
        assert!((ft1.sin_val(1, 2) - ft2.sin_val(1, 2)).abs() > 1e-6);
    }

    #[test]
    fn freq_table_layout_stored() {
        let cfg = ExtendedRopeConfig::new(8, 4).with_layout(RotationLayout::RotaryHalf);
        let ft = build_extended_freq_table(&cfg);
        assert_eq!(ft.layout, RotationLayout::RotaryHalf);
    }

    // ================================================================
    // §3  Interleaved RoPE (scalar)
    // ================================================================

    #[test]
    fn interleaved_matches_naive() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);

        for pos in 0..8 {
            let mut a = make_data(dim);
            let mut b = a.clone();
            apply_rope_interleaved(&mut a, &ft, pos);
            naive_rope_interleaved(&mut b, dim, pos, 10_000.0);
            assert!(max_abs_diff(&a, &b) < 1e-5, "pos={pos}");
        }
    }

    #[test]
    fn interleaved_pos_zero_identity() {
        let dim = 8;
        let cfg = ExtendedRopeConfig::new(dim, 4);
        let ft = build_extended_freq_table(&cfg);
        let mut x = make_data(dim);
        let orig = x.clone();
        apply_rope_interleaved(&mut x, &ft, 0);
        assert!(max_abs_diff(&x, &orig) < 1e-6);
    }

    #[test]
    fn interleaved_preserves_norm() {
        let dim = 32;
        let cfg = ExtendedRopeConfig::new(dim, 64);
        let ft = build_extended_freq_table(&cfg);
        for pos in [0, 1, 10, 50] {
            let mut x = make_data(dim);
            let before = vec_norm(&x);
            apply_rope_interleaved(&mut x, &ft, pos);
            let after = vec_norm(&x);
            assert!((before - after).abs() < 1e-4, "pos={pos}");
        }
    }

    #[test]
    fn interleaved_different_positions_differ() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 64);
        let ft = build_extended_freq_table(&cfg);
        let mut a = make_data(dim);
        let mut b = make_data(dim);
        apply_rope_interleaved(&mut a, &ft, 5);
        apply_rope_interleaved(&mut b, &ft, 10);
        assert!(max_abs_diff(&a, &b) > 1e-3);
    }

    // ================================================================
    // §4  Rotary-half RoPE (scalar)
    // ================================================================

    #[test]
    fn rotary_half_matches_naive() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32).with_layout(RotationLayout::RotaryHalf);
        let ft = build_extended_freq_table(&cfg);

        for pos in 0..8 {
            let mut a = make_data(dim);
            let mut b = a.clone();
            apply_rope_rotary_half(&mut a, &ft, pos);
            naive_rope_rotary_half(&mut b, dim, pos, 10_000.0);
            assert!(max_abs_diff(&a, &b) < 1e-5, "pos={pos}");
        }
    }

    #[test]
    fn rotary_half_preserves_norm() {
        let dim = 32;
        let cfg = ExtendedRopeConfig::new(dim, 64).with_layout(RotationLayout::RotaryHalf);
        let ft = build_extended_freq_table(&cfg);
        for pos in [0, 1, 10, 50] {
            let mut x = make_data(dim);
            let before = vec_norm(&x);
            apply_rope_rotary_half(&mut x, &ft, pos);
            let after = vec_norm(&x);
            assert!((before - after).abs() < 1e-4, "pos={pos}");
        }
    }

    #[test]
    fn rotary_half_pos_zero_identity() {
        let dim = 8;
        let cfg = ExtendedRopeConfig::new(dim, 4).with_layout(RotationLayout::RotaryHalf);
        let ft = build_extended_freq_table(&cfg);
        let mut x = make_data(dim);
        let orig = x.clone();
        apply_rope_rotary_half(&mut x, &ft, 0);
        assert!(max_abs_diff(&x, &orig) < 1e-6);
    }

    #[test]
    fn rotary_half_differs_from_interleaved() {
        let dim = 16;
        let cfg_i = ExtendedRopeConfig::new(dim, 32);
        let cfg_r = ExtendedRopeConfig::new(dim, 32).with_layout(RotationLayout::RotaryHalf);
        let ft_i = build_extended_freq_table(&cfg_i);
        let ft_r = build_extended_freq_table(&cfg_r);
        let mut a = make_data(dim);
        let mut b = a.clone();
        apply_rope_interleaved(&mut a, &ft_i, 5);
        apply_rope_rotary_half(&mut b, &ft_r, 5);
        // Same frequencies but different pair mapping → different result
        assert!(max_abs_diff(&a, &b) > 1e-3);
    }

    // ================================================================
    // §5  Layout dispatch
    // ================================================================

    #[test]
    fn dispatch_selects_interleaved() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);
        let mut a = make_data(dim);
        let mut b = a.clone();
        apply_rope_single(&mut a, &ft, 3);
        apply_rope_interleaved(&mut b, &ft, 3);
        assert!(max_abs_diff(&a, &b) < 1e-7);
    }

    #[test]
    fn dispatch_selects_rotary_half() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32).with_layout(RotationLayout::RotaryHalf);
        let ft = build_extended_freq_table(&cfg);
        let mut a = make_data(dim);
        let mut b = a.clone();
        apply_rope_single(&mut a, &ft, 3);
        apply_rope_rotary_half(&mut b, &ft, 3);
        assert!(max_abs_diff(&a, &b) < 1e-7);
    }

    // ================================================================
    // §6  Inverse RoPE
    // ================================================================

    #[test]
    fn inverse_interleaved_roundtrip() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);
        for pos in [0, 1, 5, 20] {
            let orig = make_data(dim);
            let mut x = orig.clone();
            apply_rope_interleaved(&mut x, &ft, pos);
            inverse_rope_interleaved(&mut x, &ft, pos);
            assert!(max_abs_diff(&x, &orig) < 1e-5, "pos={pos}");
        }
    }

    #[test]
    fn inverse_rotary_half_roundtrip() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32).with_layout(RotationLayout::RotaryHalf);
        let ft = build_extended_freq_table(&cfg);
        for pos in [0, 1, 5, 20] {
            let orig = make_data(dim);
            let mut x = orig.clone();
            apply_rope_rotary_half(&mut x, &ft, pos);
            inverse_rope_rotary_half(&mut x, &ft, pos);
            assert!(max_abs_diff(&x, &orig) < 1e-5, "pos={pos}");
        }
    }

    #[test]
    fn inverse_dispatch_roundtrip() {
        for layout in [RotationLayout::Interleaved, RotationLayout::RotaryHalf] {
            let dim = 16;
            let cfg = ExtendedRopeConfig::new(dim, 32).with_layout(layout);
            let ft = build_extended_freq_table(&cfg);
            let orig = make_data(dim);
            let mut x = orig.clone();
            apply_rope_single(&mut x, &ft, 7);
            inverse_rope_single(&mut x, &ft, 7);
            assert!(max_abs_diff(&x, &orig) < 1e-5, "layout={layout:?}");
        }
    }

    // ================================================================
    // §7  AVX2 dispatch
    // ================================================================

    #[test]
    fn avx2_dispatch_matches_scalar_interleaved() {
        let dim = 32;
        let cfg = ExtendedRopeConfig::new(dim, 64);
        let ft = build_extended_freq_table(&cfg);
        for pos in [0, 1, 10, 50] {
            let mut a = make_data(dim);
            let mut b = a.clone();
            apply_rope_interleaved(&mut a, &ft, pos);
            apply_rope_dispatch(&mut b, &ft, pos);
            assert!(max_abs_diff(&a, &b) < 1e-5, "pos={pos}");
        }
    }

    #[test]
    fn avx2_dispatch_matches_scalar_rotary_half() {
        let dim = 32;
        let cfg = ExtendedRopeConfig::new(dim, 64).with_layout(RotationLayout::RotaryHalf);
        let ft = build_extended_freq_table(&cfg);
        for pos in [0, 1, 10, 50] {
            let mut a = make_data(dim);
            let mut b = a.clone();
            apply_rope_rotary_half(&mut a, &ft, pos);
            apply_rope_dispatch(&mut b, &ft, pos);
            assert!(max_abs_diff(&a, &b) < 1e-5, "pos={pos}");
        }
    }

    #[test]
    fn avx2_dispatch_tail_handling() {
        // dim=12 → half=6, not divisible by 4 for interleaved
        let dim = 12;
        let cfg = ExtendedRopeConfig::new(dim, 16);
        let ft = build_extended_freq_table(&cfg);
        let mut a = make_data(dim);
        let mut b = a.clone();
        apply_rope_interleaved(&mut a, &ft, 3);
        apply_rope_dispatch(&mut b, &ft, 3);
        assert!(max_abs_diff(&a, &b) < 1e-5);
    }

    #[test]
    fn avx2_dispatch_small_dim() {
        // dim=4 → half=2, too small for AVX2 interleaved (needs 4 pairs = 8)
        let dim = 4;
        let cfg = ExtendedRopeConfig::new(dim, 8);
        let ft = build_extended_freq_table(&cfg);
        let mut a = make_data(dim);
        let mut b = a.clone();
        apply_rope_interleaved(&mut a, &ft, 2);
        apply_rope_dispatch(&mut b, &ft, 2);
        assert!(max_abs_diff(&a, &b) < 1e-6);
    }

    // ================================================================
    // §8  NTK-aware scaling
    // ================================================================

    #[test]
    fn ntk_scaling_increases_base() {
        let base_cfg = ExtendedRopeConfig::new(16, 32);
        let ntk_cfg =
            ExtendedRopeConfig::new(16, 32).with_scaling(ExtendedScaling::NtkAware { alpha: 2.0 });
        let b1 = compute_effective_base(&base_cfg, 32);
        let b2 = compute_effective_base(&ntk_cfg, 32);
        assert!(b2 > b1, "NTK should increase base: {b2} > {b1}");
    }

    #[test]
    fn ntk_alpha_one_is_identity() {
        let cfg_none = ExtendedRopeConfig::new(16, 32);
        let cfg_ntk =
            ExtendedRopeConfig::new(16, 32).with_scaling(ExtendedScaling::NtkAware { alpha: 1.0 });
        let ft1 = build_extended_freq_table(&cfg_none);
        let ft2 = build_extended_freq_table(&cfg_ntk);
        assert!(max_abs_diff(&ft1.cos, &ft2.cos) < 1e-5);
    }

    #[test]
    fn ntk_preserves_norm() {
        let dim = 32;
        let cfg =
            ExtendedRopeConfig::new(dim, 64).with_scaling(ExtendedScaling::NtkAware { alpha: 4.0 });
        let ft = build_extended_freq_table(&cfg);
        let mut x = make_data(dim);
        let before = vec_norm(&x);
        apply_rope_dispatch(&mut x, &ft, 10);
        let after = vec_norm(&x);
        assert!((before - after).abs() < 1e-4);
    }

    #[test]
    fn ntk_scaled_base_formula() {
        let b = ntk_scaled_base(10_000.0, 2.0, 64);
        let expected = 10_000.0_f32 * 2.0_f32.powf(64.0 / 62.0);
        assert!((b - expected).abs() < 1.0);
    }

    #[test]
    fn ntk_larger_alpha_slower_rotation() {
        let dim = 16;
        let cfg_lo =
            ExtendedRopeConfig::new(dim, 32).with_scaling(ExtendedScaling::NtkAware { alpha: 2.0 });
        let cfg_hi =
            ExtendedRopeConfig::new(dim, 32).with_scaling(ExtendedScaling::NtkAware { alpha: 8.0 });
        let ft_lo = build_extended_freq_table(&cfg_lo);
        let ft_hi = build_extended_freq_table(&cfg_hi);
        // pair > 0: higher alpha → larger base → slower rotation → smaller |sin|
        // (pair 0 always has inv_freq=1.0 regardless of base)
        assert!(ft_hi.sin_val(1, 4).abs() < ft_lo.sin_val(1, 4).abs());
    }

    #[test]
    fn compute_ntk_alpha_roundtrip() {
        let alpha = compute_ntk_alpha(4096, 16384, 64);
        let base = ntk_scaled_base(10_000.0, alpha, 64);
        // The effective context ratio should match
        let expected_base = ntk_scaled_base(10_000.0, alpha, 64);
        assert!((base - expected_base).abs() < 1e-3);
        assert!(alpha > 1.0, "extending context should need alpha > 1");
    }

    // ================================================================
    // §9  Dynamic NTK
    // ================================================================

    #[test]
    fn dynamic_ntk_within_ctx_is_standard() {
        let cfg_none = ExtendedRopeConfig::new(16, 32);
        let cfg_dyn = ExtendedRopeConfig::new(16, 32)
            .with_scaling(ExtendedScaling::DynamicNtk { trained_ctx: 64 });
        // seq_len=32 < trained_ctx=64 → no change
        let b1 = compute_effective_base(&cfg_none, 32);
        let b2 = compute_effective_base(&cfg_dyn, 32);
        assert!((b1 - b2).abs() < 1e-6);
    }

    #[test]
    fn dynamic_ntk_beyond_ctx_scales() {
        let cfg = ExtendedRopeConfig::new(16, 128)
            .with_scaling(ExtendedScaling::DynamicNtk { trained_ctx: 32 });
        let b_in = compute_effective_base(&cfg, 32);
        let b_out = compute_effective_base(&cfg, 128);
        assert!(b_out > b_in, "beyond trained_ctx base should grow");
    }

    #[test]
    fn dynamic_ntk_rebuild() {
        let cfg = ExtendedRopeConfig::new(16, 64)
            .with_scaling(ExtendedScaling::DynamicNtk { trained_ctx: 16 });
        let ft1 = rebuild_table_for_dynamic_ntk(&cfg, 16);
        let ft2 = rebuild_table_for_dynamic_ntk(&cfg, 48);
        // Tables should differ because base changed
        assert!(max_abs_diff(&ft1.sin, &ft2.sin) > 1e-4);
    }

    #[test]
    fn dynamic_ntk_preserves_norm() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 64)
            .with_scaling(ExtendedScaling::DynamicNtk { trained_ctx: 16 });
        let ft = build_extended_freq_table_for_seq(&cfg, 48);
        let mut x = make_data(dim);
        let before = vec_norm(&x);
        apply_rope_dispatch(&mut x, &ft, 30);
        let after = vec_norm(&x);
        assert!((before - after).abs() < 1e-4);
    }

    // ================================================================
    // §10  YaRN scaling
    // ================================================================

    #[test]
    fn yarn_high_freq_unscaled() {
        let dim = 32;
        let cfg_none = ExtendedRopeConfig::new(dim, 64);
        let cfg_yarn = ExtendedRopeConfig::new(dim, 64).with_scaling(ExtendedScaling::YaRN {
            factor: 4.0,
            trained_ctx: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: 1.0,
        });
        let ft_none = build_extended_freq_table(&cfg_none);
        let ft_yarn = build_extended_freq_table(&cfg_yarn);
        // Pair 0 is highest frequency — should be unscaled
        assert!(
            (ft_none.sin_val(1, 0) - ft_yarn.sin_val(1, 0)).abs() < 1e-5,
            "high-freq pair should match"
        );
    }

    #[test]
    fn yarn_low_freq_scaled() {
        // Use small trained_ctx so some pairs fall in the low-frequency band
        let dim = 32;
        let cfg_none = ExtendedRopeConfig::new(dim, 64);
        let cfg_yarn = ExtendedRopeConfig::new(dim, 64).with_scaling(ExtendedScaling::YaRN {
            factor: 4.0,
            trained_ctx: 32,
            beta_fast: 4.0,
            beta_slow: 0.5,
            attn_factor: 1.0,
        });
        let ft_none = build_extended_freq_table(&cfg_none);
        let ft_yarn = build_extended_freq_table(&cfg_yarn);
        let last = dim / 2 - 1;
        // Last pair is lowest frequency — should differ from unscaled
        let diff = (ft_none.sin_val(1, last) - ft_yarn.sin_val(1, last)).abs();
        assert!(diff > 1e-8, "low-freq pair should be scaled: diff={diff}");
    }

    #[test]
    fn yarn_blend_weights_boundaries() {
        // Use small trained_ctx so lo < half_dim and hi covers some pairs
        let weights = yarn_blend_weights(16, 32, 4.0, 0.5);
        assert_eq!(weights.len(), 16);
        // First weight should be 0 (unscaled)
        assert!(weights[0] < 0.01, "first weight should be ~0: {}", weights[0]);
        // Last weight should be 1 (fully scaled)
        assert!(weights[15] > 0.99, "last weight should be ~1: {}", weights[15]);
    }

    #[test]
    fn yarn_blend_weights_monotonic() {
        let weights = yarn_blend_weights(16, 4096, 32.0, 1.0);
        for i in 1..weights.len() {
            assert!(weights[i] >= weights[i - 1] - 1e-6, "weights should be non-decreasing at {i}");
        }
    }

    #[test]
    fn yarn_attention_factor_identity() {
        let f = yarn_attention_factor(1.0, 4096, 64);
        assert!((f - 1.0).abs() < 1e-6, "factor=1 should give attn_factor=1");
    }

    #[test]
    fn yarn_attention_factor_grows_with_scale() {
        let f1 = yarn_attention_factor(2.0, 4096, 64);
        let f4 = yarn_attention_factor(4.0, 4096, 64);
        assert!(f4 > f1, "larger factor should give larger attn correction");
    }

    #[test]
    fn yarn_preserves_norm() {
        let dim = 32;
        let cfg = ExtendedRopeConfig::new(dim, 64).with_scaling(ExtendedScaling::YaRN {
            factor: 4.0,
            trained_ctx: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: 1.0,
        });
        let ft = build_extended_freq_table(&cfg);
        let mut x = make_data(dim);
        let before = vec_norm(&x);
        apply_rope_dispatch(&mut x, &ft, 10);
        let after = vec_norm(&x);
        assert!((before - after).abs() < 1e-4);
    }

    // ================================================================
    // §11  Linear scaling
    // ================================================================

    #[test]
    fn linear_scaling_slows_rotation() {
        let dim = 16;
        let cfg_none = ExtendedRopeConfig::new(dim, 32);
        let cfg_lin =
            ExtendedRopeConfig::new(dim, 32).with_scaling(ExtendedScaling::Linear { factor: 2.0 });
        let ft_none = build_extended_freq_table(&cfg_none);
        let ft_lin = build_extended_freq_table(&cfg_lin);
        // Linear factor=2 → position effectively halved → less rotation
        assert!(ft_lin.sin_val(1, 0).abs() < ft_none.sin_val(1, 0).abs());
    }

    #[test]
    fn linear_factor_one_is_identity() {
        let cfg_none = ExtendedRopeConfig::new(16, 32);
        let cfg_lin =
            ExtendedRopeConfig::new(16, 32).with_scaling(ExtendedScaling::Linear { factor: 1.0 });
        let ft1 = build_extended_freq_table(&cfg_none);
        let ft2 = build_extended_freq_table(&cfg_lin);
        assert!(max_abs_diff(&ft1.cos, &ft2.cos) < 1e-6);
    }

    #[test]
    fn linear_preserves_norm() {
        let dim = 16;
        let cfg =
            ExtendedRopeConfig::new(dim, 32).with_scaling(ExtendedScaling::Linear { factor: 4.0 });
        let ft = build_extended_freq_table(&cfg);
        let mut x = make_data(dim);
        let before = vec_norm(&x);
        apply_rope_dispatch(&mut x, &ft, 5);
        let after = vec_norm(&x);
        assert!((before - after).abs() < 1e-4);
    }

    // ================================================================
    // §12  Custom base frequency
    // ================================================================

    #[test]
    fn custom_base_overrides_config() {
        let cfg = ExtendedRopeConfig::new(16, 32)
            .with_base_freq(10_000.0)
            .with_scaling(ExtendedScaling::CustomBase { base: 500_000.0 });
        let eff = compute_effective_base(&cfg, 32);
        assert!((eff - 500_000.0).abs() < 1e-3);
    }

    #[test]
    fn custom_base_changes_frequencies() {
        let cfg1 = ExtendedRopeConfig::new(16, 32);
        let cfg2 = ExtendedRopeConfig::new(16, 32)
            .with_scaling(ExtendedScaling::CustomBase { base: 100.0 });
        let ft1 = build_extended_freq_table(&cfg1);
        let ft2 = build_extended_freq_table(&cfg2);
        assert!(max_abs_diff(&ft1.sin, &ft2.sin) > 0.01);
    }

    #[test]
    fn custom_base_preserves_norm() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32)
            .with_scaling(ExtendedScaling::CustomBase { base: 500_000.0 });
        let ft = build_extended_freq_table(&cfg);
        let mut x = make_data(dim);
        let before = vec_norm(&x);
        apply_rope_dispatch(&mut x, &ft, 5);
        let after = vec_norm(&x);
        assert!((before - after).abs() < 1e-4);
    }

    // ================================================================
    // §13  Batched RoPE
    // ================================================================

    #[test]
    fn batched_matches_per_head() {
        let dim = 16;
        let heads = 4;
        let seq = 3;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);
        let positions: Vec<usize> = (0..seq).collect();

        let total = seq * heads * dim;
        let mut batched = (0..total).map(|i| (i as f32 + 1.0) * 0.01).collect::<Vec<_>>();
        let mut manual = batched.clone();

        apply_rope_batched(&mut batched, &ft, &positions, heads, dim);

        for s in 0..seq {
            for h in 0..heads {
                let off = (s * heads + h) * dim;
                apply_rope_dispatch(&mut manual[off..off + dim], &ft, positions[s]);
            }
        }

        assert!(max_abs_diff(&batched, &manual) < 1e-5);
    }

    #[test]
    fn batched_single_head_single_pos() {
        let dim = 8;
        let cfg = ExtendedRopeConfig::new(dim, 16);
        let ft = build_extended_freq_table(&cfg);
        let mut x = make_data(dim);
        let mut y = x.clone();
        apply_rope_batched(&mut x, &ft, &[3], 1, dim);
        apply_rope_dispatch(&mut y, &ft, 3);
        assert!(max_abs_diff(&x, &y) < 1e-6);
    }

    #[test]
    fn batched_preserves_total_norm() {
        let dim = 16;
        let heads = 2;
        let seq = 4;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);
        let positions: Vec<usize> = (0..seq).collect();
        let mut data: Vec<f32> = (0..seq * heads * dim).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let before = vec_norm(&data);
        apply_rope_batched(&mut data, &ft, &positions, heads, dim);
        let after = vec_norm(&data);
        assert!((before - after).abs() < 0.01);
    }

    // ================================================================
    // §14  QK batched RoPE
    // ================================================================

    #[test]
    fn qk_batched_applies_to_both() {
        let dim = 16;
        let n_q = 4;
        let n_kv = 2;
        let seq = 2;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);
        let positions: Vec<usize> = (0..seq).collect();

        let mut q: Vec<f32> = (0..seq * n_q * dim).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let mut k: Vec<f32> = (0..seq * n_kv * dim).map(|i| (i as f32 + 1.0) * 0.02).collect();
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();

        apply_rope_qk_batched(&mut q, &mut k, &ft, &positions, n_q, n_kv, dim);

        apply_rope_batched(&mut q_ref, &ft, &positions, n_q, dim);
        apply_rope_batched(&mut k_ref, &ft, &positions, n_kv, dim);

        assert!(max_abs_diff(&q, &q_ref) < 1e-5);
        assert!(max_abs_diff(&k, &k_ref) < 1e-5);
    }

    #[test]
    fn qk_batched_query_key_differ() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);
        let mut q = make_data(dim);
        let mut k = q.clone();
        apply_rope_qk_batched(&mut q, &mut k, &ft, &[0], 1, 1, dim);
        // At pos 0, both should be identity
        assert!(max_abs_diff(&q, &k) < 1e-6);

        let mut q2 = make_data(dim);
        let mut k2 = q2.clone();
        apply_rope_qk_batched(&mut q2, &mut k2, &ft, &[5], 1, 1, dim);
        // At pos 5, both get same rotation (same data, same pos)
        assert!(max_abs_diff(&q2, &k2) < 1e-6);
    }

    // ================================================================
    // §15  Fused RoPE + attention score
    // ================================================================

    #[test]
    fn fused_score_matches_separate() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);

        let mut q = make_data(dim);
        let mut k: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) * 0.1).collect();

        let mut q2 = q.clone();
        let mut k2 = k.clone();

        let score = fused_rope_attention_score(&mut q, &mut k, &ft, 2, 5, dim);

        apply_rope_dispatch(&mut q2, &ft, 2);
        apply_rope_dispatch(&mut k2, &ft, 5);
        let expected = vec_dot(&q2, &k2) / (dim as f32).sqrt();

        assert!((score - expected).abs() < 1e-4, "score={score} expected={expected}");
    }

    #[test]
    fn fused_score_symmetric_positions() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);
        let mut q = make_data(dim);
        let mut k = q.clone();
        let score = fused_rope_attention_score(&mut q, &mut k, &ft, 3, 3, dim);
        // q==k same pos → score = norm²/sqrt(dim)
        assert!(score > 0.0, "self-attention should be positive");
    }

    #[test]
    fn fused_scores_batched_length() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);
        let mut q = make_data(dim);
        let mut keys: Vec<f32> = (0..3 * dim).map(|i| (i as f32 + 0.5) * 0.1).collect();
        let scores =
            fused_rope_attention_scores_batched(&mut q, &mut keys, &ft, 2, &[0, 5, 10], dim);
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn fused_scores_batched_correctness() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);
        let q_orig: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let k_orig: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) * 0.1).collect();

        // Single fused
        let mut q = q_orig.clone();
        let mut k = k_orig.clone();
        let score = fused_rope_attention_score(&mut q, &mut k, &ft, 2, 5, dim);

        // Batched fused (single key)
        let mut q2 = q_orig.clone();
        let mut keys = k_orig.clone();
        let scores = fused_rope_attention_scores_batched(&mut q2, &mut keys, &ft, 2, &[5], dim);

        assert!((score - scores[0]).abs() < 1e-4);
    }

    // ================================================================
    // §16  Inverse-frequency extraction
    // ================================================================

    #[test]
    fn extract_inv_freq_length() {
        let cfg = ExtendedRopeConfig::new(32, 64);
        let freqs = extract_inv_frequencies(&cfg);
        assert_eq!(freqs.len(), 16);
    }

    #[test]
    fn extract_inv_freq_descending() {
        let cfg = ExtendedRopeConfig::new(32, 64);
        let freqs = extract_inv_frequencies(&cfg);
        for i in 1..freqs.len() {
            assert!(freqs[i] <= freqs[i - 1], "inv_freq should be descending");
        }
    }

    #[test]
    fn extract_inv_freq_ntk_differs() {
        let cfg1 = ExtendedRopeConfig::new(16, 32);
        let cfg2 =
            ExtendedRopeConfig::new(16, 32).with_scaling(ExtendedScaling::NtkAware { alpha: 4.0 });
        let f1 = extract_inv_frequencies(&cfg1);
        let f2 = extract_inv_frequencies(&cfg2);
        assert!(max_abs_diff(&f1, &f2) > 0.01);
    }

    // ================================================================
    // §17  Large dimensions
    // ================================================================

    #[test]
    fn large_dim_128_interleaved() {
        let dim = 128;
        let cfg = ExtendedRopeConfig::new(dim, 16);
        let ft = build_extended_freq_table(&cfg);
        let mut x = make_data(dim);
        let orig = x.clone();
        apply_rope_dispatch(&mut x, &ft, 5);
        // Should change values
        assert!(max_abs_diff(&x, &orig) > 0.01);
        // Norm preserved
        assert!((vec_norm(&x) - vec_norm(&orig)).abs() < 1e-3);
    }

    #[test]
    fn large_dim_128_rotary_half() {
        let dim = 128;
        let cfg = ExtendedRopeConfig::new(dim, 16).with_layout(RotationLayout::RotaryHalf);
        let ft = build_extended_freq_table(&cfg);
        let mut x = make_data(dim);
        let orig = x.clone();
        apply_rope_dispatch(&mut x, &ft, 5);
        assert!(max_abs_diff(&x, &orig) > 0.01);
        assert!((vec_norm(&x) - vec_norm(&orig)).abs() < 1e-3);
    }

    #[test]
    fn large_dim_256_roundtrip() {
        let dim = 256;
        let cfg = ExtendedRopeConfig::new(dim, 8);
        let ft = build_extended_freq_table(&cfg);
        let orig = make_data(dim);
        let mut x = orig.clone();
        apply_rope_dispatch(&mut x, &ft, 3);
        inverse_rope_single(&mut x, &ft, 3);
        assert!(max_abs_diff(&x, &orig) < 1e-4);
    }

    // ================================================================
    // §18  Mixed scaling with layout
    // ================================================================

    #[test]
    fn ntk_with_rotary_half() {
        let dim = 32;
        let cfg = ExtendedRopeConfig::new(dim, 32)
            .with_scaling(ExtendedScaling::NtkAware { alpha: 4.0 })
            .with_layout(RotationLayout::RotaryHalf);
        let ft = build_extended_freq_table(&cfg);
        let mut x = make_data(dim);
        let before = vec_norm(&x);
        apply_rope_dispatch(&mut x, &ft, 10);
        let after = vec_norm(&x);
        assert!((before - after).abs() < 1e-4);
    }

    #[test]
    fn dynamic_ntk_with_rotary_half() {
        let dim = 32;
        let cfg = ExtendedRopeConfig::new(dim, 64)
            .with_scaling(ExtendedScaling::DynamicNtk { trained_ctx: 16 })
            .with_layout(RotationLayout::RotaryHalf);
        let ft = build_extended_freq_table_for_seq(&cfg, 48);
        let orig = make_data(dim);
        let mut x = orig.clone();
        apply_rope_dispatch(&mut x, &ft, 10);
        inverse_rope_single(&mut x, &ft, 10);
        assert!(max_abs_diff(&x, &orig) < 1e-4);
    }

    #[test]
    fn yarn_with_rotary_half() {
        let dim = 32;
        let cfg = ExtendedRopeConfig::new(dim, 64)
            .with_scaling(ExtendedScaling::YaRN {
                factor: 4.0,
                trained_ctx: 4096,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attn_factor: 1.0,
            })
            .with_layout(RotationLayout::RotaryHalf);
        let ft = build_extended_freq_table(&cfg);
        let mut x = make_data(dim);
        let before = vec_norm(&x);
        apply_rope_dispatch(&mut x, &ft, 10);
        let after = vec_norm(&x);
        assert!((before - after).abs() < 1e-4);
    }

    // ================================================================
    // §19  Relative position encoding property
    // ================================================================

    #[test]
    fn relative_position_dot_product() {
        // Key RoPE property: q(pos_a) · k(pos_b) depends only on (pos_a - pos_b)
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 64);
        let ft = build_extended_freq_table(&cfg);

        let base_q = make_data(dim);
        let base_k: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) * 0.1).collect();

        // (pos_a=5, pos_b=3) → relative = 2
        let mut q1 = base_q.clone();
        let mut k1 = base_k.clone();
        apply_rope_dispatch(&mut q1, &ft, 5);
        apply_rope_dispatch(&mut k1, &ft, 3);
        let dot1 = vec_dot(&q1, &k1);

        // (pos_a=10, pos_b=8) → relative = 2
        let mut q2 = base_q.clone();
        let mut k2 = base_k.clone();
        apply_rope_dispatch(&mut q2, &ft, 10);
        apply_rope_dispatch(&mut k2, &ft, 8);
        let dot2 = vec_dot(&q2, &k2);

        assert!((dot1 - dot2).abs() < 1e-3, "dot1={dot1} dot2={dot2}");
    }

    #[test]
    fn relative_pos_different_offsets() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 128);
        let ft = build_extended_freq_table(&cfg);
        let base_q = make_data(dim);
        let base_k: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) * 0.1).collect();

        // relative distance = 7 at different absolute positions
        let offsets = [(0, 7), (20, 27), (50, 57)];
        let dots: Vec<f32> = offsets
            .iter()
            .map(|&(qa, ka)| {
                let mut q = base_q.clone();
                let mut k = base_k.clone();
                apply_rope_dispatch(&mut q, &ft, qa);
                apply_rope_dispatch(&mut k, &ft, ka);
                vec_dot(&q, &k)
            })
            .collect();

        for i in 1..dots.len() {
            assert!(
                (dots[0] - dots[i]).abs() < 1e-2,
                "relative pos property: {} vs {}",
                dots[0],
                dots[i]
            );
        }
    }

    // ================================================================
    // §20  Edge cases & misc
    // ================================================================

    #[test]
    fn min_dim_two() {
        let dim = 2;
        let cfg = ExtendedRopeConfig::new(dim, 8);
        let ft = build_extended_freq_table(&cfg);
        let mut x = vec![1.0, 0.0];
        apply_rope_dispatch(&mut x, &ft, 1);
        assert!((vec_norm(&x) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn multiple_apply_accumulates() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);
        let mut x = make_data(dim);
        apply_rope_dispatch(&mut x, &ft, 3);
        apply_rope_dispatch(&mut x, &ft, 3);
        // Applying pos=3 twice = rotating by 2*angle(3) = angle(6)
        let mut y = make_data(dim);
        apply_rope_dispatch(&mut y, &ft, 6);
        // Standard RoPE is linear in position, so double-apply equals double-pos
        assert!(max_abs_diff(&x, &y) < 1e-4);
    }

    #[test]
    fn all_zeros_stays_zero() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);
        let mut x = vec![0.0; dim];
        apply_rope_dispatch(&mut x, &ft, 5);
        assert!(x.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn cosine_similarity_nearby_positions() {
        let dim = 32;
        let cfg = ExtendedRopeConfig::new(dim, 128);
        let ft = build_extended_freq_table(&cfg);
        let base = make_data(dim);

        let mut near = base.clone();
        let mut far = base.clone();
        apply_rope_dispatch(&mut near, &ft, 1);
        apply_rope_dispatch(&mut far, &ft, 50);

        let mut origin = base.clone();
        apply_rope_dispatch(&mut origin, &ft, 0);

        let sim_near = cosine_sim(&origin, &near);
        let sim_far = cosine_sim(&origin, &far);
        assert!(sim_near > sim_far, "nearby positions should be more similar");
    }

    #[test]
    fn scaling_enum_default() {
        let s = ExtendedScaling::default();
        assert_eq!(s, ExtendedScaling::None);
    }

    #[test]
    fn layout_enum_default() {
        let l = RotationLayout::default();
        assert_eq!(l, RotationLayout::Interleaved);
    }

    #[test]
    fn freq_table_for_seq_differs_when_dynamic() {
        let cfg = ExtendedRopeConfig::new(16, 32)
            .with_scaling(ExtendedScaling::DynamicNtk { trained_ctx: 8 });
        let ft1 = build_extended_freq_table_for_seq(&cfg, 8);
        let ft2 = build_extended_freq_table_for_seq(&cfg, 24);
        assert!(max_abs_diff(&ft1.sin, &ft2.sin) > 1e-4);
    }

    #[test]
    fn batched_rope_empty_positions() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);
        let mut data: Vec<f32> = vec![];
        apply_rope_batched(&mut data, &ft, &[], 4, dim);
        assert!(data.is_empty());
    }

    #[test]
    fn fused_score_zero_query() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 32);
        let ft = build_extended_freq_table(&cfg);
        let mut q = vec![0.0; dim];
        let mut k = make_data(dim);
        let score = fused_rope_attention_score(&mut q, &mut k, &ft, 0, 0, dim);
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn ntk_alpha_for_double_ctx() {
        let alpha = compute_ntk_alpha(4096, 8192, 128);
        assert!(alpha > 1.0 && alpha < 3.0, "alpha={alpha}");
    }

    #[test]
    fn yarn_blend_all_unscaled_large_beta_fast() {
        // Very large beta_fast → all pairs in high-frequency band
        let weights = yarn_blend_weights(8, 4096, 10000.0, 0.001);
        // With huge beta_fast, lo becomes very large → all pairs are "high frequency"
        assert!(weights.iter().all(|w| *w < 0.01 || *w > 0.99));
    }

    #[test]
    fn interleaved_large_position() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 1024);
        let ft = build_extended_freq_table(&cfg);
        let mut x = make_data(dim);
        apply_rope_dispatch(&mut x, &ft, 1000);
        assert!((vec_norm(&x) - vec_norm(&make_data(dim))).abs() < 1e-3);
    }

    #[test]
    fn rotary_half_large_position() {
        let dim = 16;
        let cfg = ExtendedRopeConfig::new(dim, 1024).with_layout(RotationLayout::RotaryHalf);
        let ft = build_extended_freq_table(&cfg);
        let mut x = make_data(dim);
        apply_rope_dispatch(&mut x, &ft, 1000);
        assert!((vec_norm(&x) - vec_norm(&make_data(dim))).abs() < 1e-3);
    }

    #[test]
    fn batched_multi_scaling_norm() {
        let dim = 32;
        let heads = 4;
        let seq = 8;
        let cfg =
            ExtendedRopeConfig::new(dim, 64).with_scaling(ExtendedScaling::NtkAware { alpha: 2.0 });
        let ft = build_extended_freq_table(&cfg);
        let positions: Vec<usize> = (0..seq).collect();
        let mut data: Vec<f32> = (0..seq * heads * dim).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let before = vec_norm(&data);
        apply_rope_batched(&mut data, &ft, &positions, heads, dim);
        let after = vec_norm(&data);
        assert!((before - after).abs() < 0.1);
    }
}
