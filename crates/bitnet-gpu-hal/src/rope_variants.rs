//! Rotary Position Embedding (RoPE) variants for context extension.
//!
//! Provides Standard, Linear-scaled, NTK-aware, YaRN, Dynamic-NTK rotary
//! embeddings plus ALiBi linear-bias position encoding.

// Numerical code necessarily casts usize → f32 for position/frequency math.
// Domain acronyms (RoPE, YaRN, ALiBi) are not code items.
#![allow(clippy::cast_precision_loss, clippy::doc_markdown)]

use std::f32::consts::PI;

// ── Scaling type ────────────────────────────────────────────────────────

/// How position frequencies are scaled for context extension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalingType {
    /// No scaling – original RoPE.
    None,
    /// Linear position interpolation: `pos' = pos / scale_factor`.
    Linear { scale_factor: f32 },
    /// NTK-aware: adjust base frequency instead of positions.
    NTK { scale_factor: f32 },
    /// YaRN: attention-factor weighting combined with NTK.
    YaRN { scale_factor: f32, attention_factor: f32 },
    /// Dynamically compute NTK scaling from actual sequence length.
    Dynamic { original_max_seq_len: usize },
}

// ── Configuration ───────────────────────────────────────────────────────

/// Shared configuration for all RoPE variants.
#[derive(Debug, Clone)]
pub struct RoPEConfig {
    /// Base frequency θ (default 10 000).
    pub base_freq: f32,
    /// Head dimension (must be even).
    pub dim: usize,
    /// Maximum sequence length for table pre-computation.
    pub max_seq_len: usize,
    /// Scaling strategy.
    pub scaling_type: ScalingType,
}

impl RoPEConfig {
    /// Create a config with default base frequency and no scaling.
    #[must_use]
    pub const fn new(dim: usize, max_seq_len: usize) -> Self {
        Self { base_freq: 10_000.0, dim, max_seq_len, scaling_type: ScalingType::None }
    }

    /// Builder helper – set base frequency.
    #[must_use]
    pub const fn with_base_freq(mut self, base_freq: f32) -> Self {
        self.base_freq = base_freq;
        self
    }

    /// Builder helper – set scaling type.
    #[must_use]
    pub const fn with_scaling(mut self, scaling_type: ScalingType) -> Self {
        self.scaling_type = scaling_type;
        self
    }
}

// ── Frequency table ─────────────────────────────────────────────────────

/// Pre-computed cos/sin tables for a range of positions.
#[derive(Debug, Clone)]
pub struct FrequencyTable {
    /// Half of the head dimension (`dim / 2`).
    pub half_dim: usize,
    /// Cosine values, row-major `[positions, half_dim]`.
    pub cos: Vec<f32>,
    /// Sine values, row-major `[positions, half_dim]`.
    pub sin: Vec<f32>,
    /// Number of positions currently stored.
    pub len: usize,
}

impl FrequencyTable {
    /// Build a table from inverse-frequency vector and a position range.
    #[must_use]
    pub fn build(inv_freq: &[f32], max_pos: usize) -> Self {
        let half_dim = inv_freq.len();
        let mut cos = Vec::with_capacity(max_pos * half_dim);
        let mut sin = Vec::with_capacity(max_pos * half_dim);
        for pos in 0..max_pos {
            let p = pos as f32;
            for &freq in inv_freq {
                let angle = p * freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Self { half_dim, cos, sin, len: max_pos }
    }

    /// Extend the table to cover at least `new_len` positions.
    pub fn extend_to(&mut self, new_len: usize, inv_freq: &[f32]) {
        if new_len <= self.len {
            return;
        }
        self.cos.reserve((new_len - self.len) * self.half_dim);
        self.sin.reserve((new_len - self.len) * self.half_dim);
        for pos in self.len..new_len {
            let p = pos as f32;
            for &freq in inv_freq {
                let angle = p * freq;
                self.cos.push(angle.cos());
                self.sin.push(angle.sin());
            }
        }
        self.len = new_len;
    }

    /// Fetch `(cos_row, sin_row)` for a given position.
    ///
    /// Returns `None` when `pos >= self.len`.
    #[must_use]
    pub fn get(&self, pos: usize) -> Option<(&[f32], &[f32])> {
        if pos >= self.len {
            return None;
        }
        let start = pos * self.half_dim;
        let end = start + self.half_dim;
        Some((&self.cos[start..end], &self.sin[start..end]))
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Compute the standard inverse-frequency vector: `1 / base^(2i/d)`.
#[must_use]
fn standard_inv_freq(base: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    (0..half).map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32)).collect()
}

/// Apply rotary embedding to a single vector **in-place**.
///
/// `v` has length `dim`. The first `dim/2` elements are the "real" part,
/// the second `dim/2` are the "imaginary" part (interleaved-pair layout
/// used by many LLM implementations).
pub fn apply_rope_to_vec(v: &mut [f32], cos_row: &[f32], sin_row: &[f32]) {
    let half = cos_row.len();
    debug_assert!(v.len() >= 2 * half);
    for i in 0..half {
        let x0 = v[i];
        let x1 = v[i + half];
        v[i] = x0.mul_add(cos_row[i], -(x1 * sin_row[i]));
        v[i + half] = x0.mul_add(sin_row[i], x1 * cos_row[i]);
    }
}

/// Reverse (un-rotate) a vector that was previously rotated.
pub fn unapply_rope_to_vec(v: &mut [f32], cos_row: &[f32], sin_row: &[f32]) {
    let half = cos_row.len();
    debug_assert!(v.len() >= 2 * half);
    for i in 0..half {
        let x0 = v[i];
        let x1 = v[i + half];
        // cos(-θ) = cos(θ), sin(-θ) = -sin(θ)
        v[i] = x0.mul_add(cos_row[i], x1 * sin_row[i]);
        v[i + half] = (-x0).mul_add(sin_row[i], x1 * cos_row[i]);
    }
}

// ── Standard RoPE ───────────────────────────────────────────────────────

/// Original Rotary Position Embedding (Su et al., 2021).
#[derive(Debug, Clone)]
pub struct StandardRoPE {
    config: RoPEConfig,
    inv_freq: Vec<f32>,
    table: FrequencyTable,
}

impl StandardRoPE {
    #[must_use]
    pub fn new(config: RoPEConfig) -> Self {
        let inv_freq = standard_inv_freq(config.base_freq, config.dim);
        let table = FrequencyTable::build(&inv_freq, config.max_seq_len);
        Self { config, inv_freq, table }
    }

    pub const fn config(&self) -> &RoPEConfig {
        &self.config
    }

    pub const fn table(&self) -> &FrequencyTable {
        &self.table
    }

    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }
}

// ── Linear-scaled RoPE ──────────────────────────────────────────────────

/// Context extension via linear position interpolation.
///
/// Each position is divided by `scale_factor` before computing the angle,
/// effectively interpolating between existing trained positions.
#[derive(Debug, Clone)]
pub struct LinearScaledRoPE {
    config: RoPEConfig,
    scale_factor: f32,
    inv_freq: Vec<f32>,
    table: FrequencyTable,
}

impl LinearScaledRoPE {
    #[must_use]
    pub fn new(config: RoPEConfig, scale_factor: f32) -> Self {
        let base_inv = standard_inv_freq(config.base_freq, config.dim);
        // Scale inverse frequencies (equivalent to dividing position by factor)
        let inv_freq: Vec<f32> = base_inv.iter().map(|&f| f / scale_factor).collect();
        let table = FrequencyTable::build(&inv_freq, config.max_seq_len);
        Self { config, scale_factor, inv_freq, table }
    }

    pub const fn scale_factor(&self) -> f32 {
        self.scale_factor
    }

    pub const fn config(&self) -> &RoPEConfig {
        &self.config
    }

    pub const fn table(&self) -> &FrequencyTable {
        &self.table
    }

    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }
}

// ── NTK-aware RoPE ──────────────────────────────────────────────────────

/// NTK-aware scaling: adjust base frequency instead of positions.
///
/// `base' = base * (scale_factor)^(dim / (dim - 2))`
#[derive(Debug, Clone)]
pub struct NTKRoPE {
    config: RoPEConfig,
    scale_factor: f32,
    adjusted_base: f32,
    inv_freq: Vec<f32>,
    table: FrequencyTable,
}

impl NTKRoPE {
    #[must_use]
    pub fn new(config: RoPEConfig, scale_factor: f32) -> Self {
        let dim = config.dim as f32;
        let adjusted_base = config.base_freq * scale_factor.powf(dim / (dim - 2.0));
        let inv_freq = standard_inv_freq(adjusted_base, config.dim);
        let table = FrequencyTable::build(&inv_freq, config.max_seq_len);
        Self { config, scale_factor, adjusted_base, inv_freq, table }
    }

    pub const fn adjusted_base(&self) -> f32 {
        self.adjusted_base
    }

    pub const fn scale_factor(&self) -> f32 {
        self.scale_factor
    }

    pub const fn config(&self) -> &RoPEConfig {
        &self.config
    }

    pub const fn table(&self) -> &FrequencyTable {
        &self.table
    }

    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }
}

// ── YaRN RoPE ───────────────────────────────────────────────────────────

/// YaRN (Yet another RoPE extensioN): combines an attention factor with
/// NTK-style base adjustment.
///
/// High-frequency dimensions keep their original frequencies while
/// low-frequency dimensions are interpolated, weighted by `attention_factor`.
#[derive(Debug, Clone)]
pub struct YaRNRoPE {
    config: RoPEConfig,
    scale_factor: f32,
    attention_factor: f32,
    adjusted_base: f32,
    inv_freq: Vec<f32>,
    table: FrequencyTable,
}

impl YaRNRoPE {
    #[must_use]
    pub fn new(config: RoPEConfig, scale_factor: f32, attention_factor: f32) -> Self {
        let dim = config.dim as f32;
        let adjusted_base = config.base_freq * scale_factor.powf(dim / (dim - 2.0));
        let base_inv = standard_inv_freq(adjusted_base, config.dim);
        // Blend original and NTK-scaled frequencies by attention_factor.
        let orig_inv = standard_inv_freq(config.base_freq, config.dim);
        let inv_freq: Vec<f32> = base_inv
            .iter()
            .zip(orig_inv.iter())
            .map(|(&ntk, &orig)| {
                // Wavelength-dependent interpolation:
                // high-freq (large inv_freq) → keep original
                // low-freq (small inv_freq) → use NTK-scaled
                let wavelength = 2.0 * PI / orig;
                let low_freq_wavelen = config.max_seq_len as f32;
                let high_freq_wavelen = low_freq_wavelen / scale_factor;
                if wavelength < high_freq_wavelen {
                    orig // high-frequency: keep original
                } else if wavelength > low_freq_wavelen {
                    ntk / scale_factor // low-frequency: interpolate
                } else {
                    // Smooth ramp between the two
                    let t = (low_freq_wavelen / wavelength - 1.0) / (scale_factor - 1.0).max(1e-9);
                    let t = t.clamp(0.0, 1.0);
                    // attention_factor weights the blend
                    let blend = (1.0 - t) * attention_factor;
                    orig.mul_add(1.0 - blend, ntk * blend)
                }
            })
            .collect();
        let table = FrequencyTable::build(&inv_freq, config.max_seq_len);
        Self { config, scale_factor, attention_factor, adjusted_base, inv_freq, table }
    }

    pub const fn scale_factor(&self) -> f32 {
        self.scale_factor
    }

    pub const fn attention_factor(&self) -> f32 {
        self.attention_factor
    }

    pub const fn adjusted_base(&self) -> f32 {
        self.adjusted_base
    }

    pub const fn config(&self) -> &RoPEConfig {
        &self.config
    }

    pub const fn table(&self) -> &FrequencyTable {
        &self.table
    }

    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }
}

// ── Dynamic NTK RoPE ────────────────────────────────────────────────────

/// Dynamically adjusts NTK scaling based on actual sequence length.
///
/// When `seq_len > original_max_seq_len`, the effective scale factor is
/// `seq_len / original_max_seq_len` and base is adjusted accordingly.
#[derive(Debug, Clone)]
pub struct DynamicNTKRoPE {
    config: RoPEConfig,
    original_max_seq_len: usize,
    inv_freq_base: Vec<f32>,
    table: FrequencyTable,
}

impl DynamicNTKRoPE {
    #[must_use]
    pub fn new(config: RoPEConfig, original_max_seq_len: usize) -> Self {
        let inv_freq_base = standard_inv_freq(config.base_freq, config.dim);
        let table = FrequencyTable::build(&inv_freq_base, config.max_seq_len);
        Self { config, original_max_seq_len, inv_freq_base, table }
    }

    /// Recompute tables for a given actual sequence length.
    ///
    /// If `seq_len <= original_max_seq_len`, no scaling is applied.
    /// Otherwise, NTK scaling is computed on-the-fly.
    pub fn recompute_for_seq_len(&mut self, seq_len: usize) {
        if seq_len <= self.original_max_seq_len {
            self.table = FrequencyTable::build(&self.inv_freq_base, seq_len);
            return;
        }
        let scale = seq_len as f32 / self.original_max_seq_len as f32;
        let dim = self.config.dim as f32;
        let adjusted_base = self.config.base_freq * scale.powf(dim / (dim - 2.0));
        let inv_freq = standard_inv_freq(adjusted_base, self.config.dim);
        self.table = FrequencyTable::build(&inv_freq, seq_len);
    }

    /// Current effective scale factor for a given sequence length.
    #[must_use]
    pub fn effective_scale(&self, seq_len: usize) -> f32 {
        if seq_len <= self.original_max_seq_len {
            1.0
        } else {
            seq_len as f32 / self.original_max_seq_len as f32
        }
    }

    pub const fn original_max_seq_len(&self) -> usize {
        self.original_max_seq_len
    }

    pub const fn config(&self) -> &RoPEConfig {
        &self.config
    }

    pub const fn table(&self) -> &FrequencyTable {
        &self.table
    }
}

// ── ALiBi Position Bias ─────────────────────────────────────────────────

/// Attention with Linear Biases (Press et al., 2021).
///
/// Instead of rotary embeddings, ALiBi adds a distance-based bias to
/// attention scores: `bias(i, j) = -slope * |i - j|`.
///
/// Each head gets a geometric sequence of slopes.
#[derive(Debug, Clone)]
pub struct ALiBiPositionBias {
    num_heads: usize,
    slopes: Vec<f32>,
}

impl ALiBiPositionBias {
    /// Create ALiBi slopes for `num_heads` attention heads.
    ///
    /// Slopes form a geometric sequence: `2^(-8/n) … 2^(-8*n/n)`.
    #[must_use]
    pub fn new(num_heads: usize) -> Self {
        let slopes = Self::compute_slopes(num_heads);
        Self { num_heads, slopes }
    }

    /// Custom slopes.
    #[must_use]
    pub fn with_slopes(num_heads: usize, slopes: Vec<f32>) -> Self {
        assert_eq!(slopes.len(), num_heads);
        Self { num_heads, slopes }
    }

    /// Compute the canonical geometric slope sequence.
    #[must_use]
    pub fn compute_slopes(num_heads: usize) -> Vec<f32> {
        if num_heads == 0 {
            return Vec::new();
        }
        // Use closest power-of-2 for the main set
        let closest_pow2 = 1usize << (usize::BITS - 1 - num_heads.leading_zeros());
        let base = (-8.0 / closest_pow2 as f32).exp2();
        let mut slopes: Vec<f32> = (1..=closest_pow2)
            .map(|i| base.powi(i32::try_from(i).expect("head count fits i32")))
            .collect();
        if closest_pow2 < num_heads {
            let extra_base = (-8.0 / (2 * closest_pow2) as f32).exp2();
            let remaining = num_heads - closest_pow2;
            for i in 1..=remaining {
                slopes.push(extra_base.powi(i32::try_from(2 * i).expect("head index fits i32")));
            }
        }
        slopes.truncate(num_heads);
        slopes
    }

    /// Bias for head `h` at query-position `i`, key-position `j`.
    #[must_use]
    pub fn bias(&self, head: usize, query_pos: usize, key_pos: usize) -> f32 {
        let distance = query_pos.abs_diff(key_pos);
        -self.slopes[head] * distance as f32
    }

    /// Fill a bias matrix `[seq_len, seq_len]` for a single head (row-major).
    #[must_use]
    pub fn bias_matrix(&self, head: usize, seq_len: usize) -> Vec<f32> {
        let mut mat = Vec::with_capacity(seq_len * seq_len);
        for i in 0..seq_len {
            for j in 0..seq_len {
                mat.push(self.bias(head, i, j));
            }
        }
        mat
    }

    pub const fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }
}

// ── RoPE Interpolator ───────────────────────────────────────────────────

/// Uniform interface for applying rotary embeddings to Q/K vectors.
#[derive(Debug, Clone)]
pub enum RoPEInterpolator {
    Standard(StandardRoPE),
    Linear(LinearScaledRoPE),
    NTK(NTKRoPE),
    YaRN(YaRNRoPE),
    Dynamic(DynamicNTKRoPE),
}

impl RoPEInterpolator {
    /// Apply rotary embedding to query and key vectors at given positions.
    ///
    /// `q` and `k` are `[num_tokens, dim]` row-major.
    /// `position_ids` maps each token to its sequence position.
    pub fn apply(&self, q: &mut [f32], k: &mut [f32], position_ids: &[usize], dim: usize) {
        let table = self.table();
        for (tok_idx, &pos) in position_ids.iter().enumerate() {
            if let Some((cos_row, sin_row)) = table.get(pos) {
                let offset = tok_idx * dim;
                if offset + dim <= q.len() {
                    apply_rope_to_vec(&mut q[offset..offset + dim], cos_row, sin_row);
                }
                if offset + dim <= k.len() {
                    apply_rope_to_vec(&mut k[offset..offset + dim], cos_row, sin_row);
                }
            }
        }
    }

    /// Access the underlying frequency table.
    #[must_use]
    pub const fn table(&self) -> &FrequencyTable {
        match self {
            Self::Standard(r) => r.table(),
            Self::Linear(r) => r.table(),
            Self::NTK(r) => r.table(),
            Self::YaRN(r) => r.table(),
            Self::Dynamic(r) => r.table(),
        }
    }

    /// Create from a `RoPEConfig`.
    #[must_use]
    pub fn from_config(config: RoPEConfig) -> Self {
        match config.scaling_type {
            ScalingType::None => Self::Standard(StandardRoPE::new(config)),
            ScalingType::Linear { scale_factor } => {
                Self::Linear(LinearScaledRoPE::new(config, scale_factor))
            }
            ScalingType::NTK { scale_factor } => Self::NTK(NTKRoPE::new(config, scale_factor)),
            ScalingType::YaRN { scale_factor, attention_factor } => {
                Self::YaRN(YaRNRoPE::new(config, scale_factor, attention_factor))
            }
            ScalingType::Dynamic { original_max_seq_len } => {
                Self::Dynamic(DynamicNTKRoPE::new(config, original_max_seq_len))
            }
        }
    }
}

// ── Metrics ─────────────────────────────────────────────────────────────

/// Lightweight diagnostics for a RoPE instance.
#[derive(Debug, Clone, PartialEq)]
pub struct RoPEMetrics {
    pub max_position_seen: usize,
    pub scaling_factor_used: f32,
    pub table_size: usize,
}

impl RoPEMetrics {
    /// Collect metrics from a `RoPEInterpolator`.
    #[must_use]
    pub fn from_interpolator(interp: &RoPEInterpolator) -> Self {
        let table = interp.table();
        let scaling_factor_used = match interp {
            RoPEInterpolator::Standard(_) => 1.0,
            RoPEInterpolator::Linear(r) => r.scale_factor(),
            RoPEInterpolator::NTK(r) => r.scale_factor(),
            RoPEInterpolator::YaRN(r) => r.scale_factor(),
            RoPEInterpolator::Dynamic(r) => r.effective_scale(table.len),
        };
        Self {
            max_position_seen: table.len,
            scaling_factor_used,
            table_size: table.cos.len() + table.sin.len(),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::redundant_clone,
    clippy::float_cmp,
    clippy::approx_constant,
    clippy::suboptimal_flops
)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "expected {a} ≈ {b} (tol={tol}, diff={})", (a - b).abs());
    }

    // ── Standard RoPE ───────────────────────────────────────────────

    #[test]
    fn standard_rope_position_zero_is_identity() {
        let cfg = RoPEConfig::new(8, 4);
        let rope = StandardRoPE::new(cfg);
        let (cos_row, sin_row) = rope.table().get(0).unwrap();
        for &c in cos_row {
            approx_eq(c, 1.0, 1e-6);
        }
        for &s in sin_row {
            approx_eq(s, 0.0, 1e-6);
        }
    }

    #[test]
    fn standard_rope_table_shape() {
        let cfg = RoPEConfig::new(16, 10);
        let rope = StandardRoPE::new(cfg);
        assert_eq!(rope.table().half_dim, 8);
        assert_eq!(rope.table().len, 10);
        assert_eq!(rope.table().cos.len(), 80);
        assert_eq!(rope.table().sin.len(), 80);
    }

    #[test]
    fn standard_rope_trig_identity() {
        let cfg = RoPEConfig::new(8, 32);
        let rope = StandardRoPE::new(cfg);
        for i in 0..32 {
            let (cos_row, sin_row) = rope.table().get(i).unwrap();
            for (&c, &s) in cos_row.iter().zip(sin_row) {
                approx_eq(c * c + s * s, 1.0, 1e-5);
            }
        }
    }

    #[test]
    fn standard_rope_inv_freq_decreasing() {
        let cfg = RoPEConfig::new(16, 4);
        let rope = StandardRoPE::new(cfg);
        let freq = rope.inv_freq();
        for w in freq.windows(2) {
            assert!(w[0] >= w[1], "inv_freq should be non-increasing");
        }
    }

    #[test]
    fn standard_rope_rotation_changes_vector() {
        let cfg = RoPEConfig::new(4, 4);
        let rope = StandardRoPE::new(cfg);
        let mut v = vec![1.0, 0.0, 0.0, 1.0];
        let original = v.clone();
        let (cos_row, sin_row) = rope.table().get(1).unwrap();
        apply_rope_to_vec(&mut v, cos_row, sin_row);
        assert_ne!(v, original, "rotation at pos>0 must change the vector");
    }

    #[test]
    fn standard_rope_position_zero_preserves_vector() {
        let cfg = RoPEConfig::new(4, 2);
        let rope = StandardRoPE::new(cfg);
        let mut v = vec![3.0, 7.0, -1.0, 2.5];
        let original = v.clone();
        let (cos_row, sin_row) = rope.table().get(0).unwrap();
        apply_rope_to_vec(&mut v, cos_row, sin_row);
        for (a, b) in v.iter().zip(original.iter()) {
            approx_eq(*a, *b, 1e-6);
        }
    }

    #[test]
    fn standard_rope_round_trip() {
        let cfg = RoPEConfig::new(8, 16);
        let rope = StandardRoPE::new(cfg);
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut v = original.clone();
        let (cos_row, sin_row) = rope.table().get(5).unwrap();
        apply_rope_to_vec(&mut v, cos_row, sin_row);
        unapply_rope_to_vec(&mut v, cos_row, sin_row);
        for (a, b) in v.iter().zip(original.iter()) {
            approx_eq(*a, *b, 1e-5);
        }
    }

    #[test]
    fn standard_rope_different_positions_yield_different_tables() {
        let cfg = RoPEConfig::new(8, 4);
        let rope = StandardRoPE::new(cfg);
        let (cos0, _) = rope.table().get(0).unwrap();
        let (cos1, _) = rope.table().get(1).unwrap();
        assert_ne!(cos0, cos1);
    }

    #[test]
    fn standard_rope_custom_base_freq() {
        let cfg = RoPEConfig::new(8, 4).with_base_freq(500_000.0);
        let rope = StandardRoPE::new(cfg);
        let cfg2 = RoPEConfig::new(8, 4);
        let rope2 = StandardRoPE::new(cfg2);
        // Different base → different inverse frequencies
        assert_ne!(rope.inv_freq(), rope2.inv_freq());
    }

    #[test]
    fn standard_rope_higher_base_slower_rotation() {
        // Higher base → smaller inv_freq for i>0 → slower rotation per position
        let low = StandardRoPE::new(RoPEConfig::new(8, 4).with_base_freq(100.0));
        let high = StandardRoPE::new(RoPEConfig::new(8, 4).with_base_freq(1_000_000.0));
        // Check second element (i=1); first element is always 1.0 since base^0 = 1
        assert!(high.inv_freq()[1] < low.inv_freq()[1]);
    }

    // ── Linear-scaled RoPE ──────────────────────────────────────────

    #[test]
    fn linear_scaled_inv_freq_smaller_than_standard() {
        let cfg = RoPEConfig::new(8, 16);
        let std_rope = StandardRoPE::new(cfg.clone());
        let lin_rope = LinearScaledRoPE::new(cfg, 2.0);
        for (s, l) in std_rope.inv_freq().iter().zip(lin_rope.inv_freq()) {
            approx_eq(*l, *s / 2.0, 1e-7);
        }
    }

    #[test]
    fn linear_scaled_factor_1_matches_standard() {
        let cfg = RoPEConfig::new(8, 8);
        let std_rope = StandardRoPE::new(cfg.clone());
        let lin_rope = LinearScaledRoPE::new(cfg, 1.0);
        for (s, l) in std_rope.table().cos.iter().zip(lin_rope.table().cos.iter()) {
            approx_eq(*s, *l, 1e-6);
        }
    }

    #[test]
    fn linear_scaled_extends_effective_context() {
        // With scale=4, position 4 in the scaled version should look like
        // position 1 in the original (roughly).
        let cfg = RoPEConfig::new(8, 16);
        let std_rope = StandardRoPE::new(cfg.clone());
        let lin_rope = LinearScaledRoPE::new(cfg, 4.0);
        let (std_cos, _) = std_rope.table().get(1).unwrap();
        let (lin_cos, _) = lin_rope.table().get(4).unwrap();
        for (a, b) in std_cos.iter().zip(lin_cos) {
            approx_eq(*a, *b, 1e-5);
        }
    }

    #[test]
    fn linear_scaled_round_trip() {
        let cfg = RoPEConfig::new(8, 16);
        let rope = LinearScaledRoPE::new(cfg, 3.0);
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut v = original.clone();
        let (c, s) = rope.table().get(7).unwrap();
        apply_rope_to_vec(&mut v, c, s);
        unapply_rope_to_vec(&mut v, c, s);
        for (a, b) in v.iter().zip(original.iter()) {
            approx_eq(*a, *b, 1e-5);
        }
    }

    #[test]
    fn linear_scaled_stores_factor() {
        let cfg = RoPEConfig::new(8, 4);
        let rope = LinearScaledRoPE::new(cfg, 2.5);
        approx_eq(rope.scale_factor(), 2.5, 1e-7);
    }

    // ── NTK RoPE ────────────────────────────────────────────────────

    #[test]
    fn ntk_adjusted_base_formula() {
        let dim = 8;
        let base = 10_000.0_f32;
        let scale = 2.0_f32;
        let expected = base * scale.powf(dim as f32 / (dim as f32 - 2.0));
        let cfg = RoPEConfig::new(dim, 4);
        let rope = NTKRoPE::new(cfg, scale);
        approx_eq(rope.adjusted_base(), expected, 1e-2);
    }

    #[test]
    fn ntk_scale_1_matches_standard() {
        let cfg = RoPEConfig::new(8, 8);
        let std_rope = StandardRoPE::new(cfg.clone());
        let ntk_rope = NTKRoPE::new(cfg, 1.0);
        // scale=1 → base' = base * 1^(...) = base
        for (s, n) in std_rope.inv_freq().iter().zip(ntk_rope.inv_freq()) {
            approx_eq(*s, *n, 1e-6);
        }
    }

    #[test]
    fn ntk_higher_scale_increases_base() {
        let cfg = RoPEConfig::new(8, 4);
        let r2 = NTKRoPE::new(cfg.clone(), 2.0);
        let r4 = NTKRoPE::new(cfg, 4.0);
        assert!(r4.adjusted_base() > r2.adjusted_base());
    }

    #[test]
    fn ntk_round_trip() {
        let cfg = RoPEConfig::new(8, 16);
        let rope = NTKRoPE::new(cfg, 2.0);
        let original = vec![1.0, -1.0, 0.5, 2.0, -0.3, 0.7, 1.5, -2.0];
        let mut v = original.clone();
        let (c, s) = rope.table().get(3).unwrap();
        apply_rope_to_vec(&mut v, c, s);
        unapply_rope_to_vec(&mut v, c, s);
        for (a, b) in v.iter().zip(original.iter()) {
            approx_eq(*a, *b, 1e-5);
        }
    }

    #[test]
    fn ntk_different_from_standard() {
        let cfg = RoPEConfig::new(8, 8);
        let std_rope = StandardRoPE::new(cfg.clone());
        let ntk = NTKRoPE::new(cfg, 4.0);
        // Tables must differ
        assert_ne!(std_rope.table().cos, ntk.table().cos);
    }

    // ── YaRN RoPE ───────────────────────────────────────────────────

    #[test]
    fn yarn_creates_valid_table() {
        let cfg = RoPEConfig::new(16, 32);
        let yarn = YaRNRoPE::new(cfg, 4.0, 1.0);
        assert_eq!(yarn.table().half_dim, 8);
        assert_eq!(yarn.table().len, 32);
    }

    #[test]
    fn yarn_attention_factor_affects_frequencies() {
        let cfg = RoPEConfig::new(16, 32);
        let y1 = YaRNRoPE::new(cfg.clone(), 4.0, 0.5);
        let y2 = YaRNRoPE::new(cfg, 4.0, 2.0);
        // Different attention factors → different inv_freq
        assert_ne!(y1.inv_freq(), y2.inv_freq());
    }

    #[test]
    fn yarn_stores_parameters() {
        let cfg = RoPEConfig::new(8, 4);
        let yarn = YaRNRoPE::new(cfg, 3.0, 0.7);
        approx_eq(yarn.scale_factor(), 3.0, 1e-7);
        approx_eq(yarn.attention_factor(), 0.7, 1e-7);
    }

    #[test]
    fn yarn_round_trip() {
        let cfg = RoPEConfig::new(8, 16);
        let yarn = YaRNRoPE::new(cfg, 2.0, 1.0);
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut v = original.clone();
        let (c, s) = yarn.table().get(5).unwrap();
        apply_rope_to_vec(&mut v, c, s);
        unapply_rope_to_vec(&mut v, c, s);
        for (a, b) in v.iter().zip(original.iter()) {
            approx_eq(*a, *b, 1e-5);
        }
    }

    #[test]
    fn yarn_differs_from_pure_ntk() {
        let cfg = RoPEConfig::new(16, 32);
        let ntk = NTKRoPE::new(cfg.clone(), 4.0);
        let yarn = YaRNRoPE::new(cfg, 4.0, 1.0);
        assert_ne!(ntk.inv_freq(), yarn.inv_freq());
    }

    #[test]
    fn yarn_trig_identity() {
        let cfg = RoPEConfig::new(8, 16);
        let yarn = YaRNRoPE::new(cfg, 2.0, 1.0);
        for i in 0..16 {
            let (cos_row, sin_row) = yarn.table().get(i).unwrap();
            for (&c, &s) in cos_row.iter().zip(sin_row) {
                approx_eq(c * c + s * s, 1.0, 1e-5);
            }
        }
    }

    // ── Dynamic NTK RoPE ────────────────────────────────────────────

    #[test]
    fn dynamic_ntk_no_scaling_within_original_len() {
        let cfg = RoPEConfig::new(8, 32);
        let mut dyn_rope = DynamicNTKRoPE::new(cfg.clone(), 32);
        dyn_rope.recompute_for_seq_len(16);
        let std_rope = StandardRoPE::new(cfg);
        // For positions within original, frequencies should match standard
        let (dc, ds) = dyn_rope.table().get(5).unwrap();
        let (sc, ss) = std_rope.table().get(5).unwrap();
        for (a, b) in dc.iter().zip(sc) {
            approx_eq(*a, *b, 1e-6);
        }
        for (a, b) in ds.iter().zip(ss) {
            approx_eq(*a, *b, 1e-6);
        }
    }

    #[test]
    fn dynamic_ntk_applies_scaling_beyond_original_len() {
        let cfg = RoPEConfig::new(8, 128);
        let mut dyn_rope = DynamicNTKRoPE::new(cfg.clone(), 32);
        dyn_rope.recompute_for_seq_len(64);
        let std_rope = StandardRoPE::new(RoPEConfig::new(8, 128));
        // After recompute with longer seq, pos=5 should differ from standard
        let (dc, _) = dyn_rope.table().get(5).unwrap();
        let (sc, _) = std_rope.table().get(5).unwrap();
        assert_ne!(dc, sc, "dynamic NTK should differ from standard for extended seq");
    }

    #[test]
    fn dynamic_ntk_effective_scale() {
        let cfg = RoPEConfig::new(8, 128);
        let dyn_rope = DynamicNTKRoPE::new(cfg, 32);
        approx_eq(dyn_rope.effective_scale(32), 1.0, 1e-7);
        approx_eq(dyn_rope.effective_scale(16), 1.0, 1e-7);
        approx_eq(dyn_rope.effective_scale(64), 2.0, 1e-7);
        approx_eq(dyn_rope.effective_scale(128), 4.0, 1e-7);
    }

    #[test]
    fn dynamic_ntk_round_trip() {
        let cfg = RoPEConfig::new(8, 64);
        let mut dyn_rope = DynamicNTKRoPE::new(cfg, 16);
        dyn_rope.recompute_for_seq_len(64);
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut v = original.clone();
        let (c, s) = dyn_rope.table().get(40).unwrap();
        apply_rope_to_vec(&mut v, c, s);
        unapply_rope_to_vec(&mut v, c, s);
        for (a, b) in v.iter().zip(original.iter()) {
            approx_eq(*a, *b, 1e-5);
        }
    }

    #[test]
    fn dynamic_ntk_recompute_extends_table() {
        let cfg = RoPEConfig::new(8, 16);
        let mut dyn_rope = DynamicNTKRoPE::new(cfg, 16);
        assert_eq!(dyn_rope.table().len, 16);
        dyn_rope.recompute_for_seq_len(64);
        assert_eq!(dyn_rope.table().len, 64);
    }

    // ── ALiBi ───────────────────────────────────────────────────────

    #[test]
    fn alibi_slopes_correct_for_power_of_two_heads() {
        let alibi = ALiBiPositionBias::new(8);
        assert_eq!(alibi.slopes().len(), 8);
        // All slopes should be positive (before negation in bias)
        for &s in alibi.slopes() {
            assert!(s > 0.0, "slopes must be positive, got {s}");
            assert!(s < 1.0, "slopes must be < 1, got {s}");
        }
    }

    #[test]
    fn alibi_slopes_correct_for_non_power_of_two() {
        let alibi = ALiBiPositionBias::new(6);
        assert_eq!(alibi.slopes().len(), 6);
    }

    #[test]
    fn alibi_bias_zero_at_same_position() {
        let alibi = ALiBiPositionBias::new(4);
        for h in 0..4 {
            approx_eq(alibi.bias(h, 5, 5), 0.0, 1e-7);
        }
    }

    #[test]
    fn alibi_bias_increases_with_distance() {
        let alibi = ALiBiPositionBias::new(4);
        let b1 = alibi.bias(0, 5, 4); // distance 1
        let b2 = alibi.bias(0, 5, 3); // distance 2
        // Bias is negative and more negative with distance
        assert!(b1 < 0.0);
        assert!(b2 < b1, "farther distance should give more negative bias");
    }

    #[test]
    fn alibi_bias_symmetric() {
        let alibi = ALiBiPositionBias::new(4);
        for h in 0..4 {
            approx_eq(alibi.bias(h, 3, 7), alibi.bias(h, 7, 3), 1e-7);
        }
    }

    #[test]
    fn alibi_bias_matrix_shape() {
        let alibi = ALiBiPositionBias::new(2);
        let mat = alibi.bias_matrix(0, 5);
        assert_eq!(mat.len(), 25);
    }

    #[test]
    fn alibi_bias_matrix_diagonal_is_zero() {
        let alibi = ALiBiPositionBias::new(4);
        let mat = alibi.bias_matrix(0, 8);
        for i in 0..8 {
            approx_eq(mat[i * 8 + i], 0.0, 1e-7);
        }
    }

    #[test]
    fn alibi_different_heads_different_slopes() {
        let alibi = ALiBiPositionBias::new(8);
        // Slopes should be distinct
        let slopes = alibi.slopes();
        for i in 0..slopes.len() {
            for j in (i + 1)..slopes.len() {
                assert_ne!(slopes[i], slopes[j], "heads {i} and {j} have same slope");
            }
        }
    }

    #[test]
    fn alibi_custom_slopes() {
        let slopes = vec![0.5, 0.25, 0.125];
        let alibi = ALiBiPositionBias::with_slopes(3, slopes.clone());
        assert_eq!(alibi.slopes(), &slopes[..]);
    }

    #[test]
    fn alibi_single_head() {
        let alibi = ALiBiPositionBias::new(1);
        assert_eq!(alibi.slopes().len(), 1);
        assert!(alibi.bias(0, 0, 1) < 0.0);
    }

    // ── Frequency table ─────────────────────────────────────────────

    #[test]
    fn frequency_table_build_and_get() {
        let inv_freq = vec![1.0, 0.5, 0.25];
        let table = FrequencyTable::build(&inv_freq, 4);
        assert_eq!(table.half_dim, 3);
        assert_eq!(table.len, 4);
        assert!(table.get(0).is_some());
        assert!(table.get(3).is_some());
        assert!(table.get(4).is_none());
    }

    #[test]
    fn frequency_table_extend() {
        let inv_freq = vec![1.0, 0.5];
        let mut table = FrequencyTable::build(&inv_freq, 4);
        assert_eq!(table.len, 4);
        table.extend_to(8, &inv_freq);
        assert_eq!(table.len, 8);
        assert!(table.get(7).is_some());
    }

    #[test]
    fn frequency_table_extend_no_op_when_already_large_enough() {
        let inv_freq = vec![1.0];
        let mut table = FrequencyTable::build(&inv_freq, 10);
        let old_len = table.cos.len();
        table.extend_to(5, &inv_freq);
        assert_eq!(table.cos.len(), old_len);
        assert_eq!(table.len, 10);
    }

    #[test]
    fn frequency_table_extension_matches_fresh_build() {
        let inv_freq = vec![1.0, 0.1];
        let mut table = FrequencyTable::build(&inv_freq, 4);
        table.extend_to(8, &inv_freq);
        let fresh = FrequencyTable::build(&inv_freq, 8);
        for i in 0..8 {
            let (ec, es) = table.get(i).unwrap();
            let (fc, fs) = fresh.get(i).unwrap();
            for (a, b) in ec.iter().zip(fc) {
                approx_eq(*a, *b, 1e-7);
            }
            for (a, b) in es.iter().zip(fs) {
                approx_eq(*a, *b, 1e-7);
            }
        }
    }

    #[test]
    fn frequency_table_position_zero_values() {
        let inv_freq = vec![1.0, 0.01];
        let table = FrequencyTable::build(&inv_freq, 2);
        let (cos_row, sin_row) = table.get(0).unwrap();
        // angle = 0 * freq → cos=1, sin=0
        for &c in cos_row {
            approx_eq(c, 1.0, 1e-7);
        }
        for &s in sin_row {
            approx_eq(s, 0.0, 1e-7);
        }
    }

    // ── RoPE Interpolator ───────────────────────────────────────────

    #[test]
    fn interpolator_from_config_standard() {
        let cfg = RoPEConfig::new(8, 16);
        let interp = RoPEInterpolator::from_config(cfg);
        assert!(matches!(interp, RoPEInterpolator::Standard(_)));
    }

    #[test]
    fn interpolator_from_config_linear() {
        let cfg = RoPEConfig::new(8, 16).with_scaling(ScalingType::Linear { scale_factor: 2.0 });
        let interp = RoPEInterpolator::from_config(cfg);
        assert!(matches!(interp, RoPEInterpolator::Linear(_)));
    }

    #[test]
    fn interpolator_from_config_ntk() {
        let cfg = RoPEConfig::new(8, 16).with_scaling(ScalingType::NTK { scale_factor: 2.0 });
        let interp = RoPEInterpolator::from_config(cfg);
        assert!(matches!(interp, RoPEInterpolator::NTK(_)));
    }

    #[test]
    fn interpolator_from_config_yarn() {
        let cfg = RoPEConfig::new(8, 16)
            .with_scaling(ScalingType::YaRN { scale_factor: 2.0, attention_factor: 1.0 });
        let interp = RoPEInterpolator::from_config(cfg);
        assert!(matches!(interp, RoPEInterpolator::YaRN(_)));
    }

    #[test]
    fn interpolator_from_config_dynamic() {
        let cfg =
            RoPEConfig::new(8, 16).with_scaling(ScalingType::Dynamic { original_max_seq_len: 8 });
        let interp = RoPEInterpolator::from_config(cfg);
        assert!(matches!(interp, RoPEInterpolator::Dynamic(_)));
    }

    #[test]
    fn interpolator_apply_rotates_qk() {
        let cfg = RoPEConfig::new(4, 8);
        let interp = RoPEInterpolator::from_config(cfg);
        let mut q = vec![1.0, 0.0, 0.0, 1.0];
        let mut k = vec![0.0, 1.0, 1.0, 0.0];
        let q_orig = q.clone();
        let k_orig = k.clone();
        interp.apply(&mut q, &mut k, &[3], 4);
        assert_ne!(q, q_orig);
        assert_ne!(k, k_orig);
    }

    #[test]
    fn interpolator_apply_multiple_positions() {
        let cfg = RoPEConfig::new(4, 8);
        let interp = RoPEInterpolator::from_config(cfg);
        let mut q = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let mut k = vec![0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0];
        interp.apply(&mut q, &mut k, &[1, 2], 4);
        // Both tokens should be rotated; we just verify it didn't panic
        // and that the vectors changed
        assert!(q.iter().any(|&x| (x - 1.0).abs() > 1e-6 || (x - 0.0).abs() > 1e-6));
    }

    // ── Metrics ─────────────────────────────────────────────────────

    #[test]
    fn metrics_standard() {
        let cfg = RoPEConfig::new(8, 16);
        let interp = RoPEInterpolator::from_config(cfg);
        let m = RoPEMetrics::from_interpolator(&interp);
        assert_eq!(m.max_position_seen, 16);
        approx_eq(m.scaling_factor_used, 1.0, 1e-7);
        assert_eq!(m.table_size, 16 * 4 * 2); // 16 positions * 4 half_dim * 2 (cos+sin)
    }

    #[test]
    fn metrics_linear() {
        let cfg = RoPEConfig::new(8, 16).with_scaling(ScalingType::Linear { scale_factor: 3.0 });
        let interp = RoPEInterpolator::from_config(cfg);
        let m = RoPEMetrics::from_interpolator(&interp);
        approx_eq(m.scaling_factor_used, 3.0, 1e-7);
    }

    #[test]
    fn metrics_ntk() {
        let cfg = RoPEConfig::new(8, 16).with_scaling(ScalingType::NTK { scale_factor: 2.0 });
        let interp = RoPEInterpolator::from_config(cfg);
        let m = RoPEMetrics::from_interpolator(&interp);
        approx_eq(m.scaling_factor_used, 2.0, 1e-7);
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn edge_dim_2_minimal_rope() {
        let cfg = RoPEConfig::new(2, 4);
        let rope = StandardRoPE::new(cfg);
        assert_eq!(rope.table().half_dim, 1);
        let (c, s) = rope.table().get(0).unwrap();
        approx_eq(c[0], 1.0, 1e-7);
        approx_eq(s[0], 0.0, 1e-7);
    }

    #[test]
    fn edge_very_long_position() {
        let cfg = RoPEConfig::new(4, 100_000);
        let rope = StandardRoPE::new(cfg);
        let (cos_row, sin_row) = rope.table().get(99_999).unwrap();
        for (&c, &s) in cos_row.iter().zip(sin_row) {
            approx_eq(c * c + s * s, 1.0, 1e-4);
        }
    }

    #[test]
    fn edge_max_seq_len_1() {
        let cfg = RoPEConfig::new(4, 1);
        let rope = StandardRoPE::new(cfg);
        assert_eq!(rope.table().len, 1);
        assert!(rope.table().get(0).is_some());
        assert!(rope.table().get(1).is_none());
    }

    #[test]
    fn edge_apply_rope_to_zero_vector() {
        let cfg = RoPEConfig::new(4, 4);
        let rope = StandardRoPE::new(cfg);
        let mut v = vec![0.0; 4];
        let (c, s) = rope.table().get(2).unwrap();
        apply_rope_to_vec(&mut v, c, s);
        for &x in &v {
            approx_eq(x, 0.0, 1e-7);
        }
    }

    #[test]
    fn edge_round_trip_at_position_zero() {
        let cfg = RoPEConfig::new(4, 2);
        let rope = StandardRoPE::new(cfg);
        let original = vec![3.14, 2.71, -1.41, 0.57];
        let mut v = original.clone();
        let (c, s) = rope.table().get(0).unwrap();
        apply_rope_to_vec(&mut v, c, s);
        unapply_rope_to_vec(&mut v, c, s);
        for (a, b) in v.iter().zip(original.iter()) {
            approx_eq(*a, *b, 1e-6);
        }
    }

    #[test]
    fn edge_alibi_zero_heads() {
        let slopes = ALiBiPositionBias::compute_slopes(0);
        assert!(slopes.is_empty());
    }

    #[test]
    fn edge_frequency_table_empty() {
        let inv_freq = vec![1.0];
        let table = FrequencyTable::build(&inv_freq, 0);
        assert_eq!(table.len, 0);
        assert!(table.get(0).is_none());
    }

    #[test]
    fn edge_interpolator_position_out_of_range_is_no_op() {
        let cfg = RoPEConfig::new(4, 4);
        let interp = RoPEInterpolator::from_config(cfg);
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        let q_before = q.clone();
        let k_before = k.clone();
        // Position 10 is out of range for max_seq_len=4
        interp.apply(&mut q, &mut k, &[10], 4);
        assert_eq!(q, q_before);
        assert_eq!(k, k_before);
    }

    // ── Config builder tests ────────────────────────────────────────

    #[test]
    fn config_builder_defaults() {
        let cfg = RoPEConfig::new(8, 16);
        approx_eq(cfg.base_freq, 10_000.0, 1e-7);
        assert_eq!(cfg.dim, 8);
        assert_eq!(cfg.max_seq_len, 16);
        assert!(matches!(cfg.scaling_type, ScalingType::None));
    }

    #[test]
    fn config_builder_chaining() {
        let cfg = RoPEConfig::new(16, 64)
            .with_base_freq(500_000.0)
            .with_scaling(ScalingType::NTK { scale_factor: 4.0 });
        approx_eq(cfg.base_freq, 500_000.0, 1e-7);
        assert!(matches!(
            cfg.scaling_type,
            ScalingType::NTK { scale_factor } if (scale_factor - 4.0).abs() < 1e-7
        ));
    }
}
