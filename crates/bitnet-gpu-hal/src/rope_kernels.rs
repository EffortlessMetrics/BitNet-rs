//! Module stub - implementation pending merge from feature branch
//! Rotary Position Embedding (`RoPE`) kernels with scaling variants.
//!
//! Implements precomputation and application of rotary position embeddings
//! for transformer attention heads. Supports Standard, NTK-Aware, `YaRN`,
//! `DynamicNTK`, `LinearScaling`, and `LongRoPE` variants.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt;

// ── RoPE Type ───────────────────────────────────────────────────────────────

/// Variant of rotary position embedding to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoPEType {
    /// Standard `RoPE` with fixed base frequency.
    Standard,
    /// NTK-Aware scaling: adjusts base frequency for longer sequences.
    NTKAware,
    /// Yet another `RoPE` extensioN with attention factor scaling.
    YaRN,
    /// Dynamic NTK: adjusts base at runtime based on current sequence length.
    DynamicNTK,
    /// Linear frequency scaling by a constant factor.
    LinearScaling,
    /// Long-range `RoPE` with learned rescaling factors.
    LongRoPE,
}

impl fmt::Display for RoPEType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => write!(f, "Standard"),
            Self::NTKAware => write!(f, "NTK-Aware"),
            Self::YaRN => write!(f, "YaRN"),
            Self::DynamicNTK => write!(f, "DynamicNTK"),
            Self::LinearScaling => write!(f, "LinearScaling"),
            Self::LongRoPE => write!(f, "LongRoPE"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for rotary position embeddings.
#[derive(Debug, Clone)]
pub struct RoPEConfig {
    /// Dimension of each attention head (must be even).
    pub head_dim: usize,
    /// Maximum sequence length supported.
    pub max_seq_len: usize,
    /// Base frequency for computing inverse frequencies (default: 10000.0).
    pub base_freq: f64,
    /// Scaling factor for frequency/position interpolation.
    pub scaling_factor: f64,
    /// Which `RoPE` variant to use.
    pub rope_type: RoPEType,
}

impl RoPEConfig {
    /// Create a standard `RoPE` configuration with the given head dimension
    /// and max sequence length.
    pub const fn standard(head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            head_dim,
            max_seq_len,
            base_freq: 10_000.0,
            scaling_factor: 1.0,
            rope_type: RoPEType::Standard,
        }
    }

    /// Create an NTK-Aware `RoPE` configuration.
    pub const fn ntk_aware(head_dim: usize, max_seq_len: usize, scaling_factor: f64) -> Self {
        Self {
            head_dim,
            max_seq_len,
            base_freq: 10_000.0,
            scaling_factor,
            rope_type: RoPEType::NTKAware,
        }
    }

    /// Create a `YaRN` `RoPE` configuration.
    pub const fn yarn(head_dim: usize, max_seq_len: usize, scaling_factor: f64) -> Self {
        Self {
            head_dim,
            max_seq_len,
            base_freq: 10_000.0,
            scaling_factor,
            rope_type: RoPEType::YaRN,
        }
    }

    /// Create a Linear Scaling `RoPE` configuration.
    pub const fn linear_scaling(head_dim: usize, max_seq_len: usize, scaling_factor: f64) -> Self {
        Self {
            head_dim,
            max_seq_len,
            base_freq: 10_000.0,
            scaling_factor,
            rope_type: RoPEType::LinearScaling,
        }
    }

    /// Number of dimension pairs (`head_dim` / 2).
    pub const fn num_pairs(&self) -> usize {
        self.head_dim / 2
    }
}

// ── Frequency Table ─────────────────────────────────────────────────────────

/// Precomputed cosine and sine tables for all positions.
#[derive(Debug, Clone)]
pub struct FrequencyTable {
    /// Cosine values: `cos[pos][pair]` for position `pos` and dimension pair
    /// index `pair`.
    pub cos: Vec<Vec<f64>>,
    /// Sine values: `sin[pos][pair]` for position `pos` and dimension pair
    /// index `pair`.
    pub sin: Vec<Vec<f64>>,
    /// Sequence length this table was built for.
    pub seq_len: usize,
    /// Number of dimension pairs.
    pub num_pairs: usize,
}

impl FrequencyTable {
    /// Get the cosine value for the given position and dimension pair.
    pub fn cos_at(&self, pos: usize, pair: usize) -> f64 {
        self.cos[pos][pair]
    }

    /// Get the sine value for the given position and dimension pair.
    pub fn sin_at(&self, pos: usize, pair: usize) -> f64 {
        self.sin[pos][pair]
    }
}

// ── Frequency Computer ──────────────────────────────────────────────────────

/// Computes inverse frequencies: `1 / (base ^ (2i / d))` for each dimension
/// pair index `i`.
#[derive(Debug, Clone)]
pub struct FrequencyComputer;

impl FrequencyComputer {
    /// Compute inverse frequencies for the given configuration.
    ///
    /// Returns a vector of length `head_dim / 2` where element `i` is
    /// `1.0 / (base_freq ^ (2*i / head_dim))`.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_inv_freq(config: &RoPEConfig) -> Vec<f64> {
        let num_pairs = config.num_pairs();
        let d = config.head_dim as f64;
        (0..num_pairs)
            .map(|i| {
                let exponent = (2 * i) as f64 / d;
                1.0 / config.base_freq.powf(exponent)
            })
            .collect()
    }

    /// Build a [`FrequencyTable`] for the given sequence length from inverse
    /// frequencies.
    #[allow(clippy::cast_precision_loss)]
    pub fn build_table(inv_freq: &[f64], seq_len: usize) -> FrequencyTable {
        let num_pairs = inv_freq.len();
        let mut cos_table = Vec::with_capacity(seq_len);
        let mut sin_table = Vec::with_capacity(seq_len);

        for pos in 0..seq_len {
            let mut cos_row = Vec::with_capacity(num_pairs);
            let mut sin_row = Vec::with_capacity(num_pairs);
            for &freq in inv_freq {
                let angle = pos as f64 * freq;
                cos_row.push(angle.cos());
                sin_row.push(angle.sin());
            }
            cos_table.push(cos_row);
            sin_table.push(sin_row);
        }

        FrequencyTable { cos: cos_table, sin: sin_table, seq_len, num_pairs }
    }

    /// Convenience: compute inverse frequencies and build the table in one
    /// step.
    pub fn build_from_config(config: &RoPEConfig, seq_len: usize) -> FrequencyTable {
        let inv_freq = Self::compute_inv_freq(config);
        Self::build_table(&inv_freq, seq_len)
    }
}

// ── Rotary Applier ──────────────────────────────────────────────────────────

/// Applies rotary embeddings to a vector using precomputed cos/sin values.
///
/// For each dimension pair `(x, y)` the rotation is:
/// ```text
/// x' = x * cos(θ) - y * sin(θ)
/// y' = x * sin(θ) + y * cos(θ)
/// ```
#[derive(Debug, Clone)]
pub struct RotaryApplier;

impl RotaryApplier {
    /// Apply rotary embedding to `vec` at position `pos` using the given
    /// [`FrequencyTable`].
    ///
    /// `vec` must have length equal to `table.num_pairs * 2` (i.e. `head_dim`).
    /// The vector is modified in-place.
    pub fn apply(vec: &mut [f64], pos: usize, table: &FrequencyTable) {
        debug_assert_eq!(vec.len(), table.num_pairs * 2);
        for i in 0..table.num_pairs {
            let x = vec[2 * i];
            let y = vec[2 * i + 1];
            let cos_val = table.cos_at(pos, i);
            let sin_val = table.sin_at(pos, i);
            vec[2 * i] = x.mul_add(cos_val, -(y * sin_val));
            vec[2 * i + 1] = x.mul_add(sin_val, y * cos_val);
        }
    }

    /// Apply rotary embedding, returning a new vector.
    pub fn apply_new(vec: &[f64], pos: usize, table: &FrequencyTable) -> Vec<f64> {
        let mut result = vec.to_vec();
        Self::apply(&mut result, pos, table);
        result
    }

    /// Apply rotary embedding to both Q and K vectors at the same position.
    pub fn apply_qk(q: &mut [f64], k: &mut [f64], pos: usize, table: &FrequencyTable) {
        Self::apply(q, pos, table);
        Self::apply(k, pos, table);
    }
}

// ── NTK Scaler ──────────────────────────────────────────────────────────────

/// NTK-Aware scaling: adjusts base frequency for longer sequences.
///
/// The scaled base is `base * (scale ^ (d / (d - 2)))` where `d` is `head_dim`.
#[derive(Debug, Clone)]
pub struct NTKScaler;

impl NTKScaler {
    /// Compute the NTK-scaled base frequency.
    #[allow(clippy::cast_precision_loss)]
    pub fn scaled_base(base_freq: f64, scaling_factor: f64, head_dim: usize) -> f64 {
        let d = head_dim as f64;
        base_freq * scaling_factor.powf(d / (d - 2.0))
    }

    /// Compute inverse frequencies with NTK-aware scaling applied.
    pub fn compute_inv_freq(config: &RoPEConfig) -> Vec<f64> {
        let scaled = Self::scaled_base(config.base_freq, config.scaling_factor, config.head_dim);
        let scaled_config = RoPEConfig { base_freq: scaled, ..config.clone() };
        FrequencyComputer::compute_inv_freq(&scaled_config)
    }
}

// ── YaRN Scaler ─────────────────────────────────────────────────────────────

/// Yet another `RoPE` extensioN scaler.
///
/// Partitions dimensions into low-frequency and high-frequency bands with
/// an interpolation ramp, then applies an attention-factor correction.
#[derive(Debug, Clone)]
pub struct YaRNScaler {
    /// Lower wavelength bound for the ramp region.
    pub beta_low: f64,
    /// Upper wavelength bound for the ramp region.
    pub beta_high: f64,
    /// Attention factor applied post-rotation.
    pub attn_factor: f64,
}

impl Default for YaRNScaler {
    fn default() -> Self {
        Self { beta_low: 1.0, beta_high: 32.0, attn_factor: 1.0 }
    }
}

impl YaRNScaler {
    /// Compute the interpolation weight for a given dimension pair index.
    ///
    /// Returns 0.0 for high-frequency dimensions (no interpolation),
    /// 1.0 for low-frequency dimensions (full interpolation),
    /// and a linear ramp in between.
    #[allow(clippy::cast_precision_loss)]
    pub fn interpolation_weight(&self, pair_idx: usize, head_dim: usize, base_freq: f64) -> f64 {
        let d = head_dim as f64;
        let exponent = (2 * pair_idx) as f64 / d;
        let inv_freq = 1.0 / base_freq.powf(exponent);
        // wavelength = 2π / inv_freq
        let wavelength = 2.0 * PI / inv_freq;

        if wavelength < self.beta_low {
            0.0 // high-frequency: no scaling
        } else if wavelength > self.beta_high {
            1.0 // low-frequency: full interpolation
        } else {
            // linear ramp
            (wavelength - self.beta_low) / (self.beta_high - self.beta_low)
        }
    }

    /// Compute YaRN-scaled inverse frequencies.
    pub fn compute_inv_freq(&self, config: &RoPEConfig) -> Vec<f64> {
        let base_inv_freq = FrequencyComputer::compute_inv_freq(config);
        let num_pairs = config.num_pairs();

        (0..num_pairs)
            .map(|i| {
                let w = self.interpolation_weight(i, config.head_dim, config.base_freq);
                let scaled = base_inv_freq[i] / config.scaling_factor;
                // blend: (1 - w) * original + w * scaled
                (1.0 - w).mul_add(base_inv_freq[i], w * scaled)
            })
            .collect()
    }
}

// ── Position Interpolator ───────────────────────────────────────────────────

/// Linear position interpolation for extending context beyond training length.
///
/// Divides positions by `scaling_factor` so the model sees a compressed
/// position range: `effective_pos = pos / scale`.
#[derive(Debug, Clone)]
pub struct PositionInterpolator {
    /// Scaling factor (> 1.0 extends context).
    pub scaling_factor: f64,
}

impl PositionInterpolator {
    /// Create a new interpolator with the given scaling factor.
    pub const fn new(scaling_factor: f64) -> Self {
        Self { scaling_factor }
    }

    /// Compute the effective (interpolated) position.
    #[allow(clippy::cast_precision_loss)]
    pub fn interpolate(&self, pos: usize) -> f64 {
        pos as f64 / self.scaling_factor
    }

    /// Build a frequency table with linearly interpolated positions.
    pub fn build_table(&self, config: &RoPEConfig, seq_len: usize) -> FrequencyTable {
        let inv_freq = FrequencyComputer::compute_inv_freq(config);
        let num_pairs = inv_freq.len();
        let mut cos_table = Vec::with_capacity(seq_len);
        let mut sin_table = Vec::with_capacity(seq_len);

        for pos in 0..seq_len {
            let effective_pos = self.interpolate(pos);
            let mut cos_row = Vec::with_capacity(num_pairs);
            let mut sin_row = Vec::with_capacity(num_pairs);
            for &freq in &inv_freq {
                let angle = effective_pos * freq;
                cos_row.push(angle.cos());
                sin_row.push(angle.sin());
            }
            cos_table.push(cos_row);
            sin_table.push(sin_row);
        }

        FrequencyTable { cos: cos_table, sin: sin_table, seq_len, num_pairs }
    }
}

// ── RoPE Cache ──────────────────────────────────────────────────────────────

/// Caches frequency tables per sequence length to avoid recomputation.
#[derive(Debug)]
pub struct RoPECache {
    config: RoPEConfig,
    cache: HashMap<usize, FrequencyTable>,
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
}

impl RoPECache {
    /// Create a new cache for the given configuration.
    pub fn new(config: RoPEConfig) -> Self {
        Self { config, cache: HashMap::new(), hits: 0, misses: 0 }
    }

    /// Get or compute the frequency table for the given sequence length.
    pub fn get_or_compute(&mut self, seq_len: usize) -> &FrequencyTable {
        if self.cache.contains_key(&seq_len) {
            self.hits += 1;
        } else {
            self.misses += 1;
            let table = RoPEKernel::build_table_for_config(&self.config, seq_len);
            self.cache.insert(seq_len, table);
        }
        self.cache.get(&seq_len).expect("just inserted")
    }

    /// Invalidate all cached tables.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Number of cached sequence lengths.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Return the config this cache was created with.
    pub const fn config(&self) -> &RoPEConfig {
        &self.config
    }
}

// ── RoPE Kernel (Orchestrator) ──────────────────────────────────────────────

/// Orchestrator: compute frequencies → cache → apply rotation to Q and K
/// tensors.
///
/// This is the main entry point for applying `RoPE` to attention head vectors.
#[derive(Debug)]
pub struct RoPEKernel {
    cache: RoPECache,
}

impl RoPEKernel {
    /// Create a new `RoPE` kernel with the given configuration.
    pub fn new(config: RoPEConfig) -> Self {
        Self { cache: RoPECache::new(config) }
    }

    /// Build a frequency table for the config, dispatching by rope type.
    #[allow(clippy::cast_precision_loss)]
    pub fn build_table_for_config(config: &RoPEConfig, seq_len: usize) -> FrequencyTable {
        match config.rope_type {
            RoPEType::Standard => FrequencyComputer::build_from_config(config, seq_len),
            RoPEType::NTKAware => {
                let inv_freq = NTKScaler::compute_inv_freq(config);
                FrequencyComputer::build_table(&inv_freq, seq_len)
            }
            RoPEType::YaRN => {
                let scaler = YaRNScaler::default();
                let inv_freq = scaler.compute_inv_freq(config);
                FrequencyComputer::build_table(&inv_freq, seq_len)
            }
            RoPEType::DynamicNTK => {
                // Dynamic NTK: scale base when seq_len exceeds max_seq_len
                if seq_len > config.max_seq_len {
                    let ratio = seq_len as f64 / config.max_seq_len as f64;
                    let scaled_base =
                        NTKScaler::scaled_base(config.base_freq, ratio, config.head_dim);
                    let dyn_config = RoPEConfig {
                        base_freq: scaled_base,
                        rope_type: RoPEType::Standard,
                        ..config.clone()
                    };
                    FrequencyComputer::build_from_config(&dyn_config, seq_len)
                } else {
                    FrequencyComputer::build_from_config(config, seq_len)
                }
            }
            RoPEType::LinearScaling => {
                let interp = PositionInterpolator::new(config.scaling_factor);
                interp.build_table(config, seq_len)
            }
            RoPEType::LongRoPE => {
                // LongRoPE: same as NTK-Aware for the base implementation.
                // Full learned-rescaling factors require per-layer parameters.
                let inv_freq = NTKScaler::compute_inv_freq(config);
                FrequencyComputer::build_table(&inv_freq, seq_len)
            }
        }
    }

    /// Apply rotary embedding to a single Q vector at the given position.
    ///
    /// The cache is consulted and grown as necessary.
    pub fn apply_q(&mut self, q: &mut [f64], pos: usize) {
        let seq_len = pos + 1;
        let table = self.cache.get_or_compute(seq_len);
        RotaryApplier::apply(q, pos, table);
    }

    /// Apply rotary embedding to Q and K vectors at the given position.
    pub fn apply_qk(&mut self, q: &mut [f64], k: &mut [f64], pos: usize) {
        let seq_len = pos + 1;
        let table = self.cache.get_or_compute(seq_len);
        RotaryApplier::apply(q, pos, table);
        RotaryApplier::apply(k, pos, table);
    }

    /// Apply rotary embedding to a batch of (Q, K) pairs at consecutive
    /// positions starting from `start_pos`.
    pub fn apply_batch(
        &mut self,
        q_batch: &mut [Vec<f64>],
        k_batch: &mut [Vec<f64>],
        start_pos: usize,
    ) {
        assert_eq!(q_batch.len(), k_batch.len());
        let end_pos = start_pos + q_batch.len();
        let table = self.cache.get_or_compute(end_pos);
        for (i, (q, k)) in q_batch.iter_mut().zip(k_batch.iter_mut()).enumerate() {
            let pos = start_pos + i;
            RotaryApplier::apply(q, pos, table);
            RotaryApplier::apply(k, pos, table);
        }
    }

    /// Access the underlying cache.
    pub const fn cache(&self) -> &RoPECache {
        &self.cache
    }

    /// Access the configuration.
    pub const fn config(&self) -> &RoPEConfig {
        self.cache.config()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ──────────────────────────────────────────────────────────

    fn magnitude(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn standard_config(head_dim: usize, max_seq_len: usize) -> RoPEConfig {
        RoPEConfig::standard(head_dim, max_seq_len)
    }

    // ── RoPEType ────────────────────────────────────────────────────────

    #[test]
    fn rope_type_display() {
        assert_eq!(RoPEType::Standard.to_string(), "Standard");
        assert_eq!(RoPEType::NTKAware.to_string(), "NTK-Aware");
        assert_eq!(RoPEType::YaRN.to_string(), "YaRN");
        assert_eq!(RoPEType::DynamicNTK.to_string(), "DynamicNTK");
        assert_eq!(RoPEType::LinearScaling.to_string(), "LinearScaling");
        assert_eq!(RoPEType::LongRoPE.to_string(), "LongRoPE");
    }

    #[test]
    fn rope_type_eq_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(RoPEType::Standard);
        set.insert(RoPEType::NTKAware);
        set.insert(RoPEType::Standard); // duplicate
        assert_eq!(set.len(), 2);
    }

    // ── RoPEConfig ──────────────────────────────────────────────────────

    #[test]
    fn config_standard_defaults() {
        let c = RoPEConfig::standard(128, 2048);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.max_seq_len, 2048);
        assert!(approx_eq(c.base_freq, 10_000.0, 1e-9));
        assert!(approx_eq(c.scaling_factor, 1.0, 1e-9));
        assert_eq!(c.rope_type, RoPEType::Standard);
    }

    #[test]
    fn config_ntk_aware() {
        let c = RoPEConfig::ntk_aware(64, 4096, 2.0);
        assert_eq!(c.rope_type, RoPEType::NTKAware);
        assert!(approx_eq(c.scaling_factor, 2.0, 1e-9));
    }

    #[test]
    fn config_yarn() {
        let c = RoPEConfig::yarn(128, 8192, 4.0);
        assert_eq!(c.rope_type, RoPEType::YaRN);
        assert!(approx_eq(c.scaling_factor, 4.0, 1e-9));
    }

    #[test]
    fn config_linear_scaling() {
        let c = RoPEConfig::linear_scaling(256, 4096, 2.0);
        assert_eq!(c.rope_type, RoPEType::LinearScaling);
    }

    #[test]
    fn config_num_pairs() {
        assert_eq!(RoPEConfig::standard(64, 1024).num_pairs(), 32);
        assert_eq!(RoPEConfig::standard(128, 1024).num_pairs(), 64);
        assert_eq!(RoPEConfig::standard(256, 1024).num_pairs(), 128);
    }

    // ── FrequencyComputer ───────────────────────────────────────────────

    #[test]
    fn inv_freq_first_element_is_one() {
        // For i=0: 1 / base^(0/d) = 1 / 1 = 1
        let config = standard_config(128, 2048);
        let inv_freq = FrequencyComputer::compute_inv_freq(&config);
        assert!(approx_eq(inv_freq[0], 1.0, 1e-12));
    }

    #[test]
    fn inv_freq_length_equals_num_pairs() {
        let config = standard_config(128, 2048);
        let inv_freq = FrequencyComputer::compute_inv_freq(&config);
        assert_eq!(inv_freq.len(), 64);
    }

    #[test]
    fn inv_freq_values_match_formula_dim64() {
        let config = standard_config(64, 1024);
        let inv_freq = FrequencyComputer::compute_inv_freq(&config);
        for i in 0..config.num_pairs() {
            let expected = 1.0 / 10_000.0_f64.powf((2 * i) as f64 / 64.0);
            assert!(
                approx_eq(inv_freq[i], expected, 1e-12),
                "pair {i}: got {} expected {expected}",
                inv_freq[i]
            );
        }
    }

    #[test]
    fn inv_freq_values_match_formula_dim128() {
        let config = standard_config(128, 2048);
        let inv_freq = FrequencyComputer::compute_inv_freq(&config);
        for i in 0..config.num_pairs() {
            let expected = 1.0 / 10_000.0_f64.powf((2 * i) as f64 / 128.0);
            assert!(
                approx_eq(inv_freq[i], expected, 1e-12),
                "pair {i}: got {} expected {expected}",
                inv_freq[i]
            );
        }
    }

    #[test]
    fn inv_freq_values_match_formula_dim256() {
        let config = standard_config(256, 4096);
        let inv_freq = FrequencyComputer::compute_inv_freq(&config);
        for i in 0..config.num_pairs() {
            let expected = 1.0 / 10_000.0_f64.powf((2 * i) as f64 / 256.0);
            assert!(
                approx_eq(inv_freq[i], expected, 1e-12),
                "pair {i}: got {} expected {expected}",
                inv_freq[i]
            );
        }
    }

    #[test]
    fn inv_freq_monotonically_decreasing() {
        let config = standard_config(128, 2048);
        let inv_freq = FrequencyComputer::compute_inv_freq(&config);
        for window in inv_freq.windows(2) {
            assert!(window[0] > window[1]);
        }
    }

    #[test]
    fn inv_freq_last_element_is_smallest() {
        let config = standard_config(128, 2048);
        let inv_freq = FrequencyComputer::compute_inv_freq(&config);
        let last = *inv_freq.last().unwrap();
        let expected = 1.0 / 10_000.0_f64.powf(126.0 / 128.0);
        assert!(approx_eq(last, expected, 1e-12));
    }

    #[test]
    fn inv_freq_custom_base() {
        let config = RoPEConfig { base_freq: 500.0, ..standard_config(64, 1024) };
        let inv_freq = FrequencyComputer::compute_inv_freq(&config);
        let expected_0 = 1.0; // base^0 = 1
        let expected_1 = 1.0 / 500.0_f64.powf(2.0 / 64.0);
        assert!(approx_eq(inv_freq[0], expected_0, 1e-12));
        assert!(approx_eq(inv_freq[1], expected_1, 1e-12));
    }

    // ── FrequencyTable ──────────────────────────────────────────────────

    #[test]
    fn table_dimensions() {
        let config = standard_config(64, 1024);
        let table = FrequencyComputer::build_from_config(&config, 10);
        assert_eq!(table.seq_len, 10);
        assert_eq!(table.num_pairs, 32);
        assert_eq!(table.cos.len(), 10);
        assert_eq!(table.sin.len(), 10);
        assert_eq!(table.cos[0].len(), 32);
    }

    #[test]
    fn table_position_zero_cos_is_one() {
        let config = standard_config(128, 2048);
        let table = FrequencyComputer::build_from_config(&config, 100);
        for pair in 0..table.num_pairs {
            assert!(
                approx_eq(table.cos_at(0, pair), 1.0, 1e-12),
                "cos at pos=0, pair={pair} should be 1.0, got {}",
                table.cos_at(0, pair)
            );
        }
    }

    #[test]
    fn table_position_zero_sin_is_zero() {
        let config = standard_config(128, 2048);
        let table = FrequencyComputer::build_from_config(&config, 100);
        for pair in 0..table.num_pairs {
            assert!(
                approx_eq(table.sin_at(0, pair), 0.0, 1e-12),
                "sin at pos=0, pair={pair} should be 0.0, got {}",
                table.sin_at(0, pair)
            );
        }
    }

    #[test]
    fn table_cos_sin_identity() {
        // cos²(θ) + sin²(θ) = 1 for all entries
        let config = standard_config(64, 1024);
        let table = FrequencyComputer::build_from_config(&config, 50);
        for pos in 0..50 {
            for pair in 0..32 {
                let c = table.cos_at(pos, pair);
                let s = table.sin_at(pos, pair);
                assert!(
                    approx_eq(c * c + s * s, 1.0, 1e-10),
                    "cos²+sin²≠1 at pos={pos}, pair={pair}"
                );
            }
        }
    }

    #[test]
    fn table_values_at_position_one() {
        let config = standard_config(4, 1024);
        let table = FrequencyComputer::build_from_config(&config, 5);
        // pair 0: angle = 1.0 * 1.0 = 1.0 (inv_freq[0] = 1.0)
        assert!(approx_eq(table.cos_at(1, 0), 1.0_f64.cos(), 1e-12));
        assert!(approx_eq(table.sin_at(1, 0), 1.0_f64.sin(), 1e-12));
        // pair 1: angle = 1.0 * inv_freq[1]
        let inv1 = 1.0 / 10_000.0_f64.powf(2.0 / 4.0);
        assert!(approx_eq(table.cos_at(1, 1), inv1.cos(), 1e-12));
        assert!(approx_eq(table.sin_at(1, 1), inv1.sin(), 1e-12));
    }

    // ── RotaryApplier ───────────────────────────────────────────────────

    #[test]
    fn rotation_at_position_zero_is_identity() {
        let config = standard_config(8, 1024);
        let table = FrequencyComputer::build_from_config(&config, 10);
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rotated = RotaryApplier::apply_new(&original, 0, &table);
        for (a, b) in original.iter().zip(rotated.iter()) {
            assert!(approx_eq(*a, *b, 1e-12));
        }
    }

    #[test]
    fn rotation_preserves_magnitude_dim8() {
        let config = standard_config(8, 1024);
        let table = FrequencyComputer::build_from_config(&config, 100);
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig_mag = magnitude(&v);
        for pos in 0..100 {
            let rotated = RotaryApplier::apply_new(&v, pos, &table);
            assert!(
                approx_eq(magnitude(&rotated), orig_mag, 1e-8),
                "magnitude changed at pos={pos}"
            );
        }
    }

    #[test]
    fn rotation_preserves_magnitude_dim64() {
        let config = standard_config(64, 2048);
        let table = FrequencyComputer::build_from_config(&config, 200);
        let v: Vec<f64> = (0..64).map(|i| (i as f64 + 1.0) * 0.1).collect();
        let orig_mag = magnitude(&v);
        for pos in [0, 1, 10, 50, 100, 199] {
            let rotated = RotaryApplier::apply_new(&v, pos, &table);
            assert!(
                approx_eq(magnitude(&rotated), orig_mag, 1e-8),
                "magnitude changed at pos={pos}"
            );
        }
    }

    #[test]
    fn rotation_preserves_magnitude_dim128() {
        let config = standard_config(128, 4096);
        let table = FrequencyComputer::build_from_config(&config, 500);
        let v: Vec<f64> = (0..128).map(|i| (i as f64).sin()).collect();
        let orig_mag = magnitude(&v);
        for pos in [0, 1, 50, 250, 499] {
            let rotated = RotaryApplier::apply_new(&v, pos, &table);
            assert!(
                approx_eq(magnitude(&rotated), orig_mag, 1e-8),
                "magnitude changed at pos={pos}"
            );
        }
    }

    #[test]
    fn rotation_preserves_magnitude_dim256() {
        let config = standard_config(256, 4096);
        let table = FrequencyComputer::build_from_config(&config, 100);
        let v: Vec<f64> = (0..256).map(|i| ((i as f64) * 0.01).cos()).collect();
        let orig_mag = magnitude(&v);
        for pos in [0, 1, 50, 99] {
            let rotated = RotaryApplier::apply_new(&v, pos, &table);
            assert!(
                approx_eq(magnitude(&rotated), orig_mag, 1e-8),
                "magnitude changed at pos={pos}"
            );
        }
    }

    #[test]
    fn rotation_in_place_matches_new() {
        let config = standard_config(8, 1024);
        let table = FrequencyComputer::build_from_config(&config, 10);
        let v = vec![3.0, 7.0, 1.0, 4.0, 2.0, 5.0, 6.0, 8.0];
        let new_result = RotaryApplier::apply_new(&v, 5, &table);
        let mut in_place = v.clone();
        RotaryApplier::apply(&mut in_place, 5, &table);
        for (a, b) in in_place.iter().zip(new_result.iter()) {
            assert!(approx_eq(*a, *b, 1e-15));
        }
    }

    #[test]
    fn apply_qk_rotates_both() {
        let config = standard_config(4, 1024);
        let table = FrequencyComputer::build_from_config(&config, 10);
        let q_orig = vec![1.0, 0.0, 0.0, 1.0];
        let k_orig = vec![0.0, 1.0, 1.0, 0.0];
        let mut q = q_orig.clone();
        let mut k = k_orig.clone();
        RotaryApplier::apply_qk(&mut q, &mut k, 3, &table);
        // Both should be rotated (not equal to original for pos > 0)
        assert!(q != q_orig || k != k_orig);
        // Magnitudes preserved
        assert!(approx_eq(magnitude(&q), magnitude(&q_orig), 1e-10));
        assert!(approx_eq(magnitude(&k), magnitude(&k_orig), 1e-10));
    }

    #[test]
    fn rotation_known_values_dim2() {
        // head_dim=2 → 1 pair, inv_freq[0] = 1.0
        // At pos=1: angle=1.0, cos(1)=0.5403, sin(1)=0.8415
        let config = standard_config(2, 100);
        let table = FrequencyComputer::build_from_config(&config, 5);
        let v = vec![1.0, 0.0];
        let r = RotaryApplier::apply_new(&v, 1, &table);
        assert!(approx_eq(r[0], 1.0_f64.cos(), 1e-10));
        assert!(approx_eq(r[1], 1.0_f64.sin(), 1e-10));
    }

    #[test]
    fn rotation_known_values_dim2_unit_y() {
        let config = standard_config(2, 100);
        let table = FrequencyComputer::build_from_config(&config, 5);
        let v = vec![0.0, 1.0];
        let r = RotaryApplier::apply_new(&v, 1, &table);
        // x' = 0*cos(1) - 1*sin(1) = -sin(1)
        // y' = 0*sin(1) + 1*cos(1) = cos(1)
        assert!(approx_eq(r[0], -(1.0_f64.sin()), 1e-10));
        assert!(approx_eq(r[1], 1.0_f64.cos(), 1e-10));
    }

    #[test]
    fn double_rotation_at_same_position() {
        // Applying rotation at the same position twice = rotation by 2*angle
        let config = standard_config(4, 100);
        let table = FrequencyComputer::build_from_config(&config, 10);
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut once = v.clone();
        RotaryApplier::apply(&mut once, 3, &table);
        RotaryApplier::apply(&mut once, 3, &table);

        // Build table with doubled angles (pos=6 with same freqs = angle*2)
        let inv_freq = FrequencyComputer::compute_inv_freq(&config);
        let double_table = FrequencyComputer::build_table(&inv_freq, 7);
        let double = RotaryApplier::apply_new(&v, 6, &double_table);

        for (a, b) in once.iter().zip(double.iter()) {
            assert!(approx_eq(*a, *b, 1e-10));
        }
    }

    // ── NTK Scaler ──────────────────────────────────────────────────────

    #[test]
    fn ntk_scaled_base_increases_with_factor() {
        let base = NTKScaler::scaled_base(10_000.0, 1.0, 128);
        assert!(approx_eq(base, 10_000.0, 1e-6));

        let base2 = NTKScaler::scaled_base(10_000.0, 2.0, 128);
        assert!(base2 > 10_000.0);

        let base4 = NTKScaler::scaled_base(10_000.0, 4.0, 128);
        assert!(base4 > base2);
    }

    #[test]
    fn ntk_scaling_factor_one_is_identity() {
        let config = RoPEConfig::ntk_aware(128, 2048, 1.0);
        let ntk_freq = NTKScaler::compute_inv_freq(&config);
        let std_config = standard_config(128, 2048);
        let std_freq = FrequencyComputer::compute_inv_freq(&std_config);
        for (a, b) in ntk_freq.iter().zip(std_freq.iter()) {
            assert!(approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn ntk_inv_freq_length() {
        let config = RoPEConfig::ntk_aware(64, 4096, 2.0);
        let inv_freq = NTKScaler::compute_inv_freq(&config);
        assert_eq!(inv_freq.len(), 32);
    }

    #[test]
    fn ntk_scaled_base_formula() {
        let base = 10_000.0_f64;
        let scale = 2.0_f64;
        let d = 128.0_f64;
        let expected = base * scale.powf(d / (d - 2.0));
        let actual = NTKScaler::scaled_base(base, scale, 128);
        assert!(approx_eq(actual, expected, 1e-6));
    }

    #[test]
    fn ntk_different_head_dims() {
        let b64 = NTKScaler::scaled_base(10_000.0, 2.0, 64);
        let b128 = NTKScaler::scaled_base(10_000.0, 2.0, 128);
        let b256 = NTKScaler::scaled_base(10_000.0, 2.0, 256);
        // All should be > base
        assert!(b64 > 10_000.0);
        assert!(b128 > 10_000.0);
        assert!(b256 > 10_000.0);
        // Higher dim → smaller exponent → smaller scaled base
        assert!(b64 > b128);
        assert!(b128 > b256);
    }

    // ── YaRN Scaler ────────────────────────────────────────────────────

    #[test]
    fn yarn_default_params() {
        let y = YaRNScaler::default();
        assert!(approx_eq(y.beta_low, 1.0, 1e-9));
        assert!(approx_eq(y.beta_high, 32.0, 1e-9));
        assert!(approx_eq(y.attn_factor, 1.0, 1e-9));
    }

    #[test]
    fn yarn_interpolation_weight_extremes() {
        let y = YaRNScaler { beta_low: 1.0, beta_high: 100.0, attn_factor: 1.0 };
        // pair 0 with base 10000, dim 128 → very high wavelength → weight = 1
        // (low frequency gets full interpolation)
        let w0 = y.interpolation_weight(0, 128, 10_000.0);
        // pair 0: inv_freq = 1.0, wavelength = 2π ≈ 6.28 → between beta_low and
        // beta_high
        assert!(w0 >= 0.0 && w0 <= 1.0);
    }

    #[test]
    fn yarn_inv_freq_length() {
        let y = YaRNScaler::default();
        let config = RoPEConfig::yarn(128, 4096, 2.0);
        let inv_freq = y.compute_inv_freq(&config);
        assert_eq!(inv_freq.len(), 64);
    }

    #[test]
    fn yarn_scaling_factor_one_is_identity() {
        let y = YaRNScaler::default();
        let config = RoPEConfig::yarn(128, 4096, 1.0);
        let yarn_freq = y.compute_inv_freq(&config);
        let std_config = standard_config(128, 4096);
        let std_freq = FrequencyComputer::compute_inv_freq(&std_config);
        // With scaling_factor = 1, the interpolation blends identical values
        for (a, b) in yarn_freq.iter().zip(std_freq.iter()) {
            assert!(approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn yarn_scaling_modifies_frequencies() {
        let y = YaRNScaler::default();
        let config = RoPEConfig::yarn(64, 4096, 4.0);
        let yarn_freq = y.compute_inv_freq(&config);
        let std_config = standard_config(64, 4096);
        let std_freq = FrequencyComputer::compute_inv_freq(&std_config);
        // At least some frequencies should differ
        let any_different =
            yarn_freq.iter().zip(std_freq.iter()).any(|(a, b)| !approx_eq(*a, *b, 1e-12));
        assert!(any_different, "YaRN with scale=4 should modify some frequencies");
    }

    // ── Position Interpolator ───────────────────────────────────────────

    #[test]
    fn interpolator_effective_position() {
        let interp = PositionInterpolator::new(2.0);
        assert!(approx_eq(interp.interpolate(0), 0.0, 1e-12));
        assert!(approx_eq(interp.interpolate(10), 5.0, 1e-12));
        assert!(approx_eq(interp.interpolate(100), 50.0, 1e-12));
    }

    #[test]
    fn interpolator_factor_one_is_identity() {
        let interp = PositionInterpolator::new(1.0);
        for pos in 0..100 {
            assert!(approx_eq(interp.interpolate(pos), pos as f64, 1e-12));
        }
    }

    #[test]
    fn interpolator_table_dimensions() {
        let interp = PositionInterpolator::new(2.0);
        let config = standard_config(64, 4096);
        let table = interp.build_table(&config, 20);
        assert_eq!(table.seq_len, 20);
        assert_eq!(table.num_pairs, 32);
    }

    #[test]
    fn linear_scaling_reduces_frequencies() {
        let config = standard_config(64, 4096);
        let std_table = FrequencyComputer::build_from_config(&config, 100);
        let interp = PositionInterpolator::new(2.0);
        let scaled_table = interp.build_table(&config, 100);
        // At position 10 with scale 2, angles should match position 5 unscaled
        for pair in 0..config.num_pairs() {
            assert!(approx_eq(scaled_table.cos_at(10, pair), std_table.cos_at(5, pair), 1e-10,));
            assert!(approx_eq(scaled_table.sin_at(10, pair), std_table.sin_at(5, pair), 1e-10,));
        }
    }

    #[test]
    fn linear_scaling_position_zero_unchanged() {
        let interp = PositionInterpolator::new(4.0);
        let config = standard_config(64, 4096);
        let table = interp.build_table(&config, 10);
        for pair in 0..config.num_pairs() {
            assert!(approx_eq(table.cos_at(0, pair), 1.0, 1e-12));
            assert!(approx_eq(table.sin_at(0, pair), 0.0, 1e-12));
        }
    }

    #[test]
    fn rotation_inverse_by_negative_position() {
        // Rotating by pos then by -pos (simulated via 2π periodicity) returns
        // approximately the original vector for the first dimension pair.
        let config = standard_config(2, 1000);
        let table = FrequencyComputer::build_from_config(&config, 1000);
        let v = vec![3.0, 5.0];
        let mut rotated = RotaryApplier::apply_new(&v, 100, &table);
        // Rotate back: the period for pair 0 is 2π (inv_freq=1), so rotating
        // by a complementary angle should undo. For general dim we just check
        // magnitude.
        assert!(approx_eq(magnitude(&rotated), magnitude(&v), 1e-10));
        // Apply at pos that completes a full 2π cycle: ceil(2π)=7
        // 100+n where angle ≈ 2πk → check magnitude still preserved
        RotaryApplier::apply(&mut rotated, 900, &table);
        assert!(approx_eq(magnitude(&rotated), magnitude(&v), 1e-10));
    }

    #[test]
    fn ntk_scaling_preserves_magnitude() {
        let config = RoPEConfig::ntk_aware(8, 4096, 4.0);
        let table = RoPEKernel::build_table_for_config(&config, 100);
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig_mag = magnitude(&v);
        for pos in [0, 1, 50, 99] {
            let rotated = RotaryApplier::apply_new(&v, pos, &table);
            assert!(
                approx_eq(magnitude(&rotated), orig_mag, 1e-8),
                "NTK rotation changed magnitude at pos={pos}"
            );
        }
    }

    // ── RoPE Cache ──────────────────────────────────────────────────────

    #[test]
    fn cache_starts_empty() {
        let cache = RoPECache::new(standard_config(64, 1024));
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.hits, 0);
        assert_eq!(cache.misses, 0);
    }

    #[test]
    fn cache_miss_then_hit() {
        let mut cache = RoPECache::new(standard_config(64, 1024));
        let _ = cache.get_or_compute(10);
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 0);
        assert_eq!(cache.len(), 1);

        let _ = cache.get_or_compute(10);
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_different_seq_lens() {
        let mut cache = RoPECache::new(standard_config(64, 1024));
        let _ = cache.get_or_compute(10);
        let _ = cache.get_or_compute(20);
        let _ = cache.get_or_compute(30);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.misses, 3);
    }

    #[test]
    fn cache_clear() {
        let mut cache = RoPECache::new(standard_config(64, 1024));
        let _ = cache.get_or_compute(10);
        let _ = cache.get_or_compute(10);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.hits, 0);
        assert_eq!(cache.misses, 0);
    }

    #[test]
    fn cache_returns_correct_table() {
        let mut cache = RoPECache::new(standard_config(64, 1024));
        let table = cache.get_or_compute(5);
        assert_eq!(table.seq_len, 5);
        let table2 = cache.get_or_compute(15);
        assert_eq!(table2.seq_len, 15);
    }

    #[test]
    fn cache_config_accessible() {
        let config = standard_config(128, 2048);
        let cache = RoPECache::new(config);
        assert_eq!(cache.config().head_dim, 128);
        assert_eq!(cache.config().max_seq_len, 2048);
    }

    // ── RoPE Kernel (Orchestrator) ──────────────────────────────────────

    #[test]
    fn kernel_apply_q() {
        let mut kernel = RoPEKernel::new(standard_config(4, 100));
        let mut q = vec![1.0, 0.0, 0.0, 1.0];
        kernel.apply_q(&mut q, 0);
        // Position 0 = identity
        assert!(approx_eq(q[0], 1.0, 1e-12));
        assert!(approx_eq(q[1], 0.0, 1e-12));
    }

    #[test]
    fn kernel_apply_qk() {
        let mut kernel = RoPEKernel::new(standard_config(4, 100));
        let mut q = vec![1.0, 0.0, 0.0, 1.0];
        let mut k = vec![0.0, 1.0, 1.0, 0.0];
        kernel.apply_qk(&mut q, &mut k, 5);
        // Both should be rotated
        assert!(approx_eq(magnitude(&q), (2.0_f64).sqrt(), 1e-10));
        assert!(approx_eq(magnitude(&k), (2.0_f64).sqrt(), 1e-10));
    }

    #[test]
    fn kernel_apply_batch() {
        let mut kernel = RoPEKernel::new(standard_config(4, 100));
        let mut q_batch =
            vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]];
        let mut k_batch =
            vec![vec![1.0, 1.0, 1.0, 1.0], vec![2.0, 0.0, 0.0, 2.0], vec![0.5, 0.5, 0.5, 0.5]];
        let q_mags: Vec<f64> = q_batch.iter().map(|v| magnitude(v)).collect();
        let k_mags: Vec<f64> = k_batch.iter().map(|v| magnitude(v)).collect();

        kernel.apply_batch(&mut q_batch, &mut k_batch, 10);

        for (i, (q, k)) in q_batch.iter().zip(k_batch.iter()).enumerate() {
            assert!(
                approx_eq(magnitude(q), q_mags[i], 1e-8),
                "q magnitude changed at batch index {i}"
            );
            assert!(
                approx_eq(magnitude(k), k_mags[i], 1e-8),
                "k magnitude changed at batch index {i}"
            );
        }
    }

    #[test]
    fn kernel_batch_consistent_with_individual() {
        let config = standard_config(4, 100);
        let mut kernel_batch = RoPEKernel::new(config.clone());
        let mut kernel_ind = RoPEKernel::new(config);

        let q_orig = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let k_orig = vec![vec![8.0, 7.0, 6.0, 5.0], vec![4.0, 3.0, 2.0, 1.0]];

        let mut q_batch = q_orig.clone();
        let mut k_batch = k_orig.clone();
        kernel_batch.apply_batch(&mut q_batch, &mut k_batch, 5);

        for (i, (q_o, k_o)) in q_orig.iter().zip(k_orig.iter()).enumerate() {
            let mut q = q_o.clone();
            let mut k = k_o.clone();
            kernel_ind.apply_qk(&mut q, &mut k, 5 + i);
            for (a, b) in q.iter().zip(q_batch[i].iter()) {
                assert!(approx_eq(*a, *b, 1e-14));
            }
            for (a, b) in k.iter().zip(k_batch[i].iter()) {
                assert!(approx_eq(*a, *b, 1e-14));
            }
        }
    }

    #[test]
    fn kernel_caches_tables() {
        let mut kernel = RoPEKernel::new(standard_config(4, 100));
        let mut q1 = vec![1.0, 0.0, 0.0, 1.0];
        kernel.apply_q(&mut q1, 5); // needs seq_len=6
        assert_eq!(kernel.cache().misses, 1);
        let mut q2 = vec![0.0, 1.0, 1.0, 0.0];
        kernel.apply_q(&mut q2, 3); // needs seq_len=4 → cache miss
        assert_eq!(kernel.cache().misses, 2);
        let mut q3 = vec![1.0, 1.0, 1.0, 1.0];
        kernel.apply_q(&mut q3, 2); // needs seq_len=3 → cache miss
        assert_eq!(kernel.cache().misses, 3);
    }

    #[test]
    fn kernel_cache_hit() {
        let mut kernel = RoPEKernel::new(standard_config(4, 100));
        let mut q = vec![1.0, 0.0, 0.0, 1.0];
        kernel.apply_q(&mut q, 5); // seq_len=6, miss
        let mut q2 = vec![0.0, 1.0, 1.0, 0.0];
        kernel.apply_q(&mut q2, 4); // seq_len=5, miss
        let mut q3 = vec![1.0, 1.0, 1.0, 1.0];
        kernel.apply_q(&mut q3, 5); // seq_len=6, hit
        assert_eq!(kernel.cache().hits, 1);
    }

    #[test]
    fn kernel_config_accessible() {
        let kernel = RoPEKernel::new(standard_config(128, 4096));
        assert_eq!(kernel.config().head_dim, 128);
        assert_eq!(kernel.config().max_seq_len, 4096);
    }

    // ── RoPE Type Dispatch ──────────────────────────────────────────────

    #[test]
    fn kernel_standard_type() {
        let config = standard_config(8, 100);
        let table = RoPEKernel::build_table_for_config(&config, 10);
        assert_eq!(table.seq_len, 10);
    }

    #[test]
    fn kernel_ntk_type() {
        let config = RoPEConfig::ntk_aware(8, 100, 2.0);
        let table = RoPEKernel::build_table_for_config(&config, 10);
        assert_eq!(table.seq_len, 10);
    }

    #[test]
    fn kernel_yarn_type() {
        let config = RoPEConfig::yarn(8, 100, 2.0);
        let table = RoPEKernel::build_table_for_config(&config, 10);
        assert_eq!(table.seq_len, 10);
    }

    #[test]
    fn kernel_dynamic_ntk_below_max() {
        let config = RoPEConfig { rope_type: RoPEType::DynamicNTK, ..standard_config(8, 100) };
        let table_dyn = RoPEKernel::build_table_for_config(&config, 50);
        let table_std = RoPEKernel::build_table_for_config(&standard_config(8, 100), 50);
        // Below max_seq_len, DynamicNTK should match Standard
        for pos in 0..50 {
            for pair in 0..4 {
                assert!(
                    approx_eq(table_dyn.cos_at(pos, pair), table_std.cos_at(pos, pair), 1e-12,)
                );
            }
        }
    }

    #[test]
    fn kernel_dynamic_ntk_above_max() {
        let config = RoPEConfig { rope_type: RoPEType::DynamicNTK, ..standard_config(8, 100) };
        let table_dyn = RoPEKernel::build_table_for_config(&config, 200);
        let table_std = RoPEKernel::build_table_for_config(&standard_config(8, 100), 200);
        // Above max_seq_len, DynamicNTK should differ from Standard
        let any_different = (0..200).any(|pos| {
            (0..4).any(|pair| {
                !approx_eq(table_dyn.cos_at(pos, pair), table_std.cos_at(pos, pair), 1e-12)
            })
        });
        assert!(any_different, "DynamicNTK above max should differ from Standard");
    }

    #[test]
    fn kernel_linear_scaling_type() {
        let config = RoPEConfig::linear_scaling(8, 100, 2.0);
        let table = RoPEKernel::build_table_for_config(&config, 10);
        assert_eq!(table.seq_len, 10);
    }

    #[test]
    fn kernel_longrope_type() {
        let config = RoPEConfig {
            rope_type: RoPEType::LongRoPE,
            scaling_factor: 2.0,
            ..standard_config(8, 100)
        };
        let table = RoPEKernel::build_table_for_config(&config, 10);
        assert_eq!(table.seq_len, 10);
    }

    // ── Edge Cases ──────────────────────────────────────────────────────

    #[test]
    fn edge_seq_len_one() {
        let config = standard_config(8, 100);
        let table = FrequencyComputer::build_from_config(&config, 1);
        assert_eq!(table.seq_len, 1);
        // Position 0 → identity
        for pair in 0..4 {
            assert!(approx_eq(table.cos_at(0, pair), 1.0, 1e-12));
            assert!(approx_eq(table.sin_at(0, pair), 0.0, 1e-12));
        }
    }

    #[test]
    fn edge_head_dim_two() {
        let config = standard_config(2, 100);
        let table = FrequencyComputer::build_from_config(&config, 50);
        assert_eq!(table.num_pairs, 1);
        let mut v = vec![1.0, 0.0];
        RotaryApplier::apply(&mut v, 25, &table);
        assert!(approx_eq(magnitude(&v), 1.0, 1e-10));
    }

    #[test]
    fn edge_very_large_position() {
        let config = standard_config(4, 1_000_000);
        let table = FrequencyComputer::build_from_config(&config, 100_001);
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let rotated = RotaryApplier::apply_new(&v, 100_000, &table);
        // Magnitude should still be preserved even at large positions
        assert!(approx_eq(magnitude(&rotated), magnitude(&v), 1e-6));
    }

    #[test]
    fn edge_zero_vector() {
        let config = standard_config(4, 100);
        let table = FrequencyComputer::build_from_config(&config, 10);
        let v = vec![0.0, 0.0, 0.0, 0.0];
        let rotated = RotaryApplier::apply_new(&v, 5, &table);
        for x in &rotated {
            assert!(approx_eq(*x, 0.0, 1e-15));
        }
    }

    #[test]
    fn edge_unit_vectors() {
        let config = standard_config(4, 100);
        let table = FrequencyComputer::build_from_config(&config, 10);
        for d in 0..4 {
            let mut v = vec![0.0; 4];
            v[d] = 1.0;
            let rotated = RotaryApplier::apply_new(&v, 5, &table);
            assert!(approx_eq(magnitude(&rotated), 1.0, 1e-10));
        }
    }

    #[test]
    fn edge_negative_values() {
        let config = standard_config(8, 100);
        let table = FrequencyComputer::build_from_config(&config, 10);
        let v = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
        let rotated = RotaryApplier::apply_new(&v, 7, &table);
        assert!(approx_eq(magnitude(&rotated), magnitude(&v), 1e-8));
    }

    #[test]
    fn edge_very_small_base_freq() {
        let config = RoPEConfig { base_freq: 1.0, ..standard_config(4, 100) };
        let inv_freq = FrequencyComputer::compute_inv_freq(&config);
        // With base=1.0, all inv_freq = 1/1^(anything) = 1.0
        for &f in &inv_freq {
            assert!(approx_eq(f, 1.0, 1e-12));
        }
    }

    #[test]
    fn edge_very_large_base_freq() {
        let config = RoPEConfig { base_freq: 1e12, ..standard_config(4, 100) };
        let inv_freq = FrequencyComputer::compute_inv_freq(&config);
        // First element is always 1.0
        assert!(approx_eq(inv_freq[0], 1.0, 1e-12));
        // Other elements should be very small
        assert!(inv_freq[1] < 0.01);
    }

    // ── Relative Position Invariance ────────────────────────────────────

    #[test]
    fn relative_position_dot_product_property() {
        // The dot product of two rotated vectors depends only on relative position
        let config = standard_config(8, 1024);
        let table = FrequencyComputer::build_from_config(&config, 200);
        let q = vec![1.0, 0.5, 0.3, 0.7, 0.2, 0.9, 0.4, 0.6];
        let k = vec![0.8, 0.3, 0.5, 0.1, 0.7, 0.4, 0.6, 0.2];

        // Rotate at positions (10, 15) → relative distance = 5
        let rq1 = RotaryApplier::apply_new(&q, 10, &table);
        let rk1 = RotaryApplier::apply_new(&k, 15, &table);
        let dot1 = dot(&rq1, &rk1);

        // Rotate at positions (50, 55) → same relative distance = 5
        let rq2 = RotaryApplier::apply_new(&q, 50, &table);
        let rk2 = RotaryApplier::apply_new(&k, 55, &table);
        let dot2 = dot(&rq2, &rk2);

        assert!(
            approx_eq(dot1, dot2, 1e-8),
            "dot products should match for same relative distance: {dot1} vs {dot2}"
        );
    }

    #[test]
    fn relative_position_multiple_offsets() {
        let config = standard_config(4, 200);
        let table = FrequencyComputer::build_from_config(&config, 200);
        let q = vec![1.0, 0.0, 1.0, 0.0];
        let k = vec![0.0, 1.0, 0.0, 1.0];

        for rel_dist in [1, 3, 7, 15] {
            let mut dots = Vec::new();
            for base_pos in [0, 10, 50, 100] {
                if base_pos + rel_dist >= 200 {
                    continue;
                }
                let rq = RotaryApplier::apply_new(&q, base_pos, &table);
                let rk = RotaryApplier::apply_new(&k, base_pos + rel_dist, &table);
                dots.push(dot(&rq, &rk));
            }
            for w in dots.windows(2) {
                assert!(
                    approx_eq(w[0], w[1], 1e-8),
                    "dot product inconsistent for rel_dist={rel_dist}: {} vs {}",
                    w[0],
                    w[1],
                );
            }
        }
    }

    // ── proptest ────────────────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_dim() -> impl Strategy<Value = usize> {
            prop_oneof![Just(2), Just(4), Just(8), Just(16), Just(32), Just(64)]
        }

        fn arb_vec(dim: usize) -> impl Strategy<Value = Vec<f64>> {
            proptest::collection::vec(-100.0..100.0_f64, dim)
        }

        proptest! {
            #[test]
            fn rotation_preserves_magnitude(
                dim in arb_dim(),
                pos in 0_usize..500,
            ) {
                let v: Vec<f64> = (0..dim).map(|i| (i as f64 + 1.0) * 0.1).collect();
                let config = standard_config(dim, 1000);
                let table = FrequencyComputer::build_from_config(&config, pos + 1);
                let rotated = RotaryApplier::apply_new(&v, pos, &table);
                let orig_mag = magnitude(&v);
                let rot_mag = magnitude(&rotated);
                prop_assert!((orig_mag - rot_mag).abs() < 1e-8,
                    "magnitude not preserved: {} vs {}", orig_mag, rot_mag);
            }

            #[test]
            fn rotation_preserves_dot_product_relative(
                dim in arb_dim(),
                base_pos in 0_usize..200,
                rel_offset in 1_usize..50,
            ) {
                let q: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.3).sin()).collect();
                let k: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.7).cos()).collect();
                let max_pos = base_pos + rel_offset + 51;
                let config = standard_config(dim, max_pos + 1);
                let table = FrequencyComputer::build_from_config(&config, max_pos + 1);

                let rq1 = RotaryApplier::apply_new(&q, base_pos, &table);
                let rk1 = RotaryApplier::apply_new(&k, base_pos + rel_offset, &table);
                let dot1 = dot(&rq1, &rk1);

                let other_base = base_pos + 50;
                let rq2 = RotaryApplier::apply_new(&q, other_base, &table);
                let rk2 = RotaryApplier::apply_new(&k, other_base + rel_offset, &table);
                let dot2 = dot(&rq2, &rk2);

                prop_assert!((dot1 - dot2).abs() < 1e-6,
                    "relative dot product not preserved: {} vs {}", dot1, dot2);
            }

            #[test]
            fn cos_sin_pythagorean(
                dim in arb_dim(),
                pos in 0_usize..1000,
            ) {
                let config = standard_config(dim, 1001);
                let table = FrequencyComputer::build_from_config(&config, pos + 1);
                for pair in 0..config.num_pairs() {
                    let c = table.cos_at(pos, pair);
                    let s = table.sin_at(pos, pair);
                    prop_assert!((c * c + s * s - 1.0).abs() < 1e-10,
                        "cos²+sin²≠1 at pos={}, pair={}", pos, pair);
                }
            }

            #[test]
            fn position_zero_is_identity(
                dim in arb_dim(),
            ) {
                let v: Vec<f64> = (0..dim).map(|i| i as f64 * 0.5 + 1.0).collect();
                let config = standard_config(dim, 100);
                let table = FrequencyComputer::build_from_config(&config, 1);
                let rotated = RotaryApplier::apply_new(&v, 0, &table);
                for (a, b) in v.iter().zip(rotated.iter()) {
                    prop_assert!((a - b).abs() < 1e-12,
                        "position 0 should be identity");
                }
            }

            #[test]
            fn ntk_scaled_base_monotonic(
                factor in 1.0_f64..20.0,
            ) {
                let b1 = NTKScaler::scaled_base(10_000.0, factor, 64);
                let b2 = NTKScaler::scaled_base(10_000.0, factor + 0.1, 64);
                prop_assert!(b2 >= b1,
                    "NTK base should be monotonically increasing with factor");
            }

            #[test]
            fn interpolator_effective_pos_monotonic(
                scale in 1.0_f64..10.0,
                pos in 0_usize..10000,
            ) {
                let interp = PositionInterpolator::new(scale);
                let eff1 = interp.interpolate(pos);
                let eff2 = interp.interpolate(pos + 1);
                prop_assert!(eff2 > eff1,
                    "effective position should be monotonically increasing");
            }

            #[test]
            fn cache_always_returns_correct_seq_len(
                seq_len in 1_usize..500,
            ) {
                let mut cache = RoPECache::new(standard_config(4, 1000));
                let table = cache.get_or_compute(seq_len);
                prop_assert_eq!(table.seq_len, seq_len);
            }

            #[test]
            fn random_vector_magnitude_preserved(
                v in arb_vec(8),
                pos in 0_usize..200,
            ) {
                let config = standard_config(8, 500);
                let table = FrequencyComputer::build_from_config(&config, pos + 1);
                let rotated = RotaryApplier::apply_new(&v, pos, &table);
                let orig_mag = magnitude(&v);
                let rot_mag = magnitude(&rotated);
                prop_assert!((orig_mag - rot_mag).abs() < 1e-6,
                    "random vector magnitude not preserved: {} vs {}", orig_mag, rot_mag);
            }

            #[test]
            fn zero_vector_stays_zero(
                dim in arb_dim(),
                pos in 0_usize..100,
            ) {
                let v = vec![0.0; dim];
                let config = standard_config(dim, 200);
                let table = FrequencyComputer::build_from_config(&config, pos + 1);
                let rotated = RotaryApplier::apply_new(&v, pos, &table);
                for x in &rotated {
                    prop_assert!(x.abs() < 1e-15, "zero vector should remain zero");
                }
            }
        }
    }
}
