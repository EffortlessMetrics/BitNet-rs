//! OpenCL-optimized position encoding library for transformer inference.
//!
//! # Overview
//!
//! Implements all major position encoding schemes used in modern LLMs:
//!
//! - **Sinusoidal** — original "Attention Is All You Need" fixed sin/cos PE
//! - **Learned** — lookup table of trained position embeddings
//! - **Rotary (RoPE)** — rotary position embeddings with configurable theta
//! - **ALiBi** — attention linear biases (no position embedding needed)
//! - **Relative** — T5-style relative position bucket encoding
//! - **DynamicNTK-RoPE** — NTK-aware RoPE with dynamic base scaling
//! - **YaRN-RoPE** — YaRN extension blending linear + NTK interpolation
//! - **PositionInterpolation** — linear scaling for context extension
//!
//! # CPU reference
//!
//! All encodings provide pure-CPU scalar reference implementations for
//! correctness testing and non-GPU environments. No OpenCL runtime required.
//!
//! # OpenCL kernel
//!
//! [`POSITION_ENCODING_CL`] contains OpenCL C source for sinusoidal and RoPE
//! generation, ready for GPU dispatch on Intel / AMD OpenCL devices.

use std::f32::consts::PI;
use std::fmt;
use std::time::{Duration, Instant};

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C source for sinusoidal and rotary position encoding generation.
///
/// The sinusoidal kernel computes `PE(pos, 2i) = sin(pos / 10000^(2i/d))`
/// and `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))` for all positions in a batch.
///
/// The RoPE kernel applies rotary embeddings in-place to Q/K tensors.
pub const POSITION_ENCODING_CL: &str = r#"
// Sinusoidal position encoding: writes [seq_len, dim] row-major
__kernel void sinusoidal_encoding(
    __global float* output,
    const int seq_len,
    const int dim)
{
    int pos = get_global_id(0);
    int i   = get_global_id(1);
    if (pos >= seq_len || i >= dim) return;

    int half_dim = dim / 2;
    int pair     = i / 2;
    float freq   = 1.0f / pow(10000.0f, (float)(2 * pair) / (float)dim);
    float angle  = (float)pos * freq;

    // Even index -> sin, odd index -> cos
    output[pos * dim + i] = (i % 2 == 0) ? sin(angle) : cos(angle);
}

// RoPE: rotate (x0, x1) pairs in-place
// data layout: [seq_len, num_heads, head_dim]
__kernel void rope_encoding(
    __global float* data,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float theta,
    const int offset)
{
    int pos  = get_global_id(0);
    int head = get_global_id(1);
    int pair = get_global_id(2);
    if (pos >= seq_len || head >= num_heads || pair >= head_dim / 2) return;

    float freq  = 1.0f / pow(theta, (float)(2 * pair) / (float)head_dim);
    float angle = (float)(pos + offset) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    int base = (pos * num_heads + head) * head_dim + 2 * pair;
    float x0 = data[base];
    float x1 = data[base + 1];
    data[base]     = x0 * cos_a - x1 * sin_a;
    data[base + 1] = x0 * sin_a + x1 * cos_a;
}
"#;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for position encodings.
#[derive(Debug, Clone)]
pub struct EncodingConfig {
    /// Maximum sequence length supported.
    pub max_seq_len: usize,
    /// Encoding dimension (must be even for sinusoidal / RoPE).
    pub dim: usize,
    /// Base frequency for RoPE variants (default: 10000.0).
    pub theta: f32,
    /// Scaling factor for position interpolation / NTK variants.
    pub scaling_factor: f32,
}

impl EncodingConfig {
    /// Create a new encoding config.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `dim` is zero or
    /// `max_seq_len` is zero.
    pub fn new(max_seq_len: usize, dim: usize) -> Result<Self> {
        if dim == 0 || max_seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "dim and max_seq_len must be > 0".into(),
            }
            .into());
        }
        Ok(Self { max_seq_len, dim, theta: 10_000.0, scaling_factor: 1.0 })
    }

    /// Set the RoPE theta base frequency.
    #[must_use]
    pub fn with_theta(mut self, theta: f32) -> Self {
        self.theta = theta;
        self
    }

    /// Set the scaling factor for interpolation variants.
    #[must_use]
    pub fn with_scaling_factor(mut self, factor: f32) -> Self {
        self.scaling_factor = factor;
        self
    }
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self { max_seq_len: 2048, dim: 64, theta: 10_000.0, scaling_factor: 1.0 }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Statistics for position encoding operations.
#[derive(Debug, Clone)]
pub struct EncodingStats {
    /// Number of cache hits (pre-computed table reuse).
    pub cache_hits: u64,
    /// Number of cache misses (required fresh computation).
    pub cache_misses: u64,
    /// Total computation time across all encode calls.
    pub computation_time: Duration,
}

impl EncodingStats {
    /// Create empty stats.
    pub fn new() -> Self {
        Self { cache_hits: 0, cache_misses: 0, computation_time: Duration::ZERO }
    }

    /// Cache hit rate as a fraction in [0, 1].
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else { self.cache_hits as f64 / total as f64 }
    }

    fn record_hit(&mut self) {
        self.cache_hits += 1;
    }

    fn record_miss(&mut self, elapsed: Duration) {
        self.cache_misses += 1;
        self.computation_time += elapsed;
    }
}

impl Default for EncodingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for EncodingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "hits={} misses={} hit_rate={:.1}% time={:.2}ms",
            self.cache_hits,
            self.cache_misses,
            self.hit_rate() * 100.0,
            self.computation_time.as_secs_f64() * 1000.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Sinusoidal Encoding
// ---------------------------------------------------------------------------

/// Original Transformer sinusoidal position encoding.
///
/// Generates fixed (non-learned) encodings where even indices use sine and
/// odd indices use cosine, with geometrically decreasing frequencies:
///
/// ```text
/// PE(pos, 2i)   = sin(pos / theta^(2i / dim))
/// PE(pos, 2i+1) = cos(pos / theta^(2i / dim))
/// ```
#[derive(Debug, Clone)]
pub struct SinusoidalEncoding {
    config: EncodingConfig,
    /// Cached encoding table `[max_seq_len, dim]` (lazily computed).
    cache: Option<Vec<f32>>,
    stats: EncodingStats,
}

impl SinusoidalEncoding {
    /// Create a new sinusoidal encoding (cache is not pre-computed).
    pub fn new(config: EncodingConfig) -> Self {
        Self { config, cache: None, stats: EncodingStats::new() }
    }

    /// Pre-compute the full encoding table up to `max_seq_len`.
    pub fn precompute(&mut self) {
        let start = Instant::now();
        let table =
            sinusoidal_table_ref(self.config.max_seq_len, self.config.dim, self.config.theta);
        self.cache = Some(table);
        self.stats.record_miss(start.elapsed());
    }

    /// Encode positions `[0, seq_len)` into `output` of shape
    /// `[seq_len, dim]`.
    ///
    /// Uses the cached table when available, otherwise computes on the fly.
    pub fn encode(&mut self, seq_len: usize, output: &mut [f32]) -> Result<()> {
        let expected = seq_len * self.config.dim;
        if output.len() < expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output length {} < seq_len({}) * dim({})",
                    output.len(),
                    seq_len,
                    self.config.dim,
                ),
            }
            .into());
        }
        if seq_len > self.config.max_seq_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "seq_len {} exceeds max_seq_len {}",
                    seq_len, self.config.max_seq_len,
                ),
            }
            .into());
        }

        if let Some(ref table) = self.cache {
            // Cache hit — copy from precomputed table
            output[..expected].copy_from_slice(&table[..expected]);
            self.stats.record_hit();
        } else {
            let start = Instant::now();
            sinusoidal_ref(seq_len, self.config.dim, self.config.theta, output);
            self.stats.record_miss(start.elapsed());
        }
        Ok(())
    }

    /// Read-only access to statistics.
    pub fn stats(&self) -> &EncodingStats {
        &self.stats
    }

    /// Read-only access to configuration.
    pub fn config(&self) -> &EncodingConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Learned Encoding
// ---------------------------------------------------------------------------

/// Learned position embeddings — a simple lookup table.
///
/// Each position `0..max_seq_len` has an independent `dim`-dimensional
/// vector that was learned during training.
#[derive(Debug, Clone)]
pub struct LearnedEncoding {
    config: EncodingConfig,
    /// Weight matrix `[max_seq_len, dim]` in row-major order.
    weight: Vec<f32>,
    stats: EncodingStats,
}

impl LearnedEncoding {
    /// Create a learned encoding from a pre-trained weight table.
    ///
    /// # Errors
    ///
    /// Returns an error if the weight length does not match
    /// `max_seq_len * dim`.
    pub fn new(config: EncodingConfig, weight: Vec<f32>) -> Result<Self> {
        let expected = config.max_seq_len * config.dim;
        if weight.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "weight length {} != max_seq_len({}) * dim({})",
                    weight.len(),
                    config.max_seq_len,
                    config.dim,
                ),
            }
            .into());
        }
        Ok(Self { config, weight, stats: EncodingStats::new() })
    }

    /// Look up position embeddings for positions `[0, seq_len)`.
    pub fn encode(&mut self, seq_len: usize, output: &mut [f32]) -> Result<()> {
        let expected = seq_len * self.config.dim;
        if output.len() < expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output length {} < seq_len({}) * dim({})",
                    output.len(),
                    seq_len,
                    self.config.dim,
                ),
            }
            .into());
        }
        if seq_len > self.config.max_seq_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "seq_len {} exceeds max_seq_len {}",
                    seq_len, self.config.max_seq_len,
                ),
            }
            .into());
        }
        let start = Instant::now();
        output[..expected].copy_from_slice(&self.weight[..expected]);
        self.stats.record_miss(start.elapsed());
        Ok(())
    }

    /// Look up embedding for a single arbitrary position.
    pub fn lookup(&self, position: usize, output: &mut [f32]) -> Result<()> {
        if position >= self.config.max_seq_len {
            return Err(
                KernelError::InvalidArguments {
                    reason: format!(
                        "position {} >= max_seq_len {}",
                        position, self.config.max_seq_len,
                    ),
                }
                .into(),
            );
        }
        if output.len() < self.config.dim {
            return Err(KernelError::InvalidArguments {
                reason: format!("output length {} < dim {}", output.len(), self.config.dim),
            }
            .into());
        }
        let start = position * self.config.dim;
        output[..self.config.dim].copy_from_slice(&self.weight[start..start + self.config.dim]);
        Ok(())
    }

    /// Read-only access to statistics.
    pub fn stats(&self) -> &EncodingStats {
        &self.stats
    }

    /// Read-only access to configuration.
    pub fn config(&self) -> &EncodingConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Rotary Encoding (RoPE)
// ---------------------------------------------------------------------------

/// Rotary Position Embedding (RoPE).
///
/// Applies rotation to pairs of dimensions in Q/K vectors. The rotation
/// angle depends on position and dimension index, producing the property
/// that the dot product of two rotated vectors depends only on relative
/// position.
///
/// The rotation for pair `i` at position `pos` uses angle:
/// ```text
/// angle = pos / theta^(2i / dim)
/// ```
#[derive(Debug, Clone)]
pub struct RotaryEncoding {
    config: EncodingConfig,
    /// Cached cos/sin table `[max_seq_len, dim]`: even=cos, odd=sin.
    cos_sin_cache: Option<Vec<f32>>,
    stats: EncodingStats,
}

impl RotaryEncoding {
    /// Create a new RoPE encoder.
    pub fn new(config: EncodingConfig) -> Self {
        Self { config, cos_sin_cache: None, stats: EncodingStats::new() }
    }

    /// Pre-compute the cos/sin frequency table.
    pub fn precompute(&mut self) {
        let start = Instant::now();
        let table = rope_cos_sin_table(self.config.max_seq_len, self.config.dim, self.config.theta);
        self.cos_sin_cache = Some(table);
        self.stats.record_miss(start.elapsed());
    }

    /// Apply RoPE in-place to `data` of shape `[seq_len, num_heads, head_dim]`.
    ///
    /// `offset` shifts all positions (for KV-cache continuation).
    pub fn apply(
        &mut self,
        data: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        offset: usize,
    ) -> Result<()> {
        if !head_dim.is_multiple_of(2) {
            return Err(KernelError::InvalidArguments {
                reason: "head_dim must be even for RoPE".into(),
            }
            .into());
        }
        let expected = seq_len * num_heads * head_dim;
        if data.len() < expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "data length {} < seq_len({}) * heads({}) * head_dim({})",
                    data.len(),
                    seq_len,
                    num_heads,
                    head_dim,
                ),
            }
            .into());
        }

        let start = Instant::now();
        rope_apply_ref(data, seq_len, num_heads, head_dim, self.config.theta, offset);
        self.stats.record_miss(start.elapsed());
        Ok(())
    }

    /// Get the cos/sin value for a given position and dimension pair.
    ///
    /// Returns `(cos(angle), sin(angle))` for the pair at index `pair_idx`.
    pub fn get_cos_sin(&self, position: usize, pair_idx: usize) -> (f32, f32) {
        let freq = 1.0 / self.config.theta.powf(2.0 * pair_idx as f32 / self.config.dim as f32);
        let angle = position as f32 * freq;
        (angle.cos(), angle.sin())
    }

    /// Read-only access to statistics.
    pub fn stats(&self) -> &EncodingStats {
        &self.stats
    }

    /// Read-only access to configuration.
    pub fn config(&self) -> &EncodingConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// ALiBi Encoding
// ---------------------------------------------------------------------------

/// Attention with Linear Biases (ALiBi).
///
/// Instead of adding position embeddings, ALiBi adds a position-dependent
/// bias directly to attention scores. Each head `h` gets a geometric slope
/// `m_h = 2^(-8h / n_heads)`, and the bias is `-m_h * |q_pos - k_pos|`.
///
/// This produces a linear decay in attention with distance.
#[derive(Debug, Clone)]
pub struct ALiBiEncoding {
    num_heads: usize,
    /// Per-head slopes `m_h`.
    slopes: Vec<f32>,
    stats: EncodingStats,
}

impl ALiBiEncoding {
    /// Create ALiBi encoding for the given number of attention heads.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_heads` is zero.
    pub fn new(num_heads: usize) -> Result<Self> {
        if num_heads == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "num_heads must be > 0".into() }.into()
            );
        }
        let slopes = alibi_slopes(num_heads);
        Ok(Self { num_heads, slopes, stats: EncodingStats::new() })
    }

    /// Compute the ALiBi bias matrix for attention scores.
    ///
    /// `output` has shape `[num_heads, q_len, kv_len]` and receives
    /// `-slope * |q_pos - k_pos|` for each head.
    pub fn compute_bias(&mut self, q_len: usize, kv_len: usize, output: &mut [f32]) -> Result<()> {
        let expected = self.num_heads * q_len * kv_len;
        if output.len() < expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output length {} < heads({}) * q_len({}) * kv_len({})",
                    output.len(),
                    self.num_heads,
                    q_len,
                    kv_len,
                ),
            }
            .into());
        }

        let start = Instant::now();
        alibi_bias_ref(&self.slopes, q_len, kv_len, output);
        self.stats.record_miss(start.elapsed());
        Ok(())
    }

    /// Get the slope for a specific head.
    pub fn slope(&self, head: usize) -> Option<f32> {
        self.slopes.get(head).copied()
    }

    /// Get all slopes.
    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }

    /// Read-only access to statistics.
    pub fn stats(&self) -> &EncodingStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Relative Encoding (T5-style)
// ---------------------------------------------------------------------------

/// T5-style relative position bucket encoding.
///
/// Maps relative position distances to a fixed number of buckets using a
/// piecewise function: small distances map linearly, larger distances map
/// logarithmically. This allows the model to generalize to unseen distances.
#[derive(Debug, Clone)]
pub struct RelativeEncoding {
    /// Number of buckets (typically 32).
    num_buckets: usize,
    /// Maximum distance before all larger distances share the last bucket.
    max_distance: usize,
    /// Whether the encoding is bidirectional (encoder) or unidirectional
    /// (decoder).
    bidirectional: bool,
    stats: EncodingStats,
}

impl RelativeEncoding {
    /// Create a new relative position encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_buckets` is zero.
    pub fn new(num_buckets: usize, max_distance: usize, bidirectional: bool) -> Result<Self> {
        if num_buckets == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "num_buckets must be > 0".into() }.into()
            );
        }
        Ok(Self { num_buckets, max_distance, bidirectional, stats: EncodingStats::new() })
    }

    /// Compute the relative position bucket for a given relative position.
    ///
    /// `relative_position` is `key_pos - query_pos` (may be negative for
    /// causal / bidirectional encodings).
    pub fn bucket(&self, relative_position: i32) -> usize {
        relative_position_bucket(
            relative_position,
            self.num_buckets,
            self.max_distance,
            self.bidirectional,
        )
    }

    /// Compute the full bucket matrix for `[q_len, kv_len]`.
    ///
    /// `output[q * kv_len + k]` receives the bucket index for position pair
    /// `(q, k)`.
    pub fn compute_buckets(
        &mut self,
        q_len: usize,
        kv_len: usize,
        output: &mut [usize],
    ) -> Result<()> {
        let expected = q_len * kv_len;
        if output.len() < expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output length {} < q_len({}) * kv_len({})",
                    output.len(),
                    q_len,
                    kv_len,
                ),
            }
            .into());
        }

        let start = Instant::now();
        for q in 0..q_len {
            for k in 0..kv_len {
                let rel = k as i32 - q as i32;
                output[q * kv_len + k] = self.bucket(rel);
            }
        }
        self.stats.record_miss(start.elapsed());
        Ok(())
    }

    /// Read-only access to configuration.
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    /// Maximum distance threshold.
    pub fn max_distance(&self) -> usize {
        self.max_distance
    }

    /// Read-only access to statistics.
    pub fn stats(&self) -> &EncodingStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Dynamic NTK-Aware RoPE
// ---------------------------------------------------------------------------

/// NTK-aware RoPE with dynamic base scaling for extended context.
///
/// When the sequence length exceeds the original training length, the base
/// theta is scaled up to preserve high-frequency information:
///
/// ```text
/// theta' = theta * (scaling_factor * seq_len / max_seq_len) ^ (dim / (dim - 2))
/// ```
///
/// This avoids the information loss of naïve position interpolation at high
/// frequencies.
#[derive(Debug, Clone)]
pub struct DynamicNTKRoPE {
    config: EncodingConfig,
    /// Original training context length.
    original_max_seq_len: usize,
    stats: EncodingStats,
}

impl DynamicNTKRoPE {
    /// Create a dynamic NTK-RoPE encoder.
    ///
    /// - `config` — encoding config with the *desired* extended max_seq_len
    /// - `original_max_seq_len` — the original training context length
    pub fn new(config: EncodingConfig, original_max_seq_len: usize) -> Self {
        Self { config, original_max_seq_len, stats: EncodingStats::new() }
    }

    /// Compute the dynamically scaled theta for a given sequence length.
    pub fn dynamic_theta(&self, seq_len: usize) -> f32 {
        if seq_len <= self.original_max_seq_len {
            return self.config.theta;
        }
        let dim = self.config.dim as f32;
        let ratio =
            (self.config.scaling_factor * seq_len as f32) / self.original_max_seq_len as f32;
        self.config.theta * ratio.powf(dim / (dim - 2.0))
    }

    /// Apply NTK-scaled RoPE in-place to `data`.
    pub fn apply(
        &mut self,
        data: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        offset: usize,
    ) -> Result<()> {
        if !head_dim.is_multiple_of(2) {
            return Err(KernelError::InvalidArguments {
                reason: "head_dim must be even for RoPE".into(),
            }
            .into());
        }
        let expected = seq_len * num_heads * head_dim;
        if data.len() < expected {
            return Err(KernelError::InvalidArguments {
                reason: format!("data length {} < seq_len * heads * head_dim", data.len()),
            }
            .into());
        }

        let start = Instant::now();
        let theta = self.dynamic_theta(seq_len + offset);
        rope_apply_ref(data, seq_len, num_heads, head_dim, theta, offset);
        self.stats.record_miss(start.elapsed());
        Ok(())
    }

    /// Read-only access to statistics.
    pub fn stats(&self) -> &EncodingStats {
        &self.stats
    }

    /// Read-only access to configuration.
    pub fn config(&self) -> &EncodingConfig {
        &self.config
    }

    /// Original training max sequence length.
    pub fn original_max_seq_len(&self) -> usize {
        self.original_max_seq_len
    }
}

// ---------------------------------------------------------------------------
// YaRN RoPE
// ---------------------------------------------------------------------------

/// YaRN (Yet another RoPE extensioN) position encoding.
///
/// Blends linear interpolation (for low frequencies) with NTK scaling (for
/// high frequencies) using a smooth ramp function, plus an attention scaling
/// factor to compensate for the entropy increase at longer sequences.
///
/// Each frequency dimension `i` gets an interpolation factor `alpha_i`:
/// - `alpha_i = 0` → pure NTK (high-frequency preservation)
/// - `alpha_i = 1` → pure linear interpolation (low-frequency scaling)
/// - `0 < alpha_i < 1` → smooth blend
#[derive(Debug, Clone)]
pub struct YaRNRoPE {
    config: EncodingConfig,
    /// Original training context length.
    original_max_seq_len: usize,
    /// Attention scaling factor (typically `0.1 * ln(s) + 1` where
    /// `s = extended / original`).
    attention_factor: f32,
    /// Low-frequency wavelength threshold (dimensions with wavelength
    /// above this use linear interpolation).
    beta_slow: f32,
    /// High-frequency wavelength threshold (dimensions with wavelength
    /// below this use NTK scaling).
    beta_fast: f32,
    stats: EncodingStats,
}

impl YaRNRoPE {
    /// Create a YaRN-RoPE encoder with default beta thresholds.
    pub fn new(config: EncodingConfig, original_max_seq_len: usize) -> Self {
        let scale = config.max_seq_len as f32 / original_max_seq_len as f32;
        let attention_factor = 0.1 * scale.ln() + 1.0;
        Self {
            config,
            original_max_seq_len,
            attention_factor,
            beta_slow: 2.0,
            beta_fast: 32.0,
            stats: EncodingStats::new(),
        }
    }

    /// Set custom beta thresholds.
    #[must_use]
    pub fn with_betas(mut self, beta_slow: f32, beta_fast: f32) -> Self {
        self.beta_slow = beta_slow;
        self.beta_fast = beta_fast;
        self
    }

    /// Override the attention scaling factor.
    #[must_use]
    pub fn with_attention_factor(mut self, factor: f32) -> Self {
        self.attention_factor = factor;
        self
    }

    /// Compute the per-dimension interpolation factor (alpha).
    ///
    /// Returns a vector of length `dim / 2` with values in [0, 1].
    pub fn compute_alphas(&self) -> Vec<f32> {
        let half_dim = self.config.dim / 2;
        let low_freq_wavelen = self.original_max_seq_len as f32 / self.beta_slow;
        let high_freq_wavelen = self.original_max_seq_len as f32 / self.beta_fast;

        (0..half_dim)
            .map(|i| {
                let freq = 1.0 / self.config.theta.powf(2.0 * i as f32 / self.config.dim as f32);
                let wavelen = 2.0 * PI / freq;

                if wavelen < high_freq_wavelen {
                    // High frequency: no interpolation (NTK)
                    0.0
                } else if wavelen > low_freq_wavelen {
                    // Low frequency: full linear interpolation
                    1.0
                } else {
                    // Smooth ramp between thresholds
                    let alpha =
                        (wavelen - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen);
                    // Clamp for numerical safety
                    alpha.clamp(0.0, 1.0)
                }
            })
            .collect()
    }

    /// Apply YaRN-RoPE in-place to `data`.
    pub fn apply(
        &mut self,
        data: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        offset: usize,
    ) -> Result<()> {
        if !head_dim.is_multiple_of(2) {
            return Err(KernelError::InvalidArguments {
                reason: "head_dim must be even for RoPE".into(),
            }
            .into());
        }
        let expected = seq_len * num_heads * head_dim;
        if data.len() < expected {
            return Err(KernelError::InvalidArguments {
                reason: format!("data length {} < expected {}", data.len(), expected),
            }
            .into());
        }

        let start = Instant::now();
        let scale = self.config.max_seq_len as f32 / self.original_max_seq_len as f32;
        let alphas = self.compute_alphas();

        for pos in 0..seq_len {
            for head in 0..num_heads {
                let base = (pos * num_heads + head) * head_dim;
                for pair in 0..head_dim / 2 {
                    let alpha = if pair < alphas.len() { alphas[pair] } else { 1.0 };

                    let freq = 1.0 / self.config.theta.powf(2.0 * pair as f32 / head_dim as f32);

                    // Blend between standard freq and interpolated freq
                    let effective_freq = freq * (1.0 - alpha) + (freq / scale) * alpha;
                    let angle = (pos + offset) as f32 * effective_freq;
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();

                    let idx = base + 2 * pair;
                    let x0 = data[idx];
                    let x1 = data[idx + 1];
                    data[idx] = x0 * cos_a - x1 * sin_a;
                    data[idx + 1] = x0 * sin_a + x1 * cos_a;
                }
            }
        }
        self.stats.record_miss(start.elapsed());
        Ok(())
    }

    /// Get the attention scaling factor.
    pub fn attention_factor(&self) -> f32 {
        self.attention_factor
    }

    /// Read-only access to statistics.
    pub fn stats(&self) -> &EncodingStats {
        &self.stats
    }

    /// Read-only access to configuration.
    pub fn config(&self) -> &EncodingConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Position Interpolation
// ---------------------------------------------------------------------------

/// Linear position interpolation for extending context length.
///
/// Scales all positions by `original / extended` so that a model trained
/// on `original_max_seq_len` can handle `extended_max_seq_len` by
/// compressing the position indices into the original range.
///
/// Applied on top of any base position encoding (sinusoidal or RoPE).
#[derive(Debug, Clone)]
pub struct PositionInterpolation {
    config: EncodingConfig,
    /// Original training context length.
    original_max_seq_len: usize,
    stats: EncodingStats,
}

impl PositionInterpolation {
    /// Create a position interpolation encoder.
    ///
    /// - `config.max_seq_len` is the *extended* target length.
    /// - `original_max_seq_len` is the model's original training length.
    pub fn new(config: EncodingConfig, original_max_seq_len: usize) -> Self {
        Self { config, original_max_seq_len, stats: EncodingStats::new() }
    }

    /// Compute the interpolation scaling factor.
    pub fn scale(&self) -> f32 {
        self.original_max_seq_len as f32 / self.config.max_seq_len as f32
    }

    /// Interpolate a position index to the compressed range.
    pub fn interpolate_position(&self, position: usize) -> f32 {
        position as f32 * self.scale()
    }

    /// Apply interpolated RoPE in-place to `data`.
    pub fn apply_rope(
        &mut self,
        data: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        offset: usize,
    ) -> Result<()> {
        if !head_dim.is_multiple_of(2) {
            return Err(KernelError::InvalidArguments {
                reason: "head_dim must be even for RoPE".into(),
            }
            .into());
        }
        let expected = seq_len * num_heads * head_dim;
        if data.len() < expected {
            return Err(KernelError::InvalidArguments {
                reason: format!("data length {} < expected {}", data.len(), expected),
            }
            .into());
        }

        let start = Instant::now();
        let s = self.scale();
        for pos in 0..seq_len {
            let scaled_pos = (pos + offset) as f32 * s;
            for head in 0..num_heads {
                let base = (pos * num_heads + head) * head_dim;
                for pair in 0..head_dim / 2 {
                    let freq = 1.0 / self.config.theta.powf(2.0 * pair as f32 / head_dim as f32);
                    let angle = scaled_pos * freq;
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();

                    let idx = base + 2 * pair;
                    let x0 = data[idx];
                    let x1 = data[idx + 1];
                    data[idx] = x0 * cos_a - x1 * sin_a;
                    data[idx + 1] = x0 * sin_a + x1 * cos_a;
                }
            }
        }
        self.stats.record_miss(start.elapsed());
        Ok(())
    }

    /// Read-only access to statistics.
    pub fn stats(&self) -> &EncodingStats {
        &self.stats
    }

    /// Read-only access to configuration.
    pub fn config(&self) -> &EncodingConfig {
        &self.config
    }

    /// Original training max sequence length.
    pub fn original_max_seq_len(&self) -> usize {
        self.original_max_seq_len
    }
}

// ===========================================================================
// CPU reference implementations
// ===========================================================================

/// Compute a full sinusoidal encoding table `[max_seq_len, dim]`.
fn sinusoidal_table_ref(max_seq_len: usize, dim: usize, theta: f32) -> Vec<f32> {
    let mut table = vec![0.0f32; max_seq_len * dim];
    sinusoidal_ref(max_seq_len, dim, theta, &mut table);
    table
}

/// Fill `output` with sinusoidal encoding for positions `[0, seq_len)`.
///
/// Layout: `output[pos * dim + i]` with even `i` → sin, odd `i` → cos.
fn sinusoidal_ref(seq_len: usize, dim: usize, theta: f32, output: &mut [f32]) {
    let half_dim = dim / 2;
    for pos in 0..seq_len {
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / dim as f32);
            let angle = pos as f32 * freq;
            output[pos * dim + 2 * i] = angle.sin();
            output[pos * dim + 2 * i + 1] = angle.cos();
        }
        // If dim is odd, the last element is sin of the last frequency
        if !dim.is_multiple_of(2) {
            let freq = 1.0 / theta.powf(2.0 * half_dim as f32 / dim as f32);
            let angle = pos as f32 * freq;
            output[pos * dim + dim - 1] = angle.sin();
        }
    }
}

/// Compute cos/sin table for RoPE: `[max_seq_len, dim]` with
/// `table[pos * dim + 2*i] = cos(angle)`, `table[pos * dim + 2*i+1] = sin(angle)`.
fn rope_cos_sin_table(max_seq_len: usize, dim: usize, theta: f32) -> Vec<f32> {
    let half_dim = dim / 2;
    let mut table = vec![0.0f32; max_seq_len * dim];
    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / dim as f32);
            let angle = pos as f32 * freq;
            table[pos * dim + 2 * i] = angle.cos();
            table[pos * dim + 2 * i + 1] = angle.sin();
        }
    }
    table
}

/// Apply RoPE in-place (scalar CPU reference).
///
/// Data layout: `[seq_len, num_heads, head_dim]`, pairs rotated independently.
fn rope_apply_ref(
    data: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    theta: f32,
    offset: usize,
) {
    let half_dim = head_dim / 2;
    for pos in 0..seq_len {
        for head in 0..num_heads {
            let base = (pos * num_heads + head) * head_dim;
            for pair in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * pair as f32 / head_dim as f32);
                let angle = (pos + offset) as f32 * freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                let idx = base + 2 * pair;
                let x0 = data[idx];
                let x1 = data[idx + 1];
                data[idx] = x0 * cos_a - x1 * sin_a;
                data[idx + 1] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

/// Compute ALiBi slopes for `num_heads` heads.
///
/// Uses the closest power-of-2 ratio: `2^(-8 / closest_pow2)` stepped
/// geometrically across heads. For non-power-of-2 head counts, interleaves
/// two geometric series.
fn alibi_slopes(num_heads: usize) -> Vec<f32> {
    fn closest_power_of_2(n: usize) -> usize {
        let mut p = 1;
        while p < n {
            p *= 2;
        }
        // Return the power-of-2 <= n
        if p == n { n } else { p / 2 }
    }

    let cp2 = closest_power_of_2(num_heads);
    let base = 2.0_f32.powf(-(8.0 / cp2 as f32));
    let mut slopes = Vec::with_capacity(num_heads);

    // First series: powers of base
    for i in 1..=cp2 {
        slopes.push(base.powi(i as i32));
    }

    // If num_heads is not a power of 2, add interleaved series
    if cp2 < num_heads {
        let extra_base = 2.0_f32.powf(-(8.0 / (2 * cp2) as f32));
        for i in 1..=(num_heads - cp2) {
            slopes.push(extra_base.powi((2 * i - 1) as i32));
        }
    }

    slopes
}

/// Fill `output` with ALiBi bias values.
///
/// Layout: `output[h * q_len * kv_len + q * kv_len + k]`.
fn alibi_bias_ref(slopes: &[f32], q_len: usize, kv_len: usize, output: &mut [f32]) {
    for (h, &slope) in slopes.iter().enumerate() {
        for q in 0..q_len {
            for k in 0..kv_len {
                // ALiBi uses causal-relative distance:
                // bias = -slope * |q - k|  (with q relative to end of kv)
                let q_abs = kv_len.saturating_sub(q_len) + q;
                let distance = if k <= q_abs { (q_abs - k) as f32 } else { (k - q_abs) as f32 };
                output[h * q_len * kv_len + q * kv_len + k] = -slope * distance;
            }
        }
    }
}

/// Compute relative position bucket (T5-style).
///
/// For bidirectional models, half the buckets handle negative distances.
/// Small distances use linear mapping; larger distances use logarithmic
/// mapping.
fn relative_position_bucket(
    relative_position: i32,
    num_buckets: usize,
    max_distance: usize,
    bidirectional: bool,
) -> usize {
    let mut rel = relative_position;
    let mut num_b = num_buckets as i32;
    let mut offset = 0i32;

    if bidirectional {
        num_b /= 2;
        if rel > 0 {
            offset = num_b;
        } else {
            rel = -rel;
        }
    } else {
        // Unidirectional: clamp to non-positive (causal)
        rel = (-rel).max(0);
    }

    // Now rel >= 0
    let rel = rel as usize;
    let half_buckets = num_b as usize / 2;

    // Small values: linear mapping
    if rel < half_buckets {
        return (offset as usize) + rel;
    }

    // Large values: logarithmic mapping
    let max_exact = half_buckets;
    let log_ratio =
        (rel as f32 / max_exact as f32).ln() / (max_distance as f32 / max_exact as f32).ln();
    let bucket = max_exact as f32 + log_ratio * (num_b as usize - max_exact) as f32;
    let bucket = bucket.min((num_b as usize - 1) as f32) as usize;

    (offset as usize) + bucket
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for floating-point comparisons
    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // -----------------------------------------------------------------------
    // EncodingConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let cfg = EncodingConfig::default();
        assert_eq!(cfg.max_seq_len, 2048);
        assert_eq!(cfg.dim, 64);
        assert!(approx_eq(cfg.theta, 10_000.0));
        assert!(approx_eq(cfg.scaling_factor, 1.0));
    }

    #[test]
    fn test_config_new_valid() {
        let cfg = EncodingConfig::new(512, 128).unwrap();
        assert_eq!(cfg.max_seq_len, 512);
        assert_eq!(cfg.dim, 128);
    }

    #[test]
    fn test_config_zero_dim_err() {
        assert!(EncodingConfig::new(512, 0).is_err());
    }

    #[test]
    fn test_config_zero_seq_len_err() {
        assert!(EncodingConfig::new(0, 64).is_err());
    }

    #[test]
    fn test_config_with_theta() {
        let cfg = EncodingConfig::new(512, 64).unwrap().with_theta(500_000.0);
        assert!(approx_eq(cfg.theta, 500_000.0));
    }

    #[test]
    fn test_config_with_scaling_factor() {
        let cfg = EncodingConfig::new(512, 64).unwrap().with_scaling_factor(4.0);
        assert!(approx_eq(cfg.scaling_factor, 4.0));
    }

    // -----------------------------------------------------------------------
    // EncodingStats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_initial() {
        let stats = EncodingStats::new();
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.computation_time, Duration::ZERO);
    }

    #[test]
    fn test_stats_hit_rate_empty() {
        let stats = EncodingStats::new();
        assert!(stats.hit_rate() == 0.0);
    }

    #[test]
    fn test_stats_hit_rate_all_hits() {
        let mut stats = EncodingStats::new();
        stats.record_hit();
        stats.record_hit();
        stats.record_hit();
        assert!((stats.hit_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_hit_rate_mixed() {
        let mut stats = EncodingStats::new();
        stats.record_hit();
        stats.record_miss(Duration::from_millis(1));
        assert!((stats.hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_display() {
        let stats = EncodingStats::new();
        let s = format!("{stats}");
        assert!(s.contains("hits=0"));
        assert!(s.contains("misses=0"));
    }

    // -----------------------------------------------------------------------
    // Sinusoidal encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sinusoidal_position_zero() {
        let cfg = EncodingConfig::new(16, 4).unwrap();
        let mut enc = SinusoidalEncoding::new(cfg);
        let mut out = vec![0.0; 4];
        enc.encode(1, &mut out).unwrap();
        // pos=0 → all angles are 0 → sin(0)=0, cos(0)=1
        assert!(approx_eq(out[0], 0.0)); // sin(0)
        assert!(approx_eq(out[1], 1.0)); // cos(0)
        assert!(approx_eq(out[2], 0.0)); // sin(0)
        assert!(approx_eq(out[3], 1.0)); // cos(0)
    }

    #[test]
    fn test_sinusoidal_correct_frequencies() {
        let dim = 8;
        let cfg = EncodingConfig::new(64, dim).unwrap();
        let mut enc = SinusoidalEncoding::new(cfg);
        let mut out = vec![0.0; 64 * dim];
        enc.encode(64, &mut out).unwrap();

        // Check that different dimension pairs have different frequencies
        // by comparing values at position 1
        let row1 = &out[dim..2 * dim];
        // First pair (i=0) uses freq = 1/10000^(0/8) = 1.0
        let angle0: f32 = 1.0 * 1.0; // pos * freq
        assert!(approx_eq(row1[0], angle0.sin()));
        assert!(approx_eq(row1[1], angle0.cos()));

        // Second pair (i=1) uses freq = 1/10000^(2/8) = 1/10
        let freq1 = 1.0 / 10_000.0_f32.powf(2.0 / 8.0);
        let angle1 = 1.0 * freq1;
        assert!(approx_eq(row1[2], angle1.sin()));
        assert!(approx_eq(row1[3], angle1.cos()));
    }

    #[test]
    fn test_sinusoidal_different_positions() {
        let dim = 4;
        let cfg = EncodingConfig::new(16, dim).unwrap();
        let mut enc = SinusoidalEncoding::new(cfg);
        let mut out = vec![0.0; 16 * dim];
        enc.encode(16, &mut out).unwrap();

        // Positions 0 and 1 should differ
        let row0 = &out[0..dim];
        let row1 = &out[dim..2 * dim];
        assert!(!row0.iter().zip(row1.iter()).all(|(a, b)| approx_eq(*a, *b)));
    }

    #[test]
    fn test_sinusoidal_orthogonality() {
        // The dot product between different positions should vary
        // (not perfectly orthogonal, but distinct positions should not be
        // identical)
        let dim = 64;
        let cfg = EncodingConfig::new(128, dim).unwrap();
        let mut enc = SinusoidalEncoding::new(cfg);
        let mut out = vec![0.0; 128 * dim];
        enc.encode(128, &mut out).unwrap();

        // Self-dot-product at position 0
        let self_dot: f32 = (0..dim).map(|i| out[i] * out[i]).sum();

        // Cross-dot-product between positions 0 and 64
        let cross_dot: f32 = (0..dim).map(|i| out[i] * out[64 * dim + i]).sum();

        // Self-dot should be larger than cross-dot for distant positions
        assert!(self_dot > cross_dot.abs());
    }

    #[test]
    fn test_sinusoidal_with_cache() {
        let cfg = EncodingConfig::new(32, 8).unwrap();
        let mut enc = SinusoidalEncoding::new(cfg.clone());
        enc.precompute();

        let mut cached_out = vec![0.0; 32 * 8];
        enc.encode(32, &mut cached_out).unwrap();

        let mut fresh = SinusoidalEncoding::new(cfg);
        let mut fresh_out = vec![0.0; 32 * 8];
        fresh.encode(32, &mut fresh_out).unwrap();

        for (a, b) in cached_out.iter().zip(fresh_out.iter()) {
            assert!(approx_eq(*a, *b), "cached vs fresh mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_sinusoidal_cache_stats() {
        let cfg = EncodingConfig::new(16, 4).unwrap();
        let mut enc = SinusoidalEncoding::new(cfg);
        enc.precompute();

        let mut out = vec![0.0; 16 * 4];
        enc.encode(16, &mut out).unwrap();
        assert_eq!(enc.stats().cache_hits, 1);
        enc.encode(16, &mut out).unwrap();
        assert_eq!(enc.stats().cache_hits, 2);
    }

    #[test]
    fn test_sinusoidal_output_too_small() {
        let cfg = EncodingConfig::new(16, 4).unwrap();
        let mut enc = SinusoidalEncoding::new(cfg);
        let mut out = vec![0.0; 2]; // too small for 1 * 4
        assert!(enc.encode(1, &mut out).is_err());
    }

    #[test]
    fn test_sinusoidal_seq_exceeds_max() {
        let cfg = EncodingConfig::new(8, 4).unwrap();
        let mut enc = SinusoidalEncoding::new(cfg);
        let mut out = vec![0.0; 100];
        assert!(enc.encode(10, &mut out).is_err());
    }

    #[test]
    fn test_sinusoidal_dim_2() {
        let cfg = EncodingConfig::new(4, 2).unwrap();
        let mut enc = SinusoidalEncoding::new(cfg);
        let mut out = vec![0.0; 4 * 2];
        enc.encode(4, &mut out).unwrap();

        // pos=1, dim=2: angle = 1 * 1.0 = 1.0
        assert!(approx_eq(out[2], 1.0_f32.sin()));
        assert!(approx_eq(out[3], 1.0_f32.cos()));
    }

    #[test]
    fn test_sinusoidal_values_bounded() {
        let cfg = EncodingConfig::new(100, 16).unwrap();
        let mut enc = SinusoidalEncoding::new(cfg);
        let mut out = vec![0.0; 100 * 16];
        enc.encode(100, &mut out).unwrap();
        for &v in &out {
            assert!((-1.0..=1.0).contains(&v), "value {v} out of [-1, 1]");
        }
    }

    #[test]
    fn test_sinusoidal_config_accessor() {
        let cfg = EncodingConfig::new(128, 32).unwrap();
        let enc = SinusoidalEncoding::new(cfg);
        assert_eq!(enc.config().dim, 32);
        assert_eq!(enc.config().max_seq_len, 128);
    }

    // -----------------------------------------------------------------------
    // Learned encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_learned_lookup_correctness() {
        let dim = 4;
        let max_seq = 8;
        let cfg = EncodingConfig::new(max_seq, dim).unwrap();
        let weight: Vec<f32> = (0..max_seq * dim).map(|i| i as f32 * 0.1).collect();
        let mut enc = LearnedEncoding::new(cfg, weight.clone()).unwrap();

        let mut out = vec![0.0; 3 * dim];
        enc.encode(3, &mut out).unwrap();

        // First 3 positions should match weight directly
        for i in 0..3 * dim {
            assert!(approx_eq(out[i], weight[i]), "mismatch at {i}: {} vs {}", out[i], weight[i]);
        }
    }

    #[test]
    fn test_learned_single_lookup() {
        let dim = 4;
        let cfg = EncodingConfig::new(8, dim).unwrap();
        let weight: Vec<f32> = (0..8 * dim).map(|i| i as f32).collect();
        let enc = LearnedEncoding::new(cfg, weight).unwrap();

        let mut out = vec![0.0; dim];
        enc.lookup(3, &mut out).unwrap();
        // Position 3 starts at index 12
        assert!(approx_eq(out[0], 12.0));
        assert!(approx_eq(out[1], 13.0));
        assert!(approx_eq(out[2], 14.0));
        assert!(approx_eq(out[3], 15.0));
    }

    #[test]
    fn test_learned_weight_mismatch_err() {
        let cfg = EncodingConfig::new(8, 4).unwrap();
        let weight = vec![0.0; 10]; // wrong size
        assert!(LearnedEncoding::new(cfg, weight).is_err());
    }

    #[test]
    fn test_learned_seq_exceeds_max() {
        let cfg = EncodingConfig::new(4, 2).unwrap();
        let weight = vec![0.0; 8];
        let mut enc = LearnedEncoding::new(cfg, weight).unwrap();
        let mut out = vec![0.0; 20];
        assert!(enc.encode(5, &mut out).is_err());
    }

    #[test]
    fn test_learned_lookup_out_of_range() {
        let cfg = EncodingConfig::new(4, 2).unwrap();
        let weight = vec![0.0; 8];
        let enc = LearnedEncoding::new(cfg, weight).unwrap();
        let mut out = vec![0.0; 2];
        assert!(enc.lookup(4, &mut out).is_err());
    }

    #[test]
    fn test_learned_output_too_small() {
        let cfg = EncodingConfig::new(4, 4).unwrap();
        let weight = vec![0.0; 16];
        let mut enc = LearnedEncoding::new(cfg, weight).unwrap();
        let mut out = vec![0.0; 3]; // too small for 1 position
        assert!(enc.encode(1, &mut out).is_err());
    }

    #[test]
    fn test_learned_config_accessor() {
        let cfg = EncodingConfig::new(16, 8).unwrap();
        let weight = vec![0.0; 128];
        let enc = LearnedEncoding::new(cfg, weight).unwrap();
        assert_eq!(enc.config().dim, 8);
    }

    #[test]
    fn test_learned_lookup_output_too_small() {
        let cfg = EncodingConfig::new(4, 4).unwrap();
        let weight = vec![0.0; 16];
        let enc = LearnedEncoding::new(cfg, weight).unwrap();
        let mut out = vec![0.0; 2]; // too small
        assert!(enc.lookup(0, &mut out).is_err());
    }

    // -----------------------------------------------------------------------
    // Rotary encoding (RoPE) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rope_identity_at_position_zero() {
        // At position 0, all angles are 0 → cos=1, sin=0 → no rotation
        let cfg = EncodingConfig::new(16, 4).unwrap();
        let mut rope = RotaryEncoding::new(cfg);
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut data = original.clone();
        rope.apply(&mut data, 1, 1, 4, 0).unwrap();
        for (a, b) in data.iter().zip(original.iter()) {
            assert!(approx_eq(*a, *b), "pos=0 should be identity: {a} vs {b}");
        }
    }

    #[test]
    fn test_rope_relative_position_property() {
        // RoPE's key property: dot(RoPE(q, pos_q), RoPE(k, pos_k)) depends
        // only on (pos_q - pos_k).
        let dim = 4;
        let cfg = EncodingConfig::new(128, dim).unwrap();

        // Create two vectors
        let q = vec![1.0, 0.5, -0.3, 0.7];
        let k = vec![0.2, -0.4, 0.6, 0.1];

        // Apply RoPE at positions (10, 5) → relative = 5
        let mut q1 = q.clone();
        let mut k1 = k.clone();
        RotaryEncoding::new(cfg.clone()).apply(&mut q1, 1, 1, dim, 10).unwrap();
        RotaryEncoding::new(cfg.clone()).apply(&mut k1, 1, 1, dim, 5).unwrap();
        let dot1: f32 = q1.iter().zip(k1.iter()).map(|(a, b)| a * b).sum();

        // Apply RoPE at positions (20, 15) → relative = 5
        let mut q2 = q.clone();
        let mut k2 = k.clone();
        RotaryEncoding::new(cfg.clone()).apply(&mut q2, 1, 1, dim, 20).unwrap();
        RotaryEncoding::new(cfg).apply(&mut k2, 1, 1, dim, 15).unwrap();
        let dot2: f32 = q2.iter().zip(k2.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (dot1 - dot2).abs() < 1e-4,
            "RoPE relative position property violated: {dot1} vs {dot2}",
        );
    }

    #[test]
    fn test_rope_preserves_norm() {
        // Rotation should preserve the L2 norm of each pair
        let dim = 8;
        let cfg = EncodingConfig::new(64, dim).unwrap();
        let mut rope = RotaryEncoding::new(cfg);
        let data_orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = data_orig.clone();
        rope.apply(&mut data, 1, 1, dim, 7).unwrap();

        // Check norm of each pair
        for pair in 0..dim / 2 {
            let orig_norm = (data_orig[2 * pair].powi(2) + data_orig[2 * pair + 1].powi(2)).sqrt();
            let new_norm = (data[2 * pair].powi(2) + data[2 * pair + 1].powi(2)).sqrt();
            assert!(
                (orig_norm - new_norm).abs() < 1e-5,
                "pair {pair}: norm changed from {orig_norm} to {new_norm}",
            );
        }
    }

    #[test]
    fn test_rope_multi_head() {
        let head_dim = 4;
        let num_heads = 2;
        let cfg = EncodingConfig::new(16, head_dim).unwrap();
        let mut rope = RotaryEncoding::new(cfg);
        let mut data = vec![1.0; num_heads * head_dim];
        rope.apply(&mut data, 1, num_heads, head_dim, 3).unwrap();

        // Both heads at same position should get same rotation
        let head0 = &data[0..head_dim];
        let head1 = &data[head_dim..2 * head_dim];
        for (a, b) in head0.iter().zip(head1.iter()) {
            assert!(approx_eq(*a, *b), "heads should match: {a} vs {b}");
        }
    }

    #[test]
    fn test_rope_odd_head_dim_err() {
        let cfg = EncodingConfig::new(16, 4).unwrap();
        let mut rope = RotaryEncoding::new(cfg);
        let mut data = vec![0.0; 3];
        assert!(rope.apply(&mut data, 1, 1, 3, 0).is_err());
    }

    #[test]
    fn test_rope_data_too_small_err() {
        let cfg = EncodingConfig::new(16, 4).unwrap();
        let mut rope = RotaryEncoding::new(cfg);
        let mut data = vec![0.0; 2]; // need 4
        assert!(rope.apply(&mut data, 1, 1, 4, 0).is_err());
    }

    #[test]
    fn test_rope_get_cos_sin() {
        let cfg = EncodingConfig::new(16, 4).unwrap();
        let rope = RotaryEncoding::new(cfg);
        let (cos_val, sin_val) = rope.get_cos_sin(0, 0);
        assert!(approx_eq(cos_val, 1.0));
        assert!(approx_eq(sin_val, 0.0));
    }

    #[test]
    fn test_rope_precompute() {
        let cfg = EncodingConfig::new(16, 4).unwrap();
        let mut rope = RotaryEncoding::new(cfg);
        rope.precompute();
        assert!(rope.cos_sin_cache.is_some());
        // Stats should record the precomputation as a miss
        assert_eq!(rope.stats().cache_misses, 1);
    }

    #[test]
    fn test_rope_offset() {
        let dim = 4;
        let cfg = EncodingConfig::new(64, dim).unwrap();

        // Apply at position 0 with offset=5 should equal position 5 with
        // offset=0
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let mut d1 = input.clone();
        RotaryEncoding::new(cfg.clone()).apply(&mut d1, 1, 1, dim, 5).unwrap();

        // Position 5 with offset=0: encode 1 position starting at 5
        // We need to simulate this by applying with seq_len=6 and taking
        // position 5. Use single-position with offset instead.
        let mut d2 = input.clone();
        rope_apply_ref(&mut d2, 1, 1, dim, 10_000.0, 5);

        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!(approx_eq(*a, *b), "offset equivalence: {a} vs {b}");
        }
    }

    #[test]
    fn test_rope_with_custom_theta() {
        // With very large theta, higher-frequency pairs rotate very slowly
        // Pair 0 uses freq = 1/theta^(0/dim) = 1.0 regardless of theta,
        // so we check pair 1 where freq = 1/theta^(2/4) = 1/sqrt(theta).
        let cfg = EncodingConfig::new(16, 4).unwrap().with_theta(500_000.0);
        let mut rope = RotaryEncoding::new(cfg);
        let mut data = vec![0.0, 0.0, 1.0, 0.0];
        rope.apply(&mut data, 1, 1, 4, 1).unwrap();
        // Pair 1: freq = 1/sqrt(500000) ≈ 0.00141, angle ≈ 0.00141
        // cos(0.00141) ≈ 1.0 → data[2] ≈ 1.0
        assert!(
            (data[2] - 1.0).abs() < 0.001,
            "high theta should produce near-identity rotation: {}",
            data[2],
        );
    }

    // -----------------------------------------------------------------------
    // ALiBi encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_alibi_slopes_power_of_2() {
        let enc = ALiBiEncoding::new(8).unwrap();
        let slopes = enc.slopes();
        assert_eq!(slopes.len(), 8);
        // Each slope should be a power of 2^(-1)
        // For 8 heads: base = 2^(-8/8) = 2^(-1) = 0.5
        assert!(approx_eq(slopes[0], 0.5));
        assert!(approx_eq(slopes[1], 0.25));
        assert!(approx_eq(slopes[2], 0.125));
    }

    #[test]
    fn test_alibi_slopes_non_power_of_2() {
        let enc = ALiBiEncoding::new(6).unwrap();
        let slopes = enc.slopes();
        assert_eq!(slopes.len(), 6);
        // Should still produce valid positive slopes
        for &s in slopes {
            assert!(s > 0.0, "slope should be positive: {s}");
            assert!(s <= 1.0, "slope should be <= 1: {s}");
        }
    }

    #[test]
    fn test_alibi_linear_decay() {
        // For 1 head, check that bias increases linearly with distance
        let q_len = 4;
        let kv_len = 4;
        let mut bias = vec![0.0; q_len * kv_len];
        let mut enc = ALiBiEncoding::new(1).unwrap();
        enc.compute_bias(q_len, kv_len, &mut bias).unwrap();

        // Check diagonal (distance = 0) → bias = 0
        for i in 0..q_len {
            assert!(approx_eq(bias[i * kv_len + i], 0.0), "diagonal should be 0");
        }

        // Check that bias decreases with distance from diagonal
        // bias[q=3, k=2] should be > bias[q=3, k=1] (less negative)
        let b32 = bias[3 * kv_len + 2]; // distance 1
        let b31 = bias[3 * kv_len + 1]; // distance 2
        assert!(b32 > b31, "bias should decrease with distance: d=1 ({b32}) vs d=2 ({b31})");

        // Verify linearity: difference should be constant
        let diff1 = b32 - bias[3 * kv_len + 3]; // distance 1 - distance 0
        let diff2 = b31 - b32; // distance 2 - distance 1
        assert!(approx_eq(diff1, diff2), "decay should be linear: {diff1} vs {diff2}");
    }

    #[test]
    fn test_alibi_zero_heads_err() {
        assert!(ALiBiEncoding::new(0).is_err());
    }

    #[test]
    fn test_alibi_output_too_small() {
        let mut enc = ALiBiEncoding::new(2).unwrap();
        let mut out = vec![0.0; 3]; // need 2 * 2 * 2 = 8
        assert!(enc.compute_bias(2, 2, &mut out).is_err());
    }

    #[test]
    fn test_alibi_single_position() {
        let mut enc = ALiBiEncoding::new(1).unwrap();
        let mut bias = vec![0.0; 1];
        enc.compute_bias(1, 1, &mut bias).unwrap();
        assert!(approx_eq(bias[0], 0.0)); // distance = 0
    }

    #[test]
    fn test_alibi_slope_accessor() {
        let enc = ALiBiEncoding::new(4).unwrap();
        assert!(enc.slope(0).is_some());
        assert!(enc.slope(3).is_some());
        assert!(enc.slope(4).is_none());
    }

    #[test]
    fn test_alibi_slopes_monotonically_decreasing_for_pow2() {
        let enc = ALiBiEncoding::new(8).unwrap();
        let slopes = enc.slopes();
        for i in 1..slopes.len() {
            assert!(
                slopes[i] < slopes[i - 1],
                "slopes should decrease: {} >= {}",
                slopes[i],
                slopes[i - 1],
            );
        }
    }

    // -----------------------------------------------------------------------
    // Relative encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_relative_bucket_zero_distance() {
        let enc = RelativeEncoding::new(32, 128, true).unwrap();
        let b = enc.bucket(0);
        // Distance 0 should map to bucket 0 (or offset bucket for positive)
        assert!(b < enc.num_buckets());
    }

    #[test]
    fn test_relative_bucket_positive() {
        let enc = RelativeEncoding::new(32, 128, true).unwrap();
        let b1 = enc.bucket(1);
        let b2 = enc.bucket(2);
        // Both should be in the "positive" half (buckets >= 16 for
        // bidirectional with 32 buckets)
        assert!(b1 >= 16);
        assert!(b2 >= 16);
    }

    #[test]
    fn test_relative_bucket_negative() {
        let enc = RelativeEncoding::new(32, 128, true).unwrap();
        let b = enc.bucket(-1);
        // Negative distances should be in first half (buckets < 16)
        assert!(b < 16);
    }

    #[test]
    fn test_relative_bucket_symmetry_bidirectional() {
        let enc = RelativeEncoding::new(32, 128, true).unwrap();
        // For bidirectional, positive and negative distances should map to
        // different halves but with similar bucket indices within their half
        let bp = enc.bucket(5);
        let bn = enc.bucket(-5);
        // Both should be the same distance from their half's start
        assert_eq!(bp - 16, bn);
    }

    #[test]
    fn test_relative_bucket_unidirectional() {
        let enc = RelativeEncoding::new(32, 128, false).unwrap();
        let b0 = enc.bucket(0);
        let b_neg = enc.bucket(-5);
        // Unidirectional clamps positive to 0
        assert_eq!(b0, 0);
        assert!(b_neg > 0);
    }

    #[test]
    fn test_relative_bucket_large_distance() {
        let enc = RelativeEncoding::new(32, 128, true).unwrap();
        let b = enc.bucket(1000);
        // Should be clamped to max bucket in the positive half
        assert!(b < enc.num_buckets());
    }

    #[test]
    fn test_relative_compute_buckets() {
        let mut enc = RelativeEncoding::new(32, 128, true).unwrap();
        let q_len = 3;
        let kv_len = 3;
        let mut buckets = vec![0usize; q_len * kv_len];
        enc.compute_buckets(q_len, kv_len, &mut buckets).unwrap();

        // Diagonal (distance = 0)
        for i in 0..q_len {
            let b = buckets[i * kv_len + i];
            assert!(b < enc.num_buckets());
        }
    }

    #[test]
    fn test_relative_buckets_output_too_small() {
        let mut enc = RelativeEncoding::new(32, 128, true).unwrap();
        let mut out = vec![0usize; 3]; // need 2 * 2 = 4
        assert!(enc.compute_buckets(2, 2, &mut out).is_err());
    }

    #[test]
    fn test_relative_zero_buckets_err() {
        assert!(RelativeEncoding::new(0, 128, true).is_err());
    }

    #[test]
    fn test_relative_bucket_monotonic() {
        // For increasing positive distances, buckets should be non-decreasing
        let enc = RelativeEncoding::new(32, 128, true).unwrap();
        let mut prev = enc.bucket(1);
        for d in 2..50 {
            let cur = enc.bucket(d);
            assert!(cur >= prev, "bucket should be non-decreasing: d={d}, prev={prev}, cur={cur}");
            prev = cur;
        }
    }

    #[test]
    fn test_relative_max_distance_accessor() {
        let enc = RelativeEncoding::new(32, 256, true).unwrap();
        assert_eq!(enc.max_distance(), 256);
    }

    // -----------------------------------------------------------------------
    // Dynamic NTK-RoPE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ntk_theta_within_training_length() {
        let cfg = EncodingConfig::new(4096, 64).unwrap();
        let ntk = DynamicNTKRoPE::new(cfg, 2048);
        // seq_len=1024 <= original=2048 → theta unchanged
        assert!(approx_eq(ntk.dynamic_theta(1024), 10_000.0));
    }

    #[test]
    fn test_ntk_theta_extended() {
        let cfg = EncodingConfig::new(8192, 64).unwrap();
        let ntk = DynamicNTKRoPE::new(cfg, 2048);
        // seq_len=4096 > original=2048 → theta should increase
        let theta = ntk.dynamic_theta(4096);
        assert!(theta > 10_000.0, "dynamic theta should increase: {theta}");
    }

    #[test]
    fn test_ntk_theta_scales_with_length() {
        let cfg = EncodingConfig::new(16384, 64).unwrap();
        let ntk = DynamicNTKRoPE::new(cfg, 2048);
        let t1 = ntk.dynamic_theta(4096);
        let t2 = ntk.dynamic_theta(8192);
        assert!(t2 > t1, "theta should increase with seq_len: {t1} vs {t2}");
    }

    #[test]
    fn test_ntk_apply() {
        let cfg = EncodingConfig::new(4096, 4).unwrap();
        let mut ntk = DynamicNTKRoPE::new(cfg, 2048);
        let mut data = vec![1.0, 0.0, 1.0, 0.0];
        ntk.apply(&mut data, 1, 1, 4, 0).unwrap();
        // At position 0, should be identity regardless of theta
        assert!(approx_eq(data[0], 1.0));
        assert!(approx_eq(data[1], 0.0));
    }

    #[test]
    fn test_ntk_apply_odd_dim_err() {
        let cfg = EncodingConfig::new(4096, 4).unwrap();
        let mut ntk = DynamicNTKRoPE::new(cfg, 2048);
        let mut data = vec![0.0; 3];
        assert!(ntk.apply(&mut data, 1, 1, 3, 0).is_err());
    }

    #[test]
    fn test_ntk_original_max_seq_accessor() {
        let cfg = EncodingConfig::new(8192, 64).unwrap();
        let ntk = DynamicNTKRoPE::new(cfg, 2048);
        assert_eq!(ntk.original_max_seq_len(), 2048);
    }

    #[test]
    fn test_ntk_preserves_norm() {
        let cfg = EncodingConfig::new(8192, 4).unwrap();
        let mut ntk = DynamicNTKRoPE::new(cfg, 2048);
        let orig = vec![3.0, 4.0, 1.0, 2.0];
        let mut data = orig.clone();
        ntk.apply(&mut data, 1, 1, 4, 100).unwrap();

        for pair in 0..2 {
            let n1 = (orig[2 * pair].powi(2) + orig[2 * pair + 1].powi(2)).sqrt();
            let n2 = (data[2 * pair].powi(2) + data[2 * pair + 1].powi(2)).sqrt();
            assert!((n1 - n2).abs() < 1e-4, "NTK-RoPE should preserve norm: {n1} vs {n2}");
        }
    }

    // -----------------------------------------------------------------------
    // YaRN-RoPE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_yarn_alphas_boundary() {
        let cfg = EncodingConfig::new(8192, 64).unwrap();
        let yarn = YaRNRoPE::new(cfg, 2048);
        let alphas = yarn.compute_alphas();
        assert_eq!(alphas.len(), 32);
        for &a in &alphas {
            assert!((0.0..=1.0).contains(&a), "alpha should be in [0, 1]: {a}");
        }
    }

    #[test]
    fn test_yarn_high_freq_no_interpolation() {
        // Very high frequency dimensions (first pair) should have alpha ≈ 0
        let cfg = EncodingConfig::new(8192, 64).unwrap();
        let yarn = YaRNRoPE::new(cfg, 2048);
        let alphas = yarn.compute_alphas();
        // First pair has highest frequency, should be 0 (NTK mode)
        assert!(alphas[0] < 0.5, "high-freq alpha should be near 0: {}", alphas[0]);
    }

    #[test]
    fn test_yarn_low_freq_full_interpolation() {
        // Very low frequency dimensions (last pairs) should have alpha ≈ 1
        let cfg = EncodingConfig::new(8192, 64).unwrap();
        let yarn = YaRNRoPE::new(cfg, 2048);
        let alphas = yarn.compute_alphas();
        let last = *alphas.last().unwrap();
        assert!(last > 0.5, "low-freq alpha should be near 1: {last}");
    }

    #[test]
    fn test_yarn_attention_factor() {
        let cfg = EncodingConfig::new(8192, 64).unwrap();
        let yarn = YaRNRoPE::new(cfg, 2048);
        let af = yarn.attention_factor();
        // For scale=4: 0.1 * ln(4) + 1 ≈ 1.139
        assert!(af > 1.0, "attention factor should be > 1: {af}");
    }

    #[test]
    fn test_yarn_custom_betas() {
        let cfg = EncodingConfig::new(8192, 64).unwrap();
        let yarn = YaRNRoPE::new(cfg, 2048).with_betas(1.0, 64.0);
        let alphas = yarn.compute_alphas();
        // Different betas should change the interpolation profile
        assert!(!alphas.is_empty());
    }

    #[test]
    fn test_yarn_apply_identity_at_pos_zero() {
        let cfg = EncodingConfig::new(8192, 4).unwrap();
        let mut yarn = YaRNRoPE::new(cfg, 2048);
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let mut data = orig.clone();
        yarn.apply(&mut data, 1, 1, 4, 0).unwrap();
        for (a, b) in data.iter().zip(orig.iter()) {
            assert!(approx_eq(*a, *b), "pos=0 should be identity: {a} vs {b}");
        }
    }

    #[test]
    fn test_yarn_apply_preserves_norm() {
        let cfg = EncodingConfig::new(8192, 4).unwrap();
        let mut yarn = YaRNRoPE::new(cfg, 2048);
        let orig = vec![3.0, 4.0, 1.0, 2.0];
        let mut data = orig.clone();
        yarn.apply(&mut data, 1, 1, 4, 10).unwrap();
        for pair in 0..2 {
            let n1 = (orig[2 * pair].powi(2) + orig[2 * pair + 1].powi(2)).sqrt();
            let n2 = (data[2 * pair].powi(2) + data[2 * pair + 1].powi(2)).sqrt();
            assert!((n1 - n2).abs() < 1e-4, "YaRN should preserve norm: {n1} vs {n2}");
        }
    }

    #[test]
    fn test_yarn_apply_odd_dim_err() {
        let cfg = EncodingConfig::new(8192, 4).unwrap();
        let mut yarn = YaRNRoPE::new(cfg, 2048);
        let mut data = vec![0.0; 3];
        assert!(yarn.apply(&mut data, 1, 1, 3, 0).is_err());
    }

    #[test]
    fn test_yarn_with_attention_factor() {
        let cfg = EncodingConfig::new(8192, 64).unwrap();
        let yarn = YaRNRoPE::new(cfg, 2048).with_attention_factor(2.0);
        assert!(approx_eq(yarn.attention_factor(), 2.0));
    }

    // -----------------------------------------------------------------------
    // Position interpolation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pi_scale_factor() {
        let cfg = EncodingConfig::new(8192, 64).unwrap();
        let pi = PositionInterpolation::new(cfg, 2048);
        // scale = 2048 / 8192 = 0.25
        assert!(approx_eq(pi.scale(), 0.25));
    }

    #[test]
    fn test_pi_interpolate_position() {
        let cfg = EncodingConfig::new(8192, 64).unwrap();
        let pi = PositionInterpolation::new(cfg, 2048);
        // Position 4096 → 4096 * 0.25 = 1024
        assert!(approx_eq(pi.interpolate_position(4096), 1024.0));
    }

    #[test]
    fn test_pi_position_zero() {
        let cfg = EncodingConfig::new(8192, 64).unwrap();
        let pi = PositionInterpolation::new(cfg, 2048);
        assert!(approx_eq(pi.interpolate_position(0), 0.0));
    }

    #[test]
    fn test_pi_apply_identity_at_pos_zero() {
        let cfg = EncodingConfig::new(8192, 4).unwrap();
        let mut pi = PositionInterpolation::new(cfg, 2048);
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let mut data = orig.clone();
        pi.apply_rope(&mut data, 1, 1, 4, 0).unwrap();
        for (a, b) in data.iter().zip(orig.iter()) {
            assert!(approx_eq(*a, *b), "pos=0 should be identity: {a} vs {b}");
        }
    }

    #[test]
    fn test_pi_preserves_norm() {
        let cfg = EncodingConfig::new(8192, 4).unwrap();
        let mut pi = PositionInterpolation::new(cfg, 2048);
        let orig = vec![5.0, 3.0, 1.0, 7.0];
        let mut data = orig.clone();
        pi.apply_rope(&mut data, 1, 1, 4, 50).unwrap();
        for pair in 0..2 {
            let n1 = (orig[2 * pair].powi(2) + orig[2 * pair + 1].powi(2)).sqrt();
            let n2 = (data[2 * pair].powi(2) + data[2 * pair + 1].powi(2)).sqrt();
            assert!((n1 - n2).abs() < 1e-4, "PI should preserve norm: {n1} vs {n2}");
        }
    }

    #[test]
    fn test_pi_odd_dim_err() {
        let cfg = EncodingConfig::new(8192, 4).unwrap();
        let mut pi = PositionInterpolation::new(cfg, 2048);
        let mut data = vec![0.0; 5];
        assert!(pi.apply_rope(&mut data, 1, 1, 5, 0).is_err());
    }

    #[test]
    fn test_pi_original_max_seq_accessor() {
        let cfg = EncodingConfig::new(8192, 64).unwrap();
        let pi = PositionInterpolation::new(cfg, 2048);
        assert_eq!(pi.original_max_seq_len(), 2048);
    }

    // -----------------------------------------------------------------------
    // OpenCL kernel source tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_opencl_source_not_empty() {
        assert!(!POSITION_ENCODING_CL.is_empty());
    }

    #[test]
    fn test_opencl_source_contains_sinusoidal() {
        assert!(POSITION_ENCODING_CL.contains("sinusoidal_encoding"));
    }

    #[test]
    fn test_opencl_source_contains_rope() {
        assert!(POSITION_ENCODING_CL.contains("rope_encoding"));
    }

    #[test]
    fn test_opencl_source_contains_kernel_qualifier() {
        assert!(POSITION_ENCODING_CL.contains("__kernel"));
    }

    // -----------------------------------------------------------------------
    // Property / edge-case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encoding_dimension_matches_config() {
        // Sinusoidal output length should be seq_len * dim
        let dim = 16;
        let seq = 10;
        let cfg = EncodingConfig::new(seq, dim).unwrap();
        let mut enc = SinusoidalEncoding::new(cfg);
        let mut out = vec![0.0; seq * dim];
        enc.encode(seq, &mut out).unwrap();
        // Every element should have been written (no zeros at the end
        // because sin/cos at various positions won't all be zero)
        let nonzero = out.iter().filter(|&&v| v.abs() > 1e-10).count();
        assert!(nonzero > 0);
    }

    #[test]
    fn test_sinusoidal_deterministic() {
        let cfg = EncodingConfig::new(32, 8).unwrap();
        let mut enc1 = SinusoidalEncoding::new(cfg.clone());
        let mut enc2 = SinusoidalEncoding::new(cfg);
        let mut out1 = vec![0.0; 32 * 8];
        let mut out2 = vec![0.0; 32 * 8];
        enc1.encode(32, &mut out1).unwrap();
        enc2.encode(32, &mut out2).unwrap();
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    #[test]
    fn test_rope_double_rotation() {
        // Applying RoPE twice at position p should equal once at position 2p
        let dim = 4;
        let cfg = EncodingConfig::new(64, dim).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let mut once = input.clone();
        RotaryEncoding::new(cfg.clone()).apply(&mut once, 1, 1, dim, 10).unwrap();

        let mut twice = input.clone();
        RotaryEncoding::new(cfg.clone()).apply(&mut twice, 1, 1, dim, 5).unwrap();
        RotaryEncoding::new(cfg).apply(&mut twice, 1, 1, dim, 5).unwrap();

        for (a, b) in once.iter().zip(twice.iter()) {
            assert!((a - b).abs() < 1e-4, "double rotation: {a} vs {b}");
        }
    }

    #[test]
    fn test_alibi_multi_head_independence() {
        // Each head should have different bias magnitudes
        let mut enc = ALiBiEncoding::new(4).unwrap();
        let mut bias = vec![0.0; 4 * 4 * 4];
        enc.compute_bias(4, 4, &mut bias).unwrap();

        // Compare head 0 and head 1 at same position pair
        let h0_bias = bias[0 * 16 + 0 * 4 + 3]; // head 0, q=0, k=3
        let h1_bias = bias[1 * 16 + 0 * 4 + 3]; // head 1, q=0, k=3
        assert!((h0_bias - h1_bias).abs() > 1e-6, "different heads should have different biases");
    }

    #[test]
    fn test_relative_bucket_range() {
        let enc = RelativeEncoding::new(32, 128, true).unwrap();
        for d in -200..200 {
            let b = enc.bucket(d);
            assert!(b < enc.num_buckets(), "bucket {b} out of range for distance {d}");
        }
    }

    #[test]
    fn test_all_encodings_handle_seq_len_1() {
        // Every encoding should work for a single position
        let cfg = EncodingConfig::new(16, 4).unwrap();

        let mut sin_enc = SinusoidalEncoding::new(cfg.clone());
        let mut out = vec![0.0; 4];
        sin_enc.encode(1, &mut out).unwrap();

        let weight = vec![0.0; 64];
        let mut learn = LearnedEncoding::new(cfg.clone(), weight).unwrap();
        let mut out2 = vec![0.0; 4];
        learn.encode(1, &mut out2).unwrap();

        let mut rope = RotaryEncoding::new(cfg.clone());
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        rope.apply(&mut data, 1, 1, 4, 0).unwrap();

        let mut alibi = ALiBiEncoding::new(1).unwrap();
        let mut bias = vec![0.0; 1];
        alibi.compute_bias(1, 1, &mut bias).unwrap();

        let mut rel = RelativeEncoding::new(32, 128, true).unwrap();
        let mut buckets = vec![0usize; 1];
        rel.compute_buckets(1, 1, &mut buckets).unwrap();
    }

    #[test]
    fn test_ntk_vs_standard_rope_at_short_sequence() {
        // When seq_len <= original, NTK should behave like standard RoPE
        let cfg = EncodingConfig::new(4096, 4).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let mut std_data = input.clone();
        RotaryEncoding::new(cfg.clone()).apply(&mut std_data, 1, 1, 4, 5).unwrap();

        let mut ntk_data = input.clone();
        DynamicNTKRoPE::new(cfg, 4096).apply(&mut ntk_data, 1, 1, 4, 5).unwrap();

        for (a, b) in std_data.iter().zip(ntk_data.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "NTK should match standard RoPE within training: {a} vs {b}",
            );
        }
    }

    #[test]
    fn test_pi_compressed_range() {
        // All interpolated positions should fit within original range
        let cfg = EncodingConfig::new(8192, 64).unwrap();
        let pi = PositionInterpolation::new(cfg, 2048);
        for pos in 0..8192 {
            let interp = pi.interpolate_position(pos);
            assert!(interp <= 2048.0, "interpolated position {interp} > original max 2048");
        }
    }

    #[test]
    fn test_sinusoidal_max_position() {
        // Encoding at max_seq_len - 1 should succeed
        let max = 64;
        let dim = 4;
        let cfg = EncodingConfig::new(max, dim).unwrap();
        let mut enc = SinusoidalEncoding::new(cfg);
        let mut out = vec![0.0; max * dim];
        enc.encode(max, &mut out).unwrap();
        // Last position should have valid values
        let last_row = &out[(max - 1) * dim..max * dim];
        assert!(last_row.iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_rope_seq_len_2_positions() {
        // Two sequential positions should produce different rotations
        let dim = 4;
        let cfg = EncodingConfig::new(16, dim).unwrap();
        let mut rope = RotaryEncoding::new(cfg);

        let input = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mut data = input.clone();
        rope.apply(&mut data, 2, 1, dim, 0).unwrap();

        let pos0 = &data[0..dim];
        let pos1 = &data[dim..2 * dim];
        assert!(
            !pos0.iter().zip(pos1.iter()).all(|(a, b)| approx_eq(*a, *b)),
            "different positions should produce different rotations",
        );
    }

    #[test]
    fn test_stats_default_impl() {
        let stats = EncodingStats::default();
        assert_eq!(stats.cache_hits, 0);
    }
}
