//! OpenCL RoPE frequency table generator with caching and extension support.
//!
//! Pre-computes cos/sin frequency tables for Rotary Position Embeddings,
//! targeting Intel Arc A770 and other OpenCL-capable GPUs. Supports multiple
//! RoPE variants including standard, NTK, YaRN, Dynamic NTK, and CodeLlama.
//!
//! # Architecture
//!
//! - **[`RopeConfig`]** — configuration: head_dim, max_seq_len, base, scaling,
//!   rope type.
//! - **[`RopeType`]** — variant selector (Standard, NTK, YaRN, Dynamic,
//!   CodeLlama).
//! - **[`FreqTable`]** — pre-computed cos/sin tables for all positions ×
//!   dimensions.
//! - **[`FreqTableCache`]** — LRU cache keyed by `RopeConfig`.
//! - **Scalers** — [`NtkScaler`], [`YarnScaler`], [`DynamicNtk`] implement
//!   frequency scaling for extended context.
//! - **[`TableGenerator`]** — orchestrates table generation with A770 buffer
//!   packing.
//! - **OpenCL kernel source** — GPU kernel for applying pre-computed RoPE.
//! - **CPU reference** — scalar fallback for testing and non-GPU paths.

use std::collections::HashMap;
use std::f32::consts::PI;
use std::fmt;

// ---------------------------------------------------------------------------
// RoPE type enum
// ---------------------------------------------------------------------------

/// Selects the RoPE variant for frequency computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RopeType {
    /// Standard RoPE with fixed base frequency.
    Standard,
    /// Neural Tangent Kernel scaling for longer contexts.
    NTK,
    /// Yet another RoPE extensioN — smooth interpolation.
    YaRN,
    /// Dynamic NTK that scales base with sequence length.
    Dynamic,
    /// CodeLlama-style extended context RoPE.
    CodeLlama,
}

impl fmt::Display for RopeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => write!(f, "Standard"),
            Self::NTK => write!(f, "NTK"),
            Self::YaRN => write!(f, "YaRN"),
            Self::Dynamic => write!(f, "Dynamic"),
            Self::CodeLlama => write!(f, "CodeLlama"),
        }
    }
}

// ---------------------------------------------------------------------------
// RoPE scaling type
// ---------------------------------------------------------------------------

/// Scaling strategy applied to RoPE frequencies.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ScalingType {
    /// No scaling — standard RoPE.
    #[default]
    None,
    /// Linear frequency scaling by the given factor.
    Linear(f32),
    /// NTK-aware scaling with the given factor.
    NtkAware(f32),
    /// YaRN scaling with attention factor and frequency bounds.
    Yarn { factor: f32, attention_factor: f32, beta_fast: f32, beta_slow: f32 },
    /// Dynamic NTK that adjusts base with sequence length.
    DynamicNtk { factor: f32, original_max_seq_len: usize },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for RoPE table generation.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeConfig {
    /// Dimension of each attention head (≥ 1).
    pub head_dim: usize,
    /// Maximum sequence length to pre-compute tables for.
    pub max_seq_len: usize,
    /// Base frequency (default 10 000).
    pub base: f32,
    /// Scaling strategy.
    pub scaling_type: ScalingType,
    /// RoPE variant.
    pub rope_type: RopeType,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            head_dim: 64,
            max_seq_len: 2048,
            base: 10_000.0,
            scaling_type: ScalingType::None,
            rope_type: RopeType::Standard,
        }
    }
}

/// Errors from RoPE table operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeTableError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Cache miss — no table found for the given config.
    CacheMiss,
}

impl fmt::Display for RopeTableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid RoPE config: {msg}"),
            Self::CacheMiss => write!(f, "RoPE table cache miss"),
        }
    }
}

impl std::error::Error for RopeTableError {}

impl RopeConfig {
    /// Create a new config with the given parameters.
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        base: f32,
        scaling_type: ScalingType,
        rope_type: RopeType,
    ) -> Self {
        Self { head_dim, max_seq_len, base, scaling_type, rope_type }
    }

    /// Validate configuration, returning an error if invalid.
    pub fn validate(&self) -> Result<(), RopeTableError> {
        if self.head_dim == 0 {
            return Err(RopeTableError::InvalidConfig("head_dim must be > 0".into()));
        }
        if self.max_seq_len == 0 {
            return Err(RopeTableError::InvalidConfig("max_seq_len must be > 0".into()));
        }
        if self.base <= 0.0 || !self.base.is_finite() {
            return Err(RopeTableError::InvalidConfig(format!(
                "base must be positive and finite, got {}",
                self.base
            )));
        }
        Ok(())
    }

    /// Compute the effective half-dimension used for frequency indexing.
    pub fn half_dim(&self) -> usize {
        self.head_dim / 2
    }

    /// Build a cache key from this config for use in [`FreqTableCache`].
    fn cache_key(&self) -> CacheKey {
        CacheKey {
            head_dim: self.head_dim,
            max_seq_len: self.max_seq_len,
            base_bits: self.base.to_bits(),
            rope_type: self.rope_type,
            scaling_tag: scaling_tag(&self.scaling_type),
        }
    }
}

/// Compact cache key derived from [`RopeConfig`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    head_dim: usize,
    max_seq_len: usize,
    base_bits: u32,
    rope_type: RopeType,
    scaling_tag: u64,
}

/// Deterministic tag for a scaling type, suitable for hashing.
fn scaling_tag(s: &ScalingType) -> u64 {
    match s {
        ScalingType::None => 0,
        ScalingType::Linear(f) => 1 ^ (f.to_bits() as u64),
        ScalingType::NtkAware(f) => 2 ^ (f.to_bits() as u64),
        ScalingType::Yarn { factor, attention_factor, beta_fast, beta_slow } => {
            3 ^ (factor.to_bits() as u64)
                ^ ((attention_factor.to_bits() as u64) << 8)
                ^ ((beta_fast.to_bits() as u64) << 16)
                ^ ((beta_slow.to_bits() as u64) << 24)
        }
        ScalingType::DynamicNtk { factor, original_max_seq_len } => {
            4 ^ (factor.to_bits() as u64) ^ ((*original_max_seq_len as u64) << 32)
        }
    }
}

// ---------------------------------------------------------------------------
// Frequency table
// ---------------------------------------------------------------------------

/// Pre-computed cos/sin frequency table.
///
/// Layout: `cos[pos * half_dim + i]` = cos(pos × θ_i), where
/// `half_dim = head_dim / 2` and θ_i = base^(−2i / head_dim).
///
/// For odd `head_dim`, `half_dim = head_dim / 2` (integer division) and
/// the last dimension element is left unrotated.
#[derive(Debug, Clone)]
pub struct FreqTable {
    /// Cosine values, length = max_seq_len × half_dim.
    pub cos: Vec<f32>,
    /// Sine values, length = max_seq_len × half_dim.
    pub sin: Vec<f32>,
    /// Half-dimension used for indexing.
    pub half_dim: usize,
    /// Number of sequence positions covered.
    pub seq_len: usize,
    /// The config that produced this table.
    pub config: RopeConfig,
}

impl FreqTable {
    /// Get (cos, sin) for a given position and frequency index.
    pub fn get(&self, pos: usize, freq_idx: usize) -> Option<(f32, f32)> {
        if pos >= self.seq_len || freq_idx >= self.half_dim {
            return None;
        }
        let idx = pos * self.half_dim + freq_idx;
        Some((self.cos[idx], self.sin[idx]))
    }

    /// Total number of entries (positions × half_dim).
    pub fn len(&self) -> usize {
        self.cos.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.cos.is_empty()
    }

    /// Pack the cos/sin tables into a single interleaved buffer suitable
    /// for GPU upload. Layout: `[cos_0, sin_0, cos_1, sin_1, ...]`.
    pub fn pack_interleaved(&self) -> Vec<f32> {
        let mut buf = Vec::with_capacity(self.cos.len() * 2);
        for i in 0..self.cos.len() {
            buf.push(self.cos[i]);
            buf.push(self.sin[i]);
        }
        buf
    }

    /// Pack cos/sin tables contiguously: `[all_cos..., all_sin...]`.
    /// Preferred for A770 coalesced memory access patterns.
    pub fn pack_contiguous(&self) -> Vec<f32> {
        let mut buf = Vec::with_capacity(self.cos.len() * 2);
        buf.extend_from_slice(&self.cos);
        buf.extend_from_slice(&self.sin);
        buf
    }
}

// ---------------------------------------------------------------------------
// Inverse-frequency computation
// ---------------------------------------------------------------------------

/// Compute the inverse-frequency vector θ_i = base^(−2i / d).
pub fn compute_inv_freq(head_dim: usize, base: f32) -> Vec<f32> {
    let half = head_dim / 2;
    if half == 0 {
        return vec![];
    }
    (0..half)
        .map(|i| {
            let exponent = -((2 * i) as f32) / (head_dim as f32);
            base.powf(exponent)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// NTK scaler
// ---------------------------------------------------------------------------

/// Neural Tangent Kernel scaling for extended context windows.
///
/// Scales the base frequency so that low-frequency components are preserved
/// while high-frequency ones are compressed:
///   scaled_base = base × factor^(d / (d − 2))
pub struct NtkScaler {
    /// Scaling factor (typically context_extension_ratio).
    pub factor: f32,
}

impl NtkScaler {
    pub fn new(factor: f32) -> Self {
        Self { factor }
    }

    /// Apply NTK scaling to an inverse-frequency vector.
    pub fn scale(&self, inv_freq: &[f32]) -> Vec<f32> {
        if inv_freq.is_empty() {
            return vec![];
        }
        let d = (inv_freq.len() * 2) as f32;
        let base_scale = self.factor.powf(d / (d - 2.0));
        inv_freq.iter().map(|&f| f / base_scale).collect()
    }

    /// Compute NTK-scaled inverse frequencies from scratch.
    pub fn compute(&self, head_dim: usize, base: f32) -> Vec<f32> {
        let inv = compute_inv_freq(head_dim, base);
        self.scale(&inv)
    }
}

// ---------------------------------------------------------------------------
// YaRN scaler
// ---------------------------------------------------------------------------

/// Yet another RoPE extensioN — smooth interpolation between low and high
/// frequency regions with an attention scaling factor.
pub struct YarnScaler {
    /// Overall scaling factor.
    pub factor: f32,
    /// Attention factor (multiplied into the final scaled frequencies).
    pub attention_factor: f32,
    /// High-frequency boundary (wavelength threshold).
    pub beta_fast: f32,
    /// Low-frequency boundary (wavelength threshold).
    pub beta_slow: f32,
}

impl YarnScaler {
    pub fn new(factor: f32, attention_factor: f32, beta_fast: f32, beta_slow: f32) -> Self {
        Self { factor, attention_factor, beta_fast, beta_slow }
    }

    /// Apply YaRN scaling to an inverse-frequency vector.
    pub fn scale(&self, inv_freq: &[f32]) -> Vec<f32> {
        if inv_freq.is_empty() {
            return vec![];
        }
        let low_freq_wavelen = 2.0 * PI / self.beta_slow;
        let high_freq_wavelen = 2.0 * PI / self.beta_fast;
        let d = (inv_freq.len() * 2) as f32;

        inv_freq
            .iter()
            .map(|&freq| {
                let wavelen = 2.0 * PI / freq;
                let scaled = if wavelen < high_freq_wavelen {
                    // High frequency — keep unchanged
                    freq
                } else if wavelen > low_freq_wavelen {
                    // Low frequency — fully scale down
                    freq / self.factor
                } else {
                    // Smooth ramp
                    let t = (d * (wavelen - high_freq_wavelen).ln()
                        / (low_freq_wavelen - high_freq_wavelen))
                        .clamp(0.0, 1.0);
                    let smooth = (1.0 - t.cos()) / 2.0;
                    freq * (1.0 - smooth) + (freq / self.factor) * smooth
                };
                scaled * self.attention_factor
            })
            .collect()
    }

    /// Compute YaRN-scaled inverse frequencies from scratch.
    pub fn compute(&self, head_dim: usize, base: f32) -> Vec<f32> {
        let inv = compute_inv_freq(head_dim, base);
        self.scale(&inv)
    }
}

// ---------------------------------------------------------------------------
// Dynamic NTK scaler
// ---------------------------------------------------------------------------

/// Dynamic NTK scaling — adjusts base frequency based on the current
/// sequence length, allowing the model to handle progressively longer
/// contexts without retraining.
pub struct DynamicNtk {
    /// Extension ratio.
    pub factor: f32,
    /// Original maximum sequence length the model was trained on.
    pub original_max_seq_len: usize,
}

impl DynamicNtk {
    pub fn new(factor: f32, original_max_seq_len: usize) -> Self {
        Self { factor, original_max_seq_len }
    }

    /// Compute an effective base for the given current sequence length.
    ///
    /// If `current_seq_len <= original_max_seq_len`, returns the original
    /// base unchanged. Otherwise, scales the base proportionally.
    pub fn effective_base(&self, base: f32, head_dim: usize, current_seq_len: usize) -> f32 {
        if current_seq_len <= self.original_max_seq_len {
            return base;
        }
        let ratio = (self.factor * current_seq_len as f32) / self.original_max_seq_len as f32;
        let d = head_dim as f32;
        base * ratio.powf(d / (d - 2.0))
    }

    /// Compute dynamic-NTK-scaled inverse frequencies for a given sequence
    /// length.
    pub fn compute(&self, head_dim: usize, base: f32, current_seq_len: usize) -> Vec<f32> {
        let eff_base = self.effective_base(base, head_dim, current_seq_len);
        compute_inv_freq(head_dim, eff_base)
    }
}

// ---------------------------------------------------------------------------
// CodeLlama RoPE helper
// ---------------------------------------------------------------------------

/// Compute CodeLlama-style extended RoPE frequencies.
///
/// CodeLlama uses a higher base (typically 1 000 000) and applies a
/// correction factor to the inverse frequencies for positions beyond the
/// original training window.
pub fn codellama_inv_freq(head_dim: usize, base: f32, factor: f32) -> Vec<f32> {
    let inv = compute_inv_freq(head_dim, base);
    // CodeLlama simply scales the base; the factor is baked in
    if (factor - 1.0).abs() < f32::EPSILON {
        inv
    } else {
        let d = head_dim as f32;
        let scaled_base = base * factor.powf(d / (d - 2.0));
        compute_inv_freq(head_dim, scaled_base)
    }
}

// ---------------------------------------------------------------------------
// Table generator
// ---------------------------------------------------------------------------

/// Generates [`FreqTable`]s for any RoPE variant with optional A770 buffer
/// packing.
pub struct TableGenerator;

impl TableGenerator {
    /// Generate a frequency table from the given configuration.
    pub fn generate(config: &RopeConfig) -> Result<FreqTable, RopeTableError> {
        config.validate()?;
        let inv_freq = Self::compute_inv_freq_for_config(config);
        Ok(Self::build_table(&inv_freq, config))
    }

    /// Generate a table and return it packed in interleaved layout for GPU
    /// upload.
    pub fn generate_packed_interleaved(config: &RopeConfig) -> Result<Vec<f32>, RopeTableError> {
        let table = Self::generate(config)?;
        Ok(table.pack_interleaved())
    }

    /// Generate a table and return it in contiguous layout (A770-optimised).
    pub fn generate_packed_contiguous(config: &RopeConfig) -> Result<Vec<f32>, RopeTableError> {
        let table = Self::generate(config)?;
        Ok(table.pack_contiguous())
    }

    /// Compute inverse frequencies for the given config, dispatching to the
    /// appropriate scaler.
    fn compute_inv_freq_for_config(config: &RopeConfig) -> Vec<f32> {
        match (&config.rope_type, &config.scaling_type) {
            (RopeType::Standard, ScalingType::None) => {
                compute_inv_freq(config.head_dim, config.base)
            }
            (RopeType::Standard, ScalingType::Linear(f)) => {
                let inv = compute_inv_freq(config.head_dim, config.base);
                inv.iter().map(|&v| v / f).collect()
            }
            (RopeType::NTK, ScalingType::NtkAware(f)) | (RopeType::NTK, ScalingType::Linear(f)) => {
                NtkScaler::new(*f).compute(config.head_dim, config.base)
            }
            (RopeType::NTK, ScalingType::None) => compute_inv_freq(config.head_dim, config.base),
            (
                RopeType::YaRN,
                ScalingType::Yarn { factor, attention_factor, beta_fast, beta_slow },
            ) => YarnScaler::new(*factor, *attention_factor, *beta_fast, *beta_slow)
                .compute(config.head_dim, config.base),
            (RopeType::YaRN, _) => compute_inv_freq(config.head_dim, config.base),
            (RopeType::Dynamic, ScalingType::DynamicNtk { factor, original_max_seq_len }) => {
                DynamicNtk::new(*factor, *original_max_seq_len).compute(
                    config.head_dim,
                    config.base,
                    config.max_seq_len,
                )
            }
            (RopeType::Dynamic, _) => compute_inv_freq(config.head_dim, config.base),
            (RopeType::CodeLlama, ScalingType::Linear(f)) => {
                codellama_inv_freq(config.head_dim, config.base, *f)
            }
            (RopeType::CodeLlama, _) => codellama_inv_freq(config.head_dim, config.base, 1.0),
            // Fallback: standard frequencies
            (_, _) => compute_inv_freq(config.head_dim, config.base),
        }
    }

    /// Build a `FreqTable` from pre-computed inverse frequencies.
    fn build_table(inv_freq: &[f32], config: &RopeConfig) -> FreqTable {
        let half_dim = inv_freq.len();
        let total = config.max_seq_len * half_dim;
        let mut cos = Vec::with_capacity(total);
        let mut sin = Vec::with_capacity(total);

        for pos in 0..config.max_seq_len {
            for &freq in inv_freq {
                let angle = (pos as f32) * freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }

        FreqTable { cos, sin, half_dim, seq_len: config.max_seq_len, config: config.clone() }
    }
}

// ---------------------------------------------------------------------------
// Frequency table cache (LRU)
// ---------------------------------------------------------------------------

/// LRU cache for pre-computed [`FreqTable`]s, keyed by [`RopeConfig`].
pub struct FreqTableCache {
    entries: HashMap<CacheKey, CacheEntry>,
    order: Vec<CacheKey>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

struct CacheEntry {
    table: FreqTable,
}

impl FreqTableCache {
    /// Create a new cache with the given maximum number of entries.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: Vec::new(),
            capacity: capacity.max(1),
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a table by config, returning `None` on cache miss.
    pub fn get(&mut self, config: &RopeConfig) -> Option<&FreqTable> {
        let key = config.cache_key();
        if self.entries.contains_key(&key) {
            self.hits += 1;
            // Move to most-recently-used position
            self.order.retain(|k| k != &key);
            self.order.push(key.clone());
            Some(&self.entries.get(&key).unwrap().table)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a table into the cache, evicting the LRU entry if at capacity.
    pub fn insert(&mut self, config: &RopeConfig, table: FreqTable) {
        let key = config.cache_key();
        if self.entries.contains_key(&key) {
            self.order.retain(|k| k != &key);
        } else if self.entries.len() >= self.capacity {
            // Evict LRU
            if let Some(lru_key) = self.order.first().cloned() {
                self.entries.remove(&lru_key);
                self.order.remove(0);
            }
        }
        self.order.push(key.clone());
        self.entries.insert(key, CacheEntry { table });
    }

    /// Get or generate: returns cached table if available, otherwise
    /// generates, caches, and returns it.
    pub fn get_or_generate(&mut self, config: &RopeConfig) -> Result<&FreqTable, RopeTableError> {
        let key = config.cache_key();
        if !self.entries.contains_key(&key) {
            let table = TableGenerator::generate(config)?;
            self.insert(config, table);
        } else {
            self.hits += 1;
            self.order.retain(|k| k != &key);
            self.order.push(key.clone());
        }
        Ok(&self.entries.get(&key).unwrap().table)
    }

    /// Number of entries currently cached.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cache hit count.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache miss count.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Maximum number of entries this cache can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// CPU reference: apply RoPE from a FreqTable
// ---------------------------------------------------------------------------

/// Apply RoPE rotation to a vector using pre-computed cos/sin values.
///
/// Rotates pairs (x[i], x[i + half]) as:
///   x'[i]        = x[i] × cos[i] − x[i+half] × sin[i]
///   x'[i + half] = x[i] × sin[i] + x[i+half] × cos[i]
pub fn cpu_apply_rope(vec: &mut [f32], cos: &[f32], sin: &[f32], half_dim: usize) {
    for i in 0..half_dim {
        if i + half_dim >= vec.len() {
            break;
        }
        let x0 = vec[i];
        let x1 = vec[i + half_dim];
        vec[i] = x0 * cos[i] - x1 * sin[i];
        vec[i + half_dim] = x0 * sin[i] + x1 * cos[i];
    }
}

/// Apply RoPE to Q/K tensors for a batch of positions using a [`FreqTable`].
///
/// `q` and `k` are shaped `[num_tokens × num_heads × head_dim]`.
/// `positions` contains one position per token.
pub fn cpu_apply_rope_batch(
    q: &mut [f32],
    k: &mut [f32],
    table: &FreqTable,
    positions: &[usize],
    num_heads: usize,
    head_dim: usize,
) -> Result<(), RopeTableError> {
    let half = table.half_dim;
    let token_stride = num_heads * head_dim;

    for (tok, &pos) in positions.iter().enumerate() {
        if pos >= table.seq_len {
            return Err(RopeTableError::InvalidConfig(format!(
                "position {pos} exceeds table seq_len {}",
                table.seq_len
            )));
        }
        let cos = &table.cos[pos * half..(pos + 1) * half];
        let sin = &table.sin[pos * half..(pos + 1) * half];

        for h in 0..num_heads {
            let offset = tok * token_stride + h * head_dim;
            if offset + head_dim <= q.len() {
                cpu_apply_rope(&mut q[offset..offset + head_dim], cos, sin, half);
            }
            if offset + head_dim <= k.len() {
                cpu_apply_rope(&mut k[offset..offset + head_dim], cos, sin, half);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C kernel source for applying pre-computed RoPE tables on GPU.
///
/// Designed for Intel Arc A770 with 512-bit EU width. The kernel reads from
/// a contiguous buffer layout `[all_cos..., all_sin...]` to maximise
/// coalesced access across sub-groups.
pub const OPENCL_ROPE_TABLE_SRC: &str = r#"
// Apply pre-computed RoPE cos/sin tables to Q and K vectors.
//
// Buffer layout (contiguous):
//   rope_table[0 .. seq_len*half_dim-1]           = cos values
//   rope_table[seq_len*half_dim .. 2*seq_len*half_dim-1] = sin values
//
// Each work-item processes one (head, freq_idx) pair for one token.
__kernel void apply_rope_table(
    __global float* q,              // [num_tokens * num_heads * head_dim]
    __global float* k,              // [num_tokens * num_heads * head_dim]
    __global const float* rope_table, // contiguous cos/sin table
    __global const int* positions,  // [num_tokens]
    const int num_heads,
    const int head_dim,
    const int half_dim,
    const int seq_len,
    const int num_tokens
) {
    int gid = get_global_id(0);
    int total_pairs = num_tokens * num_heads * half_dim;
    if (gid >= total_pairs) return;

    int pair_idx = gid % half_dim;
    int remaining = gid / half_dim;
    int head_idx = remaining % num_heads;
    int tok_idx = remaining / num_heads;

    int pos = positions[tok_idx];
    int table_idx = pos * half_dim + pair_idx;

    float cos_val = rope_table[table_idx];
    float sin_val = rope_table[seq_len * half_dim + table_idx];

    int base_offset = tok_idx * num_heads * head_dim + head_idx * head_dim;
    int lo = base_offset + pair_idx;
    int hi = base_offset + pair_idx + half_dim;

    // Rotate Q
    float q0 = q[lo];
    float q1 = q[hi];
    q[lo] = q0 * cos_val - q1 * sin_val;
    q[hi] = q0 * sin_val + q1 * cos_val;

    // Rotate K
    float k0 = k[lo];
    float k1 = k[hi];
    k[lo] = k0 * cos_val - k1 * sin_val;
    k[hi] = k0 * sin_val + k1 * cos_val;
}

// Pre-compute RoPE frequency table on GPU.
//
// Each work-item computes cos/sin for one (position, freq_idx) pair.
__kernel void compute_rope_table(
    __global float* cos_out,   // [max_seq_len * half_dim]
    __global float* sin_out,   // [max_seq_len * half_dim]
    __global const float* inv_freq, // [half_dim]
    const int half_dim,
    const int max_seq_len
) {
    int gid = get_global_id(0);
    if (gid >= max_seq_len * half_dim) return;

    int freq_idx = gid % half_dim;
    int pos = gid / half_dim;

    float angle = (float)pos * inv_freq[freq_idx];
    cos_out[gid] = cos(angle);
    sin_out[gid] = sin(angle);
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers --

    fn default_config() -> RopeConfig {
        RopeConfig::default()
    }

    fn config_with(head_dim: usize, seq_len: usize, base: f32) -> RopeConfig {
        RopeConfig::new(head_dim, seq_len, base, ScalingType::None, RopeType::Standard)
    }

    // ===================================================================
    // RopeConfig validation
    // ===================================================================

    #[test]
    fn test_config_default_is_valid() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn test_config_head_dim_zero_invalid() {
        let c = config_with(0, 128, 10_000.0);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_seq_len_zero_invalid() {
        let c = config_with(64, 0, 10_000.0);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_negative_base_invalid() {
        let c = config_with(64, 128, -1.0);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_nan_base_invalid() {
        let c = config_with(64, 128, f32::NAN);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_inf_base_invalid() {
        let c = config_with(64, 128, f32::INFINITY);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_half_dim() {
        assert_eq!(config_with(64, 128, 10_000.0).half_dim(), 32);
        assert_eq!(config_with(128, 128, 10_000.0).half_dim(), 64);
        assert_eq!(config_with(1, 128, 10_000.0).half_dim(), 0);
    }

    // ===================================================================
    // Inverse frequency computation
    // ===================================================================

    #[test]
    fn test_inv_freq_length() {
        let inv = compute_inv_freq(64, 10_000.0);
        assert_eq!(inv.len(), 32);
    }

    #[test]
    fn test_inv_freq_first_element_is_one() {
        let inv = compute_inv_freq(64, 10_000.0);
        assert!((inv[0] - 1.0).abs() < 1e-6, "θ_0 should be base^0 = 1.0");
    }

    #[test]
    fn test_inv_freq_monotonically_decreasing() {
        let inv = compute_inv_freq(64, 10_000.0);
        for i in 1..inv.len() {
            assert!(inv[i] < inv[i - 1], "inv_freq should be decreasing");
        }
    }

    #[test]
    fn test_inv_freq_head_dim_1_empty() {
        let inv = compute_inv_freq(1, 10_000.0);
        assert!(inv.is_empty());
    }

    #[test]
    fn test_inv_freq_head_dim_2() {
        let inv = compute_inv_freq(2, 10_000.0);
        assert_eq!(inv.len(), 1);
        assert!((inv[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_inv_freq_different_bases() {
        let inv_10k = compute_inv_freq(64, 10_000.0);
        let inv_500k = compute_inv_freq(64, 500_000.0);
        let inv_1m = compute_inv_freq(64, 1_000_000.0);
        // Higher base → slower decay (values closer to 1), but each
        // θ_i = base^(-2i/d) is actually *smaller* for larger bases
        // when -2i/d < 0, because a larger base raised to a negative
        // exponent yields a smaller result. Verify they differ.
        let differs_500k =
            inv_10k.iter().zip(inv_500k.iter()).skip(1).any(|(a, b)| (a - b).abs() > 1e-6);
        let differs_1m =
            inv_500k.iter().zip(inv_1m.iter()).skip(1).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs_500k, "500k base should differ from 10k");
        assert!(differs_1m, "1M base should differ from 500k");
    }

    // ===================================================================
    // Standard RoPE table generation
    // ===================================================================

    #[test]
    fn test_standard_table_dimensions() {
        let cfg = config_with(64, 128, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        assert_eq!(table.half_dim, 32);
        assert_eq!(table.seq_len, 128);
        assert_eq!(table.cos.len(), 128 * 32);
        assert_eq!(table.sin.len(), 128 * 32);
    }

    #[test]
    fn test_standard_table_position_zero() {
        let cfg = config_with(64, 128, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        // At position 0, angle = 0 for all freqs → cos=1, sin=0
        for i in 0..table.half_dim {
            let (c, s) = table.get(0, i).unwrap();
            assert!((c - 1.0).abs() < 1e-6, "cos(0) should be 1.0");
            assert!(s.abs() < 1e-6, "sin(0) should be 0.0");
        }
    }

    #[test]
    fn test_cos_sq_plus_sin_sq_equals_one() {
        let cfg = config_with(64, 256, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        for pos in 0..table.seq_len {
            for i in 0..table.half_dim {
                let (c, s) = table.get(pos, i).unwrap();
                let sum = c * c + s * s;
                assert!((sum - 1.0).abs() < 1e-5, "cos²+sin²≠1 at pos={pos}, i={i}: {sum}");
            }
        }
    }

    #[test]
    fn test_table_get_out_of_bounds_returns_none() {
        let cfg = config_with(64, 16, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        assert!(table.get(16, 0).is_none()); // pos out of range
        assert!(table.get(0, 32).is_none()); // freq out of range
        assert!(table.get(100, 100).is_none());
    }

    #[test]
    fn test_table_len_and_is_empty() {
        let cfg = config_with(64, 16, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        assert_eq!(table.len(), 16 * 32);
        assert!(!table.is_empty());
    }

    #[test]
    fn test_table_seq_len_1() {
        let cfg = config_with(64, 1, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        assert_eq!(table.seq_len, 1);
        assert_eq!(table.cos.len(), 32);
        // Position 0 → all cos=1, sin=0
        for i in 0..32 {
            let (c, s) = table.get(0, i).unwrap();
            assert!((c - 1.0).abs() < 1e-6);
            assert!(s.abs() < 1e-6);
        }
    }

    #[test]
    fn test_head_dim_odd_handling() {
        // Odd head_dim: half_dim = head_dim / 2 (integer division)
        let cfg = config_with(65, 16, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        assert_eq!(table.half_dim, 32); // 65/2 = 32
    }

    #[test]
    fn test_head_dim_1_generates_empty_table() {
        let cfg = config_with(1, 16, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        assert_eq!(table.half_dim, 0);
        assert!(table.is_empty());
    }

    // ===================================================================
    // Buffer packing (A770 support)
    // ===================================================================

    #[test]
    fn test_pack_interleaved_layout() {
        let cfg = config_with(4, 2, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        let packed = table.pack_interleaved();
        assert_eq!(packed.len(), table.cos.len() * 2);
        // Verify interleaving: [cos0, sin0, cos1, sin1, ...]
        for i in 0..table.cos.len() {
            assert_eq!(packed[2 * i], table.cos[i]);
            assert_eq!(packed[2 * i + 1], table.sin[i]);
        }
    }

    #[test]
    fn test_pack_contiguous_layout() {
        let cfg = config_with(4, 2, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        let packed = table.pack_contiguous();
        assert_eq!(packed.len(), table.cos.len() * 2);
        let n = table.cos.len();
        assert_eq!(&packed[..n], &table.cos[..]);
        assert_eq!(&packed[n..], &table.sin[..]);
    }

    // ===================================================================
    // NTK scaler
    // ===================================================================

    #[test]
    fn test_ntk_scaler_changes_frequencies() {
        let inv_std = compute_inv_freq(64, 10_000.0);
        let ntk = NtkScaler::new(2.0);
        let inv_ntk = ntk.scale(&inv_std);
        assert_eq!(inv_ntk.len(), inv_std.len());
        // NTK divides by a positive factor > 1, so all freqs should be smaller
        for (s, n) in inv_std.iter().zip(inv_ntk.iter()) {
            assert!(n < s, "NTK-scaled freq should be smaller");
        }
    }

    #[test]
    fn test_ntk_scaler_factor_one_nearly_identity() {
        let inv_std = compute_inv_freq(64, 10_000.0);
        let ntk = NtkScaler::new(1.0);
        let inv_ntk = ntk.scale(&inv_std);
        for (s, n) in inv_std.iter().zip(inv_ntk.iter()) {
            // factor=1 → base_scale = 1^(d/(d-2)) = 1
            assert!((s - n).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ntk_scaler_empty_input() {
        let ntk = NtkScaler::new(2.0);
        assert!(ntk.scale(&[]).is_empty());
    }

    #[test]
    fn test_ntk_compute_from_scratch() {
        let ntk = NtkScaler::new(4.0);
        let inv = ntk.compute(128, 10_000.0);
        assert_eq!(inv.len(), 64);
        // All positive
        for &v in &inv {
            assert!(v > 0.0);
        }
    }

    // ===================================================================
    // YaRN scaler
    // ===================================================================

    #[test]
    fn test_yarn_scaler_with_attention_factor() {
        let inv_std = compute_inv_freq(64, 10_000.0);
        let yarn = YarnScaler::new(2.0, 0.5, 32.0, 1.0);
        let inv_yarn = yarn.scale(&inv_std);
        assert_eq!(inv_yarn.len(), inv_std.len());
        // With attention_factor=0.5, all outputs should be roughly half
        // or less compared to standard (modulo the ramp region)
        for &v in &inv_yarn {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_yarn_scaler_attention_factor_one() {
        let inv_std = compute_inv_freq(64, 10_000.0);
        let yarn = YarnScaler::new(1.0, 1.0, 32.0, 1.0);
        let inv_yarn = yarn.scale(&inv_std);
        // factor=1 → no frequency scaling, attention_factor=1 → no
        // attn scaling; but ramp region might still alter some values.
        // At minimum, the high-frequency region should be unchanged.
        // (First element is highest freq = 1.0, wavelen = 2π)
        assert!((inv_yarn[0] - inv_std[0]).abs() < 1e-4);
    }

    #[test]
    fn test_yarn_scaler_empty_input() {
        let yarn = YarnScaler::new(2.0, 1.0, 32.0, 1.0);
        assert!(yarn.scale(&[]).is_empty());
    }

    #[test]
    fn test_yarn_produces_different_table_than_standard() {
        let cfg_std = config_with(64, 128, 10_000.0);
        let cfg_yarn = RopeConfig::new(
            64,
            128,
            10_000.0,
            ScalingType::Yarn {
                factor: 2.0,
                attention_factor: 1.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
            RopeType::YaRN,
        );
        let t_std = TableGenerator::generate(&cfg_std).unwrap();
        let t_yarn = TableGenerator::generate(&cfg_yarn).unwrap();
        // Tables should differ
        let differs = t_std.cos.iter().zip(t_yarn.cos.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs, "YaRN table should differ from standard");
    }

    #[test]
    fn test_yarn_cos_sq_sin_sq_invariant() {
        let cfg = RopeConfig::new(
            64,
            64,
            10_000.0,
            ScalingType::Yarn {
                factor: 4.0,
                attention_factor: 0.7,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
            RopeType::YaRN,
        );
        let table = TableGenerator::generate(&cfg).unwrap();
        for pos in 0..table.seq_len {
            for i in 0..table.half_dim {
                let (c, s) = table.get(pos, i).unwrap();
                let sum = c * c + s * s;
                assert!((sum - 1.0).abs() < 1e-5, "YaRN cos²+sin²≠1 at pos={pos}, i={i}: {sum}");
            }
        }
    }

    // ===================================================================
    // Dynamic NTK
    // ===================================================================

    #[test]
    fn test_dynamic_ntk_below_threshold_uses_original_base() {
        let dyn_ntk = DynamicNtk::new(2.0, 2048);
        let eff = dyn_ntk.effective_base(10_000.0, 64, 1024);
        assert!((eff - 10_000.0).abs() < 1e-3);
    }

    #[test]
    fn test_dynamic_ntk_above_threshold_scales_base() {
        let dyn_ntk = DynamicNtk::new(2.0, 2048);
        let eff = dyn_ntk.effective_base(10_000.0, 64, 4096);
        assert!(eff > 10_000.0, "effective base should increase");
    }

    #[test]
    fn test_dynamic_ntk_at_boundary() {
        let dyn_ntk = DynamicNtk::new(2.0, 2048);
        let eff = dyn_ntk.effective_base(10_000.0, 64, 2048);
        assert!((eff - 10_000.0).abs() < 1e-3, "at boundary, no scaling");
    }

    #[test]
    fn test_dynamic_ntk_longer_seq_larger_base() {
        let dyn_ntk = DynamicNtk::new(2.0, 2048);
        let eff_4k = dyn_ntk.effective_base(10_000.0, 64, 4096);
        let eff_8k = dyn_ntk.effective_base(10_000.0, 64, 8192);
        assert!(eff_8k > eff_4k, "longer seq should yield larger base");
    }

    #[test]
    fn test_dynamic_ntk_compute() {
        let dyn_ntk = DynamicNtk::new(2.0, 2048);
        let inv = dyn_ntk.compute(64, 10_000.0, 4096);
        assert_eq!(inv.len(), 32);
        for &v in &inv {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_dynamic_ntk_table_via_generator() {
        let cfg = RopeConfig::new(
            64,
            128,
            10_000.0,
            ScalingType::DynamicNtk { factor: 2.0, original_max_seq_len: 64 },
            RopeType::Dynamic,
        );
        let table = TableGenerator::generate(&cfg).unwrap();
        // Should produce valid cos/sin
        for pos in 0..table.seq_len {
            for i in 0..table.half_dim {
                let (c, s) = table.get(pos, i).unwrap();
                let sum = c * c + s * s;
                assert!((sum - 1.0).abs() < 1e-5);
            }
        }
    }

    // ===================================================================
    // CodeLlama RoPE
    // ===================================================================

    #[test]
    fn test_codellama_inv_freq_factor_one_same_as_standard() {
        let std_inv = compute_inv_freq(64, 1_000_000.0);
        let cl_inv = codellama_inv_freq(64, 1_000_000.0, 1.0);
        for (a, b) in std_inv.iter().zip(cl_inv.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_codellama_inv_freq_factor_gt1_differs() {
        let cl_1 = codellama_inv_freq(64, 1_000_000.0, 1.0);
        let cl_2 = codellama_inv_freq(64, 1_000_000.0, 2.0);
        let differs = cl_1.iter().zip(cl_2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs);
    }

    #[test]
    fn test_codellama_table_generation() {
        let cfg =
            RopeConfig::new(64, 128, 1_000_000.0, ScalingType::Linear(2.0), RopeType::CodeLlama);
        let table = TableGenerator::generate(&cfg).unwrap();
        assert_eq!(table.half_dim, 32);
        assert_eq!(table.seq_len, 128);
    }

    // ===================================================================
    // FreqTableCache
    // ===================================================================

    #[test]
    fn test_cache_miss_then_hit() {
        let mut cache = FreqTableCache::new(4);
        let cfg = default_config();
        assert!(cache.get(&cfg).is_none());
        assert_eq!(cache.misses(), 1);

        let table = TableGenerator::generate(&cfg).unwrap();
        cache.insert(&cfg, table);
        assert!(cache.get(&cfg).is_some());
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = FreqTableCache::new(2);

        let c1 = config_with(64, 128, 10_000.0);
        let c2 = config_with(128, 128, 10_000.0);
        let c3 = config_with(32, 128, 10_000.0);

        cache.insert(&c1, TableGenerator::generate(&c1).unwrap());
        cache.insert(&c2, TableGenerator::generate(&c2).unwrap());
        assert_eq!(cache.len(), 2);

        // Inserting c3 should evict c1 (LRU)
        cache.insert(&c3, TableGenerator::generate(&c3).unwrap());
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&c1).is_none()); // evicted
        assert!(cache.get(&c2).is_some()); // still present
        assert!(cache.get(&c3).is_some()); // just inserted
    }

    #[test]
    fn test_cache_lru_access_refreshes_entry() {
        let mut cache = FreqTableCache::new(2);

        let c1 = config_with(64, 128, 10_000.0);
        let c2 = config_with(128, 128, 10_000.0);
        let c3 = config_with(32, 128, 10_000.0);

        cache.insert(&c1, TableGenerator::generate(&c1).unwrap());
        cache.insert(&c2, TableGenerator::generate(&c2).unwrap());

        // Access c1 to refresh it (make c2 the LRU)
        let _ = cache.get(&c1);

        // Now insert c3 — should evict c2 (LRU), not c1
        cache.insert(&c3, TableGenerator::generate(&c3).unwrap());
        assert!(cache.get(&c1).is_some()); // refreshed → still present
        assert!(cache.get(&c2).is_none()); // was LRU → evicted
    }

    #[test]
    fn test_cache_get_or_generate() {
        let mut cache = FreqTableCache::new(4);
        let cfg = default_config();
        let table = cache.get_or_generate(&cfg).unwrap();
        assert_eq!(table.half_dim, 32);
        assert_eq!(cache.len(), 1);

        // Second call should hit cache
        let _ = cache.get_or_generate(&cfg).unwrap();
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = FreqTableCache::new(4);
        let cfg = default_config();
        cache.insert(&cfg, TableGenerator::generate(&cfg).unwrap());
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn test_cache_capacity() {
        let cache = FreqTableCache::new(8);
        assert_eq!(cache.capacity(), 8);
    }

    #[test]
    fn test_cache_capacity_min_one() {
        let cache = FreqTableCache::new(0);
        assert_eq!(cache.capacity(), 1);
    }

    #[test]
    fn test_cache_different_configs_are_separate() {
        let mut cache = FreqTableCache::new(8);
        let c1 = config_with(64, 128, 10_000.0);
        let c2 = config_with(64, 128, 500_000.0);
        cache.insert(&c1, TableGenerator::generate(&c1).unwrap());
        cache.insert(&c2, TableGenerator::generate(&c2).unwrap());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_same_config_overwrites() {
        let mut cache = FreqTableCache::new(8);
        let cfg = default_config();
        cache.insert(&cfg, TableGenerator::generate(&cfg).unwrap());
        cache.insert(&cfg, TableGenerator::generate(&cfg).unwrap());
        assert_eq!(cache.len(), 1);
    }

    // ===================================================================
    // CPU apply RoPE
    // ===================================================================

    #[test]
    fn test_cpu_apply_rope_identity_at_position_zero() {
        let cfg = config_with(4, 16, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        let half = table.half_dim; // 2
        let cos = &table.cos[0..half];
        let sin = &table.sin[0..half];
        // cos=[1,1], sin=[0,0] → rotation is identity
        let mut vec = vec![1.0, 2.0, 3.0, 4.0];
        let orig = vec.clone();
        cpu_apply_rope(&mut vec, cos, sin, half);
        for (a, b) in vec.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cpu_apply_rope_batch_basic() {
        let cfg = config_with(4, 16, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        let num_heads = 1;
        let head_dim = 4;
        let mut q = vec![1.0; head_dim];
        let mut k = vec![1.0; head_dim];
        let positions = vec![0_usize];
        cpu_apply_rope_batch(&mut q, &mut k, &table, &positions, num_heads, head_dim).unwrap();
        // At pos 0 → identity rotation → values unchanged
        for &v in &q {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cpu_apply_rope_batch_position_out_of_range() {
        let cfg = config_with(4, 4, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        let mut q = vec![1.0; 4];
        let mut k = vec![1.0; 4];
        let positions = vec![10_usize]; // out of range
        assert!(cpu_apply_rope_batch(&mut q, &mut k, &table, &positions, 1, 4).is_err());
    }

    // ===================================================================
    // Various base values
    // ===================================================================

    #[test]
    fn test_base_10000_table() {
        let cfg = config_with(64, 32, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        assert_eq!(table.half_dim, 32);
    }

    #[test]
    fn test_base_500000_table() {
        let cfg = config_with(64, 32, 500_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        // Higher base → slower frequency decay → different values
        let std_table = TableGenerator::generate(&config_with(64, 32, 10_000.0)).unwrap();
        let differs = table.cos.iter().zip(std_table.cos.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs);
    }

    #[test]
    fn test_base_1000000_table() {
        let cfg = config_with(64, 32, 1_000_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        // cos²+sin²=1 invariant still holds
        for pos in 0..table.seq_len {
            for i in 0..table.half_dim {
                let (c, s) = table.get(pos, i).unwrap();
                assert!((c * c + s * s - 1.0).abs() < 1e-5);
            }
        }
    }

    // ===================================================================
    // Large sequence length
    // ===================================================================

    #[test]
    fn test_large_seq_len() {
        let cfg = config_with(64, 8192, 10_000.0);
        let table = TableGenerator::generate(&cfg).unwrap();
        assert_eq!(table.seq_len, 8192);
        // Spot-check last position
        let (c, s) = table.get(8191, 0).unwrap();
        assert!((c * c + s * s - 1.0).abs() < 1e-5);
    }

    // ===================================================================
    // RopeType display
    // ===================================================================

    #[test]
    fn test_rope_type_display() {
        assert_eq!(format!("{}", RopeType::Standard), "Standard");
        assert_eq!(format!("{}", RopeType::NTK), "NTK");
        assert_eq!(format!("{}", RopeType::YaRN), "YaRN");
        assert_eq!(format!("{}", RopeType::Dynamic), "Dynamic");
        assert_eq!(format!("{}", RopeType::CodeLlama), "CodeLlama");
    }

    // ===================================================================
    // Property: cos²+sin²=1 for all RoPE variants
    // ===================================================================

    #[test]
    fn test_ntk_cos_sq_sin_sq_invariant() {
        let cfg = RopeConfig::new(64, 64, 10_000.0, ScalingType::NtkAware(4.0), RopeType::NTK);
        let table = TableGenerator::generate(&cfg).unwrap();
        for pos in 0..table.seq_len {
            for i in 0..table.half_dim {
                let (c, s) = table.get(pos, i).unwrap();
                assert!((c * c + s * s - 1.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_dynamic_cos_sq_sin_sq_invariant() {
        let cfg = RopeConfig::new(
            64,
            64,
            10_000.0,
            ScalingType::DynamicNtk { factor: 2.0, original_max_seq_len: 32 },
            RopeType::Dynamic,
        );
        let table = TableGenerator::generate(&cfg).unwrap();
        for pos in 0..table.seq_len {
            for i in 0..table.half_dim {
                let (c, s) = table.get(pos, i).unwrap();
                assert!((c * c + s * s - 1.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_codellama_cos_sq_sin_sq_invariant() {
        let cfg =
            RopeConfig::new(64, 64, 1_000_000.0, ScalingType::Linear(2.0), RopeType::CodeLlama);
        let table = TableGenerator::generate(&cfg).unwrap();
        for pos in 0..table.seq_len {
            for i in 0..table.half_dim {
                let (c, s) = table.get(pos, i).unwrap();
                assert!((c * c + s * s - 1.0).abs() < 1e-5);
            }
        }
    }

    // ===================================================================
    // OpenCL kernel source smoke test
    // ===================================================================

    #[test]
    fn test_opencl_kernel_source_is_non_empty() {
        assert!(!OPENCL_ROPE_TABLE_SRC.is_empty());
    }

    #[test]
    fn test_opencl_kernel_contains_apply_rope_table() {
        assert!(OPENCL_ROPE_TABLE_SRC.contains("apply_rope_table"));
    }

    #[test]
    fn test_opencl_kernel_contains_compute_rope_table() {
        assert!(OPENCL_ROPE_TABLE_SRC.contains("compute_rope_table"));
    }

    // ===================================================================
    // ScalingType default
    // ===================================================================

    #[test]
    fn test_scaling_type_default_is_none() {
        assert_eq!(ScalingType::default(), ScalingType::None);
    }

    // ===================================================================
    // Error display
    // ===================================================================

    #[test]
    fn test_error_display_invalid_config() {
        let e = RopeTableError::InvalidConfig("bad".into());
        assert!(format!("{e}").contains("bad"));
    }

    #[test]
    fn test_error_display_cache_miss() {
        let e = RopeTableError::CacheMiss;
        assert!(format!("{e}").contains("cache miss"));
    }

    // ===================================================================
    // Packed generation convenience methods
    // ===================================================================

    #[test]
    fn test_generate_packed_interleaved() {
        let cfg = config_with(4, 4, 10_000.0);
        let packed = TableGenerator::generate_packed_interleaved(&cfg).unwrap();
        let table = TableGenerator::generate(&cfg).unwrap();
        assert_eq!(packed.len(), table.cos.len() * 2);
    }

    #[test]
    fn test_generate_packed_contiguous() {
        let cfg = config_with(4, 4, 10_000.0);
        let packed = TableGenerator::generate_packed_contiguous(&cfg).unwrap();
        let table = TableGenerator::generate(&cfg).unwrap();
        assert_eq!(packed.len(), table.cos.len() * 2);
    }

    // ===================================================================
    // NTK table through TableGenerator
    // ===================================================================

    #[test]
    fn test_ntk_table_differs_from_standard() {
        let cfg_std = config_with(64, 128, 10_000.0);
        let cfg_ntk = RopeConfig::new(64, 128, 10_000.0, ScalingType::NtkAware(4.0), RopeType::NTK);
        let t_std = TableGenerator::generate(&cfg_std).unwrap();
        let t_ntk = TableGenerator::generate(&cfg_ntk).unwrap();
        let differs = t_std.cos.iter().zip(t_ntk.cos.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs);
    }

    // ===================================================================
    // Linear scaling
    // ===================================================================

    #[test]
    fn test_linear_scaling_differs_from_standard() {
        let cfg_std = config_with(64, 128, 10_000.0);
        let cfg_lin =
            RopeConfig::new(64, 128, 10_000.0, ScalingType::Linear(2.0), RopeType::Standard);
        let t_std = TableGenerator::generate(&cfg_std).unwrap();
        let t_lin = TableGenerator::generate(&cfg_lin).unwrap();
        let differs = t_std.cos.iter().zip(t_lin.cos.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs);
    }
}
