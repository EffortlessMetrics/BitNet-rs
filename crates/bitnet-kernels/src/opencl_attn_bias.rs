//! OpenCL-accelerated attention bias patterns for transformer models.
//!
//! Provides CPU reference implementations and OpenCL kernel sources for
//! various attention bias patterns commonly used in modern LLMs:
//!
//! - **Causal mask** — triangular mask preventing future-token attention
//! - **ALiBi** — Attention with Linear Biases (Press et al., 2022)
//! - **Relative position** — learned relative position buckets (T5-style)
//! - **T5 bias buckets** — logarithmic distance bucketing
//! - **Rotary bias** — placeholder for RoPE-derived biases
//! - **Composite bias** — additive combination of multiple patterns
//!
//! All bias matrices are `[num_heads, seq_len, seq_len]` in row-major order.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ── OpenCL kernel source ─────────────────────────────────────────

/// OpenCL C kernel that applies a pre-computed bias matrix to attention
/// scores in-place: `scores[h][i][j] += bias[h][i][j]`.
pub const ATTN_BIAS_CL: &str = r#"
__kernel void apply_bias(
    __global float* scores,       // [num_heads, seq_len, seq_len]
    __global const float* bias,   // [num_heads, seq_len, seq_len]
    const int seq_len
) {
    int gid = get_global_id(0);
    int total = get_global_size(0);
    if (gid < total) {
        scores[gid] += bias[gid];
    }
}

__kernel void generate_causal_mask(
    __global float* mask,    // [seq_len, seq_len]
    const int seq_len,
    const float neg_inf,
    const int prefix_len     // positions [0, prefix_len) are unmasked
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < seq_len && j < seq_len) {
        // Allow attending to prefix and causal positions
        mask[i * seq_len + j] = (j <= i || j < prefix_len) ? 0.0f : neg_inf;
    }
}

__kernel void generate_alibi_bias(
    __global float* bias,    // [num_heads, seq_len, seq_len]
    __global const float* slopes, // [num_heads]
    const int num_heads,
    const int seq_len
) {
    int h = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);
    if (h < num_heads && i < seq_len && j < seq_len) {
        int dist = j - i;
        bias[h * seq_len * seq_len + i * seq_len + j] = slopes[h] * (float)dist;
    }
}
"#;

// ── Bias Pattern Enum ────────────────────────────────────────────

/// Enumerates the supported attention bias patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BiasPattern {
    /// No bias applied (identity).
    NoBias,
    /// Lower-triangular causal mask (−∞ for future positions).
    CausalMask,
    /// Attention with Linear Biases — slopes decrease geometrically.
    ALiBiSlopes,
    /// Learned relative position buckets (T5-style).
    RelativePosition,
    /// T5 logarithmic distance bucketing.
    T5Bias,
    /// Placeholder for RoPE-derived position biases.
    RotaryBias,
}

impl fmt::Display for BiasPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoBias => write!(f, "NoBias"),
            Self::CausalMask => write!(f, "CausalMask"),
            Self::ALiBiSlopes => write!(f, "ALiBiSlopes"),
            Self::RelativePosition => write!(f, "RelativePosition"),
            Self::T5Bias => write!(f, "T5Bias"),
            Self::RotaryBias => write!(f, "RotaryBias"),
        }
    }
}

// ── ALiBi Generator ──────────────────────────────────────────────

/// Generates ALiBi (Attention with Linear Biases) slopes and bias
/// matrices.
///
/// Each head receives a slope `m_h = 2^(-8h/n)` where `h` is the
/// 1-indexed head number and `n` is the total head count. The bias
/// for position pair `(i, j)` is `m_h * (j - i)`.
#[derive(Debug, Clone)]
pub struct ALiBiGenerator {
    num_heads: usize,
    slopes: Vec<f32>,
}

impl ALiBiGenerator {
    /// Create a generator for `num_heads` attention heads.
    ///
    /// # Panics
    ///
    /// Panics if `num_heads` is zero.
    pub fn new(num_heads: usize) -> Self {
        assert!(num_heads > 0, "num_heads must be > 0");
        let slopes = Self::compute_slopes(num_heads);
        Self { num_heads, slopes }
    }

    /// Compute the geometric ALiBi slopes: `2^(-8 * h / n)` for
    /// `h` in `1..=n`.
    fn compute_slopes(num_heads: usize) -> Vec<f32> {
        let ratio = 8.0_f64 / num_heads as f64;
        (1..=num_heads).map(|h| 2.0_f64.powf(-ratio * h as f64) as f32).collect()
    }

    /// Return the slopes for all heads.
    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }

    /// Number of heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Generate the full bias matrix `[num_heads, seq_len, seq_len]`.
    ///
    /// `bias[h][i][j] = slope[h] * (j - i)`
    pub fn generate(&self, seq_len: usize) -> Vec<f32> {
        let mut bias = vec![0.0_f32; self.num_heads * seq_len * seq_len];
        for (h, &slope) in self.slopes.iter().enumerate() {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let dist = j as i64 - i as i64;
                    bias[h * seq_len * seq_len + i * seq_len + j] = slope * dist as f32;
                }
            }
        }
        bias
    }

    /// Generate bias for a single head.
    pub fn generate_head(&self, head: usize, seq_len: usize) -> Vec<f32> {
        assert!(head < self.num_heads, "head index out of range");
        let slope = self.slopes[head];
        let mut bias = vec![0.0_f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let dist = j as i64 - i as i64;
                bias[i * seq_len + j] = slope * dist as f32;
            }
        }
        bias
    }
}

// ── T5 Bias Buckets ──────────────────────────────────────────────

/// Implements T5-style logarithmic distance bucketing for relative
/// position bias.
///
/// Distances are mapped to a fixed number of buckets:
/// - Bucket 0: exact positions
/// - Buckets 1..half: linear for small distances
/// - Buckets half..num_buckets: logarithmic for large distances
///
/// Separate buckets for positive vs. negative relative positions when
/// `bidirectional` is true.
#[derive(Debug, Clone)]
pub struct T5BiasBuckets {
    num_buckets: usize,
    max_distance: usize,
    bidirectional: bool,
}

impl T5BiasBuckets {
    /// Create a bucket configuration.
    ///
    /// # Panics
    ///
    /// Panics if `num_buckets` is zero or `max_distance` is zero.
    pub fn new(num_buckets: usize, max_distance: usize, bidirectional: bool) -> Self {
        assert!(num_buckets > 0, "num_buckets must be > 0");
        assert!(max_distance > 0, "max_distance must be > 0");
        Self { num_buckets, max_distance, bidirectional }
    }

    /// Map a signed relative position `(j - i)` to a bucket index.
    pub fn relative_position_bucket(&self, relative_position: i64) -> usize {
        let mut rp = relative_position;
        let num_buckets = self.num_buckets;
        let max_distance = self.max_distance as f64;

        let mut bucket: usize = 0;

        if self.bidirectional {
            let half = num_buckets / 2;
            if rp > 0 {
                bucket += half;
            } else {
                rp = -rp;
            }
            let rp_abs = rp as usize;
            let max_exact = half / 2;
            if rp_abs < max_exact {
                bucket += rp_abs;
            } else {
                let log_val = (rp_abs as f64 / max_exact as f64).ln()
                    / (max_distance / max_exact as f64).ln();
                let b = max_exact as f64 + log_val * (half - max_exact) as f64;
                bucket += b.min((half - 1) as f64) as usize;
            }
        } else {
            rp = -rp.min(0);
            let rp_abs = rp as usize;
            let max_exact = num_buckets / 2;
            if rp_abs < max_exact {
                bucket = rp_abs;
            } else {
                let log_val = (rp_abs as f64 / max_exact as f64).ln()
                    / (max_distance / max_exact as f64).ln();
                let b = max_exact as f64 + log_val * (num_buckets - max_exact) as f64;
                bucket = b.min((num_buckets - 1) as f64) as usize;
            }
        }
        bucket
    }

    /// Generate the full bucket index matrix `[seq_len, seq_len]`.
    pub fn compute_bucket_matrix(&self, seq_len: usize) -> Vec<usize> {
        let mut buckets = vec![0usize; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let rp = j as i64 - i as i64;
                buckets[i * seq_len + j] = self.relative_position_bucket(rp);
            }
        }
        buckets
    }

    /// Number of buckets.
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    /// Maximum distance.
    pub fn max_distance(&self) -> usize {
        self.max_distance
    }

    /// Whether bidirectional bucketing is enabled.
    pub fn is_bidirectional(&self) -> bool {
        self.bidirectional
    }
}

// ── Relative Position Bias ───────────────────────────────────────

/// Learned relative position bias using T5-style bucketing.
///
/// Stores a learned bias table `[num_heads, num_buckets]` and uses
/// [`T5BiasBuckets`] to map position pairs to bucket indices.
#[derive(Debug, Clone)]
pub struct RelativePositionBias {
    /// Learned bias values: `[num_heads, num_buckets]`.
    pub bias_table: Vec<f32>,
    num_heads: usize,
    buckets: T5BiasBuckets,
}

impl RelativePositionBias {
    /// Create a new relative position bias with the given learned
    /// table.
    ///
    /// `bias_table` must have length `num_heads * num_buckets`.
    ///
    /// # Panics
    ///
    /// Panics if the table length does not match.
    pub fn new(num_heads: usize, buckets: T5BiasBuckets, bias_table: Vec<f32>) -> Self {
        assert_eq!(
            bias_table.len(),
            num_heads * buckets.num_buckets(),
            "bias_table length mismatch"
        );
        Self { bias_table, num_heads, buckets }
    }

    /// Create with a zero-initialized table (useful for testing).
    pub fn zeros(num_heads: usize, buckets: T5BiasBuckets) -> Self {
        let len = num_heads * buckets.num_buckets();
        Self { bias_table: vec![0.0; len], num_heads, buckets }
    }

    /// Generate the full bias matrix `[num_heads, seq_len, seq_len]`.
    pub fn generate(&self, seq_len: usize) -> Vec<f32> {
        let bucket_matrix = self.buckets.compute_bucket_matrix(seq_len);
        let nb = self.buckets.num_buckets();
        let mut bias = vec![0.0_f32; self.num_heads * seq_len * seq_len];
        for h in 0..self.num_heads {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let bucket = bucket_matrix[i * seq_len + j];
                    bias[h * seq_len * seq_len + i * seq_len + j] =
                        self.bias_table[h * nb + bucket];
                }
            }
        }
        bias
    }

    /// Number of heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}

// ── Causal Mask Generator ────────────────────────────────────────

/// Generates a lower-triangular causal mask.
///
/// `mask[i][j] = 0.0` when `j <= i` (allowed) and `NEG_INF` when
/// `j > i` (masked). An optional prefix length allows the first
/// `prefix_len` positions to be attended to by all positions.
#[derive(Debug, Clone)]
pub struct CausalMaskGenerator {
    prefix_len: usize,
}

/// Value used to represent −∞ in attention masks.
pub const NEG_INF: f32 = -1e9;

impl CausalMaskGenerator {
    /// Create a causal mask generator with no prefix.
    pub fn new() -> Self {
        Self { prefix_len: 0 }
    }

    /// Create a causal mask generator with a prefix of `prefix_len`
    /// unmasked positions.
    pub fn with_prefix(prefix_len: usize) -> Self {
        Self { prefix_len }
    }

    /// Generate the mask matrix `[seq_len, seq_len]`.
    pub fn generate(&self, seq_len: usize) -> Vec<f32> {
        let mut mask = vec![0.0_f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i && j >= self.prefix_len {
                    mask[i * seq_len + j] = NEG_INF;
                }
            }
        }
        mask
    }

    /// Generate the mask broadcast across `num_heads` heads:
    /// `[num_heads, seq_len, seq_len]`.
    pub fn generate_multi_head(&self, num_heads: usize, seq_len: usize) -> Vec<f32> {
        let single = self.generate(seq_len);
        let head_size = seq_len * seq_len;
        let mut out = vec![0.0_f32; num_heads * head_size];
        for h in 0..num_heads {
            out[h * head_size..(h + 1) * head_size].copy_from_slice(&single);
        }
        out
    }

    /// Prefix length.
    pub fn prefix_len(&self) -> usize {
        self.prefix_len
    }
}

impl Default for CausalMaskGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Composite Bias ───────────────────────────────────────────────

/// Combines multiple bias patterns additively.
///
/// Given a list of bias matrices (all with shape
/// `[num_heads, seq_len, seq_len]`), produces the element-wise sum.
#[derive(Debug, Clone)]
pub struct CompositeBias {
    components: Vec<(BiasPattern, Vec<f32>)>,
    num_heads: usize,
    seq_len: usize,
}

impl CompositeBias {
    /// Create a new composite bias aggregator.
    pub fn new(num_heads: usize, seq_len: usize) -> Self {
        Self { components: Vec::new(), num_heads, seq_len }
    }

    /// Add a bias component.
    ///
    /// # Panics
    ///
    /// Panics if the bias length does not match
    /// `num_heads * seq_len * seq_len`.
    pub fn add(&mut self, pattern: BiasPattern, bias: Vec<f32>) {
        let expected = self.num_heads * self.seq_len * self.seq_len;
        assert_eq!(bias.len(), expected, "bias length {} != expected {}", bias.len(), expected,);
        self.components.push((pattern, bias));
    }

    /// Compute the element-wise sum of all components.
    pub fn combine(&self) -> Vec<f32> {
        let size = self.num_heads * self.seq_len * self.seq_len;
        let mut result = vec![0.0_f32; size];
        for (_, bias) in &self.components {
            for (r, b) in result.iter_mut().zip(bias.iter()) {
                *r += *b;
            }
        }
        result
    }

    /// Number of registered components.
    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// List the patterns in this composite.
    pub fn patterns(&self) -> Vec<BiasPattern> {
        self.components.iter().map(|(p, _)| *p).collect()
    }
}

// ── Bias Cache ───────────────────────────────────────────────────

/// Cache key for bias matrices.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BiasCacheKey {
    pattern: BiasPattern,
    seq_len: usize,
    num_heads: usize,
}

/// Caches generated bias matrices keyed by
/// `(pattern, seq_len, num_heads)`.
///
/// Avoids recomputing expensive bias matrices when the same
/// configuration is requested repeatedly.
pub struct BiasCache {
    entries: HashMap<BiasCacheKey, Vec<f32>>,
    hits: u64,
    misses: u64,
    max_entries: usize,
}

impl fmt::Debug for BiasCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BiasCache")
            .field("entries", &self.entries.len())
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .finish()
    }
}

impl BiasCache {
    /// Create a new bias cache with the specified maximum number of
    /// entries.
    pub fn new(max_entries: usize) -> Self {
        Self { entries: HashMap::new(), hits: 0, misses: 0, max_entries }
    }

    /// Look up a cached bias matrix.
    pub fn get(
        &mut self,
        pattern: BiasPattern,
        seq_len: usize,
        num_heads: usize,
    ) -> Option<&Vec<f32>> {
        let key = BiasCacheKey { pattern, seq_len, num_heads };
        if self.entries.contains_key(&key) {
            self.hits += 1;
            self.entries.get(&key)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a bias matrix into the cache. Evicts oldest entry if
    /// at capacity (FIFO-ish via HashMap iteration order).
    pub fn insert(
        &mut self,
        pattern: BiasPattern,
        seq_len: usize,
        num_heads: usize,
        bias: Vec<f32>,
    ) {
        if self.entries.len() >= self.max_entries
            && let Some(key) = self.entries.keys().next().cloned()
        {
            self.entries.remove(&key);
        }
        let key = BiasCacheKey { pattern, seq_len, num_heads };
        self.entries.insert(key, bias);
    }

    /// Total cache hits.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Total cache misses.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Current number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries and reset counters.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Hit rate as a fraction in [0, 1].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    /// Total memory used by cached bias matrices (bytes).
    pub fn memory_bytes(&self) -> usize {
        self.entries.values().map(|v| v.len() * std::mem::size_of::<f32>()).sum()
    }
}

// ── Bias Stats ───────────────────────────────────────────────────

/// Statistics about bias generation and caching.
#[derive(Debug, Clone)]
pub struct BiasStats {
    /// Memory used by cached biases (bytes).
    pub memory_bytes: usize,
    /// Time spent generating biases.
    pub generation_time: Duration,
    /// Cache hit rate `[0, 1]`.
    pub cache_hit_rate: f64,
    /// Number of bias matrices generated.
    pub matrices_generated: u64,
}

impl BiasStats {
    /// Collect stats from a [`BiasCache`].
    pub fn from_cache(cache: &BiasCache, gen_time: Duration) -> Self {
        Self {
            memory_bytes: cache.memory_bytes(),
            generation_time: gen_time,
            cache_hit_rate: cache.hit_rate(),
            matrices_generated: cache.misses(),
        }
    }
}

impl fmt::Display for BiasStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BiasStats {{ memory: {} B, gen_time: {:?}, hit_rate: {:.2}%, \
             generated: {} }}",
            self.memory_bytes,
            self.generation_time,
            self.cache_hit_rate * 100.0,
            self.matrices_generated,
        )
    }
}

// ── CPU Reference: generate_no_bias ──────────────────────────────

/// Generate an all-zeros bias matrix (no bias).
pub fn generate_no_bias(num_heads: usize, seq_len: usize) -> Vec<f32> {
    vec![0.0_f32; num_heads * seq_len * seq_len]
}

// ── CPU Reference: apply_bias_inplace ────────────────────────────

/// Apply a bias matrix to attention scores in-place (CPU reference).
///
/// Both `scores` and `bias` must have length
/// `num_heads * seq_len * seq_len`.
pub fn apply_bias_inplace(scores: &mut [f32], bias: &[f32]) {
    assert_eq!(scores.len(), bias.len(), "scores and bias must have the same length");
    for (s, b) in scores.iter_mut().zip(bias.iter()) {
        *s += *b;
    }
}

// ── CPU Reference: generate_rotary_bias ──────────────────────────

/// Placeholder for RoPE-derived bias (returns zeros).
pub fn generate_rotary_bias(num_heads: usize, seq_len: usize) -> Vec<f32> {
    vec![0.0_f32; num_heads * seq_len * seq_len]
}

// ── Convenience: generate bias for any pattern ───────────────────

/// Generate a bias matrix for the given pattern using default
/// parameters.
///
/// This is a convenience entry point for testing and benchmarking.
/// For production use, prefer the individual generators.
pub fn generate_bias(pattern: BiasPattern, num_heads: usize, seq_len: usize) -> Vec<f32> {
    match pattern {
        BiasPattern::NoBias => generate_no_bias(num_heads, seq_len),
        BiasPattern::CausalMask => {
            CausalMaskGenerator::new().generate_multi_head(num_heads, seq_len)
        }
        BiasPattern::ALiBiSlopes => ALiBiGenerator::new(num_heads).generate(seq_len),
        BiasPattern::RelativePosition | BiasPattern::T5Bias => {
            let buckets = T5BiasBuckets::new(32, 128, true);
            RelativePositionBias::zeros(num_heads, buckets).generate(seq_len)
        }
        BiasPattern::RotaryBias => generate_rotary_bias(num_heads, seq_len),
    }
}

/// Measure generation time for a bias pattern.
pub fn timed_generate(
    pattern: BiasPattern,
    num_heads: usize,
    seq_len: usize,
) -> (Vec<f32>, Duration) {
    let start = Instant::now();
    let bias = generate_bias(pattern, num_heads, seq_len);
    (bias, start.elapsed())
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BiasPattern enum tests ───────────────────────────────

    #[test]
    fn test_bias_pattern_display() {
        assert_eq!(BiasPattern::NoBias.to_string(), "NoBias");
        assert_eq!(BiasPattern::CausalMask.to_string(), "CausalMask");
        assert_eq!(BiasPattern::ALiBiSlopes.to_string(), "ALiBiSlopes");
        assert_eq!(BiasPattern::RelativePosition.to_string(), "RelativePosition");
        assert_eq!(BiasPattern::T5Bias.to_string(), "T5Bias");
        assert_eq!(BiasPattern::RotaryBias.to_string(), "RotaryBias");
    }

    #[test]
    fn test_bias_pattern_eq_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(BiasPattern::NoBias);
        set.insert(BiasPattern::CausalMask);
        set.insert(BiasPattern::ALiBiSlopes);
        assert_eq!(set.len(), 3);
        assert!(set.contains(&BiasPattern::NoBias));
    }

    #[test]
    fn test_bias_pattern_clone_copy() {
        let a = BiasPattern::CausalMask;
        let b = a;
        assert_eq!(a, b);
    }

    // ── No-bias tests ────────────────────────────────────────

    #[test]
    fn test_no_bias_all_zeros() {
        let bias = generate_no_bias(4, 8);
        assert!(bias.iter().all(|&v| v == 0.0));
        assert_eq!(bias.len(), 4 * 8 * 8);
    }

    #[test]
    fn test_no_bias_single_head_single_pos() {
        let bias = generate_no_bias(1, 1);
        assert_eq!(bias, vec![0.0]);
    }

    #[test]
    fn test_generate_bias_nobias_variant() {
        let bias = generate_bias(BiasPattern::NoBias, 2, 4);
        assert!(bias.iter().all(|&v| v == 0.0));
    }

    // ── Causal mask tests ────────────────────────────────────

    #[test]
    fn test_causal_mask_triangular_structure() {
        let cmg = CausalMaskGenerator::new();
        let mask = cmg.generate(4);
        // Lower-triangular + diagonal should be 0
        for i in 0..4 {
            for j in 0..4 {
                if j <= i {
                    assert_eq!(mask[i * 4 + j], 0.0, "mask[{i}][{j}] should be 0");
                } else {
                    assert_eq!(mask[i * 4 + j], NEG_INF, "mask[{i}][{j}] should be NEG_INF");
                }
            }
        }
    }

    #[test]
    fn test_causal_mask_property_j_gt_i_is_neg_inf() {
        let cmg = CausalMaskGenerator::new();
        for seq_len in [1, 2, 5, 16, 33] {
            let mask = cmg.generate(seq_len);
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let val = mask[i * seq_len + j];
                    if j > i {
                        assert_eq!(val, NEG_INF, "seq={seq_len} mask[{i}][{j}]");
                    } else {
                        assert_eq!(val, 0.0, "seq={seq_len} mask[{i}][{j}]");
                    }
                }
            }
        }
    }

    #[test]
    fn test_causal_mask_seq_len_1() {
        let cmg = CausalMaskGenerator::new();
        let mask = cmg.generate(1);
        assert_eq!(mask, vec![0.0]);
    }

    #[test]
    fn test_causal_mask_with_prefix() {
        let cmg = CausalMaskGenerator::with_prefix(2);
        let mask = cmg.generate(4);
        // Positions 0 and 1 (prefix) should always be unmasked
        for i in 0..4 {
            assert_eq!(mask[i * 4 + 0], 0.0);
            assert_eq!(mask[i * 4 + 1], 0.0);
        }
        // Position 2 should be masked for i=0 (since j=2 > i=0
        // AND j >= prefix_len)
        assert_eq!(mask[0 * 4 + 2], NEG_INF);
        // Position 2 should be unmasked for i=2 (j <= i)
        assert_eq!(mask[2 * 4 + 2], 0.0);
    }

    #[test]
    fn test_causal_mask_multi_head_broadcast() {
        let cmg = CausalMaskGenerator::new();
        let mh = cmg.generate_multi_head(3, 4);
        let single = cmg.generate(4);
        assert_eq!(mh.len(), 3 * 16);
        for h in 0..3 {
            assert_eq!(&mh[h * 16..(h + 1) * 16], single.as_slice(),);
        }
    }

    #[test]
    fn test_causal_mask_default() {
        let cmg = CausalMaskGenerator::default();
        assert_eq!(cmg.prefix_len(), 0);
    }

    #[test]
    fn test_causal_mask_prefix_len_accessor() {
        let cmg = CausalMaskGenerator::with_prefix(5);
        assert_eq!(cmg.prefix_len(), 5);
    }

    // ── ALiBi tests ──────────────────────────────────────────

    #[test]
    fn test_alibi_slopes_geometric_sequence() {
        let alibi = ALiBiGenerator::new(8);
        let slopes = alibi.slopes();
        // Each slope should be 2^(-8*h/8) = 2^(-h)
        for h in 0..8 {
            let expected = 2.0_f64.powi(-((h + 1) as i32)) as f32;
            assert!(
                (slopes[h] - expected).abs() < 1e-6,
                "slope[{h}] = {}, expected {}",
                slopes[h],
                expected,
            );
        }
    }

    #[test]
    fn test_alibi_slopes_geometric_ratio() {
        // Consecutive slopes should have constant ratio
        let alibi = ALiBiGenerator::new(4);
        let slopes = alibi.slopes();
        let ratio = slopes[1] / slopes[0];
        for i in 1..3 {
            let r = slopes[i + 1] / slopes[i];
            assert!((r - ratio).abs() < 1e-5, "ratio not constant: {} vs {}", r, ratio,);
        }
    }

    #[test]
    fn test_alibi_slopes_decreasing() {
        let alibi = ALiBiGenerator::new(16);
        let slopes = alibi.slopes();
        for i in 1..16 {
            assert!(slopes[i] < slopes[i - 1], "slopes not decreasing at {}", i,);
        }
    }

    #[test]
    fn test_alibi_bias_linear_in_distance() {
        let alibi = ALiBiGenerator::new(4);
        let bias = alibi.generate(8);
        // For each head, bias[i][j] = slope * (j - i)
        let slopes = alibi.slopes();
        for h in 0..4 {
            for i in 0..8 {
                for j in 0..8 {
                    let expected = slopes[h] * (j as i64 - i as i64) as f32;
                    let actual = bias[h * 64 + i * 8 + j];
                    assert!(
                        (actual - expected).abs() < 1e-6,
                        "h={h} i={i} j={j}: {} vs {}",
                        actual,
                        expected,
                    );
                }
            }
        }
    }

    #[test]
    fn test_alibi_diagonal_is_zero() {
        let alibi = ALiBiGenerator::new(4);
        let bias = alibi.generate(8);
        for h in 0..4 {
            for i in 0..8 {
                assert_eq!(bias[h * 64 + i * 8 + i], 0.0, "diagonal should be zero at h={h} i={i}",);
            }
        }
    }

    #[test]
    fn test_alibi_single_head() {
        let alibi = ALiBiGenerator::new(1);
        assert_eq!(alibi.num_heads(), 1);
        let slopes = alibi.slopes();
        // 2^(-8/1 * 1) = 2^(-8) = 1/256
        let expected = 1.0_f32 / 256.0;
        assert!(
            (slopes[0] - expected).abs() < 1e-8,
            "slope = {}, expected {}",
            slopes[0],
            expected,
        );
    }

    #[test]
    fn test_alibi_generate_head() {
        let alibi = ALiBiGenerator::new(4);
        let full = alibi.generate(6);
        for h in 0..4 {
            let head_bias = alibi.generate_head(h, 6);
            assert_eq!(head_bias.len(), 36);
            for idx in 0..36 {
                assert_eq!(head_bias[idx], full[h * 36 + idx], "mismatch at head {h} idx {idx}",);
            }
        }
    }

    #[test]
    fn test_alibi_negative_distance_negative_bias() {
        // For j < i, distance is negative, so bias should be
        // negative (slope is positive)
        let alibi = ALiBiGenerator::new(2);
        let bias = alibi.generate(4);
        let slopes = alibi.slopes();
        for h in 0..2 {
            for i in 1..4 {
                let val = bias[h * 16 + i * 4 + 0]; // j=0 < i
                assert!(val < 0.0, "bias should be negative for j<i: h={h} i={i}");
                let expected = slopes[h] * (0i64 - i as i64) as f32;
                assert!((val - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_alibi_symmetry_of_magnitude() {
        let alibi = ALiBiGenerator::new(4);
        let bias = alibi.generate(8);
        // |bias[h][i][j]| == |bias[h][j][i]| because both have
        // |distance| = |j-i|
        for h in 0..4 {
            for i in 0..8 {
                for j in 0..8 {
                    let a = bias[h * 64 + i * 8 + j].abs();
                    let b = bias[h * 64 + j * 8 + i].abs();
                    assert!((a - b).abs() < 1e-6, "magnitude asymmetry at h={h} i={i} j={j}",);
                }
            }
        }
    }

    // ── T5 Bias Buckets tests ────────────────────────────────

    #[test]
    fn test_t5_bucket_zero_distance() {
        let buckets = T5BiasBuckets::new(32, 128, true);
        // Distance 0 should map to bucket 0 (negative side)
        assert_eq!(buckets.relative_position_bucket(0), 0);
    }

    #[test]
    fn test_t5_bucket_positive_direction() {
        let buckets = T5BiasBuckets::new(32, 128, true);
        let b = buckets.relative_position_bucket(1);
        // Positive => offset by half (16)
        assert!(b >= 16, "positive distance bucket should be >= half");
    }

    #[test]
    fn test_t5_bucket_monotonic_positive() {
        let buckets = T5BiasBuckets::new(32, 128, true);
        let mut prev = buckets.relative_position_bucket(1);
        for d in 2..=64 {
            let cur = buckets.relative_position_bucket(d);
            assert!(cur >= prev, "buckets should be monotonically non-decreasing");
            prev = cur;
        }
    }

    #[test]
    fn test_t5_bucket_monotonic_negative() {
        let buckets = T5BiasBuckets::new(32, 128, true);
        let mut prev = buckets.relative_position_bucket(-1);
        for d in 2..=64 {
            let cur = buckets.relative_position_bucket(-d);
            assert!(cur >= prev, "negative buckets non-decreasing");
            prev = cur;
        }
    }

    #[test]
    fn test_t5_bucket_within_range() {
        let buckets = T5BiasBuckets::new(32, 128, true);
        for d in -200..=200 {
            let b = buckets.relative_position_bucket(d);
            assert!(b < 32, "bucket {} out of range for d={}", b, d,);
        }
    }

    #[test]
    fn test_t5_bucket_unidirectional() {
        let buckets = T5BiasBuckets::new(32, 128, false);
        // Unidirectional: negative positions mapped to 0
        let b0 = buckets.relative_position_bucket(-5);
        let b1 = buckets.relative_position_bucket(0);
        // Both should be valid bucket indices
        assert!(b0 < 32);
        assert!(b1 < 32);
    }

    #[test]
    fn test_t5_bucket_large_distance_caps() {
        let buckets = T5BiasBuckets::new(32, 128, true);
        let b1 = buckets.relative_position_bucket(1000);
        let b2 = buckets.relative_position_bucket(10000);
        // Both should be capped to the same max bucket
        assert_eq!(b1, b2);
        assert!(b1 < 32);
    }

    #[test]
    fn test_t5_bucket_matrix_dimensions() {
        let buckets = T5BiasBuckets::new(32, 128, true);
        let mat = buckets.compute_bucket_matrix(8);
        assert_eq!(mat.len(), 64);
    }

    #[test]
    fn test_t5_bucket_matrix_diagonal_zero() {
        let buckets = T5BiasBuckets::new(32, 128, true);
        let mat = buckets.compute_bucket_matrix(8);
        for i in 0..8 {
            assert_eq!(mat[i * 8 + i], buckets.relative_position_bucket(0),);
        }
    }

    #[test]
    fn test_t5_bucket_accessors() {
        let b = T5BiasBuckets::new(32, 128, true);
        assert_eq!(b.num_buckets(), 32);
        assert_eq!(b.max_distance(), 128);
        assert!(b.is_bidirectional());
    }

    // ── RelativePositionBias tests ───────────────────────────

    #[test]
    fn test_relative_position_bias_zeros() {
        let buckets = T5BiasBuckets::new(32, 128, true);
        let rpb = RelativePositionBias::zeros(4, buckets);
        let bias = rpb.generate(8);
        assert_eq!(bias.len(), 4 * 64);
        assert!(bias.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_relative_position_bias_learned() {
        let buckets = T5BiasBuckets::new(4, 16, true);
        // 2 heads, 4 buckets => table of length 8
        let table: Vec<f32> = (0..8).map(|i| (i + 1) as f32 * 0.1).collect();
        let rpb = RelativePositionBias::new(2, buckets, table);
        let bias = rpb.generate(4);
        assert_eq!(bias.len(), 2 * 16);
        // Should contain non-zero values (table is non-zero)
        assert!(bias.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_relative_position_bias_num_heads() {
        let buckets = T5BiasBuckets::new(32, 128, true);
        let rpb = RelativePositionBias::zeros(8, buckets);
        assert_eq!(rpb.num_heads(), 8);
    }

    // ── Composite Bias tests ─────────────────────────────────

    #[test]
    fn test_composite_bias_empty_is_zeros() {
        let cb = CompositeBias::new(2, 4);
        let result = cb.combine();
        assert_eq!(result.len(), 2 * 16);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_composite_bias_single_component() {
        let mut cb = CompositeBias::new(1, 2);
        let bias = vec![1.0, 2.0, 3.0, 4.0];
        cb.add(BiasPattern::NoBias, bias.clone());
        assert_eq!(cb.combine(), bias);
    }

    #[test]
    fn test_composite_bias_sum_of_two() {
        let mut cb = CompositeBias::new(1, 2);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        cb.add(BiasPattern::CausalMask, a);
        cb.add(BiasPattern::ALiBiSlopes, b);
        let result = cb.combine();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_composite_bias_is_additive() {
        let num_heads = 2;
        let seq_len = 4;
        let size = num_heads * seq_len * seq_len;

        let alibi = ALiBiGenerator::new(num_heads).generate(seq_len);
        let causal = CausalMaskGenerator::new().generate_multi_head(num_heads, seq_len);

        let mut cb = CompositeBias::new(num_heads, seq_len);
        cb.add(BiasPattern::ALiBiSlopes, alibi.clone());
        cb.add(BiasPattern::CausalMask, causal.clone());

        let combined = cb.combine();
        for idx in 0..size {
            let expected = alibi[idx] + causal[idx];
            assert!(
                (combined[idx] - expected).abs() < 1e-6,
                "idx={idx}: {} vs {}",
                combined[idx],
                expected,
            );
        }
    }

    #[test]
    fn test_composite_bias_num_components() {
        let mut cb = CompositeBias::new(1, 2);
        assert_eq!(cb.num_components(), 0);
        cb.add(BiasPattern::NoBias, vec![0.0; 4]);
        assert_eq!(cb.num_components(), 1);
        cb.add(BiasPattern::CausalMask, vec![0.0; 4]);
        assert_eq!(cb.num_components(), 2);
    }

    #[test]
    fn test_composite_bias_patterns_list() {
        let mut cb = CompositeBias::new(1, 2);
        cb.add(BiasPattern::CausalMask, vec![0.0; 4]);
        cb.add(BiasPattern::ALiBiSlopes, vec![0.0; 4]);
        let p = cb.patterns();
        assert_eq!(p, vec![BiasPattern::CausalMask, BiasPattern::ALiBiSlopes]);
    }

    // ── Bias Cache tests ─────────────────────────────────────

    #[test]
    fn test_cache_miss_then_hit() {
        let mut cache = BiasCache::new(16);
        assert!(cache.get(BiasPattern::NoBias, 4, 2).is_none());
        assert_eq!(cache.misses(), 1);

        cache.insert(BiasPattern::NoBias, 4, 2, vec![0.0; 32]);
        assert!(cache.get(BiasPattern::NoBias, 4, 2).is_some());
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = BiasCache::new(16);
        cache.get(BiasPattern::NoBias, 4, 2); // miss
        cache.insert(BiasPattern::NoBias, 4, 2, vec![0.0; 32]);
        cache.get(BiasPattern::NoBias, 4, 2); // hit
        cache.get(BiasPattern::NoBias, 4, 2); // hit
        // 2 hits / 3 total = 0.667
        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = BiasCache::new(2);
        cache.insert(BiasPattern::NoBias, 4, 1, vec![0.0; 16]);
        cache.insert(BiasPattern::CausalMask, 4, 1, vec![1.0; 16]);
        assert_eq!(cache.len(), 2);
        // Inserting a 3rd should evict one
        cache.insert(BiasPattern::ALiBiSlopes, 4, 1, vec![2.0; 16]);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = BiasCache::new(16);
        cache.insert(BiasPattern::NoBias, 4, 2, vec![0.0; 32]);
        cache.get(BiasPattern::NoBias, 4, 2);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn test_cache_memory_bytes() {
        let mut cache = BiasCache::new(16);
        cache.insert(BiasPattern::NoBias, 2, 1, vec![0.0; 4]);
        // 4 floats × 4 bytes = 16
        assert_eq!(cache.memory_bytes(), 16);
    }

    #[test]
    fn test_cache_different_keys_independent() {
        let mut cache = BiasCache::new(16);
        cache.insert(BiasPattern::NoBias, 4, 2, vec![0.0; 32]);
        cache.insert(BiasPattern::CausalMask, 4, 2, vec![1.0; 32]);
        let a = cache.get(BiasPattern::NoBias, 4, 2).unwrap().clone();
        let b = cache.get(BiasPattern::CausalMask, 4, 2).unwrap().clone();
        assert_ne!(a, b);
    }

    #[test]
    fn test_cache_empty_initial_state() {
        let cache = BiasCache::new(8);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
        assert_eq!(cache.memory_bytes(), 0);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    // ── BiasStats tests ──────────────────────────────────────

    #[test]
    fn test_bias_stats_from_cache() {
        let mut cache = BiasCache::new(16);
        cache.get(BiasPattern::NoBias, 4, 2); // miss
        cache.insert(BiasPattern::NoBias, 4, 2, vec![0.0; 32]);
        cache.get(BiasPattern::NoBias, 4, 2); // hit

        let stats = BiasStats::from_cache(&cache, Duration::from_millis(10));
        assert_eq!(stats.memory_bytes, 128);
        assert_eq!(stats.matrices_generated, 1);
        assert!((stats.cache_hit_rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bias_stats_display() {
        let stats = BiasStats {
            memory_bytes: 1024,
            generation_time: Duration::from_millis(5),
            cache_hit_rate: 0.75,
            matrices_generated: 3,
        };
        let s = stats.to_string();
        assert!(s.contains("1024"));
        assert!(s.contains("75.00%"));
    }

    // ── apply_bias_inplace tests ─────────────────────────────

    #[test]
    fn test_apply_bias_inplace() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![0.1, 0.2, 0.3, 0.4];
        apply_bias_inplace(&mut scores, &bias);
        assert!((scores[0] - 1.1).abs() < 1e-6);
        assert!((scores[3] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn test_apply_bias_inplace_zeros() {
        let mut scores = vec![1.0, 2.0];
        let bias = vec![0.0, 0.0];
        apply_bias_inplace(&mut scores, &bias);
        assert_eq!(scores, vec![1.0, 2.0]);
    }

    // ── Rotary bias placeholder tests ────────────────────────

    #[test]
    fn test_rotary_bias_is_zeros() {
        let bias = generate_rotary_bias(4, 8);
        assert_eq!(bias.len(), 4 * 64);
        assert!(bias.iter().all(|&v| v == 0.0));
    }

    // ── generate_bias convenience tests ──────────────────────

    #[test]
    fn test_generate_bias_causal() {
        let bias = generate_bias(BiasPattern::CausalMask, 2, 4);
        assert_eq!(bias.len(), 2 * 16);
    }

    #[test]
    fn test_generate_bias_alibi() {
        let bias = generate_bias(BiasPattern::ALiBiSlopes, 4, 8);
        assert_eq!(bias.len(), 4 * 64);
    }

    #[test]
    fn test_generate_bias_relative_position() {
        let bias = generate_bias(BiasPattern::RelativePosition, 2, 4);
        assert_eq!(bias.len(), 2 * 16);
    }

    #[test]
    fn test_generate_bias_t5() {
        let bias = generate_bias(BiasPattern::T5Bias, 2, 4);
        assert_eq!(bias.len(), 2 * 16);
    }

    #[test]
    fn test_generate_bias_rotary() {
        let bias = generate_bias(BiasPattern::RotaryBias, 2, 4);
        assert_eq!(bias.len(), 2 * 16);
        assert!(bias.iter().all(|&v| v == 0.0));
    }

    // ── timed_generate tests ─────────────────────────────────

    #[test]
    fn test_timed_generate_returns_valid() {
        let (bias, dur) = timed_generate(BiasPattern::NoBias, 2, 4);
        assert_eq!(bias.len(), 2 * 16);
        assert!(dur.as_nanos() > 0 || dur == Duration::ZERO);
    }

    // ── Edge-case tests ──────────────────────────────────────

    #[test]
    fn test_alibi_large_head_count() {
        let alibi = ALiBiGenerator::new(128);
        let slopes = alibi.slopes();
        assert_eq!(slopes.len(), 128);
        // All slopes should be positive
        assert!(slopes.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn test_causal_mask_large_seq() {
        let cmg = CausalMaskGenerator::new();
        let mask = cmg.generate(256);
        assert_eq!(mask.len(), 256 * 256);
        // Check a few spot values
        assert_eq!(mask[0], 0.0); // [0][0]
        assert_eq!(mask[1], NEG_INF); // [0][1]
        assert_eq!(mask[256 + 1], 0.0); // [1][1]
    }

    #[test]
    fn test_no_bias_very_long_sequence() {
        let bias = generate_no_bias(1, 512);
        assert_eq!(bias.len(), 512 * 512);
        assert!(bias.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_alibi_seq_len_1() {
        let alibi = ALiBiGenerator::new(4);
        let bias = alibi.generate(1);
        // All zeros since distance is always 0
        assert_eq!(bias.len(), 4);
        assert!(bias.iter().all(|&v| v == 0.0));
    }

    // ── OpenCL kernel source tests ───────────────────────────

    #[test]
    fn test_opencl_source_not_empty() {
        assert!(!ATTN_BIAS_CL.is_empty());
    }

    #[test]
    fn test_opencl_source_contains_kernels() {
        assert!(ATTN_BIAS_CL.contains("__kernel"));
        assert!(ATTN_BIAS_CL.contains("apply_bias"));
        assert!(ATTN_BIAS_CL.contains("generate_causal_mask"));
        assert!(ATTN_BIAS_CL.contains("generate_alibi_bias"));
    }

    // ── Debug/Display impl tests ─────────────────────────────

    #[test]
    fn test_cache_debug() {
        let cache = BiasCache::new(8);
        let dbg = format!("{cache:?}");
        assert!(dbg.contains("BiasCache"));
        assert!(dbg.contains("entries"));
    }

    #[test]
    fn test_alibi_generator_debug() {
        let alibi = ALiBiGenerator::new(2);
        let dbg = format!("{alibi:?}");
        assert!(dbg.contains("ALiBiGenerator"));
    }

    #[test]
    fn test_t5_buckets_debug() {
        let b = T5BiasBuckets::new(32, 128, true);
        let dbg = format!("{b:?}");
        assert!(dbg.contains("T5BiasBuckets"));
    }
}
