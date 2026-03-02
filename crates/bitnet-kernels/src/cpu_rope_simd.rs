//! SIMD-accelerated Rotary Position Embedding (RoPE) for CPU inference.
//!
//! Provides precomputed sin/cos caches, runtime SIMD dispatch (AVX2 / NEON /
//! scalar fallback), and extended-context scaling variants (NTK-aware, YaRN).
//!
//! # Dispatch strategy
//!
//! [`apply_rope`] selects the fastest available path at runtime:
//!
//! | Architecture | Feature | Path                |
//! |--------------|---------|---------------------|
//! | x86_64       | AVX2    | [`apply_rope_avx2`] |
//! | aarch64      | NEON    | [`apply_rope_neon`] |
//! | *any*        | —       | scalar fallback     |

use std::f32::consts::PI;
use std::fmt;

// ── Error type ──────────────────────────────────────────────────────

/// Errors produced by RoPE operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RoPEError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Position exceeds precomputed cache bounds.
    PositionOutOfRange { requested: usize, max_cached: usize },
    /// Tensor slice length does not match expected head_dim layout.
    DimensionMismatch { expected: usize, actual: usize },
}

impl fmt::Display for RoPEError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid RoPE config: {msg}"),
            Self::PositionOutOfRange { requested, max_cached } => {
                write!(f, "position {requested} exceeds max cached {max_cached}")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for RoPEError {}

// ── Configuration ───────────────────────────────────────────────────

/// RoPE configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct RoPEConfig {
    /// Per-head embedding dimension (must be even, > 0).
    pub head_dim: usize,
    /// Maximum sequence length to precompute.
    pub max_seq_len: usize,
    /// Base rotation frequency (default `10_000.0`).
    pub base: f32,
    /// Optional global frequency scaling factor.
    pub scaling_factor: Option<f32>,
}

impl RoPEConfig {
    /// Create a default config with the given dimensions.
    pub fn new(head_dim: usize, max_seq_len: usize) -> Self {
        Self { head_dim, max_seq_len, base: 10_000.0, scaling_factor: None }
    }
}

// ── Scaling variants ────────────────────────────────────────────────

/// Extended-context scaling variant.
#[derive(Debug, Clone, PartialEq)]
pub enum RoPEScalingVariant {
    /// Standard RoPE (no scaling beyond the optional factor in config).
    Standard,
    /// NTK-aware scaling: adjusts base frequency to preserve high-frequency
    /// components while extending context.
    NtkAware { factor: f32 },
    /// YaRN (Yet another RoPE extensioN): frequency-dependent interpolation
    /// with smooth ramp between low and high frequency bands.
    YaRN { factor: f32, original_max_seq_len: usize, beta_fast: f32, beta_slow: f32 },
}

// ── Cache ───────────────────────────────────────────────────────────

/// Precomputed sin/cos tables for RoPE.
///
/// Layout: separate `cos_table` and `sin_table`, each of length
/// `max_seq_len * half_dim`. Index for position `p` and pair `i`:
/// `p * half_dim + i`.
#[derive(Debug, Clone)]
pub struct RoPECache {
    pub cos_table: Vec<f32>,
    pub sin_table: Vec<f32>,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub base: f32,
}

impl RoPECache {
    /// Half the head dimension (number of rotation pairs).
    #[inline]
    pub fn half_dim(&self) -> usize {
        self.head_dim / 2
    }

    /// Get the (cos, sin) pair for a given position and pair index.
    #[inline]
    pub fn get(&self, position: usize, pair: usize) -> (f32, f32) {
        let idx = position * self.half_dim() + pair;
        (self.cos_table[idx], self.sin_table[idx])
    }
}

// ── Cache construction ──────────────────────────────────────────────

/// Validate common config fields.
fn validate_config(config: &RoPEConfig) -> Result<(), RoPEError> {
    if config.head_dim == 0 || !config.head_dim.is_multiple_of(2) {
        return Err(RoPEError::InvalidConfig("head_dim must be even and > 0".into()));
    }
    if config.max_seq_len == 0 {
        return Err(RoPEError::InvalidConfig("max_seq_len must be > 0".into()));
    }
    if config.base <= 0.0 || !config.base.is_finite() {
        return Err(RoPEError::InvalidConfig("base must be a positive finite number".into()));
    }
    Ok(())
}

/// Build a RoPE cache from configuration (standard or linear-scaled).
pub fn build_rope_cache(config: &RoPEConfig) -> Result<RoPECache, RoPEError> {
    validate_config(config)?;

    let half_dim = config.head_dim / 2;
    let scale = config.scaling_factor.unwrap_or(1.0);
    let len = config.max_seq_len * half_dim;
    let mut cos_table = Vec::with_capacity(len);
    let mut sin_table = Vec::with_capacity(len);

    for pos in 0..config.max_seq_len {
        for i in 0..half_dim {
            let freq = inv_freq(i, config.head_dim, config.base) * scale;
            let angle = pos as f32 * freq;
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
    }

    Ok(RoPECache {
        cos_table,
        sin_table,
        head_dim: config.head_dim,
        max_seq_len: config.max_seq_len,
        base: config.base,
    })
}

/// Build a cache using NTK-aware scaling.
pub fn build_rope_cache_ntk(
    config: &RoPEConfig,
    variant: &RoPEScalingVariant,
) -> Result<RoPECache, RoPEError> {
    let factor = match variant {
        RoPEScalingVariant::NtkAware { factor } => *factor,
        _ => return Err(RoPEError::InvalidConfig("expected NtkAware variant".into())),
    };
    if factor <= 0.0 {
        return Err(RoPEError::InvalidConfig("NTK factor must be > 0".into()));
    }
    validate_config(config)?;

    let dim = config.head_dim as f32;
    let adjusted_base = config.base * factor.powf(dim / (dim - 2.0));
    let half_dim = config.head_dim / 2;
    let len = config.max_seq_len * half_dim;
    let mut cos_table = Vec::with_capacity(len);
    let mut sin_table = Vec::with_capacity(len);

    for pos in 0..config.max_seq_len {
        for i in 0..half_dim {
            let freq = inv_freq(i, config.head_dim, adjusted_base);
            let angle = pos as f32 * freq;
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
    }

    Ok(RoPECache {
        cos_table,
        sin_table,
        head_dim: config.head_dim,
        max_seq_len: config.max_seq_len,
        base: adjusted_base,
    })
}

/// Build a cache using YaRN scaling.
pub fn build_rope_cache_yarn(
    config: &RoPEConfig,
    variant: &RoPEScalingVariant,
) -> Result<RoPECache, RoPEError> {
    let (factor, original_max, beta_fast, beta_slow) = match variant {
        RoPEScalingVariant::YaRN { factor, original_max_seq_len, beta_fast, beta_slow } => {
            (*factor, *original_max_seq_len, *beta_fast, *beta_slow)
        }
        _ => return Err(RoPEError::InvalidConfig("expected YaRN variant".into())),
    };
    if factor <= 0.0 {
        return Err(RoPEError::InvalidConfig("YaRN factor must be > 0".into()));
    }
    validate_config(config)?;

    let half_dim = config.head_dim / 2;
    let dim_f = config.head_dim as f32;
    let len = config.max_seq_len * half_dim;
    let mut cos_table = Vec::with_capacity(len);
    let mut sin_table = Vec::with_capacity(len);

    // Compute low/high frequency bounds.
    let low_freq = 1.0 / (beta_slow / (2.0 * PI) * (original_max as f32));
    let high_freq = 1.0 / (beta_fast / (2.0 * PI) * (original_max as f32));

    for pos in 0..config.max_seq_len {
        for i in 0..half_dim {
            let base_freq = inv_freq(i, config.head_dim, config.base);
            let freq = yarn_scaled_freq(base_freq, factor, low_freq, high_freq, dim_f);
            let angle = pos as f32 * freq;
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
    }

    Ok(RoPECache {
        cos_table,
        sin_table,
        head_dim: config.head_dim,
        max_seq_len: config.max_seq_len,
        base: config.base,
    })
}

// ── Apply functions ─────────────────────────────────────────────────

/// Apply RoPE to query and key tensors using runtime SIMD dispatch.
///
/// `q` and `k` are flat slices: `[num_positions * head_dim]`.
/// `positions` has length `num_positions`.
pub fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    positions: &[usize],
    cache: &RoPECache,
) -> Result<(), RoPEError> {
    let n = positions.len();
    let hd = cache.head_dim;

    if q.len() != n * hd {
        return Err(RoPEError::DimensionMismatch { expected: n * hd, actual: q.len() });
    }
    if k.len() != n * hd {
        return Err(RoPEError::DimensionMismatch { expected: n * hd, actual: k.len() });
    }
    for &pos in positions {
        if pos >= cache.max_seq_len {
            return Err(RoPEError::PositionOutOfRange {
                requested: pos,
                max_cached: cache.max_seq_len - 1,
            });
        }
    }

    // Runtime dispatch
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: we checked the feature flag above.
            unsafe { apply_rope_avx2_inner(q, k, positions, cache) };
            return Ok(());
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // Safety: we checked the feature flag above.
            unsafe { apply_rope_neon_inner(q, k, positions, cache) };
            return Ok(());
        }
    }

    apply_rope_scalar(q, k, positions, cache);
    Ok(())
}

/// Scalar fallback — always available.
fn apply_rope_scalar(q: &mut [f32], k: &mut [f32], positions: &[usize], cache: &RoPECache) {
    let hd = cache.head_dim;
    let half = cache.half_dim();

    for (idx, &pos) in positions.iter().enumerate() {
        let base = idx * hd;
        let cache_base = pos * half;
        for i in 0..half {
            let cos = cache.cos_table[cache_base + i];
            let sin = cache.sin_table[cache_base + i];
            rotate_pair_in_slice(q, base + 2 * i, cos, sin);
            rotate_pair_in_slice(k, base + 2 * i, cos, sin);
        }
    }
}

/// AVX2-accelerated RoPE application (public entry point).
///
/// Falls back to scalar if AVX2 is not available at runtime.
pub fn apply_rope_avx2(
    q: &mut [f32],
    k: &mut [f32],
    positions: &[usize],
    cache: &RoPECache,
) -> Result<(), RoPEError> {
    validate_rope_args(q, k, positions, cache)?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { apply_rope_avx2_inner(q, k, positions, cache) };
            return Ok(());
        }
    }

    // Fallback to scalar when AVX2 is unavailable.
    apply_rope_scalar(q, k, positions, cache);
    Ok(())
}

/// NEON-accelerated RoPE application (public entry point).
///
/// Falls back to scalar if NEON is not available at runtime.
pub fn apply_rope_neon(
    q: &mut [f32],
    k: &mut [f32],
    positions: &[usize],
    cache: &RoPECache,
) -> Result<(), RoPEError> {
    validate_rope_args(q, k, positions, cache)?;

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { apply_rope_neon_inner(q, k, positions, cache) };
            return Ok(());
        }
    }

    apply_rope_scalar(q, k, positions, cache);
    Ok(())
}

/// Apply RoPE in-place to a single tensor slice at a given position.
pub fn apply_rope_inplace(
    tensor: &mut [f32],
    pos: usize,
    head_dim: usize,
    cache: &RoPECache,
) -> Result<(), RoPEError> {
    if head_dim != cache.head_dim {
        return Err(RoPEError::DimensionMismatch { expected: cache.head_dim, actual: head_dim });
    }
    if tensor.len() < head_dim {
        return Err(RoPEError::DimensionMismatch { expected: head_dim, actual: tensor.len() });
    }
    if pos >= cache.max_seq_len {
        return Err(RoPEError::PositionOutOfRange {
            requested: pos,
            max_cached: cache.max_seq_len - 1,
        });
    }

    let half = cache.half_dim();
    let cache_base = pos * half;
    for i in 0..half {
        let cos = cache.cos_table[cache_base + i];
        let sin = cache.sin_table[cache_base + i];
        rotate_pair_in_slice(tensor, 2 * i, cos, sin);
    }
    Ok(())
}

/// Apply NTK scaling to an existing cache, rebuilding tables in-place.
pub fn apply_ntk_scaling(cache: &mut RoPECache, scaling_factor: f32) {
    let half = cache.half_dim();
    let dim_f = cache.head_dim as f32;
    let adjusted_base = cache.base * scaling_factor.powf(dim_f / (dim_f - 2.0));

    for pos in 0..cache.max_seq_len {
        for i in 0..half {
            let freq = inv_freq(i, cache.head_dim, adjusted_base);
            let angle = pos as f32 * freq;
            let idx = pos * half + i;
            cache.cos_table[idx] = angle.cos();
            cache.sin_table[idx] = angle.sin();
        }
    }
    cache.base = adjusted_base;
}

// ── AVX2 inner implementation ───────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_rope_avx2_inner(
    q: &mut [f32],
    k: &mut [f32],
    positions: &[usize],
    cache: &RoPECache,
) {
    use std::arch::x86_64::*;

    let hd = cache.head_dim;
    let half = cache.half_dim();

    for (idx, &pos) in positions.iter().enumerate() {
        let base = idx * hd;
        let cache_base = pos * half;

        let mut i = 0;
        while i + 4 <= half {
            // Safety: AVX2 is available (checked by caller via target_feature).
            unsafe {
                let cos_v = _mm256_loadu_ps(cache.cos_table.as_ptr().add(cache_base + i));
                let sin_v = _mm256_loadu_ps(cache.sin_table.as_ptr().add(cache_base + i));

                let q_lo = _mm256_loadu_ps(q.as_ptr().add(base + 2 * i));
                let k_lo = _mm256_loadu_ps(k.as_ptr().add(base + 2 * i));

                let perm_deinterleave = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
                let shuf_q = _mm256_permutevar8x32_ps(q_lo, perm_deinterleave);
                let q_even = _mm256_extractf128_ps(shuf_q, 0);
                let q_odd = _mm256_extractf128_ps(shuf_q, 1);

                let shuf_k = _mm256_permutevar8x32_ps(k_lo, perm_deinterleave);
                let k_even = _mm256_extractf128_ps(shuf_k, 0);
                let k_odd = _mm256_extractf128_ps(shuf_k, 1);

                let cos4 = _mm256_castps256_ps128(cos_v);
                let sin4 = _mm256_castps256_ps128(sin_v);

                let qe = _mm_sub_ps(_mm_mul_ps(q_even, cos4), _mm_mul_ps(q_odd, sin4));
                let qo = _mm_add_ps(_mm_mul_ps(q_even, sin4), _mm_mul_ps(q_odd, cos4));
                let ke = _mm_sub_ps(_mm_mul_ps(k_even, cos4), _mm_mul_ps(k_odd, sin4));
                let ko = _mm_add_ps(_mm_mul_ps(k_even, sin4), _mm_mul_ps(k_odd, cos4));

                let perm_interleave = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
                let q_merged = _mm256_insertf128_ps(_mm256_castps128_ps256(qe), qo, 1);
                let q_out = _mm256_permutevar8x32_ps(q_merged, perm_interleave);
                _mm256_storeu_ps(q.as_mut_ptr().add(base + 2 * i), q_out);

                let k_merged = _mm256_insertf128_ps(_mm256_castps128_ps256(ke), ko, 1);
                let k_out = _mm256_permutevar8x32_ps(k_merged, perm_interleave);
                _mm256_storeu_ps(k.as_mut_ptr().add(base + 2 * i), k_out);
            }
            i += 4;
        }

        // Scalar tail for remaining pairs.
        while i < half {
            let cos = cache.cos_table[cache_base + i];
            let sin = cache.sin_table[cache_base + i];
            rotate_pair_in_slice(q, base + 2 * i, cos, sin);
            rotate_pair_in_slice(k, base + 2 * i, cos, sin);
            i += 1;
        }
    }
}

// ── NEON inner implementation ───────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_rope_neon_inner(
    q: &mut [f32],
    k: &mut [f32],
    positions: &[usize],
    cache: &RoPECache,
) {
    use std::arch::aarch64::*;

    let hd = cache.head_dim;
    let half = cache.half_dim();

    for (idx, &pos) in positions.iter().enumerate() {
        let base = idx * hd;
        let cache_base = pos * half;

        let mut i = 0;
        while i + 2 <= half {
            // Safety: NEON is available (checked by caller via target_feature).
            unsafe {
                let cos_v = vld1q_f32(cache.cos_table.as_ptr().add(cache_base + i));
                let sin_v = vld1q_f32(cache.sin_table.as_ptr().add(cache_base + i));

                let q_v = vld1q_f32(q.as_ptr().add(base + 2 * i));
                let k_v = vld1q_f32(k.as_ptr().add(base + 2 * i));

                let q_even = vuzp1q_f32(q_v, q_v);
                let q_odd = vuzp2q_f32(q_v, q_v);
                let k_even = vuzp1q_f32(k_v, k_v);
                let k_odd = vuzp2q_f32(k_v, k_v);

                let cos2 = vzip1q_f32(cos_v, cos_v);
                let sin2 = vzip1q_f32(sin_v, sin_v);

                let qe = vsubq_f32(vmulq_f32(q_even, cos2), vmulq_f32(q_odd, sin2));
                let qo = vaddq_f32(vmulq_f32(q_even, sin2), vmulq_f32(q_odd, cos2));
                let ke = vsubq_f32(vmulq_f32(k_even, cos2), vmulq_f32(k_odd, sin2));
                let ko = vaddq_f32(vmulq_f32(k_even, sin2), vmulq_f32(k_odd, cos2));

                let q_out = vzip1q_f32(qe, qo);
                let k_out = vzip1q_f32(ke, ko);

                vst1q_f32(q.as_mut_ptr().add(base + 2 * i), q_out);
                vst1q_f32(k.as_mut_ptr().add(base + 2 * i), k_out);
            }
            i += 2;
        }

        // Scalar tail.
        while i < half {
            let cos = cache.cos_table[cache_base + i];
            let sin = cache.sin_table[cache_base + i];
            rotate_pair_in_slice(q, base + 2 * i, cos, sin);
            rotate_pair_in_slice(k, base + 2 * i, cos, sin);
            i += 1;
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Compute the inverse frequency for dimension pair `i`.
#[inline]
fn inv_freq(i: usize, head_dim: usize, base: f32) -> f32 {
    1.0 / base.powf(2.0 * i as f32 / head_dim as f32)
}

/// Rotate a single (even, odd) pair in a slice by (cos, sin).
#[inline]
fn rotate_pair_in_slice(slice: &mut [f32], even_idx: usize, cos: f32, sin: f32) {
    let e = slice[even_idx];
    let o = slice[even_idx + 1];
    slice[even_idx] = e * cos - o * sin;
    slice[even_idx + 1] = e * sin + o * cos;
}

/// Validate common arguments for the public apply functions.
fn validate_rope_args(
    q: &[f32],
    k: &[f32],
    positions: &[usize],
    cache: &RoPECache,
) -> Result<(), RoPEError> {
    let n = positions.len();
    let hd = cache.head_dim;
    if q.len() != n * hd {
        return Err(RoPEError::DimensionMismatch { expected: n * hd, actual: q.len() });
    }
    if k.len() != n * hd {
        return Err(RoPEError::DimensionMismatch { expected: n * hd, actual: k.len() });
    }
    for &pos in positions {
        if pos >= cache.max_seq_len {
            return Err(RoPEError::PositionOutOfRange {
                requested: pos,
                max_cached: cache.max_seq_len - 1,
            });
        }
    }
    Ok(())
}

/// YaRN frequency scaling: smoothly interpolates between unscaled (high freq)
/// and linearly scaled (low freq) using a ramp function.
#[inline]
fn yarn_scaled_freq(base_freq: f32, factor: f32, low_freq: f32, high_freq: f32, _dim: f32) -> f32 {
    if base_freq >= high_freq {
        // High-frequency band: keep original.
        base_freq
    } else if base_freq <= low_freq {
        // Low-frequency band: linear scale.
        base_freq / factor
    } else {
        // Ramp region: smooth interpolation.
        let t = (base_freq - low_freq) / (high_freq - low_freq);
        let scaled = base_freq / factor;
        scaled * (1.0 - t) + base_freq * t
    }
}

// ── Dispatch info (for diagnostics) ─────────────────────────────────

/// Which SIMD backend would be selected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoPEDispatch {
    Scalar,
    Avx2,
    Neon,
}

impl fmt::Display for RoPEDispatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar => write!(f, "scalar"),
            Self::Avx2 => write!(f, "AVX2"),
            Self::Neon => write!(f, "NEON"),
        }
    }
}

/// Detect which dispatch path will be used on the current CPU.
pub fn detect_dispatch() -> RoPEDispatch {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return RoPEDispatch::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return RoPEDispatch::Neon;
        }
    }
    RoPEDispatch::Scalar
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers ----------------------------------------------------------

    fn default_config(head_dim: usize, max_seq_len: usize) -> RoPEConfig {
        RoPEConfig::new(head_dim, max_seq_len)
    }

    fn default_cache() -> RoPECache {
        build_rope_cache(&default_config(8, 64)).unwrap()
    }

    fn vec_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    // -- RoPEConfig tests -------------------------------------------------

    #[test]
    fn config_new_basic() {
        let c = RoPEConfig::new(64, 2048);
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.max_seq_len, 2048);
        assert!((c.base - 10_000.0).abs() < f32::EPSILON);
        assert!(c.scaling_factor.is_none());
    }

    #[test]
    fn config_with_scaling() {
        let c = RoPEConfig { scaling_factor: Some(2.0), ..RoPEConfig::new(64, 2048) };
        assert_eq!(c.scaling_factor, Some(2.0));
    }

    // -- RoPECache construction -------------------------------------------

    #[test]
    fn build_cache_basic() {
        let cache = default_cache();
        assert_eq!(cache.head_dim, 8);
        assert_eq!(cache.max_seq_len, 64);
        assert_eq!(cache.cos_table.len(), 64 * 4);
        assert_eq!(cache.sin_table.len(), 64 * 4);
    }

    #[test]
    fn build_cache_position_zero_is_identity() {
        let cache = default_cache();
        // At position 0, angle = 0 for all pairs → cos=1, sin=0.
        for i in 0..cache.half_dim() {
            let (c, s) = cache.get(0, i);
            assert!((c - 1.0).abs() < 1e-6, "cos[0][{i}] = {c}");
            assert!(s.abs() < 1e-6, "sin[0][{i}] = {s}");
        }
    }

    #[test]
    fn build_cache_cos_sin_squared_sum() {
        let cache = default_cache();
        for pos in 0..cache.max_seq_len {
            for i in 0..cache.half_dim() {
                let (c, s) = cache.get(pos, i);
                let sum = c * c + s * s;
                assert!((sum - 1.0).abs() < 1e-5, "cos²+sin² = {sum} at ({pos},{i})");
            }
        }
    }

    #[test]
    fn build_cache_with_scaling_factor() {
        let cfg = RoPEConfig { scaling_factor: Some(0.5), ..default_config(8, 16) };
        let cache = build_rope_cache(&cfg).unwrap();
        assert_eq!(cache.cos_table.len(), 16 * 4);
    }

    #[test]
    fn build_cache_err_zero_head_dim() {
        let cfg = RoPEConfig { head_dim: 0, ..default_config(2, 4) };
        assert!(matches!(build_rope_cache(&cfg), Err(RoPEError::InvalidConfig(_))));
    }

    #[test]
    fn build_cache_err_odd_head_dim() {
        let cfg = RoPEConfig { head_dim: 3, ..default_config(2, 4) };
        assert!(matches!(build_rope_cache(&cfg), Err(RoPEError::InvalidConfig(_))));
    }

    #[test]
    fn build_cache_err_zero_seq_len() {
        let cfg = RoPEConfig { max_seq_len: 0, ..default_config(4, 1) };
        assert!(matches!(build_rope_cache(&cfg), Err(RoPEError::InvalidConfig(_))));
    }

    #[test]
    fn build_cache_err_negative_base() {
        let cfg = RoPEConfig { base: -1.0, ..default_config(4, 4) };
        assert!(matches!(build_rope_cache(&cfg), Err(RoPEError::InvalidConfig(_))));
    }

    #[test]
    fn build_cache_err_nan_base() {
        let cfg = RoPEConfig { base: f32::NAN, ..default_config(4, 4) };
        assert!(matches!(build_rope_cache(&cfg), Err(RoPEError::InvalidConfig(_))));
    }

    // -- apply_rope (dispatch) tests --------------------------------------

    #[test]
    fn apply_rope_identity_at_pos_zero() {
        let cache = default_cache();
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut q = orig.clone();
        let mut k = orig.clone();
        apply_rope(&mut q, &mut k, &[0], &cache).unwrap();
        for (i, (&o, &r)) in orig.iter().zip(q.iter()).enumerate() {
            assert!((o - r).abs() < 1e-5, "q[{i}]: {o} != {r}");
        }
    }

    #[test]
    fn apply_rope_nonzero_pos_changes_values() {
        let cache = default_cache();
        let orig: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut q = orig.clone();
        let mut k = orig.clone();
        apply_rope(&mut q, &mut k, &[5], &cache).unwrap();
        assert!(q != orig, "values should change at pos 5");
    }

    #[test]
    fn apply_rope_multiple_positions() {
        let cache = default_cache();
        let n = 3;
        let hd = 8;
        let mut q = vec![1.0; n * hd];
        let mut k = vec![1.0; n * hd];
        apply_rope(&mut q, &mut k, &[0, 1, 2], &cache).unwrap();
        // Just verify no panic and first position is near-identity.
        assert!((q[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn apply_rope_err_q_length_mismatch() {
        let cache = default_cache();
        let mut q = vec![1.0; 4]; // wrong: expected 8
        let mut k = vec![1.0; 8];
        assert!(matches!(
            apply_rope(&mut q, &mut k, &[0], &cache),
            Err(RoPEError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn apply_rope_err_k_length_mismatch() {
        let cache = default_cache();
        let mut q = vec![1.0; 8];
        let mut k = vec![1.0; 4];
        assert!(matches!(
            apply_rope(&mut q, &mut k, &[0], &cache),
            Err(RoPEError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn apply_rope_err_position_out_of_range() {
        let cache = default_cache(); // max=64
        let mut q = vec![1.0; 8];
        let mut k = vec![1.0; 8];
        assert!(matches!(
            apply_rope(&mut q, &mut k, &[999], &cache),
            Err(RoPEError::PositionOutOfRange { .. })
        ));
    }

    // -- apply_rope_avx2 (uses scalar fallback on non-x86_64) -------------

    #[test]
    fn apply_rope_avx2_identity_at_pos_zero() {
        let cache = default_cache();
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut q = orig.clone();
        let mut k = orig.clone();
        apply_rope_avx2(&mut q, &mut k, &[0], &cache).unwrap();
        for (i, (&o, &r)) in orig.iter().zip(q.iter()).enumerate() {
            assert!((o - r).abs() < 1e-5, "avx2 q[{i}]: {o} != {r}");
        }
    }

    #[test]
    fn apply_rope_avx2_nonzero_pos() {
        let cache = default_cache();
        let orig: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut q = orig.clone();
        let mut k = orig.clone();
        apply_rope_avx2(&mut q, &mut k, &[3], &cache).unwrap();
        assert!(q != orig);
    }

    #[test]
    fn apply_rope_avx2_matches_scalar() {
        let cache = build_rope_cache(&default_config(16, 32)).unwrap();
        let data: Vec<f32> = (0..16).map(|x| x as f32 * 0.1 + 0.5).collect();
        let mut q_s = data.clone();
        let mut k_s = data.clone();
        apply_rope_scalar(&mut q_s, &mut k_s, &[7], &cache);

        let mut q_a = data.clone();
        let mut k_a = data;
        apply_rope_avx2(&mut q_a, &mut k_a, &[7], &cache).unwrap();

        for i in 0..16 {
            assert!(
                (q_s[i] - q_a[i]).abs() < 1e-5,
                "avx2 vs scalar mismatch q[{i}]: {} vs {}",
                q_s[i],
                q_a[i]
            );
            assert!((k_s[i] - k_a[i]).abs() < 1e-5, "avx2 vs scalar mismatch k[{i}]",);
        }
    }

    #[test]
    fn apply_rope_avx2_large_head_dim() {
        let cache = build_rope_cache(&default_config(128, 16)).unwrap();
        let mut q = vec![1.0; 128];
        let mut k = vec![1.0; 128];
        apply_rope_avx2(&mut q, &mut k, &[5], &cache).unwrap();
        assert!(q.iter().any(|&x| (x - 1.0).abs() > 1e-6));
    }

    // -- apply_rope_neon (uses scalar fallback on non-aarch64) ------------

    #[test]
    fn apply_rope_neon_identity_at_pos_zero() {
        let cache = default_cache();
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut q = orig.clone();
        let mut k = orig.clone();
        apply_rope_neon(&mut q, &mut k, &[0], &cache).unwrap();
        for (i, (&o, &r)) in orig.iter().zip(q.iter()).enumerate() {
            assert!((o - r).abs() < 1e-5, "neon q[{i}]: {o} != {r}");
        }
    }

    #[test]
    fn apply_rope_neon_nonzero_pos() {
        let cache = default_cache();
        let orig: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut q = orig.clone();
        let mut k = orig.clone();
        apply_rope_neon(&mut q, &mut k, &[3], &cache).unwrap();
        assert!(q != orig);
    }

    #[test]
    fn apply_rope_neon_matches_scalar() {
        let cache = build_rope_cache(&default_config(16, 32)).unwrap();
        let data: Vec<f32> = (0..16).map(|x| x as f32 * 0.1 + 0.5).collect();
        let mut q_s = data.clone();
        let mut k_s = data.clone();
        apply_rope_scalar(&mut q_s, &mut k_s, &[7], &cache);

        let mut q_n = data.clone();
        let mut k_n = data;
        apply_rope_neon(&mut q_n, &mut k_n, &[7], &cache).unwrap();

        for i in 0..16 {
            assert!(
                (q_s[i] - q_n[i]).abs() < 1e-5,
                "neon vs scalar mismatch q[{i}]: {} vs {}",
                q_s[i],
                q_n[i]
            );
        }
    }

    // -- apply_rope_inplace -----------------------------------------------

    #[test]
    fn inplace_identity_at_pos_zero() {
        let cache = default_cache();
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut t = orig.clone();
        apply_rope_inplace(&mut t, 0, 8, &cache).unwrap();
        for (i, (&o, &r)) in orig.iter().zip(t.iter()).enumerate() {
            assert!((o - r).abs() < 1e-5, "inplace[{i}]: {o} != {r}");
        }
    }

    #[test]
    fn inplace_matches_apply_rope_q() {
        let cache = build_rope_cache(&default_config(8, 32)).unwrap();
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let mut via_apply = data.clone();
        let mut dummy_k = data.clone();
        apply_rope(&mut via_apply, &mut dummy_k, &[4], &cache).unwrap();

        let mut via_inplace = data;
        apply_rope_inplace(&mut via_inplace, 4, 8, &cache).unwrap();

        for i in 0..8 {
            assert!(
                (via_apply[i] - via_inplace[i]).abs() < 1e-5,
                "inplace vs apply q[{i}]: {} vs {}",
                via_apply[i],
                via_inplace[i]
            );
        }
    }

    #[test]
    fn inplace_err_head_dim_mismatch() {
        let cache = default_cache(); // head_dim=8
        let mut t = vec![1.0; 16];
        assert!(matches!(
            apply_rope_inplace(&mut t, 0, 16, &cache),
            Err(RoPEError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn inplace_err_tensor_too_short() {
        let cache = default_cache();
        let mut t = vec![1.0; 4];
        assert!(matches!(
            apply_rope_inplace(&mut t, 0, 8, &cache),
            Err(RoPEError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn inplace_err_position_oob() {
        let cache = default_cache();
        let mut t = vec![1.0; 8];
        assert!(matches!(
            apply_rope_inplace(&mut t, 999, 8, &cache),
            Err(RoPEError::PositionOutOfRange { .. })
        ));
    }

    // -- NTK scaling ------------------------------------------------------

    #[test]
    fn ntk_scaling_changes_cache() {
        let cfg = default_config(8, 16);
        let original = build_rope_cache(&cfg).unwrap();
        let mut cache = build_rope_cache(&cfg).unwrap();
        apply_ntk_scaling(&mut cache, 2.0);
        assert!(cache.cos_table != original.cos_table, "NTK scaling should change cos values");
        assert!(cache.base > original.base, "base should increase");
    }

    #[test]
    fn ntk_scaling_preserves_cos_sin_unit() {
        let mut cache = default_cache();
        apply_ntk_scaling(&mut cache, 4.0);
        for pos in 0..cache.max_seq_len {
            for i in 0..cache.half_dim() {
                let (c, s) = cache.get(pos, i);
                let sum = c * c + s * s;
                assert!((sum - 1.0).abs() < 1e-5, "cos²+sin² after NTK = {sum}");
            }
        }
    }

    #[test]
    fn ntk_cache_vs_direct_build() {
        let cfg = default_config(8, 16);
        let variant = RoPEScalingVariant::NtkAware { factor: 2.0 };
        let direct = build_rope_cache_ntk(&cfg, &variant).unwrap();

        let mut incremental = build_rope_cache(&cfg).unwrap();
        apply_ntk_scaling(&mut incremental, 2.0);

        for i in 0..direct.cos_table.len() {
            assert!(
                (direct.cos_table[i] - incremental.cos_table[i]).abs() < 1e-5,
                "NTK direct vs incremental mismatch at cos[{i}]"
            );
            assert!(
                (direct.sin_table[i] - incremental.sin_table[i]).abs() < 1e-5,
                "NTK direct vs incremental mismatch at sin[{i}]"
            );
        }
    }

    #[test]
    fn ntk_build_err_wrong_variant() {
        let cfg = default_config(8, 16);
        let variant = RoPEScalingVariant::Standard;
        assert!(matches!(build_rope_cache_ntk(&cfg, &variant), Err(RoPEError::InvalidConfig(_))));
    }

    #[test]
    fn ntk_build_err_negative_factor() {
        let cfg = default_config(8, 16);
        let variant = RoPEScalingVariant::NtkAware { factor: -1.0 };
        assert!(matches!(build_rope_cache_ntk(&cfg, &variant), Err(RoPEError::InvalidConfig(_))));
    }

    // -- YaRN scaling -----------------------------------------------------

    #[test]
    fn yarn_build_basic() {
        let cfg = default_config(8, 16);
        let variant = RoPEScalingVariant::YaRN {
            factor: 2.0,
            original_max_seq_len: 8,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let cache = build_rope_cache_yarn(&cfg, &variant).unwrap();
        assert_eq!(cache.cos_table.len(), 16 * 4);
    }

    #[test]
    fn yarn_differs_from_standard() {
        let cfg = default_config(8, 16);
        let standard = build_rope_cache(&cfg).unwrap();
        let variant = RoPEScalingVariant::YaRN {
            factor: 4.0,
            original_max_seq_len: 8,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let yarn = build_rope_cache_yarn(&cfg, &variant).unwrap();
        let any_diff =
            standard.cos_table.iter().zip(&yarn.cos_table).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "YaRN should produce different tables");
    }

    #[test]
    fn yarn_preserves_cos_sin_unit() {
        let cfg = default_config(16, 32);
        let variant = RoPEScalingVariant::YaRN {
            factor: 2.0,
            original_max_seq_len: 16,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let cache = build_rope_cache_yarn(&cfg, &variant).unwrap();
        for pos in 0..cache.max_seq_len {
            for i in 0..cache.half_dim() {
                let (c, s) = cache.get(pos, i);
                let sum = c * c + s * s;
                assert!((sum - 1.0).abs() < 1e-5, "cos²+sin² yarn = {sum}");
            }
        }
    }

    #[test]
    fn yarn_err_wrong_variant() {
        let cfg = default_config(8, 16);
        let variant = RoPEScalingVariant::NtkAware { factor: 2.0 };
        assert!(matches!(build_rope_cache_yarn(&cfg, &variant), Err(RoPEError::InvalidConfig(_))));
    }

    #[test]
    fn yarn_err_negative_factor() {
        let cfg = default_config(8, 16);
        let variant = RoPEScalingVariant::YaRN {
            factor: -1.0,
            original_max_seq_len: 8,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        assert!(matches!(build_rope_cache_yarn(&cfg, &variant), Err(RoPEError::InvalidConfig(_))));
    }

    // -- dispatch detection -----------------------------------------------

    #[test]
    fn detect_dispatch_returns_valid() {
        let d = detect_dispatch();
        // On any platform, it should return something.
        let label = format!("{d}");
        assert!(!label.is_empty());
    }

    #[test]
    fn dispatch_display_scalar() {
        assert_eq!(format!("{}", RoPEDispatch::Scalar), "scalar");
    }

    #[test]
    fn dispatch_display_avx2() {
        assert_eq!(format!("{}", RoPEDispatch::Avx2), "AVX2");
    }

    #[test]
    fn dispatch_display_neon() {
        assert_eq!(format!("{}", RoPEDispatch::Neon), "NEON");
    }

    // -- RoPEError display ------------------------------------------------

    #[test]
    fn error_display_invalid_config() {
        let e = RoPEError::InvalidConfig("bad dim".into());
        assert!(e.to_string().contains("bad dim"));
    }

    #[test]
    fn error_display_position_oob() {
        let e = RoPEError::PositionOutOfRange { requested: 100, max_cached: 63 };
        let s = e.to_string();
        assert!(s.contains("100") && s.contains("63"));
    }

    #[test]
    fn error_display_dim_mismatch() {
        let e = RoPEError::DimensionMismatch { expected: 8, actual: 4 };
        let s = e.to_string();
        assert!(s.contains("8") && s.contains("4"));
    }

    // -- helper: inv_freq -------------------------------------------------

    #[test]
    fn inv_freq_pair_zero() {
        let f = inv_freq(0, 64, 10_000.0);
        assert!((f - 1.0).abs() < 1e-6, "inv_freq(0) should be 1.0");
    }

    #[test]
    fn inv_freq_decreases_with_index() {
        let f0 = inv_freq(0, 8, 10_000.0);
        let f1 = inv_freq(1, 8, 10_000.0);
        let f2 = inv_freq(2, 8, 10_000.0);
        assert!(f0 > f1 && f1 > f2, "inv_freq should decrease with i");
    }

    // -- helper: rotate_pair_in_slice -------------------------------------

    #[test]
    fn rotate_pair_zero_angle() {
        let mut v = [3.0_f32, 4.0];
        rotate_pair_in_slice(&mut v, 0, 1.0, 0.0); // cos=1, sin=0
        assert!((v[0] - 3.0).abs() < 1e-6);
        assert!((v[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn rotate_pair_quarter_turn() {
        let mut v = [1.0_f32, 0.0];
        let (c, s) = ((PI / 2.0).cos(), (PI / 2.0).sin());
        rotate_pair_in_slice(&mut v, 0, c, s);
        assert!(v[0].abs() < 1e-5, "a should be ~0 after π/2 rotation");
        assert!((v[1] - 1.0).abs() < 1e-5, "b should be ~1 after π/2 rotation");
    }

    // -- yarn_scaled_freq -------------------------------------------------

    #[test]
    fn yarn_high_freq_pass_through() {
        let f = yarn_scaled_freq(100.0, 2.0, 1.0, 50.0, 64.0);
        assert!((f - 100.0).abs() < 1e-6, "high-freq should pass through");
    }

    #[test]
    fn yarn_low_freq_fully_scaled() {
        let f = yarn_scaled_freq(0.5, 4.0, 1.0, 50.0, 64.0);
        assert!((f - 0.125).abs() < 1e-6, "low-freq should be /factor");
    }

    #[test]
    fn yarn_mid_freq_interpolated() {
        let f = yarn_scaled_freq(25.0, 2.0, 1.0, 50.0, 64.0);
        // Should be between 12.5 (fully scaled) and 25.0 (pass-through).
        assert!(f > 12.5 && f < 25.0, "mid-freq should be interpolated: {f}");
    }

    // -- Regression / edge-case tests -------------------------------------

    #[test]
    fn apply_rope_empty_positions() {
        let cache = default_cache();
        let mut q: Vec<f32> = vec![];
        let mut k: Vec<f32> = vec![];
        apply_rope(&mut q, &mut k, &[], &cache).unwrap();
    }

    #[test]
    fn large_seq_len_cache() {
        let cfg = default_config(4, 8192);
        let cache = build_rope_cache(&cfg).unwrap();
        assert_eq!(cache.cos_table.len(), 8192 * 2);
    }

    #[test]
    fn apply_rope_different_positions_different_output() {
        let cache = build_rope_cache(&default_config(8, 32)).unwrap();
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();

        let mut q1 = data.clone();
        let mut k1 = data.clone();
        apply_rope(&mut q1, &mut k1, &[1], &cache).unwrap();

        let mut q2 = data.clone();
        let mut k2 = data;
        apply_rope(&mut q2, &mut k2, &[7], &cache).unwrap();

        assert!(q1 != q2, "different positions → different outputs");
    }

    #[test]
    fn cache_half_dim() {
        let cache = default_cache();
        assert_eq!(cache.half_dim(), 4);
    }
}

// =====================================================================
// Property tests
// =====================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // Strategy: even head_dim in [2, 128], seq_len in [1, 256].
    fn config_strategy() -> impl Strategy<Value = RoPEConfig> {
        (1..=64_usize, 1..=256_usize).prop_map(|(half, seq)| RoPEConfig::new(half * 2, seq))
    }

    /// RoPE is a rotation — it must preserve the L2 norm of each head vector.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn norm_preservation(
            cfg in config_strategy(),
            pos in 0..256_usize,
        ) {
            let pos = pos % cfg.max_seq_len;
            let cache = build_rope_cache(&cfg).unwrap();
            let data: Vec<f32> = (0..cfg.head_dim)
                .map(|i| (i as f32 + 1.0) * 0.1)
                .collect();
            let norm_before = data.iter().map(|x| x * x).sum::<f32>().sqrt();

            let mut q = data.clone();
            let mut k = data;
            apply_rope(&mut q, &mut k, &[pos], &cache).unwrap();

            let norm_after_q = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_after_k = k.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!(
                (norm_before - norm_after_q).abs() < 1e-3,
                "q norm changed: {norm_before} → {norm_after_q}"
            );
            prop_assert!(
                (norm_before - norm_after_k).abs() < 1e-3,
                "k norm changed: {norm_before} → {norm_after_k}"
            );
        }
    }

    /// Position 0 should produce near-identity (angle = 0 → cos=1, sin=0).
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn position_zero_identity(cfg in config_strategy()) {
            let cache = build_rope_cache(&cfg).unwrap();
            let data: Vec<f32> = (0..cfg.head_dim)
                .map(|i| (i as f32 + 1.0) * 0.3)
                .collect();
            let mut q = data.clone();
            let mut k = data.clone();
            apply_rope(&mut q, &mut k, &[0], &cache).unwrap();

            for i in 0..cfg.head_dim {
                prop_assert!(
                    (data[i] - q[i]).abs() < 1e-4,
                    "pos-0 identity violated at {i}: {} vs {}",
                    data[i],
                    q[i]
                );
            }
        }
    }

    /// cos²(θ) + sin²(θ) = 1 for every entry in the cache.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn cache_unit_circle(cfg in config_strategy()) {
            let cache = build_rope_cache(&cfg).unwrap();
            for pos in 0..cfg.max_seq_len {
                for i in 0..cache.half_dim() {
                    let (c, s) = cache.get(pos, i);
                    let sum = c * c + s * s;
                    prop_assert!(
                        (sum - 1.0).abs() < 1e-4,
                        "unit circle violated at ({pos},{i}): {sum}"
                    );
                }
            }
        }
    }

    /// NTK scaling should increase the base frequency.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn ntk_increases_base(
            half in 1..=32_usize,
            factor in 1.1_f32..10.0,
        ) {
            let cfg = RoPEConfig::new(half * 2, 16);
            let mut cache = build_rope_cache(&cfg).unwrap();
            let orig_base = cache.base;
            apply_ntk_scaling(&mut cache, factor);
            prop_assert!(
                cache.base > orig_base,
                "NTK should increase base: {} -> {}",
                orig_base,
                cache.base
            );
        }
    }

    /// Applying RoPE twice at the same position should not equal the original
    /// (unless position is 0).
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn double_apply_differs(
            cfg in config_strategy(),
            pos in 1..256_usize,
        ) {
            let pos = (pos % cfg.max_seq_len).max(1);
            let cache = build_rope_cache(&cfg).unwrap();
            let orig: Vec<f32> = (0..cfg.head_dim)
                .map(|i| (i as f32 + 1.0) * 0.2)
                .collect();
            let mut q = orig.clone();
            let mut k = orig.clone();
            apply_rope(&mut q, &mut k, &[pos], &cache).unwrap();
            // Apply again at the same position.
            apply_rope(&mut q, &mut k, &[pos], &cache).unwrap();

            let any_diff = orig.iter().zip(&q).any(|(a, b)| (a - b).abs() > 1e-5);
            prop_assert!(any_diff, "double-apply at pos {pos} should differ from original");
        }
    }
}
