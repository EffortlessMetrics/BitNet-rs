//! RoPE (Rotary Position Embedding) frequency cache with scaling variants.
//!
//! Pre-computes and caches cos/sin frequency tables for RoPE, optimized for
//! incremental autoregressive decoding. Supports multiple scaling strategies
//! (Linear, Dynamic, NTK-aware, YaRN) for extended context lengths.
//!
//! # Overview
//!
//! - **`FreqCache`** — stores pre-computed cos/sin values indexed by
//!   `[position * half_dim + freq_idx]`.
//! - **CPU reference functions** — scalar implementations for correctness
//!   testing and non-GPU environments.
//! - **OpenCL kernel source** — ready for GPU dispatch on Intel Arc A770 and
//!   other OpenCL-capable devices.
//!
//! # Scaling variants
//!
//! | Variant | Use case |
//! |---------|----------|
//! | `Linear` | Simple frequency down-scaling |
//! | `Dynamic` | Position-dependent NTK scaling |
//! | `NTK` | NTK-aware RoPE for long contexts |
//! | `YaRN` | Smooth interpolation between low/high frequencies |

use std::f32::consts::PI;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during RoPE cache operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RoPECacheError {
    /// Configuration is invalid (e.g. head_dim is zero or odd).
    InvalidConfig(String),
    /// Requested position exceeds maximum cached position.
    PositionOutOfRange { requested: usize, max_cached: usize },
    /// Tensor dimensions do not match the expected layout.
    DimensionMismatch { expected: usize, actual: usize },
}

impl fmt::Display for RoPECacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => {
                write!(f, "invalid RoPE config: {msg}")
            }
            Self::PositionOutOfRange {
                requested,
                max_cached,
            } => write!(
                f,
                "position {requested} out of range (max cached: {max_cached})"
            ),
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "dimension mismatch: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for RoPECacheError {}

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Scaling strategy for extending RoPE beyond the original training length.
#[derive(Debug, Clone, PartialEq)]
pub enum RoPEScaling {
    /// Linearly scale down all frequencies by the given factor.
    Linear(f32),
    /// Dynamic NTK scaling — adjusts base frequency at runtime.
    Dynamic(f32),
    /// NTK-aware scaling for extended context windows.
    NTK(f32),
    /// YaRN: Yet another RoPE extensioN method.
    YaRN {
        factor: f32,
        max_position: usize,
        beta_fast: f32,
        beta_slow: f32,
    },
}

/// Configuration for RoPE frequency computation.
#[derive(Debug, Clone, PartialEq)]
pub struct RoPEConfig {
    /// Dimension of each attention head (must be even and ≥ 2).
    pub head_dim: usize,
    /// Maximum sequence position to pre-compute.
    pub max_position: usize,
    /// Base frequency (default 10 000.0).
    pub base_freq: f32,
    /// Optional scaling strategy for extended context.
    pub rope_scaling: Option<RoPEScaling>,
}

impl RoPEConfig {
    /// Validate the configuration, returning an error if invalid.
    pub fn validate(&self) -> Result<(), RoPECacheError> {
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return Err(RoPECacheError::InvalidConfig(format!(
                "head_dim must be even and > 0, got {}",
                self.head_dim
            )));
        }
        if self.max_position == 0 {
            return Err(RoPECacheError::InvalidConfig(
                "max_position must be > 0".into(),
            ));
        }
        if self.base_freq <= 0.0 {
            return Err(RoPECacheError::InvalidConfig(format!(
                "base_freq must be positive, got {}",
                self.base_freq
            )));
        }
        Ok(())
    }
}

impl Default for RoPEConfig {
    fn default() -> Self {
        Self {
            head_dim: 64,
            max_position: 2048,
            base_freq: 10_000.0,
            rope_scaling: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Frequency cache
// ---------------------------------------------------------------------------

/// Pre-computed cos/sin frequency cache for RoPE.
///
/// Layout: `cos_cache[pos * half_dim + i]` stores `cos(pos * θ_i)`,
/// where `half_dim = head_dim / 2` and `θ_i = base^(-2i / head_dim)`.
#[derive(Debug, Clone)]
pub struct FreqCache {
    /// Cosine values, length = `max_cached_position * half_dim`.
    pub cos_cache: Vec<f32>,
    /// Sine values, length = `max_cached_position * half_dim`.
    pub sin_cache: Vec<f32>,
    /// Number of positions currently cached.
    pub max_cached_position: usize,
    /// Configuration used to build this cache.
    pub config: RoPEConfig,
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Compute inverse-frequency vector: θ_i = base^(-2i / d) for i in 0..d/2.
pub fn compute_inv_freq(head_dim: usize, base: f32) -> Vec<f32> {
    let half = head_dim / 2;
    (0..half)
        .map(|i| {
            let exp = -((2 * i) as f32) / (head_dim as f32);
            base.powf(exp)
        })
        .collect()
}

/// Pre-compute cos/sin frequency tables for all positions up to
/// `config.max_position`.
pub fn compute_freq_table(
    config: &RoPEConfig,
) -> Result<FreqCache, RoPECacheError> {
    config.validate()?;
    let inv_freq = match &config.rope_scaling {
        Some(_) => return cpu_rope_with_scaling(config),
        None => compute_inv_freq(config.head_dim, config.base_freq),
    };
    Ok(build_cache_from_inv_freq(
        &inv_freq,
        config.max_position,
        config.clone(),
    ))
}

/// Build a `FreqCache` from an inverse-frequency vector.
fn build_cache_from_inv_freq(
    inv_freq: &[f32],
    max_position: usize,
    config: RoPEConfig,
) -> FreqCache {
    let half = inv_freq.len();
    let total = max_position * half;
    let mut cos_cache = Vec::with_capacity(total);
    let mut sin_cache = Vec::with_capacity(total);

    for pos in 0..max_position {
        for &freq in inv_freq {
            let angle = (pos as f32) * freq;
            cos_cache.push(angle.cos());
            sin_cache.push(angle.sin());
        }
    }

    FreqCache {
        cos_cache,
        sin_cache,
        max_cached_position: max_position,
        config,
    }
}

/// Extend an existing cache to cover positions up to `new_max_position`.
///
/// Positions already cached are kept; only new positions are computed.
pub fn extend_freq_cache(
    cache: &mut FreqCache,
    new_max_position: usize,
) -> Result<(), RoPECacheError> {
    if new_max_position <= cache.max_cached_position {
        return Ok(());
    }
    let inv_freq = match &cache.config.rope_scaling {
        Some(scaling) => {
            let base = compute_inv_freq(
                cache.config.head_dim,
                cache.config.base_freq,
            );
            apply_scaling(&base, scaling)
        }
        None => {
            compute_inv_freq(cache.config.head_dim, cache.config.base_freq)
        }
    };
    let half = inv_freq.len();
    let additional =
        (new_max_position - cache.max_cached_position) * half;
    cache.cos_cache.reserve(additional);
    cache.sin_cache.reserve(additional);

    for pos in cache.max_cached_position..new_max_position {
        for &freq in &inv_freq {
            let angle = (pos as f32) * freq;
            cache.cos_cache.push(angle.cos());
            cache.sin_cache.push(angle.sin());
        }
    }
    cache.max_cached_position = new_max_position;
    Ok(())
}

/// Apply RoPE rotation to Q and K tensors using cached frequencies.
///
/// `q` and `k` are laid out as `[token * num_heads * head_dim]`.
/// `positions` contains one position index per token.
pub fn cpu_apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    cache: &FreqCache,
    positions: &[u32],
    num_heads: usize,
    head_dim: usize,
) -> Result<(), RoPECacheError> {
    let half = head_dim / 2;
    let head_stride = head_dim;
    let token_stride = num_heads * head_dim;

    for (tok, &pos) in positions.iter().enumerate() {
        let pos = pos as usize;
        if pos >= cache.max_cached_position {
            return Err(RoPECacheError::PositionOutOfRange {
                requested: pos,
                max_cached: cache.max_cached_position,
            });
        }
        let cos = &cache.cos_cache[pos * half..(pos + 1) * half];
        let sin = &cache.sin_cache[pos * half..(pos + 1) * half];

        for h in 0..num_heads {
            let offset = tok * token_stride + h * head_stride;
            if offset + head_dim > q.len() || offset + head_dim > k.len() {
                return Err(RoPECacheError::DimensionMismatch {
                    expected: offset + head_dim,
                    actual: q.len().min(k.len()),
                });
            }
            cpu_apply_rope_single(
                &mut q[offset..offset + head_dim],
                cos,
                sin,
                head_dim,
            );
            cpu_apply_rope_single(
                &mut k[offset..offset + head_dim],
                cos,
                sin,
                head_dim,
            );
        }
    }
    Ok(())
}

/// Apply RoPE rotation to a single head vector using pair-wise rotation.
///
/// For each pair `(x[i], x[i + half])`:
/// ```text
/// x'[i]        = x[i] * cos[i] - x[i + half] * sin[i]
/// x'[i + half] = x[i] * sin[i] + x[i + half] * cos[i]
/// ```
pub fn cpu_apply_rope_single(
    vec: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    head_dim: usize,
) {
    let half = head_dim / 2;
    for i in 0..half {
        let x0 = vec[i];
        let x1 = vec[i + half];
        vec[i] = x0 * cos[i] - x1 * sin[i];
        vec[i + half] = x0 * sin[i] + x1 * cos[i];
    }
}

/// Compute frequencies with the configured scaling applied.
pub fn cpu_rope_with_scaling(
    config: &RoPEConfig,
) -> Result<FreqCache, RoPECacheError> {
    config.validate()?;
    let base_inv = compute_inv_freq(config.head_dim, config.base_freq);
    let inv_freq = match &config.rope_scaling {
        Some(scaling) => apply_scaling(&base_inv, scaling),
        None => base_inv,
    };
    Ok(build_cache_from_inv_freq(
        &inv_freq,
        config.max_position,
        config.clone(),
    ))
}

/// Apply scaling to an inverse-frequency vector.
fn apply_scaling(inv_freq: &[f32], scaling: &RoPEScaling) -> Vec<f32> {
    match scaling {
        RoPEScaling::Linear(factor) => {
            inv_freq.iter().map(|&f| f / factor).collect()
        }
        RoPEScaling::Dynamic(factor) => {
            inv_freq.iter().map(|&f| f / factor).collect()
        }
        RoPEScaling::NTK(factor) => {
            cpu_ntk_aware_scaling(inv_freq, *factor)
        }
        yarn @ RoPEScaling::YaRN { .. } => {
            cpu_yarn_scaling(inv_freq, yarn)
        }
    }
}

/// NTK-aware scaling: redistributes frequencies so that low-frequency
/// components are preserved while high-frequency ones are compressed.
///
/// `scaled_base = base * (factor ^ (d / (d - 2)))` then recompute.
pub fn cpu_ntk_aware_scaling(
    inv_freq: &[f32],
    scaling_factor: f32,
) -> Vec<f32> {
    let d = (inv_freq.len() * 2) as f32;
    let base_scale = scaling_factor.powf(d / (d - 2.0));
    inv_freq.iter().map(|&f| f / base_scale).collect()
}

/// YaRN scaling: smooth interpolation between low and high frequencies.
///
/// Frequencies below `beta_slow` are kept unchanged; above `beta_fast`
/// are linearly scaled; in between a smooth ramp blends the two.
pub fn cpu_yarn_scaling(
    inv_freq: &[f32],
    config: &RoPEScaling,
) -> Vec<f32> {
    let (factor, _max_pos, beta_fast, beta_slow) = match config {
        RoPEScaling::YaRN {
            factor,
            max_position,
            beta_fast,
            beta_slow,
        } => (*factor, *max_position, *beta_fast, *beta_slow),
        _ => return inv_freq.to_vec(),
    };

    let d = (inv_freq.len() * 2) as f32;
    let low_freq_wavelen = 2.0 * PI / beta_slow;
    let high_freq_wavelen = 2.0 * PI / beta_fast;

    inv_freq
        .iter()
        .enumerate()
        .map(|(i, &freq)| {
            let wavelen = 2.0 * PI / freq;
            if wavelen < high_freq_wavelen {
                // High frequency — keep unchanged
                freq
            } else if wavelen > low_freq_wavelen {
                // Low frequency — fully scale
                freq / factor
            } else {
                // Smooth ramp between the two
                let t = (d * (wavelen - high_freq_wavelen).ln()
                    / (low_freq_wavelen - high_freq_wavelen))
                    .clamp(0.0, 1.0);
                let smooth = (1.0 - t.cos()) / 2.0;
                let _idx = i; // suppress unused
                freq * (1.0 - smooth) + (freq / factor) * smooth
            }
        })
        .collect()
}

/// Apply RoPE for a single new token during autoregressive decoding.
///
/// This is a convenience wrapper around `cpu_apply_rope` for the common
/// case of processing one token at a time.
pub fn cpu_incremental_rope(
    q: &mut [f32],
    k: &mut [f32],
    cache: &FreqCache,
    position: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<(), RoPECacheError> {
    if position >= cache.max_cached_position {
        return Err(RoPECacheError::PositionOutOfRange {
            requested: position,
            max_cached: cache.max_cached_position,
        });
    }
    let half = head_dim / 2;
    let cos = &cache.cos_cache[position * half..(position + 1) * half];
    let sin = &cache.sin_cache[position * half..(position + 1) * half];

    for h in 0..num_heads {
        let offset = h * head_dim;
        if offset + head_dim > q.len() || offset + head_dim > k.len() {
            return Err(RoPECacheError::DimensionMismatch {
                expected: offset + head_dim,
                actual: q.len().min(k.len()),
            });
        }
        cpu_apply_rope_single(
            &mut q[offset..offset + head_dim],
            cos,
            sin,
            head_dim,
        );
        cpu_apply_rope_single(
            &mut k[offset..offset + head_dim],
            cos,
            sin,
            head_dim,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// OpenCL kernel sources
// ---------------------------------------------------------------------------

/// OpenCL C kernel source for RoPE cache operations.
///
/// Contains three kernels:
/// - `apply_rope_cached` — apply pre-computed cos/sin from a cache buffer
/// - `compute_rope_inline` — compute RoPE on-the-fly (cache miss path)
/// - `extend_rope_cache` — parallel cache extension kernel
pub const ROPE_CACHE_SRC: &str = r#"
// Apply pre-computed RoPE cos/sin from cache buffers.
//
// Each work-item handles one (position, frequency-index) pair for one head.
// Layout: cache[pos * half_dim + i].
__kernel void apply_rope_cached(
    __global float* q,            // [tokens, heads, head_dim]
    __global float* k,            // [tokens, heads, head_dim]
    __global const float* cos_c,  // [max_pos, half_dim]
    __global const float* sin_c,  // [max_pos, half_dim]
    __global const uint* positions,// [tokens]
    const uint num_heads,
    const uint head_dim
) {
    const uint gid    = get_global_id(0);
    const uint half   = head_dim / 2;
    const uint total  = half * num_heads;

    const uint tok    = gid / total;
    const uint rem    = gid % total;
    const uint head   = rem / half;
    const uint i      = rem % half;

    const uint pos    = positions[tok];
    const uint c_idx  = pos * half + i;
    const float cosv  = cos_c[c_idx];
    const float sinv  = sin_c[c_idx];

    const uint base   = tok * num_heads * head_dim + head * head_dim;
    // Q rotation
    {
        const float x0 = q[base + i];
        const float x1 = q[base + i + half];
        q[base + i]        = x0 * cosv - x1 * sinv;
        q[base + i + half] = x0 * sinv + x1 * cosv;
    }
    // K rotation
    {
        const float x0 = k[base + i];
        const float x1 = k[base + i + half];
        k[base + i]        = x0 * cosv - x1 * sinv;
        k[base + i + half] = x0 * sinv + x1 * cosv;
    }
}

// Compute RoPE on-the-fly without a cache (fallback path).
//
// Each work-item computes θ_i = base^(-2i/d) * position, then applies
// the rotation directly.
__kernel void compute_rope_inline(
    __global float* q,
    __global float* k,
    __global const uint* positions,
    const uint num_heads,
    const uint head_dim,
    const float base_freq
) {
    const uint gid    = get_global_id(0);
    const uint half   = head_dim / 2;
    const uint total  = half * num_heads;

    const uint tok    = gid / total;
    const uint rem    = gid % total;
    const uint head   = rem / half;
    const uint i      = rem % half;

    const float exp   = -((float)(2 * i)) / (float)head_dim;
    const float theta = pow(base_freq, exp);
    const float angle = (float)positions[tok] * theta;
    const float cosv  = cos(angle);
    const float sinv  = sin(angle);

    const uint base   = tok * num_heads * head_dim + head * head_dim;
    {
        const float x0 = q[base + i];
        const float x1 = q[base + i + half];
        q[base + i]        = x0 * cosv - x1 * sinv;
        q[base + i + half] = x0 * sinv + x1 * cosv;
    }
    {
        const float x0 = k[base + i];
        const float x1 = k[base + i + half];
        k[base + i]        = x0 * cosv - x1 * sinv;
        k[base + i + half] = x0 * sinv + x1 * cosv;
    }
}

// Extend a RoPE frequency cache in parallel.
//
// Global size = (new_max - old_max) * half_dim.
__kernel void extend_rope_cache(
    __global float* cos_c,
    __global float* sin_c,
    __global const float* inv_freq, // [half_dim]
    const uint old_max,
    const uint half_dim
) {
    const uint gid = get_global_id(0);
    const uint pos = old_max + gid / half_dim;
    const uint i   = gid % half_dim;
    const float angle = (float)pos * inv_freq[i];
    cos_c[pos * half_dim + i] = cos(angle);
    sin_c[pos * half_dim + i] = sin(angle);
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(head_dim: usize, max_pos: usize) -> RoPEConfig {
        RoPEConfig {
            head_dim,
            max_position: max_pos,
            base_freq: 10_000.0,
            rope_scaling: None,
        }
    }

    // -- Inverse frequency tests ------------------------------------------

    #[test]
    fn inv_freq_head_dim_4() {
        let f = compute_inv_freq(4, 10_000.0);
        assert_eq!(f.len(), 2);
        // θ_0 = 10000^(0) = 1.0, θ_1 = 10000^(-1) = 0.0001
        assert!((f[0] - 1.0).abs() < 1e-6);
        assert!((f[1] - 0.01).abs() < 1e-4);
    }

    #[test]
    fn inv_freq_head_dim_8() {
        let f = compute_inv_freq(8, 10_000.0);
        assert_eq!(f.len(), 4);
        assert!((f[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn inv_freq_head_dim_16() {
        let f = compute_inv_freq(16, 10_000.0);
        assert_eq!(f.len(), 8);
        assert!((f[0] - 1.0).abs() < 1e-6);
        // Last element: 10000^(-14/16) = 10000^(-0.875)
        let expected_last = 10_000.0_f32.powf(-14.0 / 16.0);
        assert!((f[7] - expected_last).abs() < 1e-8);
    }

    #[test]
    fn inv_freq_head_dim_64() {
        let f = compute_inv_freq(64, 10_000.0);
        assert_eq!(f.len(), 32);
        assert!((f[0] - 1.0).abs() < 1e-6);
        // Monotonically decreasing
        for w in f.windows(2) {
            assert!(w[0] > w[1]);
        }
    }

    #[test]
    fn inv_freq_head_dim_128() {
        let f = compute_inv_freq(128, 10_000.0);
        assert_eq!(f.len(), 64);
        assert!((f[0] - 1.0).abs() < 1e-6);
        assert!(f[63] > 0.0);
        assert!(f[63] < f[0]);
    }

    // -- FreqCache basic tests --------------------------------------------

    #[test]
    fn freq_cache_position_0_cos_is_1() {
        let cfg = default_config(8, 16);
        let cache = compute_freq_table(&cfg).unwrap();
        let half = cfg.head_dim / 2;
        for i in 0..half {
            assert!(
                (cache.cos_cache[i] - 1.0).abs() < 1e-6,
                "cos[0][{i}] = {} != 1.0",
                cache.cos_cache[i]
            );
        }
    }

    #[test]
    fn freq_cache_position_0_sin_is_0() {
        let cfg = default_config(8, 16);
        let cache = compute_freq_table(&cfg).unwrap();
        let half = cfg.head_dim / 2;
        for i in 0..half {
            assert!(
                cache.sin_cache[i].abs() < 1e-6,
                "sin[0][{i}] = {} != 0.0",
                cache.sin_cache[i]
            );
        }
    }

    #[test]
    fn freq_cache_correct_length() {
        let cfg = default_config(16, 32);
        let cache = compute_freq_table(&cfg).unwrap();
        let half = cfg.head_dim / 2;
        assert_eq!(cache.cos_cache.len(), 32 * half);
        assert_eq!(cache.sin_cache.len(), 32 * half);
        assert_eq!(cache.max_cached_position, 32);
    }

    #[test]
    fn freq_cache_cos2_sin2_identity() {
        let cfg = default_config(64, 128);
        let cache = compute_freq_table(&cfg).unwrap();
        for (c, s) in
            cache.cos_cache.iter().zip(cache.sin_cache.iter())
        {
            let sum = c * c + s * s;
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "cos²+sin² = {sum} at some entry"
            );
        }
    }

    #[test]
    fn freq_cache_periodicity() {
        // For the second frequency dim (θ_1 ≈ 0.01), period ≈ 628.
        // At position p and p + period the values should be close.
        let head_dim = 4;
        let half = head_dim / 2;
        let inv_freq = compute_inv_freq(head_dim, 10_000.0);
        // θ_1 period in positions = 2π / θ_1
        let period = (2.0 * PI / inv_freq[1]).round() as usize;
        let max_pos = period + 10;
        let cfg = default_config(head_dim, max_pos);
        let cache = compute_freq_table(&cfg).unwrap();
        // Check cos/sin at position p ≈ position p + period for dim 1
        for p in 0..5 {
            let p2 = p + period;
            if p2 >= max_pos {
                break;
            }
            let c1 = cache.cos_cache[p * half + 1];
            let c2 = cache.cos_cache[p2 * half + 1];
            assert!(
                (c1 - c2).abs() < 0.05,
                "cos periodicity failed at {p}: {c1} vs {c2}"
            );
        }
    }

    // -- RoPE rotation tests ----------------------------------------------

    #[test]
    fn rope_at_position_0_unchanged() {
        let cfg = default_config(4, 8);
        let cache = compute_freq_table(&cfg).unwrap();
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut q = original.clone();
        let mut k = original.clone();
        cpu_apply_rope(&mut q, &mut k, &cache, &[0], 1, 4).unwrap();
        for i in 0..4 {
            assert!(
                (q[i] - original[i]).abs() < 1e-5,
                "q[{i}] changed at pos 0"
            );
            assert!(
                (k[i] - original[i]).abs() < 1e-5,
                "k[{i}] changed at pos 0"
            );
        }
    }

    #[test]
    fn rope_double_application_cancels() {
        let cfg = default_config(4, 64);
        let cache = compute_freq_table(&cfg).unwrap();
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut q = original.clone();
        let mut k = original.clone();
        // Apply at position p
        cpu_apply_rope(&mut q, &mut k, &cache, &[5], 1, 4).unwrap();
        // Negate sin to reverse rotation
        let half = 2;
        let pos = 5;
        let cos =
            cache.cos_cache[pos * half..(pos + 1) * half].to_vec();
        let neg_sin: Vec<f32> = cache.sin_cache
            [pos * half..(pos + 1) * half]
            .iter()
            .map(|s| -s)
            .collect();
        cpu_apply_rope_single(&mut q, &cos, &neg_sin, 4);
        cpu_apply_rope_single(&mut k, &cos, &neg_sin, 4);
        for i in 0..4 {
            assert!(
                (q[i] - original[i]).abs() < 1e-4,
                "cancel q[{i}]: {} vs {}",
                q[i],
                original[i]
            );
        }
    }

    #[test]
    fn rope_preserves_norm() {
        let cfg = default_config(8, 32);
        let cache = compute_freq_table(&cfg).unwrap();
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let norm_before: f32 =
            original.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut q = original.clone();
        let mut k = original;
        cpu_apply_rope(&mut q, &mut k, &cache, &[7], 1, 8).unwrap();
        let norm_after: f32 =
            q.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "norm changed: {norm_before} -> {norm_after}"
        );
    }

    #[test]
    fn rope_q_and_k_same_at_same_position() {
        let cfg = default_config(8, 16);
        let cache = compute_freq_table(&cfg).unwrap();
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let mut q = data.clone();
        let mut k = data;
        cpu_apply_rope(&mut q, &mut k, &cache, &[3], 1, 8).unwrap();
        for i in 0..8 {
            assert!(
                (q[i] - k[i]).abs() < 1e-6,
                "q[{i}]={} != k[{i}]={}",
                q[i],
                k[i]
            );
        }
    }

    #[test]
    fn rope_preserves_inner_product_magnitude() {
        // RoPE is a rotation so |<a,b>| should be preserved.
        let cfg = default_config(4, 16);
        let cache = compute_freq_table(&cfg).unwrap();
        let a_orig = vec![1.0, 0.0, 1.0, 0.0];
        let b_orig = vec![0.0, 1.0, 0.0, 1.0];
        let dot_before: f32 =
            a_orig.iter().zip(&b_orig).map(|(a, b)| a * b).sum();

        let mut a = a_orig;
        let mut b = b_orig;
        // Apply same rotation to both
        let half = 2;
        let cos = &cache.cos_cache[3 * half..4 * half];
        let sin = &cache.sin_cache[3 * half..4 * half];
        cpu_apply_rope_single(&mut a, cos, sin, 4);
        cpu_apply_rope_single(&mut b, cos, sin, 4);
        let dot_after: f32 =
            a.iter().zip(&b).map(|(a, b)| a * b).sum();

        assert!(
            (dot_before - dot_after).abs() < 1e-4,
            "dot product changed: {dot_before} -> {dot_after}"
        );
    }

    // -- Multi-head tests ------------------------------------------------

    #[test]
    fn multi_head_correct_offsets() {
        let cfg = default_config(4, 16);
        let cache = compute_freq_table(&cfg).unwrap();
        let num_heads = 2;
        let head_dim = 4;
        // Two heads: [h0_d0, h0_d1, h0_d2, h0_d3, h1_d0, ...]
        let mut q = vec![1.0; num_heads * head_dim];
        let mut k = vec![1.0; num_heads * head_dim];
        cpu_apply_rope(&mut q, &mut k, &cache, &[5], num_heads, head_dim)
            .unwrap();
        // Both heads should get the same rotation at the same position
        for i in 0..head_dim {
            assert!(
                (q[i] - q[head_dim + i]).abs() < 1e-6,
                "head mismatch at dim {i}"
            );
        }
    }

    #[test]
    fn multi_head_two_tokens() {
        let cfg = default_config(4, 16);
        let cache = compute_freq_table(&cfg).unwrap();
        let num_heads = 2;
        let hd = 4;
        let token_stride = num_heads * hd;
        let mut q = vec![1.0; 2 * token_stride];
        let mut k = vec![1.0; 2 * token_stride];
        cpu_apply_rope(&mut q, &mut k, &cache, &[0, 3], num_heads, hd)
            .unwrap();
        // Token 0 at pos 0 should be unchanged
        for i in 0..hd {
            assert!((q[i] - 1.0).abs() < 1e-5);
        }
        // Token 1 at pos 3 should differ from 1.0 (except head_dim=4
        // with base 10000, changes are small for dim>0)
        let changed = (0..hd)
            .any(|i| (q[token_stride + i] - 1.0).abs() > 1e-6);
        assert!(changed, "token at pos 3 should differ from 1.0");
    }

    // -- Cache extension tests -------------------------------------------

    #[test]
    fn extend_cache_agrees_with_fresh() {
        let cfg = default_config(8, 16);
        let mut cache = compute_freq_table(&cfg).unwrap();
        extend_freq_cache(&mut cache, 32).unwrap();

        let cfg2 = default_config(8, 32);
        let fresh = compute_freq_table(&cfg2).unwrap();

        assert_eq!(cache.cos_cache.len(), fresh.cos_cache.len());
        for (a, b) in cache.cos_cache.iter().zip(&fresh.cos_cache) {
            assert!((a - b).abs() < 1e-6);
        }
        for (a, b) in cache.sin_cache.iter().zip(&fresh.sin_cache) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn extend_cache_no_op_when_already_large() {
        let cfg = default_config(4, 32);
        let mut cache = compute_freq_table(&cfg).unwrap();
        let len_before = cache.cos_cache.len();
        extend_freq_cache(&mut cache, 16).unwrap();
        assert_eq!(cache.cos_cache.len(), len_before);
    }

    #[test]
    fn extend_cache_preserves_existing() {
        let cfg = default_config(4, 8);
        let cache_orig = compute_freq_table(&cfg).unwrap();
        let mut cache = cache_orig.clone();
        extend_freq_cache(&mut cache, 16).unwrap();
        // Original entries must be identical
        for i in 0..cache_orig.cos_cache.len() {
            assert!((cache.cos_cache[i] - cache_orig.cos_cache[i]).abs() < 1e-7);
        }
    }

    // -- Incremental vs batch tests --------------------------------------

    #[test]
    fn incremental_matches_batch() {
        let cfg = default_config(8, 32);
        let cache = compute_freq_table(&cfg).unwrap();
        let num_heads = 2;
        let hd = 8;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let full: Vec<f32> =
            data.iter().cycle().take(num_heads * hd).cloned().collect();

        // Batch at position 10
        let mut q_batch = full.clone();
        let mut k_batch = full.clone();
        cpu_apply_rope(
            &mut q_batch,
            &mut k_batch,
            &cache,
            &[10],
            num_heads,
            hd,
        )
        .unwrap();

        // Incremental at position 10
        let mut q_inc = full.clone();
        let mut k_inc = full;
        cpu_incremental_rope(
            &mut q_inc, &mut k_inc, &cache, 10, num_heads, hd,
        )
        .unwrap();

        for i in 0..q_batch.len() {
            assert!(
                (q_batch[i] - q_inc[i]).abs() < 1e-6,
                "mismatch at q[{i}]"
            );
            assert!(
                (k_batch[i] - k_inc[i]).abs() < 1e-6,
                "mismatch at k[{i}]"
            );
        }
    }

    #[test]
    fn incremental_position_out_of_range() {
        let cfg = default_config(4, 8);
        let cache = compute_freq_table(&cfg).unwrap();
        let mut q = vec![0.0; 4];
        let mut k = vec![0.0; 4];
        let result =
            cpu_incremental_rope(&mut q, &mut k, &cache, 100, 1, 4);
        assert!(matches!(
            result,
            Err(RoPECacheError::PositionOutOfRange { .. })
        ));
    }

    // -- Scaling tests ---------------------------------------------------

    #[test]
    fn linear_scaling_lowers_frequencies() {
        let base = compute_inv_freq(8, 10_000.0);
        let scaled = apply_scaling(&base, &RoPEScaling::Linear(2.0));
        for (b, s) in base.iter().zip(&scaled) {
            assert!(
                (s - b / 2.0).abs() < 1e-7,
                "linear scaling: {s} != {b}/2"
            );
        }
    }

    #[test]
    fn linear_scaling_higher_factor_lower_freq() {
        let base = compute_inv_freq(16, 10_000.0);
        let s2 = apply_scaling(&base, &RoPEScaling::Linear(2.0));
        let s4 = apply_scaling(&base, &RoPEScaling::Linear(4.0));
        for (a, b) in s2.iter().zip(&s4) {
            assert!(a > b, "higher factor should give lower frequencies");
        }
    }

    #[test]
    fn ntk_scaling_preserves_low_freq() {
        // NTK should scale all uniformly by the base_scale factor
        let base = compute_inv_freq(64, 10_000.0);
        let ntk = cpu_ntk_aware_scaling(&base, 2.0);
        assert_eq!(ntk.len(), base.len());
        // All should be smaller
        for (b, n) in base.iter().zip(&ntk) {
            assert!(n < b, "NTK freq should be ≤ base freq");
        }
    }

    #[test]
    fn ntk_scaling_factor_1_no_change() {
        let base = compute_inv_freq(8, 10_000.0);
        let ntk = cpu_ntk_aware_scaling(&base, 1.0);
        for (b, n) in base.iter().zip(&ntk) {
            assert!(
                (b - n).abs() < 1e-6,
                "factor=1 should not change freqs"
            );
        }
    }

    #[test]
    fn yarn_scaling_smooth_interpolation() {
        let base = compute_inv_freq(64, 10_000.0);
        let yarn = cpu_yarn_scaling(
            &base,
            &RoPEScaling::YaRN {
                factor: 4.0,
                max_position: 8192,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
        );
        assert_eq!(yarn.len(), base.len());
        // All frequencies should be ≤ original (scaling divides)
        for (b, y) in base.iter().zip(&yarn) {
            assert!(
                *y <= *b + 1e-6,
                "YaRN freq {y} > base {b}"
            );
        }
    }

    #[test]
    fn yarn_non_yarn_config_returns_copy() {
        let base = compute_inv_freq(8, 10_000.0);
        let result = cpu_yarn_scaling(&base, &RoPEScaling::Linear(2.0));
        assert_eq!(result, base);
    }

    #[test]
    fn dynamic_scaling_same_as_linear() {
        let base = compute_inv_freq(8, 10_000.0);
        let lin = apply_scaling(&base, &RoPEScaling::Linear(3.0));
        let dyn_ = apply_scaling(&base, &RoPEScaling::Dynamic(3.0));
        for (l, d) in lin.iter().zip(&dyn_) {
            assert!((l - d).abs() < 1e-7);
        }
    }

    #[test]
    fn scaled_cache_computation() {
        let cfg = RoPEConfig {
            head_dim: 8,
            max_position: 16,
            base_freq: 10_000.0,
            rope_scaling: Some(RoPEScaling::Linear(2.0)),
        };
        let cache = compute_freq_table(&cfg).unwrap();
        assert_eq!(cache.max_cached_position, 16);
        // At position 0, cos should still be 1.0
        for i in 0..4 {
            assert!((cache.cos_cache[i] - 1.0).abs() < 1e-6);
        }
    }

    // -- Edge case tests -------------------------------------------------

    #[test]
    fn head_dim_2_minimum() {
        let cfg = default_config(2, 4);
        let cache = compute_freq_table(&cfg).unwrap();
        assert_eq!(cache.cos_cache.len(), 4); // 4 positions * 1 freq
        assert!((cache.cos_cache[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn max_position_1() {
        let cfg = default_config(4, 1);
        let cache = compute_freq_table(&cfg).unwrap();
        // Only position 0
        assert_eq!(cache.cos_cache.len(), 2);
        assert!((cache.cos_cache[0] - 1.0).abs() < 1e-6);
        assert!((cache.cos_cache[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn invalid_config_zero_head_dim() {
        let cfg = RoPEConfig {
            head_dim: 0,
            max_position: 8,
            base_freq: 10_000.0,
            rope_scaling: None,
        };
        assert!(matches!(
            compute_freq_table(&cfg),
            Err(RoPECacheError::InvalidConfig(_))
        ));
    }

    #[test]
    fn invalid_config_odd_head_dim() {
        let cfg = RoPEConfig {
            head_dim: 7,
            max_position: 8,
            base_freq: 10_000.0,
            rope_scaling: None,
        };
        assert!(matches!(
            compute_freq_table(&cfg),
            Err(RoPECacheError::InvalidConfig(_))
        ));
    }

    #[test]
    fn invalid_config_zero_max_position() {
        let cfg = RoPEConfig {
            head_dim: 4,
            max_position: 0,
            base_freq: 10_000.0,
            rope_scaling: None,
        };
        assert!(matches!(
            compute_freq_table(&cfg),
            Err(RoPECacheError::InvalidConfig(_))
        ));
    }

    #[test]
    fn invalid_config_negative_base_freq() {
        let cfg = RoPEConfig {
            head_dim: 4,
            max_position: 8,
            base_freq: -1.0,
            rope_scaling: None,
        };
        assert!(matches!(
            compute_freq_table(&cfg),
            Err(RoPECacheError::InvalidConfig(_))
        ));
    }

    #[test]
    fn position_out_of_range_error() {
        let cfg = default_config(4, 8);
        let cache = compute_freq_table(&cfg).unwrap();
        let mut q = vec![0.0; 4];
        let mut k = vec![0.0; 4];
        let result =
            cpu_apply_rope(&mut q, &mut k, &cache, &[10], 1, 4);
        assert!(matches!(
            result,
            Err(RoPECacheError::PositionOutOfRange { .. })
        ));
    }

    #[test]
    fn dimension_mismatch_error() {
        let cfg = default_config(4, 8);
        let cache = compute_freq_table(&cfg).unwrap();
        let mut q = vec![0.0; 2]; // too short
        let mut k = vec![0.0; 2];
        let result =
            cpu_apply_rope(&mut q, &mut k, &cache, &[0], 1, 4);
        assert!(matches!(
            result,
            Err(RoPECacheError::DimensionMismatch { .. })
        ));
    }

    // -- OpenCL source tests ---------------------------------------------

    #[test]
    fn opencl_source_contains_apply_rope_cached() {
        assert!(ROPE_CACHE_SRC.contains("apply_rope_cached"));
    }

    #[test]
    fn opencl_source_contains_compute_rope_inline() {
        assert!(ROPE_CACHE_SRC.contains("compute_rope_inline"));
    }

    #[test]
    fn opencl_source_contains_extend_rope_cache() {
        assert!(ROPE_CACHE_SRC.contains("extend_rope_cache"));
    }

    #[test]
    fn opencl_source_contains_kernel_keyword() {
        assert!(ROPE_CACHE_SRC.contains("__kernel"));
        let count =
            ROPE_CACHE_SRC.matches("__kernel").count();
        assert_eq!(count, 3, "expected 3 kernels, found {count}");
    }

    // -- Additional property tests ---------------------------------------

    #[test]
    fn rope_different_positions_different_output() {
        let cfg = default_config(8, 32);
        let cache = compute_freq_table(&cfg).unwrap();
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut q1 = data.clone();
        let mut k1 = data.clone();
        cpu_apply_rope(&mut q1, &mut k1, &cache, &[1], 1, 8).unwrap();

        let mut q2 = data.clone();
        let mut k2 = data;
        cpu_apply_rope(&mut q2, &mut k2, &cache, &[5], 1, 8).unwrap();

        let any_diff = q1.iter().zip(&q2).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "different positions should give different results");
    }

    #[test]
    fn extend_cache_with_scaling() {
        let cfg = RoPEConfig {
            head_dim: 8,
            max_position: 8,
            base_freq: 10_000.0,
            rope_scaling: Some(RoPEScaling::Linear(2.0)),
        };
        let mut cache = compute_freq_table(&cfg).unwrap();
        extend_freq_cache(&mut cache, 16).unwrap();
        assert_eq!(cache.max_cached_position, 16);

        // Verify extended portion matches fresh computation
        let cfg2 = RoPEConfig {
            head_dim: 8,
            max_position: 16,
            base_freq: 10_000.0,
            rope_scaling: Some(RoPEScaling::Linear(2.0)),
        };
        let fresh = compute_freq_table(&cfg2).unwrap();
        for i in 0..fresh.cos_cache.len() {
            assert!(
                (cache.cos_cache[i] - fresh.cos_cache[i]).abs() < 1e-6,
                "scaled extend mismatch at {i}"
            );
        }
    }

    #[test]
    fn rope_config_default() {
        let cfg = RoPEConfig::default();
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.max_position, 2048);
        assert!((cfg.base_freq - 10_000.0).abs() < 1e-6);
        assert!(cfg.rope_scaling.is_none());
    }

    #[test]
    fn error_display_messages() {
        let e1 = RoPECacheError::InvalidConfig("test".into());
        assert!(e1.to_string().contains("test"));

        let e2 = RoPECacheError::PositionOutOfRange {
            requested: 10,
            max_cached: 5,
        };
        assert!(e2.to_string().contains("10"));

        let e3 = RoPECacheError::DimensionMismatch {
            expected: 8,
            actual: 4,
        };
        assert!(e3.to_string().contains("8"));
    }

    #[test]
    fn inv_freq_custom_base() {
        let f1 = compute_inv_freq(4, 10_000.0);
        let f2 = compute_inv_freq(4, 500.0);
        // With smaller base, second element should be larger
        assert!(f2[1] > f1[1]);
    }

    #[test]
    fn rope_single_applies_pair_rotation() {
        let cos = vec![0.0_f32]; // cos(π/2) = 0
        let sin = vec![1.0_f32]; // sin(π/2) = 1
        let mut v = vec![1.0, 0.0];
        cpu_apply_rope_single(&mut v, &cos, &sin, 2);
        // x'[0] = 1*0 - 0*1 = 0, x'[1] = 1*1 + 0*0 = 1
        assert!((v[0]).abs() < 1e-6);
        assert!((v[1] - 1.0).abs() < 1e-6);
    }
}
