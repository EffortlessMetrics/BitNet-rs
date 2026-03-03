//! CUDA RoPE (Rotary Position Embedding) kernel parameters and launch helpers.
//!
//! This crate provides host-side types for configuring and launching CUDA RoPE
//! kernels, including:
//!
//! - **Frequency table generation** — inverse-frequency vectors for standard and
//!   scaled RoPE variants.
//! - **Forward-pass application** — parameters for the CUDA kernel that rotates
//!   Q/K head vectors in-place.
//! - **NTK-aware scaling** — Neural Tangent Kernel base-frequency adjustment for
//!   extended context lengths.
//! - **YaRN** (Yet another RoPE extensioN) — piecewise wavelength interpolation
//!   with attention scaling.
//! - **Dynamic NTK** — runtime base adjustment that recomputes when sequence
//!   length exceeds the training window.
//!
//! All floating-point maths runs on the host in `f64` for precision, then the
//! resulting tables are cast to `f32` before upload to device memory.

use std::f64::consts::PI;

// Re-export error types from bitnet-common for downstream convenience.
pub use bitnet_common::error::BitNetError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default RoPE base frequency (θ) used by LLaMA-family models.
pub const DEFAULT_ROPE_BASE: f64 = 10_000.0;

/// Default head dimension for LLaMA-2 / LLaMA-3 style models.
pub const DEFAULT_HEAD_DIM: usize = 128;

/// Default maximum sequence length for standard RoPE.
pub const DEFAULT_MAX_SEQ_LEN: usize = 4096;

/// Minimum head dimension accepted by any RoPE variant.
pub const MIN_HEAD_DIM: usize = 2;

/// Maximum head dimension accepted (guard against absurd allocations).
pub const MAX_HEAD_DIM: usize = 1024;

/// Maximum sequence length accepted.
pub const MAX_SEQ_LEN: usize = 16_777_216; // 16M

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by RoPE configuration or table generation.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeError {
    /// Head dimension is zero.
    ZeroDimension,
    /// Head dimension is not even.
    OddDimension { dim: usize },
    /// Head dimension exceeds [`MAX_HEAD_DIM`].
    DimensionTooLarge { dim: usize },
    /// Sequence length is zero.
    ZeroSequenceLength,
    /// Sequence length exceeds [`MAX_SEQ_LEN`].
    SequenceLengthTooLarge { len: usize },
    /// Base frequency is not finite.
    NonFiniteBase { base: f64 },
    /// Base frequency is non-positive.
    NonPositiveBase { base: f64 },
    /// Scaling factor is non-positive.
    NonPositiveScale { scale: f64 },
    /// YaRN alpha/beta is invalid.
    InvalidYarnParam { name: &'static str, value: f64 },
    /// Trained context length is zero.
    ZeroTrainedContext,
}

impl std::fmt::Display for RopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "head dimension must be > 0"),
            Self::OddDimension { dim } => write!(f, "head dimension must be even, got {dim}"),
            Self::DimensionTooLarge { dim } => {
                write!(f, "head dimension {dim} exceeds maximum {MAX_HEAD_DIM}")
            }
            Self::ZeroSequenceLength => write!(f, "sequence length must be > 0"),
            Self::SequenceLengthTooLarge { len } => {
                write!(f, "sequence length {len} exceeds maximum {MAX_SEQ_LEN}")
            }
            Self::NonFiniteBase { base } => write!(f, "base must be finite, got {base}"),
            Self::NonPositiveBase { base } => write!(f, "base must be > 0, got {base}"),
            Self::NonPositiveScale { scale } => {
                write!(f, "scaling factor must be > 0, got {scale}")
            }
            Self::InvalidYarnParam { name, value } => {
                write!(f, "YaRN parameter `{name}` is invalid: {value}")
            }
            Self::ZeroTrainedContext => write!(f, "trained context length must be > 0"),
        }
    }
}

impl std::error::Error for RopeError {}

// ---------------------------------------------------------------------------
// RoPE scaling method
// ---------------------------------------------------------------------------

/// Selects how positional frequencies are scaled for extended context.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScalingMethod {
    /// No scaling — standard RoPE with fixed base.
    None,
    /// Linear position interpolation (divide positions by `factor`).
    Linear { factor: f64 },
    /// NTK-aware scaling — adjust the base frequency instead of positions.
    Ntk { factor: f64 },
    /// YaRN piecewise interpolation with attention temperature scaling.
    Yarn(YarnConfig),
    /// Dynamic NTK — recompute base at runtime when `seq_len > trained_ctx`.
    DynamicNtk {
        /// The context length the model was originally trained with.
        trained_context_len: usize,
    },
}

// ---------------------------------------------------------------------------
// YaRN configuration
// ---------------------------------------------------------------------------

/// Configuration for YaRN (Yet another RoPE extensioN) scaling.
///
/// YaRN partitions frequency dimensions into three bands:
/// - **Low frequencies** (wavelength > `beta * trained_ctx`): fully interpolated.
/// - **High frequencies** (wavelength < `alpha * trained_ctx`): unchanged.
/// - **Middle band**: linearly ramped between the two extremes.
///
/// An optional attention temperature factor (`attn_factor`) rescales the
/// softmax temperature to compensate for the entropy increase from longer
/// sequences.
#[derive(Debug, Clone, PartialEq)]
pub struct YarnConfig {
    /// Extension ratio (target_ctx / trained_ctx).
    pub factor: f64,
    /// Trained (original) context length.
    pub trained_context_len: usize,
    /// Lower wavelength boundary coefficient.  Typical default: 1.0.
    pub alpha: f64,
    /// Upper wavelength boundary coefficient.  Typical default: 32.0.
    pub beta: f64,
    /// Softmax attention temperature factor. Typical default: 1.0.
    pub attn_factor: f64,
}

impl YarnConfig {
    /// Create a YaRN config with common defaults (`alpha=1, beta=32, attn=1`).
    pub fn new(factor: f64, trained_context_len: usize) -> Self {
        Self { factor, trained_context_len, alpha: 1.0, beta: 32.0, attn_factor: 1.0 }
    }
}

// ---------------------------------------------------------------------------
// Frequency table
// ---------------------------------------------------------------------------

/// Host-side RoPE frequency table ready for upload to device memory.
///
/// Layout: row-major `[max_seq_len, half_dim]` for both `sin` and `cos`.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeFrequencyTable {
    /// Half of the head dimension (`head_dim / 2`).
    pub half_dim: usize,
    /// Maximum sequence length the table covers.
    pub max_seq_len: usize,
    /// Flattened sine cache `[max_seq_len * half_dim]`.
    pub sin: Vec<f32>,
    /// Flattened cosine cache `[max_seq_len * half_dim]`.
    pub cos: Vec<f32>,
    /// Attention scaling factor (1.0 unless YaRN overrides it).
    pub attn_factor: f32,
}

// ---------------------------------------------------------------------------
// CUDA kernel launch parameters
// ---------------------------------------------------------------------------

/// Parameters for launching the CUDA RoPE forward-pass kernel.
///
/// The kernel rotates Q and K head vectors in-place using pre-computed sin/cos
/// tables.  These parameters describe the tensor shapes and strides so the
/// kernel can index correctly.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeKernelParams {
    /// Batch size.
    pub batch_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Current sequence length (number of positions to rotate).
    pub seq_len: usize,
    /// Full head dimension (must be even).
    pub head_dim: usize,
    /// Starting position offset (for incremental / KV-cache inference).
    pub position_offset: usize,
    /// Whether the input is in `[B, S, H, D]` layout (true) or
    /// `[B, H, S, D]` (false).
    pub is_neox_style: bool,
}

/// Suggested CUDA block size for the RoPE kernel.
///
/// One block handles one `(batch, head, position)` triple, with threads
/// iterating over the `half_dim` pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopeBlockConfig {
    /// Threads per block (x-dimension).
    pub threads_x: u32,
    /// Grid blocks in the x-dimension.
    pub grid_x: u32,
    /// Grid blocks in the y-dimension.
    pub grid_y: u32,
    /// Grid blocks in the z-dimension.
    pub grid_z: u32,
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate dimension and sequence length, returning `half_dim`.
fn validate_dim_seq(head_dim: usize, max_seq_len: usize) -> Result<usize, RopeError> {
    if head_dim == 0 {
        return Err(RopeError::ZeroDimension);
    }
    if !head_dim.is_multiple_of(2) {
        return Err(RopeError::OddDimension { dim: head_dim });
    }
    if head_dim > MAX_HEAD_DIM {
        return Err(RopeError::DimensionTooLarge { dim: head_dim });
    }
    if max_seq_len == 0 {
        return Err(RopeError::ZeroSequenceLength);
    }
    if max_seq_len > MAX_SEQ_LEN {
        return Err(RopeError::SequenceLengthTooLarge { len: max_seq_len });
    }
    Ok(head_dim / 2)
}

fn validate_base(base: f64) -> Result<(), RopeError> {
    if !base.is_finite() {
        return Err(RopeError::NonFiniteBase { base });
    }
    if base <= 0.0 {
        return Err(RopeError::NonPositiveBase { base });
    }
    Ok(())
}

fn validate_factor(factor: f64) -> Result<(), RopeError> {
    if !factor.is_finite() || factor <= 0.0 {
        return Err(RopeError::NonPositiveScale { scale: factor });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Inverse-frequency vector
// ---------------------------------------------------------------------------

/// Compute the standard inverse-frequency vector: `1 / (base^(2i/dim))`.
fn compute_inv_freq(half_dim: usize, head_dim: usize, base: f64) -> Vec<f64> {
    (0..half_dim).map(|i| 1.0 / base.powf((2.0 * i as f64) / head_dim as f64)).collect()
}

// ---------------------------------------------------------------------------
// Public API — frequency table builders
// ---------------------------------------------------------------------------

/// Build a standard RoPE frequency table (no scaling).
///
/// # Errors
///
/// Returns [`RopeError`] if `head_dim` is zero, odd, or exceeds limits, or if
/// `base` is non-positive / non-finite.
pub fn build_frequency_table(
    head_dim: usize,
    max_seq_len: usize,
    base: f64,
) -> Result<RopeFrequencyTable, RopeError> {
    let half_dim = validate_dim_seq(head_dim, max_seq_len)?;
    validate_base(base)?;

    let inv_freq = compute_inv_freq(half_dim, head_dim, base);
    Ok(materialize_table(half_dim, max_seq_len, &inv_freq, 1.0))
}

/// Build a frequency table with **linear position interpolation**.
///
/// Every position `p` is scaled to `p / factor` before computing the angle.
pub fn build_linear_scaled_table(
    head_dim: usize,
    max_seq_len: usize,
    base: f64,
    factor: f64,
) -> Result<RopeFrequencyTable, RopeError> {
    let half_dim = validate_dim_seq(head_dim, max_seq_len)?;
    validate_base(base)?;
    validate_factor(factor)?;

    let inv_freq = compute_inv_freq(half_dim, head_dim, base);
    Ok(materialize_table_scaled(half_dim, max_seq_len, &inv_freq, factor, 1.0))
}

/// Build a frequency table with **NTK-aware** base adjustment.
///
/// The base is scaled as `base * factor^(dim / (dim - 2))`, which spreads
/// the frequency spectrum without altering positions directly.
pub fn build_ntk_table(
    head_dim: usize,
    max_seq_len: usize,
    base: f64,
    factor: f64,
) -> Result<RopeFrequencyTable, RopeError> {
    let half_dim = validate_dim_seq(head_dim, max_seq_len)?;
    validate_base(base)?;
    validate_factor(factor)?;

    let adjusted_base = compute_ntk_base(base, factor, head_dim);
    let inv_freq = compute_inv_freq(half_dim, head_dim, adjusted_base);
    Ok(materialize_table(half_dim, max_seq_len, &inv_freq, 1.0))
}

/// Build a frequency table with **YaRN** piecewise interpolation.
///
/// Dimensions are partitioned into three bands by wavelength relative to the
/// trained context; each band receives a different interpolation ratio.
pub fn build_yarn_table(
    head_dim: usize,
    max_seq_len: usize,
    base: f64,
    cfg: &YarnConfig,
) -> Result<RopeFrequencyTable, RopeError> {
    let half_dim = validate_dim_seq(head_dim, max_seq_len)?;
    validate_base(base)?;
    validate_factor(cfg.factor)?;
    if cfg.trained_context_len == 0 {
        return Err(RopeError::ZeroTrainedContext);
    }
    if !cfg.alpha.is_finite() || cfg.alpha < 0.0 {
        return Err(RopeError::InvalidYarnParam { name: "alpha", value: cfg.alpha });
    }
    if !cfg.beta.is_finite() || cfg.beta < 0.0 {
        return Err(RopeError::InvalidYarnParam { name: "beta", value: cfg.beta });
    }
    if !cfg.attn_factor.is_finite() || cfg.attn_factor <= 0.0 {
        return Err(RopeError::InvalidYarnParam { name: "attn_factor", value: cfg.attn_factor });
    }

    let base_inv_freq = compute_inv_freq(half_dim, head_dim, base);
    let yarn_inv_freq = apply_yarn_interpolation(&base_inv_freq, cfg);
    Ok(materialize_table(half_dim, max_seq_len, &yarn_inv_freq, cfg.attn_factor as f32))
}

/// Build a frequency table with **dynamic NTK** scaling.
///
/// If `current_seq_len > trained_context_len` the base is recomputed so that
/// the effective context window covers `current_seq_len`.  Otherwise the table
/// is identical to standard RoPE.
pub fn build_dynamic_ntk_table(
    head_dim: usize,
    max_seq_len: usize,
    base: f64,
    trained_context_len: usize,
    current_seq_len: usize,
) -> Result<RopeFrequencyTable, RopeError> {
    let half_dim = validate_dim_seq(head_dim, max_seq_len)?;
    validate_base(base)?;
    if trained_context_len == 0 {
        return Err(RopeError::ZeroTrainedContext);
    }

    let effective_base = if current_seq_len > trained_context_len {
        let factor = current_seq_len as f64 / trained_context_len as f64;
        compute_ntk_base(base, factor, head_dim)
    } else {
        base
    };

    let inv_freq = compute_inv_freq(half_dim, head_dim, effective_base);
    Ok(materialize_table(half_dim, max_seq_len, &inv_freq, 1.0))
}

// ---------------------------------------------------------------------------
// Kernel launch configuration
// ---------------------------------------------------------------------------

/// Compute a suggested CUDA block/grid configuration for the RoPE kernel.
///
/// The kernel is organized so each thread-block processes one
/// `(batch, head, position)` triple.  Threads within the block iterate over
/// `half_dim` pairs.
pub fn compute_block_config(params: &RopeKernelParams) -> RopeBlockConfig {
    let half_dim = params.head_dim / 2;
    // Clamp threads to the warp-aligned ceiling of half_dim, max 256.
    let threads_x = ((half_dim as u32).next_multiple_of(32)).min(256);
    RopeBlockConfig {
        threads_x,
        grid_x: params.seq_len as u32,
        grid_y: params.num_heads as u32,
        grid_z: params.batch_size as u32,
    }
}

/// Validate [`RopeKernelParams`] before launch.
pub fn validate_kernel_params(p: &RopeKernelParams) -> Result<(), RopeError> {
    validate_dim_seq(p.head_dim, p.seq_len.max(1))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// NTK base adjustment: `base * factor ^ (dim / (dim - 2))`.
fn compute_ntk_base(base: f64, factor: f64, head_dim: usize) -> f64 {
    let dim = head_dim as f64;
    base * factor.powf(dim / (dim - 2.0))
}

/// Materialise sin/cos tables from an inverse-frequency vector.
fn materialize_table(
    half_dim: usize,
    max_seq_len: usize,
    inv_freq: &[f64],
    attn_factor: f32,
) -> RopeFrequencyTable {
    let total = max_seq_len * half_dim;
    let mut sin = Vec::with_capacity(total);
    let mut cos = Vec::with_capacity(total);

    for pos in 0..max_seq_len {
        for &freq in inv_freq {
            let angle = pos as f64 * freq;
            sin.push(angle.sin() as f32);
            cos.push(angle.cos() as f32);
        }
    }

    RopeFrequencyTable { half_dim, max_seq_len, sin, cos, attn_factor }
}

/// Materialise with linear position scaling (positions divided by `factor`).
fn materialize_table_scaled(
    half_dim: usize,
    max_seq_len: usize,
    inv_freq: &[f64],
    factor: f64,
    attn_factor: f32,
) -> RopeFrequencyTable {
    let total = max_seq_len * half_dim;
    let mut sin = Vec::with_capacity(total);
    let mut cos = Vec::with_capacity(total);

    for pos in 0..max_seq_len {
        let scaled_pos = pos as f64 / factor;
        for &freq in inv_freq {
            let angle = scaled_pos * freq;
            sin.push(angle.sin() as f32);
            cos.push(angle.cos() as f32);
        }
    }

    RopeFrequencyTable { half_dim, max_seq_len, sin, cos, attn_factor }
}

/// Compute YaRN-interpolated inverse frequencies.
///
/// For each frequency dimension the wavelength is `2π / freq`.  The dimension
/// is classified into one of three bands based on how the wavelength compares
/// to `alpha * trained_ctx` and `beta * trained_ctx`.
fn apply_yarn_interpolation(base_inv_freq: &[f64], cfg: &YarnConfig) -> Vec<f64> {
    let trained_ctx = cfg.trained_context_len as f64;
    let low_threshold = cfg.alpha * trained_ctx;
    let high_threshold = cfg.beta * trained_ctx;

    base_inv_freq
        .iter()
        .map(|&freq| {
            let wavelength = 2.0 * PI / freq;

            if wavelength < low_threshold {
                // High-frequency band — no interpolation.
                freq
            } else if wavelength > high_threshold {
                // Low-frequency band — full interpolation.
                freq / cfg.factor
            } else {
                // Middle band — linear ramp.
                let ramp = if (high_threshold - low_threshold).abs() < 1e-12 {
                    0.5
                } else {
                    (wavelength - low_threshold) / (high_threshold - low_threshold)
                };
                let interpolated = freq / cfg.factor;
                freq * (1.0 - ramp) + interpolated * ramp
            }
        })
        .collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helper ----------------------------------------------------------

    fn approx_eq(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "expected {a} ≈ {b} (tol {tol}, diff {})", (a - b).abs());
    }

    fn assert_trig_identity(sin: &[f32], cos: &[f32]) {
        for (s, c) in sin.iter().zip(cos) {
            let norm = s * s + c * c;
            approx_eq(norm, 1.0, 1e-4);
        }
    }

    // ======================================================================
    // Validation
    // ======================================================================

    #[test]
    fn rejects_zero_dimension() {
        assert_eq!(build_frequency_table(0, 8, DEFAULT_ROPE_BASE), Err(RopeError::ZeroDimension));
    }

    #[test]
    fn rejects_odd_dimension() {
        assert_eq!(
            build_frequency_table(3, 8, DEFAULT_ROPE_BASE),
            Err(RopeError::OddDimension { dim: 3 })
        );
    }

    #[test]
    fn rejects_dimension_too_large() {
        assert_eq!(
            build_frequency_table(2048, 8, DEFAULT_ROPE_BASE),
            Err(RopeError::DimensionTooLarge { dim: 2048 })
        );
    }

    #[test]
    fn rejects_zero_seq_len() {
        assert_eq!(
            build_frequency_table(64, 0, DEFAULT_ROPE_BASE),
            Err(RopeError::ZeroSequenceLength)
        );
    }

    #[test]
    fn rejects_seq_len_too_large() {
        assert_eq!(
            build_frequency_table(64, MAX_SEQ_LEN + 1, DEFAULT_ROPE_BASE),
            Err(RopeError::SequenceLengthTooLarge { len: MAX_SEQ_LEN + 1 })
        );
    }

    #[test]
    fn rejects_non_finite_base() {
        assert_eq!(
            build_frequency_table(64, 8, f64::INFINITY),
            Err(RopeError::NonFiniteBase { base: f64::INFINITY })
        );
    }

    #[test]
    fn rejects_nan_base() {
        let res = build_frequency_table(64, 8, f64::NAN);
        assert!(matches!(res, Err(RopeError::NonFiniteBase { .. })));
    }

    #[test]
    fn rejects_zero_base() {
        assert_eq!(
            build_frequency_table(64, 8, 0.0),
            Err(RopeError::NonPositiveBase { base: 0.0 })
        );
    }

    #[test]
    fn rejects_negative_base() {
        assert_eq!(
            build_frequency_table(64, 8, -1.0),
            Err(RopeError::NonPositiveBase { base: -1.0 })
        );
    }

    // ======================================================================
    // Standard frequency table
    // ======================================================================

    #[test]
    fn standard_table_shape() {
        let t = build_frequency_table(64, 128, DEFAULT_ROPE_BASE).unwrap();
        assert_eq!(t.half_dim, 32);
        assert_eq!(t.max_seq_len, 128);
        assert_eq!(t.sin.len(), 128 * 32);
        assert_eq!(t.cos.len(), 128 * 32);
    }

    #[test]
    fn standard_table_position_zero_is_zero_sin() {
        let t = build_frequency_table(64, 4, DEFAULT_ROPE_BASE).unwrap();
        for i in 0..t.half_dim {
            approx_eq(t.sin[i], 0.0, 1e-7);
            approx_eq(t.cos[i], 1.0, 1e-7);
        }
    }

    #[test]
    fn standard_table_trig_identity() {
        let t = build_frequency_table(128, 64, DEFAULT_ROPE_BASE).unwrap();
        assert_trig_identity(&t.sin, &t.cos);
    }

    #[test]
    fn standard_table_attn_factor_default() {
        let t = build_frequency_table(64, 4, DEFAULT_ROPE_BASE).unwrap();
        approx_eq(t.attn_factor, 1.0, 1e-7);
    }

    #[test]
    fn standard_table_minimum_dim() {
        let t = build_frequency_table(MIN_HEAD_DIM, 2, DEFAULT_ROPE_BASE).unwrap();
        assert_eq!(t.half_dim, 1);
        assert_eq!(t.sin.len(), 2);
    }

    #[test]
    fn standard_table_max_dim() {
        let t = build_frequency_table(MAX_HEAD_DIM, 2, DEFAULT_ROPE_BASE).unwrap();
        assert_eq!(t.half_dim, MAX_HEAD_DIM / 2);
    }

    #[test]
    fn standard_table_known_value_dim4() {
        let t = build_frequency_table(4, 2, DEFAULT_ROPE_BASE).unwrap();
        // position=1, freq_index=0 ⇒ angle = 1.0 * 1/10000^(0/4) = 1.0
        approx_eq(t.sin[t.half_dim], 1.0_f64.sin() as f32, 1e-6);
        approx_eq(t.cos[t.half_dim], 1.0_f64.cos() as f32, 1e-6);
    }

    #[test]
    fn standard_table_monotonic_freq_decay() {
        // inv_freq[i] should be strictly decreasing with i.
        let t = build_frequency_table(128, 4, DEFAULT_ROPE_BASE).unwrap();
        // At position=1 the sin values encode angle = inv_freq[i], so
        // sin[half_dim + i] should start large and decay (for small angles).
        // Just check that the first frequency gives a larger sin than the last.
        let first_sin = t.sin[t.half_dim].abs();
        let last_sin = t.sin[t.half_dim + t.half_dim - 1].abs();
        assert!(first_sin > last_sin, "first_sin={first_sin} <= last_sin={last_sin}");
    }

    // ======================================================================
    // Linear scaling
    // ======================================================================

    #[test]
    fn linear_rejects_zero_factor() {
        assert!(matches!(
            build_linear_scaled_table(64, 8, DEFAULT_ROPE_BASE, 0.0),
            Err(RopeError::NonPositiveScale { .. })
        ));
    }

    #[test]
    fn linear_rejects_negative_factor() {
        assert!(matches!(
            build_linear_scaled_table(64, 8, DEFAULT_ROPE_BASE, -2.0),
            Err(RopeError::NonPositiveScale { .. })
        ));
    }

    #[test]
    fn linear_factor_one_matches_standard() {
        let std_t = build_frequency_table(64, 16, DEFAULT_ROPE_BASE).unwrap();
        let lin_t = build_linear_scaled_table(64, 16, DEFAULT_ROPE_BASE, 1.0).unwrap();
        for (a, b) in std_t.sin.iter().zip(&lin_t.sin) {
            approx_eq(*a, *b, 1e-6);
        }
        for (a, b) in std_t.cos.iter().zip(&lin_t.cos) {
            approx_eq(*a, *b, 1e-6);
        }
    }

    #[test]
    fn linear_scaling_halves_angles() {
        // With factor=2, position p effectively becomes p/2.
        let t1 = build_frequency_table(4, 4, DEFAULT_ROPE_BASE).unwrap();
        let t2 = build_linear_scaled_table(4, 4, DEFAULT_ROPE_BASE, 2.0).unwrap();
        // Position 2 with factor=2 ⇒ effective position 1, so should match
        // standard position 1.
        let std_pos1_sin = t1.sin[t1.half_dim];
        let lin_pos2_sin = t2.sin[2 * t2.half_dim];
        approx_eq(std_pos1_sin, lin_pos2_sin, 1e-5);
    }

    #[test]
    fn linear_table_trig_identity() {
        let t = build_linear_scaled_table(64, 32, DEFAULT_ROPE_BASE, 4.0).unwrap();
        assert_trig_identity(&t.sin, &t.cos);
    }

    #[test]
    fn linear_table_shape() {
        let t = build_linear_scaled_table(32, 10, DEFAULT_ROPE_BASE, 2.0).unwrap();
        assert_eq!(t.half_dim, 16);
        assert_eq!(t.sin.len(), 10 * 16);
    }

    // ======================================================================
    // NTK-aware scaling
    // ======================================================================

    #[test]
    fn ntk_rejects_zero_factor() {
        assert!(matches!(
            build_ntk_table(64, 8, DEFAULT_ROPE_BASE, 0.0),
            Err(RopeError::NonPositiveScale { .. })
        ));
    }

    #[test]
    fn ntk_factor_one_matches_standard() {
        let std_t = build_frequency_table(64, 16, DEFAULT_ROPE_BASE).unwrap();
        let ntk_t = build_ntk_table(64, 16, DEFAULT_ROPE_BASE, 1.0).unwrap();
        for (a, b) in std_t.sin.iter().zip(&ntk_t.sin) {
            approx_eq(*a, *b, 1e-5);
        }
    }

    #[test]
    fn ntk_increases_effective_base() {
        // With factor > 1 the adjusted base should be larger.
        let base = DEFAULT_ROPE_BASE;
        let adjusted = compute_ntk_base(base, 2.0, 64);
        assert!(adjusted > base, "NTK base should increase with factor > 1");
    }

    #[test]
    fn ntk_table_trig_identity() {
        let t = build_ntk_table(128, 64, DEFAULT_ROPE_BASE, 4.0).unwrap();
        assert_trig_identity(&t.sin, &t.cos);
    }

    #[test]
    fn ntk_table_shape() {
        let t = build_ntk_table(64, 32, DEFAULT_ROPE_BASE, 2.0).unwrap();
        assert_eq!(t.half_dim, 32);
        assert_eq!(t.sin.len(), 32 * 32);
    }

    #[test]
    fn ntk_base_formula_regression() {
        // base * factor^(dim / (dim - 2)) with dim=4 ⇒ base * factor^2
        let b = compute_ntk_base(100.0, 3.0, 4);
        approx_eq(b as f32, 900.0, 1e-3);
    }

    // ======================================================================
    // YaRN
    // ======================================================================

    #[test]
    fn yarn_rejects_zero_factor() {
        let cfg = YarnConfig {
            factor: 0.0,
            trained_context_len: 4096,
            alpha: 1.0,
            beta: 32.0,
            attn_factor: 1.0,
        };
        assert!(matches!(
            build_yarn_table(64, 8, DEFAULT_ROPE_BASE, &cfg),
            Err(RopeError::NonPositiveScale { .. })
        ));
    }

    #[test]
    fn yarn_rejects_zero_trained_ctx() {
        let cfg = YarnConfig {
            factor: 2.0,
            trained_context_len: 0,
            alpha: 1.0,
            beta: 32.0,
            attn_factor: 1.0,
        };
        assert!(matches!(
            build_yarn_table(64, 8, DEFAULT_ROPE_BASE, &cfg),
            Err(RopeError::ZeroTrainedContext)
        ));
    }

    #[test]
    fn yarn_rejects_negative_alpha() {
        let cfg = YarnConfig {
            factor: 2.0,
            trained_context_len: 4096,
            alpha: -1.0,
            beta: 32.0,
            attn_factor: 1.0,
        };
        assert!(matches!(
            build_yarn_table(64, 8, DEFAULT_ROPE_BASE, &cfg),
            Err(RopeError::InvalidYarnParam { name: "alpha", .. })
        ));
    }

    #[test]
    fn yarn_rejects_negative_beta() {
        let cfg = YarnConfig {
            factor: 2.0,
            trained_context_len: 4096,
            alpha: 1.0,
            beta: -1.0,
            attn_factor: 1.0,
        };
        assert!(matches!(
            build_yarn_table(64, 8, DEFAULT_ROPE_BASE, &cfg),
            Err(RopeError::InvalidYarnParam { name: "beta", .. })
        ));
    }

    #[test]
    fn yarn_rejects_zero_attn_factor() {
        let cfg = YarnConfig {
            factor: 2.0,
            trained_context_len: 4096,
            alpha: 1.0,
            beta: 32.0,
            attn_factor: 0.0,
        };
        assert!(matches!(
            build_yarn_table(64, 8, DEFAULT_ROPE_BASE, &cfg),
            Err(RopeError::InvalidYarnParam { name: "attn_factor", .. })
        ));
    }

    #[test]
    fn yarn_rejects_nan_alpha() {
        let cfg = YarnConfig {
            factor: 2.0,
            trained_context_len: 4096,
            alpha: f64::NAN,
            beta: 32.0,
            attn_factor: 1.0,
        };
        assert!(matches!(
            build_yarn_table(64, 8, DEFAULT_ROPE_BASE, &cfg),
            Err(RopeError::InvalidYarnParam { name: "alpha", .. })
        ));
    }

    #[test]
    fn yarn_table_shape() {
        let cfg = YarnConfig::new(2.0, 4096);
        let t = build_yarn_table(64, 32, DEFAULT_ROPE_BASE, &cfg).unwrap();
        assert_eq!(t.half_dim, 32);
        assert_eq!(t.sin.len(), 32 * 32);
    }

    #[test]
    fn yarn_table_trig_identity() {
        let cfg = YarnConfig::new(4.0, 4096);
        let t = build_yarn_table(128, 64, DEFAULT_ROPE_BASE, &cfg).unwrap();
        assert_trig_identity(&t.sin, &t.cos);
    }

    #[test]
    fn yarn_attn_factor_propagated() {
        let cfg = YarnConfig {
            factor: 2.0,
            trained_context_len: 4096,
            alpha: 1.0,
            beta: 32.0,
            attn_factor: 0.7,
        };
        let t = build_yarn_table(64, 4, DEFAULT_ROPE_BASE, &cfg).unwrap();
        approx_eq(t.attn_factor, 0.7, 1e-6);
    }

    #[test]
    fn yarn_default_config_alpha_beta() {
        let cfg = YarnConfig::new(2.0, 4096);
        assert!((cfg.alpha - 1.0).abs() < 1e-12);
        assert!((cfg.beta - 32.0).abs() < 1e-12);
        assert!((cfg.attn_factor - 1.0).abs() < 1e-12);
    }

    #[test]
    fn yarn_high_freq_unchanged() {
        // With alpha=1, beta=32, trained_ctx=4096: low_threshold = 4096.
        // For dim=4, base=10000, inv_freq[0] = 1.0, wavelength = 2π ≈ 6.28.
        // wavelength < low_threshold ⇒ high-freq band, no interpolation.
        let cfg = YarnConfig::new(2.0, 4096);
        let std_t = build_frequency_table(4, 4, DEFAULT_ROPE_BASE).unwrap();
        let yarn_t = build_yarn_table(4, 4, DEFAULT_ROPE_BASE, &cfg).unwrap();
        // High-freq dimension should be identical.
        for pos in 0..4 {
            approx_eq(std_t.sin[pos * 1], yarn_t.sin[pos * 1], 1e-6);
            approx_eq(std_t.cos[pos * 1], yarn_t.cos[pos * 1], 1e-6);
        }
    }

    #[test]
    fn yarn_low_freq_interpolated() {
        // Construct a scenario where wavelength > high_threshold so full
        // interpolation applies: freq_scaled = freq / factor.
        // Use a tiny base so inv_freq is very small ⇒ huge wavelength.
        let base = 1.0001; // very small base
        let cfg = YarnConfig {
            factor: 4.0,
            trained_context_len: 1,
            alpha: 0.0,
            beta: 0.001, // tiny thresholds
            attn_factor: 1.0,
        };
        let half_dim = 1;
        let head_dim = 2;
        let base_inv = compute_inv_freq(half_dim, head_dim, base);
        let yarn_inv = apply_yarn_interpolation(&base_inv, &cfg);
        // Full interpolation: yarn_inv[0] ≈ base_inv[0] / 4.0
        let expected = base_inv[0] / 4.0;
        assert!((yarn_inv[0] - expected).abs() < 1e-10);
    }

    // ======================================================================
    // Dynamic NTK
    // ======================================================================

    #[test]
    fn dynamic_ntk_no_extension() {
        // current_seq_len <= trained_ctx ⇒ identical to standard.
        let std_t = build_frequency_table(64, 16, DEFAULT_ROPE_BASE).unwrap();
        let dyn_t = build_dynamic_ntk_table(64, 16, DEFAULT_ROPE_BASE, 4096, 2048).unwrap();
        for (a, b) in std_t.sin.iter().zip(&dyn_t.sin) {
            approx_eq(*a, *b, 1e-7);
        }
    }

    #[test]
    fn dynamic_ntk_exact_boundary() {
        // current == trained ⇒ no adjustment.
        let std_t = build_frequency_table(64, 4, DEFAULT_ROPE_BASE).unwrap();
        let dyn_t = build_dynamic_ntk_table(64, 4, DEFAULT_ROPE_BASE, 4096, 4096).unwrap();
        for (a, b) in std_t.sin.iter().zip(&dyn_t.sin) {
            approx_eq(*a, *b, 1e-7);
        }
    }

    #[test]
    fn dynamic_ntk_extends_beyond_trained() {
        // current_seq_len > trained ⇒ base should increase.
        let std_t = build_frequency_table(64, 4, DEFAULT_ROPE_BASE).unwrap();
        let dyn_t = build_dynamic_ntk_table(64, 4, DEFAULT_ROPE_BASE, 4096, 8192).unwrap();
        // inv_freq[0] is always 1.0 regardless of base, so check index 1.
        let half = std_t.half_dim;
        assert!(
            (std_t.sin[half + 1] - dyn_t.sin[half + 1]).abs() > 1e-5,
            "dynamic NTK should alter frequencies when seq_len > trained_ctx"
        );
    }

    #[test]
    fn dynamic_ntk_rejects_zero_trained_ctx() {
        assert!(matches!(
            build_dynamic_ntk_table(64, 4, DEFAULT_ROPE_BASE, 0, 1),
            Err(RopeError::ZeroTrainedContext)
        ));
    }

    #[test]
    fn dynamic_ntk_trig_identity() {
        let t = build_dynamic_ntk_table(128, 32, DEFAULT_ROPE_BASE, 4096, 8192).unwrap();
        assert_trig_identity(&t.sin, &t.cos);
    }

    #[test]
    fn dynamic_ntk_table_shape() {
        let t = build_dynamic_ntk_table(64, 16, DEFAULT_ROPE_BASE, 2048, 4096).unwrap();
        assert_eq!(t.half_dim, 32);
        assert_eq!(t.sin.len(), 16 * 32);
    }

    // ======================================================================
    // Kernel params & block config
    // ======================================================================

    #[test]
    fn block_config_small_dim() {
        let p = RopeKernelParams {
            batch_size: 1,
            num_heads: 8,
            seq_len: 32,
            head_dim: 64,
            position_offset: 0,
            is_neox_style: true,
        };
        let bc = compute_block_config(&p);
        assert_eq!(bc.threads_x, 32); // 64/2 = 32, already warp-aligned
        assert_eq!(bc.grid_x, 32);
        assert_eq!(bc.grid_y, 8);
        assert_eq!(bc.grid_z, 1);
    }

    #[test]
    fn block_config_large_dim_clamped() {
        let p = RopeKernelParams {
            batch_size: 2,
            num_heads: 32,
            seq_len: 1024,
            head_dim: 1024,
            position_offset: 0,
            is_neox_style: false,
        };
        let bc = compute_block_config(&p);
        assert_eq!(bc.threads_x, 256); // clamped
        assert_eq!(bc.grid_x, 1024);
        assert_eq!(bc.grid_y, 32);
        assert_eq!(bc.grid_z, 2);
    }

    #[test]
    fn block_config_non_warp_aligned_rounds_up() {
        let p = RopeKernelParams {
            batch_size: 1,
            num_heads: 1,
            seq_len: 1,
            head_dim: 6, // half_dim = 3, not warp-aligned
            position_offset: 0,
            is_neox_style: true,
        };
        let bc = compute_block_config(&p);
        assert_eq!(bc.threads_x, 32); // rounded up to 32
    }

    #[test]
    fn validate_kernel_params_ok() {
        let p = RopeKernelParams {
            batch_size: 1,
            num_heads: 8,
            seq_len: 128,
            head_dim: 64,
            position_offset: 0,
            is_neox_style: true,
        };
        assert!(validate_kernel_params(&p).is_ok());
    }

    #[test]
    fn validate_kernel_params_odd_dim() {
        let p = RopeKernelParams {
            batch_size: 1,
            num_heads: 1,
            seq_len: 1,
            head_dim: 7,
            position_offset: 0,
            is_neox_style: false,
        };
        assert!(matches!(validate_kernel_params(&p), Err(RopeError::OddDimension { .. })));
    }

    #[test]
    fn validate_kernel_params_zero_dim() {
        let p = RopeKernelParams {
            batch_size: 1,
            num_heads: 1,
            seq_len: 1,
            head_dim: 0,
            position_offset: 0,
            is_neox_style: false,
        };
        assert!(matches!(validate_kernel_params(&p), Err(RopeError::ZeroDimension)));
    }

    // ======================================================================
    // RopeScalingMethod enum coverage
    // ======================================================================

    #[test]
    fn scaling_method_none() {
        let m = RopeScalingMethod::None;
        assert_eq!(m, RopeScalingMethod::None);
    }

    #[test]
    fn scaling_method_linear() {
        let m = RopeScalingMethod::Linear { factor: 2.0 };
        assert_eq!(m, RopeScalingMethod::Linear { factor: 2.0 });
    }

    #[test]
    fn scaling_method_ntk() {
        let m = RopeScalingMethod::Ntk { factor: 4.0 };
        assert_eq!(m, RopeScalingMethod::Ntk { factor: 4.0 });
    }

    #[test]
    fn scaling_method_dynamic_ntk() {
        let m = RopeScalingMethod::DynamicNtk { trained_context_len: 4096 };
        assert_eq!(m, RopeScalingMethod::DynamicNtk { trained_context_len: 4096 });
    }

    #[test]
    fn scaling_method_yarn() {
        let cfg = YarnConfig::new(2.0, 4096);
        let m = RopeScalingMethod::Yarn(cfg.clone());
        assert_eq!(m, RopeScalingMethod::Yarn(cfg));
    }

    // ======================================================================
    // Error display coverage
    // ======================================================================

    #[test]
    fn error_display_zero_dim() {
        let e = RopeError::ZeroDimension;
        assert!(e.to_string().contains("must be > 0"));
    }

    #[test]
    fn error_display_odd_dim() {
        let e = RopeError::OddDimension { dim: 5 };
        assert!(e.to_string().contains("5"));
    }

    #[test]
    fn error_display_dim_too_large() {
        let e = RopeError::DimensionTooLarge { dim: 9999 };
        assert!(e.to_string().contains("9999"));
    }

    #[test]
    fn error_display_zero_seq() {
        let e = RopeError::ZeroSequenceLength;
        assert!(e.to_string().contains("sequence length"));
    }

    #[test]
    fn error_display_seq_too_large() {
        let e = RopeError::SequenceLengthTooLarge { len: 99999999 };
        assert!(e.to_string().contains("99999999"));
    }

    #[test]
    fn error_display_non_finite_base() {
        let e = RopeError::NonFiniteBase { base: f64::NAN };
        assert!(e.to_string().contains("finite"));
    }

    #[test]
    fn error_display_non_positive_base() {
        let e = RopeError::NonPositiveBase { base: -5.0 };
        assert!(e.to_string().contains("-5"));
    }

    #[test]
    fn error_display_non_positive_scale() {
        let e = RopeError::NonPositiveScale { scale: 0.0 };
        assert!(e.to_string().contains("0"));
    }

    #[test]
    fn error_display_invalid_yarn_param() {
        let e = RopeError::InvalidYarnParam { name: "alpha", value: -1.0 };
        assert!(e.to_string().contains("alpha"));
    }

    #[test]
    fn error_display_zero_trained_ctx() {
        let e = RopeError::ZeroTrainedContext;
        assert!(e.to_string().contains("trained context"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(RopeError::ZeroDimension);
        let _ = e.to_string();
    }

    // ======================================================================
    // Struct field coverage
    // ======================================================================

    #[test]
    fn rope_frequency_table_fields() {
        let t = build_frequency_table(4, 2, DEFAULT_ROPE_BASE).unwrap();
        assert_eq!(t.half_dim, 2);
        assert_eq!(t.max_seq_len, 2);
        assert_eq!(t.sin.len(), 4);
        assert_eq!(t.cos.len(), 4);
        assert!((t.attn_factor - 1.0).abs() < 1e-7);
    }

    #[test]
    fn rope_kernel_params_neox_flag() {
        let p = RopeKernelParams {
            batch_size: 1,
            num_heads: 1,
            seq_len: 1,
            head_dim: 2,
            position_offset: 42,
            is_neox_style: true,
        };
        assert!(p.is_neox_style);
        assert_eq!(p.position_offset, 42);
    }

    #[test]
    fn rope_block_config_debug() {
        let bc = RopeBlockConfig { threads_x: 32, grid_x: 1, grid_y: 1, grid_z: 1 };
        let dbg = format!("{bc:?}");
        assert!(dbg.contains("threads_x"));
    }

    #[test]
    fn rope_block_config_copy() {
        let bc = RopeBlockConfig { threads_x: 64, grid_x: 2, grid_y: 3, grid_z: 4 };
        let bc2 = bc;
        assert_eq!(bc, bc2);
    }

    // ======================================================================
    // Constants
    // ======================================================================

    #[test]
    fn default_base_is_10k() {
        assert!((DEFAULT_ROPE_BASE - 10_000.0).abs() < 1e-9);
    }

    #[test]
    fn default_head_dim_is_128() {
        assert_eq!(DEFAULT_HEAD_DIM, 128);
    }

    #[test]
    fn default_max_seq_len_is_4096() {
        assert_eq!(DEFAULT_MAX_SEQ_LEN, 4096);
    }

    #[test]
    fn min_head_dim_is_2() {
        assert_eq!(MIN_HEAD_DIM, 2);
    }

    #[test]
    fn max_head_dim_is_1024() {
        assert_eq!(MAX_HEAD_DIM, 1024);
    }

    // ======================================================================
    // Inverse-frequency internals
    // ======================================================================

    #[test]
    fn inv_freq_first_element_is_one() {
        let v = compute_inv_freq(4, 8, DEFAULT_ROPE_BASE);
        approx_eq(v[0] as f32, 1.0, 1e-7);
    }

    #[test]
    fn inv_freq_strictly_decreasing() {
        let v = compute_inv_freq(32, 64, DEFAULT_ROPE_BASE);
        for w in v.windows(2) {
            assert!(w[0] > w[1], "{} should be > {}", w[0], w[1]);
        }
    }

    // ======================================================================
    // NTK base helper
    // ======================================================================

    #[test]
    fn ntk_base_factor_one_identity() {
        let b = compute_ntk_base(DEFAULT_ROPE_BASE, 1.0, 128);
        approx_eq(b as f32, DEFAULT_ROPE_BASE as f32, 1e-2);
    }

    #[test]
    fn ntk_base_increases_with_factor() {
        let b1 = compute_ntk_base(DEFAULT_ROPE_BASE, 2.0, 128);
        let b2 = compute_ntk_base(DEFAULT_ROPE_BASE, 4.0, 128);
        assert!(b2 > b1);
    }

    // ======================================================================
    // YaRN interpolation internals
    // ======================================================================

    #[test]
    fn yarn_interp_equal_thresholds_uses_midpoint() {
        // alpha == beta ⇒ thresholds equal ⇒ ramp = 0.5 (when wavelength is
        // in the middle band).  Set trained_ctx so that the thresholds equal
        // the wavelength of our test frequency.
        // freq = 1.0 ⇒ wavelength = 2π ≈ 6.2832
        // We want low_threshold == high_threshold == wavelength.
        // threshold = alpha * trained_ctx, so trained_ctx = wavelength / alpha.
        let wavelength = 2.0 * std::f64::consts::PI;
        let trained_ctx = wavelength.ceil() as usize; // 7
        let alpha = wavelength / trained_ctx as f64; // ≈ 0.8976
        let cfg = YarnConfig {
            factor: 2.0,
            trained_context_len: trained_ctx,
            alpha,
            beta: alpha,
            attn_factor: 1.0,
        };
        let freq = vec![1.0];
        let result = apply_yarn_interpolation(&freq, &cfg);
        // thresholds are equal so the middle-band ramp defaults to 0.5
        let expected = freq[0] * 0.5 + (freq[0] / 2.0) * 0.5;
        assert!((result[0] - expected).abs() < 1e-10);
    }

    // ======================================================================
    // Clone / PartialEq for YarnConfig
    // ======================================================================

    #[test]
    fn yarn_config_clone() {
        let cfg = YarnConfig::new(3.0, 8192);
        let cfg2 = cfg.clone();
        assert_eq!(cfg, cfg2);
    }

    #[test]
    fn yarn_config_partial_eq() {
        let a = YarnConfig::new(2.0, 4096);
        let b = YarnConfig::new(2.0, 4096);
        assert_eq!(a, b);
        let c = YarnConfig::new(3.0, 4096);
        assert_ne!(a, c);
    }

    // ======================================================================
    // RopeFrequencyTable clone
    // ======================================================================

    #[test]
    fn frequency_table_clone() {
        let t = build_frequency_table(4, 2, DEFAULT_ROPE_BASE).unwrap();
        let t2 = t.clone();
        assert_eq!(t.half_dim, t2.half_dim);
        assert_eq!(t.sin, t2.sin);
    }
}
