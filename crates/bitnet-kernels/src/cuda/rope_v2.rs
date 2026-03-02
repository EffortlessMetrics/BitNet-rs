//! RoPE v2 — advanced Rotary Position Embedding variants.
//!
//! Extends the baseline [`super::rope`] module with:
//!
//! - **NTK-aware scaling** — adjusts the rotation base to extend effective
//!   context without interpolation artifacts.
//! - **GPT-NeoX layout** — split-half rotation `(x[:half], x[half:])` instead
//!   of interleaved `(x[0::2], x[1::2])`.
//! - **YaRN** — Yet another RoPE extension with per-head attention temperature
//!   (`t = 0.1 * ln(s) + 1` where `s` is the scaling factor).
//! - **Batched dispatch** — process multiple sequences in one call.
//! - **Dynamic context extension** — compute NTK factor at runtime from
//!   `current_seq_len / original_max_pos`.
//!
//! # Kernel strategy
//!
//! CPU reference implementations are **always compiled** and used as
//! fallback.  CUDA kernel source strings and GPU launch stubs are gated
//! behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use std::fmt;

use bitnet_common::{KernelError, Result};

// ───────────────────────────────────────────────────────────────────
// Error type
// ───────────────────────────────────────────────────────────────────

/// Errors specific to RoPE v2 operations.
#[derive(Debug, Clone)]
pub enum RoPEV2Error {
    /// `head_dim` is zero or not a multiple of 2.
    InvalidHeadDim(usize),
    /// `n_heads` is zero.
    InvalidNumHeads(usize),
    /// `max_seq_len` is zero.
    InvalidSeqLen(usize),
    /// The caller-supplied buffer has the wrong length.
    ShapeMismatch { expected: usize, got: usize },
    /// YaRN attention temperature must be finite and positive.
    InvalidAttentionTemp(f32),
    /// General configuration error.
    InvalidConfig(String),
}

impl fmt::Display for RoPEV2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHeadDim(d) => {
                write!(f, "head_dim must be even and non-zero, got {d}")
            }
            Self::InvalidNumHeads(n) => {
                write!(f, "n_heads must be non-zero, got {n}")
            }
            Self::InvalidSeqLen(s) => {
                write!(f, "max_seq_len must be non-zero, got {s}")
            }
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected}, got {got}")
            }
            Self::InvalidAttentionTemp(t) => {
                write!(f, "YaRN attention temp must be finite > 0, got {t}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for RoPEV2Error {}

/// Convert [`RoPEV2Error`] into [`KernelError`] so callers returning
/// `bitnet_common::Result<T>` can use `?` with one extra `.into()`.
impl From<RoPEV2Error> for KernelError {
    fn from(e: RoPEV2Error) -> Self {
        KernelError::InvalidArguments { reason: e.to_string() }
    }
}

/// Shorthand: map a `RoPEV2Error` through the two-hop chain
/// `RoPEV2Error → KernelError → BitNetError` so public functions can
/// return `bitnet_common::Result<T>`.
fn rope_err(e: RoPEV2Error) -> bitnet_common::BitNetError {
    KernelError::from(e).into()
}

// ───────────────────────────────────────────────────────────────────
// Scaling type
// ───────────────────────────────────────────────────────────────────

/// Frequency-scaling strategy applied to the inverse-frequency table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalingType {
    /// Simple linear interpolation: `inv_freq *= factor`.
    Linear,
    /// NTK-aware scaling: adjusts the rotation base itself so that
    /// high-frequency dimensions are preserved while low-frequency
    /// ones are stretched.
    NtkAware,
    /// YaRN (Yet another RoPE extensioN): NTK base plus per-head
    /// attention temperature correction.
    Yarn,
}

impl fmt::Display for ScalingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linear => write!(f, "linear"),
            Self::NtkAware => write!(f, "ntk-aware"),
            Self::Yarn => write!(f, "yarn"),
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Configuration
// ───────────────────────────────────────────────────────────────────

/// Configuration for RoPE v2 operations.
///
/// Create via [`RoPEConfig::new`] and chain builder methods:
///
/// ```ignore
/// let cfg = RoPEConfig::new(64, 8, 128)?
///     .with_base_freq(10_000.0)
///     .with_scaling(ScalingType::NtkAware, 4.0);
/// ```
#[derive(Debug, Clone)]
pub struct RoPEConfig {
    /// Per-head embedding dimension (must be even).
    pub head_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Maximum sequence length for pre-computed tables.
    pub max_seq_len: usize,
    /// Rotation base frequency (default `10_000.0`).
    pub base_freq: f32,
    /// Frequency scaling strategy.
    pub scaling_type: ScalingType,
    /// Scaling factor (default `1.0`). Interpretation depends on
    /// `scaling_type`.
    pub scaling_factor: f32,
    /// When `true`, use the GPT-NeoX interleaved layout where pairs
    /// sit at `(i, i + head_dim/2)` instead of `(2*i, 2*i+1)`.
    pub interleaved: bool,
    /// Position offset for KV-cache continuation.
    pub position_offset: usize,
}

impl RoPEConfig {
    /// Validate and create a new configuration.
    pub fn new(
        head_dim: usize,
        n_heads: usize,
        max_seq_len: usize,
    ) -> std::result::Result<Self, RoPEV2Error> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(RoPEV2Error::InvalidHeadDim(head_dim));
        }
        if n_heads == 0 {
            return Err(RoPEV2Error::InvalidNumHeads(n_heads));
        }
        if max_seq_len == 0 {
            return Err(RoPEV2Error::InvalidSeqLen(max_seq_len));
        }
        Ok(Self {
            head_dim,
            n_heads,
            max_seq_len,
            base_freq: 10_000.0,
            scaling_type: ScalingType::Linear,
            scaling_factor: 1.0,
            interleaved: false,
            position_offset: 0,
        })
    }

    /// Override the rotation base frequency.
    #[must_use]
    pub fn with_base_freq(mut self, base: f32) -> Self {
        self.base_freq = base;
        self
    }

    /// Set the scaling strategy and factor.
    #[must_use]
    pub fn with_scaling(mut self, ty: ScalingType, factor: f32) -> Self {
        self.scaling_type = ty;
        self.scaling_factor = factor;
        self
    }

    /// Enable GPT-NeoX interleaved layout.
    #[must_use]
    pub fn with_interleaved(mut self, yes: bool) -> Self {
        self.interleaved = yes;
        self
    }

    /// Override `max_seq_len`.
    #[must_use]
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Set position offset for KV-cache decode.
    #[must_use]
    pub fn with_position_offset(mut self, offset: usize) -> Self {
        self.position_offset = offset;
        self
    }

    /// Number of dimension pairs (`head_dim / 2`).
    pub fn half_dim(&self) -> usize {
        self.head_dim / 2
    }
}

// ───────────────────────────────────────────────────────────────────
// Internal helpers
// ───────────────────────────────────────────────────────────────────

/// Compute the effective base frequency after NTK scaling.
///
/// Formula: `base * ((factor * n) / (n - 2))^(n / (n - 2))`
/// where `n = head_dim`.
fn ntk_adjusted_base(base: f32, factor: f32, head_dim: usize) -> f32 {
    let n = head_dim as f32;
    if n <= 2.0 {
        return base;
    }
    let ratio = n / (n - 2.0);
    base * (factor * ratio).powf(ratio)
}

/// Build the inverse-frequency vector for the given config.
///
/// Returns a `Vec<f32>` of length `head_dim / 2`.
fn build_inv_freq(cfg: &RoPEConfig) -> Vec<f32> {
    let half = cfg.half_dim();
    let effective_base = match cfg.scaling_type {
        ScalingType::NtkAware | ScalingType::Yarn => {
            ntk_adjusted_base(cfg.base_freq, cfg.scaling_factor, cfg.head_dim)
        }
        ScalingType::Linear => cfg.base_freq,
    };
    (0..half)
        .map(|i| {
            let exp = -2.0 * (i as f32) / (cfg.head_dim as f32);
            let freq = effective_base.powf(exp);
            match cfg.scaling_type {
                ScalingType::Linear => freq * cfg.scaling_factor,
                _ => freq,
            }
        })
        .collect()
}

/// YaRN attention temperature: `t = 0.1 * ln(s) + 1`.
fn yarn_attention_temp(scaling_factor: f32) -> f32 {
    0.1 * scaling_factor.ln() + 1.0
}

// ───────────────────────────────────────────────────────────────────
// CPU reference — frequency table
// ───────────────────────────────────────────────────────────────────

/// Pre-compute the sin/cos frequency table.
///
/// Returns a flat `Vec<f32>` of length `max_seq_len * head_dim`.
/// Layout per position row: `[cos_0, sin_0, cos_1, sin_1, …]`.
pub fn compute_rope_frequencies_v2(cfg: &RoPEConfig) -> Vec<f32> {
    let inv_freq = build_inv_freq(cfg);
    let half = cfg.half_dim();
    let mut table = vec![0.0_f32; cfg.max_seq_len * cfg.head_dim];
    for pos in 0..cfg.max_seq_len {
        for i in 0..half {
            let angle = (pos as f32 + cfg.position_offset as f32) * inv_freq[i];
            let (sin, cos) = angle.sin_cos();
            table[pos * cfg.head_dim + 2 * i] = cos;
            table[pos * cfg.head_dim + 2 * i + 1] = sin;
        }
    }
    table
}

// ───────────────────────────────────────────────────────────────────
// CPU reference — standard RoPE
// ───────────────────────────────────────────────────────────────────

/// Apply standard RoPE to `input`, writing to `output`.
///
/// Both buffers must have length `n_heads * max_seq_len * head_dim`.
pub fn apply_rope_v2(cfg: &RoPEConfig, input: &[f32], output: &mut [f32]) -> Result<()> {
    let expected = cfg.n_heads * cfg.max_seq_len * cfg.head_dim;
    if input.len() != expected {
        return Err(rope_err(RoPEV2Error::ShapeMismatch { expected, got: input.len() }));
    }
    if output.len() != expected {
        return Err(rope_err(RoPEV2Error::ShapeMismatch { expected, got: output.len() }));
    }
    let table = compute_rope_frequencies_v2(cfg);
    let hd = cfg.head_dim;
    let half = cfg.half_dim();
    for h in 0..cfg.n_heads {
        for pos in 0..cfg.max_seq_len {
            let row = (h * cfg.max_seq_len + pos) * hd;
            for i in 0..half {
                let cos = table[pos * hd + 2 * i];
                let sin = table[pos * hd + 2 * i + 1];
                let (idx0, idx1) = if cfg.interleaved {
                    (row + i, row + i + half)
                } else {
                    (row + 2 * i, row + 2 * i + 1)
                };
                let x0 = input[idx0];
                let x1 = input[idx1];
                output[idx0] = x0 * cos - x1 * sin;
                output[idx1] = x0 * sin + x1 * cos;
            }
        }
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// CPU reference — NeoX layout
// ───────────────────────────────────────────────────────────────────

/// Apply RoPE with GPT-NeoX split-half layout.
///
/// Equivalent to `apply_rope_v2` with `interleaved = true`, but
/// provided as a separate entry point for clarity.
pub fn apply_rope_neox_v2(cfg: &RoPEConfig, input: &[f32], output: &mut [f32]) -> Result<()> {
    let mut neox_cfg = cfg.clone();
    neox_cfg.interleaved = true;
    apply_rope_v2(&neox_cfg, input, output)
}

// ───────────────────────────────────────────────────────────────────
// CPU reference — YaRN
// ───────────────────────────────────────────────────────────────────

/// Apply YaRN-style RoPE with attention temperature correction.
///
/// After the standard rotation the output is divided by the YaRN
/// temperature `t = 0.1 * ln(scaling_factor) + 1`.
pub fn apply_rope_yarn(cfg: &RoPEConfig, input: &[f32], output: &mut [f32]) -> Result<()> {
    let expected = cfg.n_heads * cfg.max_seq_len * cfg.head_dim;
    if input.len() != expected {
        return Err(rope_err(RoPEV2Error::ShapeMismatch { expected, got: input.len() }));
    }
    if output.len() != expected {
        return Err(rope_err(RoPEV2Error::ShapeMismatch { expected, got: output.len() }));
    }
    let temp = yarn_attention_temp(cfg.scaling_factor);
    if !temp.is_finite() || temp <= 0.0 {
        return Err(rope_err(RoPEV2Error::InvalidAttentionTemp(temp)));
    }

    // Delegate standard rotation then apply temperature.
    let yarn_cfg = RoPEConfig { scaling_type: ScalingType::Yarn, ..cfg.clone() };
    apply_rope_v2(&yarn_cfg, input, output)?;
    let inv_temp = 1.0 / temp;
    for v in output.iter_mut() {
        *v *= inv_temp;
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// CPU reference — batched
// ───────────────────────────────────────────────────────────────────

/// Apply RoPE to a batch of sequences.
///
/// `input` has shape `[batch, n_heads, seq_len, head_dim]` in
/// row-major order.
pub fn apply_rope_batched(
    cfg: &RoPEConfig,
    batch_size: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<()> {
    let per_seq = cfg.n_heads * cfg.max_seq_len * cfg.head_dim;
    let total = batch_size * per_seq;
    if input.len() != total {
        return Err(rope_err(RoPEV2Error::ShapeMismatch { expected: total, got: input.len() }));
    }
    if output.len() != total {
        return Err(rope_err(RoPEV2Error::ShapeMismatch { expected: total, got: output.len() }));
    }
    for b in 0..batch_size {
        let start = b * per_seq;
        let end = start + per_seq;
        apply_rope_v2(cfg, &input[start..end], &mut output[start..end])?;
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// Dynamic context extension
// ───────────────────────────────────────────────────────────────────

/// Compute an NTK-aware config for dynamic context extension.
///
/// If `current_seq_len ≤ original_max_pos` the original config is
/// returned unchanged (no extension needed).  Otherwise, the scaling
/// factor is set to `current_seq_len / original_max_pos` and the
/// scaling type is switched to [`ScalingType::NtkAware`].
pub fn rope_context_extension(
    base_cfg: &RoPEConfig,
    current_seq_len: usize,
    original_max_pos: usize,
) -> RoPEConfig {
    if current_seq_len <= original_max_pos {
        // Within training context — return original config unchanged.
        return base_cfg.clone();
    }
    let factor = current_seq_len as f32 / original_max_pos as f32;
    let mut ext = base_cfg.clone();
    ext.scaling_type = ScalingType::NtkAware;
    ext.scaling_factor = factor;
    ext.max_seq_len = current_seq_len;
    ext
}

// ───────────────────────────────────────────────────────────────────
// CUDA kernel sources
// ───────────────────────────────────────────────────────────────────

/// CUDA C source for the RoPE v2 forward kernel.
///
/// Supports linear, NTK-aware, and YaRN scaling via the
/// `scaling_type` parameter (0 = linear, 1 = NTK, 2 = YaRN).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ROPE_V2_FORWARD_KERNEL_SRC: &str = r#"
extern "C" __global__ void rope_v2_forward_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int head_dim,
    const int seq_len,
    const int n_heads,
    const int position_offset,
    const float base_freq,
    const float scaling_factor,
    const int scaling_type,
    const int interleaved,
    const float yarn_temp_inv
) {
    const int pos  = blockIdx.x;
    const int head = blockIdx.y;
    const int half = head_dim / 2;
    const int i    = threadIdx.x;
    if (i >= half) return;

    float eff_base = base_freq;
    if (scaling_type == 1 || scaling_type == 2) {
        float n = (float)head_dim;
        float ratio = n / (n - 2.0f);
        eff_base = base_freq * powf(scaling_factor * ratio, ratio);
    }

    float exp_val = -2.0f * (float)i / (float)head_dim;
    float inv_freq = powf(eff_base, exp_val);
    if (scaling_type == 0) inv_freq *= scaling_factor;

    float angle = (float)(pos + position_offset) * inv_freq;
    float cos_val, sin_val;
    __sincosf(angle, &sin_val, &cos_val);

    int row = (head * seq_len + pos) * head_dim;
    int idx0, idx1;
    if (interleaved) {
        idx0 = row + i;
        idx1 = row + i + half;
    } else {
        idx0 = row + 2 * i;
        idx1 = row + 2 * i + 1;
    }
    float x0 = input[idx0];
    float x1 = input[idx1];
    float o0 = x0 * cos_val - x1 * sin_val;
    float o1 = x0 * sin_val + x1 * cos_val;

    if (scaling_type == 2) {
        o0 *= yarn_temp_inv;
        o1 *= yarn_temp_inv;
    }
    output[idx0] = o0;
    output[idx1] = o1;
}
"#;

/// CUDA C source for the batched RoPE v2 kernel.
///
/// Adds a batch dimension: grid `(seq_len, n_heads, batch)`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ROPE_V2_BATCHED_KERNEL_SRC: &str = r#"
extern "C" __global__ void rope_v2_batched_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int head_dim,
    const int seq_len,
    const int n_heads,
    const int position_offset,
    const float base_freq,
    const float scaling_factor,
    const int scaling_type,
    const int interleaved,
    const float yarn_temp_inv
) {
    const int pos   = blockIdx.x;
    const int head  = blockIdx.y;
    const int batch = blockIdx.z;
    const int half  = head_dim / 2;
    const int i     = threadIdx.x;
    if (i >= half) return;

    float eff_base = base_freq;
    if (scaling_type == 1 || scaling_type == 2) {
        float n = (float)head_dim;
        float ratio = n / (n - 2.0f);
        eff_base = base_freq * powf(scaling_factor * ratio, ratio);
    }

    float exp_val = -2.0f * (float)i / (float)head_dim;
    float inv_freq = powf(eff_base, exp_val);
    if (scaling_type == 0) inv_freq *= scaling_factor;

    float angle = (float)(pos + position_offset) * inv_freq;
    float cos_val, sin_val;
    __sincosf(angle, &sin_val, &cos_val);

    int per_seq = n_heads * seq_len * head_dim;
    int row = batch * per_seq + (head * seq_len + pos) * head_dim;
    int idx0, idx1;
    if (interleaved) {
        idx0 = row + i;
        idx1 = row + i + half;
    } else {
        idx0 = row + 2 * i;
        idx1 = row + 2 * i + 1;
    }
    float x0 = input[idx0];
    float x1 = input[idx1];
    float o0 = x0 * cos_val - x1 * sin_val;
    float o1 = x0 * sin_val + x1 * cos_val;
    if (scaling_type == 2) {
        o0 *= yarn_temp_inv;
        o1 *= yarn_temp_inv;
    }
    output[idx0] = o0;
    output[idx1] = o1;
}
"#;

// ───────────────────────────────────────────────────────────────────
// GPU launch helpers
// ───────────────────────────────────────────────────────────────────

/// Launch configuration for the RoPE v2 GPU kernel.
#[derive(Debug, Clone)]
pub struct RoPEV2LaunchConfig {
    /// Grid dimensions `(seq_len, n_heads, batch)`.
    pub grid: (u32, u32, u32),
    /// Block dimensions `(threads_per_block, 1, 1)`.
    pub block: (u32, u32, u32),
}

/// Compute a GPU launch configuration for the single-sequence kernel.
pub fn launch_rope_v2(cfg: &RoPEConfig) -> Result<RoPEV2LaunchConfig> {
    let tpb = (cfg.half_dim() as u32).min(1024);
    if tpb == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "head_dim must be >= 2 for RoPE v2".into(),
        }
        .into());
    }
    Ok(RoPEV2LaunchConfig {
        grid: (cfg.max_seq_len as u32, cfg.n_heads as u32, 1),
        block: (tpb, 1, 1),
    })
}

/// Compute a GPU launch configuration for the batched kernel.
pub fn launch_rope_v2_batched(cfg: &RoPEConfig, batch_size: usize) -> Result<RoPEV2LaunchConfig> {
    let tpb = (cfg.half_dim() as u32).min(1024);
    if tpb == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "head_dim must be >= 2 for batched RoPE v2".into(),
        }
        .into());
    }
    Ok(RoPEV2LaunchConfig {
        grid: (cfg.max_seq_len as u32, cfg.n_heads as u32, batch_size as u32),
        block: (tpb, 1, 1),
    })
}

// ───────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::too_many_lines)]
mod tests {
    use super::*;

    // ── Config construction ──────────────────────────────────────

    #[test]
    fn test_config_valid() {
        let cfg = RoPEConfig::new(64, 8, 128);
        assert!(cfg.is_ok());
        let cfg = cfg.unwrap();
        assert_eq!(cfg.half_dim(), 32);
    }

    #[test]
    fn test_config_zero_head_dim() {
        assert!(RoPEConfig::new(0, 8, 128).is_err());
    }

    #[test]
    fn test_config_odd_head_dim() {
        assert!(RoPEConfig::new(3, 8, 128).is_err());
    }

    #[test]
    fn test_config_zero_n_heads() {
        assert!(RoPEConfig::new(64, 0, 128).is_err());
    }

    #[test]
    fn test_config_zero_seq_len() {
        assert!(RoPEConfig::new(64, 8, 0).is_err());
    }

    #[test]
    fn test_config_builder_methods() {
        let cfg = RoPEConfig::new(64, 8, 128)
            .unwrap()
            .with_base_freq(5000.0)
            .with_scaling(ScalingType::NtkAware, 2.0)
            .with_interleaved(true)
            .with_position_offset(10)
            .with_max_seq_len(256);
        assert_eq!(cfg.base_freq, 5000.0);
        assert_eq!(cfg.scaling_type, ScalingType::NtkAware);
        assert_eq!(cfg.scaling_factor, 2.0);
        assert!(cfg.interleaved);
        assert_eq!(cfg.position_offset, 10);
        assert_eq!(cfg.max_seq_len, 256);
    }

    // ── ScalingType ──────────────────────────────────────────────

    #[test]
    fn test_scaling_type_display() {
        assert_eq!(ScalingType::Linear.to_string(), "linear");
        assert_eq!(ScalingType::NtkAware.to_string(), "ntk-aware");
        assert_eq!(ScalingType::Yarn.to_string(), "yarn");
    }

    #[test]
    fn test_scaling_type_eq() {
        assert_eq!(ScalingType::Linear, ScalingType::Linear);
        assert_ne!(ScalingType::Linear, ScalingType::NtkAware);
    }

    // ── Error type ───────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = RoPEV2Error::InvalidHeadDim(3);
        assert!(e.to_string().contains('3'));
        let e = RoPEV2Error::ShapeMismatch { expected: 10, got: 20 };
        assert!(e.to_string().contains("10"));
    }

    #[test]
    fn test_error_to_kernel_error() {
        let e = RoPEV2Error::InvalidConfig("test".into());
        let ke: KernelError = e.into();
        assert!(ke.to_string().contains("test"));
    }

    // ── Frequency table ──────────────────────────────────────────

    #[test]
    fn test_freq_table_shape() {
        let cfg = RoPEConfig::new(4, 1, 8).unwrap();
        let table = compute_rope_frequencies_v2(&cfg);
        assert_eq!(table.len(), 8 * 4); // max_seq_len * head_dim
    }

    #[test]
    fn test_freq_table_position_zero() {
        let cfg = RoPEConfig::new(4, 1, 4).unwrap().with_max_seq_len(4);
        let table = compute_rope_frequencies_v2(&cfg);
        // Position 0: angle = 0 → cos = 1, sin = 0.
        assert!((table[0] - 1.0).abs() < 1e-6, "cos(0) should be 1");
        assert!(table[1].abs() < 1e-6, "sin(0) should be 0");
        assert!((table[2] - 1.0).abs() < 1e-6, "cos(0) should be 1");
        assert!(table[3].abs() < 1e-6, "sin(0) should be 0");
    }

    #[test]
    fn test_freq_table_base_frequency_effect() {
        let cfg_low = RoPEConfig::new(4, 1, 4).unwrap().with_max_seq_len(4).with_base_freq(100.0);
        let cfg_high =
            RoPEConfig::new(4, 1, 4).unwrap().with_max_seq_len(4).with_base_freq(100_000.0);
        let t_low = compute_rope_frequencies_v2(&cfg_low);
        let t_high = compute_rope_frequencies_v2(&cfg_high);
        // inv_freq[0] = base^0 = 1.0 regardless of base, so dim pair 0
        // gives the same angle. Check dim pair 1 instead where
        // inv_freq[1] = base^(-0.5), so higher base → smaller angle.
        let hd = 4;
        let cos_low = t_low[hd + 2]; // pos=1, dim_pair=1 cos
        let cos_high = t_high[hd + 2];
        assert!(
            (cos_high - 1.0).abs() < (cos_low - 1.0).abs(),
            "higher base should produce less rotation at dim_pair=1: \
             cos_low={cos_low}, cos_high={cos_high}"
        );
    }

    #[test]
    fn test_freq_table_ntk_differs_from_linear() {
        let cfg_lin = RoPEConfig::new(8, 1, 4)
            .unwrap()
            .with_max_seq_len(4)
            .with_scaling(ScalingType::Linear, 2.0);
        let cfg_ntk = RoPEConfig::new(8, 1, 4)
            .unwrap()
            .with_max_seq_len(4)
            .with_scaling(ScalingType::NtkAware, 2.0);
        let t_lin = compute_rope_frequencies_v2(&cfg_lin);
        let t_ntk = compute_rope_frequencies_v2(&cfg_ntk);
        let diffs: f32 = t_lin.iter().zip(&t_ntk).map(|(a, b)| (a - b).abs()).sum();
        assert!(diffs > 1e-4, "NTK table should differ from linear");
    }

    #[test]
    fn test_freq_table_ntk_scaling_factor_effect() {
        let ntk1 = RoPEConfig::new(8, 1, 4)
            .unwrap()
            .with_max_seq_len(4)
            .with_scaling(ScalingType::NtkAware, 1.0);
        let ntk4 = RoPEConfig::new(8, 1, 4)
            .unwrap()
            .with_max_seq_len(4)
            .with_scaling(ScalingType::NtkAware, 4.0);
        let t1 = compute_rope_frequencies_v2(&ntk1);
        let t4 = compute_rope_frequencies_v2(&ntk4);
        // Check dim pair 1 (index 2 in each position row) where the NTK
        // base adjustment produces different angles.
        let hd = 8;
        let cos_s1 = t1[hd + 2]; // pos=1, dim_pair=1 cos
        let cos_s4 = t4[hd + 2];
        assert!(
            (cos_s4 - 1.0).abs() < (cos_s1 - 1.0).abs(),
            "higher NTK factor should produce less rotation at dim_pair=1: \
             cos_s1={cos_s1}, cos_s4={cos_s4}"
        );
    }

    #[test]
    fn test_freq_table_interleaved_same_values() {
        let cfg = RoPEConfig::new(4, 1, 4).unwrap().with_max_seq_len(4);
        let t1 = compute_rope_frequencies_v2(&cfg);
        let cfg2 = cfg.with_interleaved(true);
        let t2 = compute_rope_frequencies_v2(&cfg2);
        // Frequency table is layout-independent.
        assert_eq!(t1, t2);
    }

    // ── Standard RoPE ────────────────────────────────────────────

    #[test]
    fn test_apply_rope_identity_at_pos_zero() {
        let cfg = RoPEConfig::new(4, 1, 1).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        apply_rope_v2(&cfg, &input, &mut output).unwrap();
        // pos=0 → angle=0 → cos=1, sin=0 → identity.
        for (o, i) in output.iter().zip(&input) {
            assert!((o - i).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_rope_preserves_norm() {
        let cfg = RoPEConfig::new(4, 2, 4).unwrap();
        let n = 2 * 4 * 4;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0; n];
        apply_rope_v2(&cfg, &input, &mut output).unwrap();
        // RoPE is a rotation → norm of each pair is preserved.
        for h in 0..2 {
            for pos in 0..4 {
                for i in 0..2 {
                    let base = (h * 4 + pos) * 4;
                    let x0 = input[base + 2 * i];
                    let x1 = input[base + 2 * i + 1];
                    let y0 = output[base + 2 * i];
                    let y1 = output[base + 2 * i + 1];
                    let norm_in = (x0 * x0 + x1 * x1).sqrt();
                    let norm_out = (y0 * y0 + y1 * y1).sqrt();
                    assert!(
                        (norm_in - norm_out).abs() < 1e-4,
                        "norm should be preserved: {norm_in} vs {norm_out}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_apply_rope_shape_mismatch_input() {
        let cfg = RoPEConfig::new(4, 1, 4).unwrap();
        let input = vec![1.0; 8]; // need 16
        let mut output = vec![0.0; 16];
        assert!(apply_rope_v2(&cfg, &input, &mut output).is_err());
    }

    #[test]
    fn test_apply_rope_shape_mismatch_output() {
        let cfg = RoPEConfig::new(4, 1, 4).unwrap();
        let input = vec![1.0; 16];
        let mut output = vec![0.0; 8];
        assert!(apply_rope_v2(&cfg, &input, &mut output).is_err());
    }

    #[test]
    fn test_apply_rope_different_positions_differ() {
        let cfg = RoPEConfig::new(4, 1, 2).unwrap();
        let input = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let mut output = vec![0.0; 8];
        apply_rope_v2(&cfg, &input, &mut output).unwrap();
        // Same input at pos 0 and pos 1 should produce different outputs.
        assert!(
            (output[0] - output[4]).abs() > 1e-6 || (output[1] - output[5]).abs() > 1e-6,
            "different positions should produce different embeddings"
        );
    }

    #[test]
    fn test_apply_rope_position_offset() {
        let cfg_no_off = RoPEConfig::new(4, 1, 1).unwrap();
        let cfg_off = RoPEConfig::new(4, 1, 1).unwrap().with_position_offset(5);
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let mut out1 = vec![0.0; 4];
        let mut out2 = vec![0.0; 4];
        apply_rope_v2(&cfg_no_off, &input, &mut out1).unwrap();
        apply_rope_v2(&cfg_off, &input, &mut out2).unwrap();
        // Offset should change the output for dim_pair 1 (dim 0 has inv_freq=1).
        let diff: f32 = out1.iter().zip(&out2).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "position offset should affect output");
    }

    // ── NeoX layout ──────────────────────────────────────────────

    #[test]
    fn test_neox_identity_at_pos_zero() {
        let cfg = RoPEConfig::new(4, 1, 1).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        apply_rope_neox_v2(&cfg, &input, &mut output).unwrap();
        // pos=0 → identity.
        for (o, i) in output.iter().zip(&input) {
            assert!((o - i).abs() < 1e-6);
        }
    }

    #[test]
    fn test_neox_differs_from_standard() {
        let cfg = RoPEConfig::new(4, 1, 4).unwrap();
        let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.3).collect();
        let mut out_std = vec![0.0; 16];
        let mut out_neox = vec![0.0; 16];
        apply_rope_v2(&cfg, &input, &mut out_std).unwrap();
        apply_rope_neox_v2(&cfg, &input, &mut out_neox).unwrap();
        let diff: f32 = out_std.iter().zip(&out_neox).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "NeoX should differ from standard");
    }

    #[test]
    fn test_neox_preserves_norm() {
        let cfg = RoPEConfig::new(4, 1, 2).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 8];
        apply_rope_neox_v2(&cfg, &input, &mut output).unwrap();
        let half = 2;
        for pos in 0..2 {
            let row = pos * 4;
            for i in 0..half {
                let x0 = input[row + i];
                let x1 = input[row + i + half];
                let y0 = output[row + i];
                let y1 = output[row + i + half];
                let norm_in = (x0 * x0 + x1 * x1).sqrt();
                let norm_out = (y0 * y0 + y1 * y1).sqrt();
                assert!((norm_in - norm_out).abs() < 1e-4, "NeoX should preserve pair norm");
            }
        }
    }

    // ── YaRN ─────────────────────────────────────────────────────

    #[test]
    fn test_yarn_temperature_formula() {
        let t = yarn_attention_temp(1.0);
        // 0.1 * ln(1) + 1 = 0 + 1 = 1.
        assert!((t - 1.0).abs() < 1e-6);
        let t4 = yarn_attention_temp(4.0);
        assert!(t4 > 1.0, "factor > 1 → temp > 1");
    }

    #[test]
    fn test_yarn_at_factor_one_matches_standard() {
        // At scaling_factor=1.0 the YaRN temperature is 1.0 so the
        // output should match standard RoPE with Yarn base adjustment.
        let cfg = RoPEConfig::new(4, 1, 4).unwrap().with_scaling(ScalingType::Yarn, 1.0);
        let n = 4 * 4;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let mut out_std = vec![0.0; n];
        let mut out_yarn = vec![0.0; n];
        apply_rope_v2(&cfg, &input, &mut out_std).unwrap();
        apply_rope_yarn(&cfg, &input, &mut out_yarn).unwrap();
        // temp = 1.0 so they should match.
        for (a, b) in out_std.iter().zip(&out_yarn) {
            assert!((a - b).abs() < 1e-5, "yarn at factor=1 should match");
        }
    }

    #[test]
    fn test_yarn_scales_output() {
        let cfg = RoPEConfig::new(4, 1, 2).unwrap().with_scaling(ScalingType::Yarn, 4.0);
        let input = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let mut out_rope = vec![0.0; 8];
        let mut out_yarn = vec![0.0; 8];
        apply_rope_v2(&cfg, &input, &mut out_rope).unwrap();
        apply_rope_yarn(&cfg, &input, &mut out_yarn).unwrap();
        let temp = yarn_attention_temp(4.0);
        // YaRN divides by temp.
        for (y, r) in out_yarn.iter().zip(&out_rope) {
            assert!((y - r / temp).abs() < 1e-5, "yarn should scale by 1/temp");
        }
    }

    #[test]
    fn test_yarn_shape_mismatch() {
        let cfg = RoPEConfig::new(4, 1, 4).unwrap().with_scaling(ScalingType::Yarn, 2.0);
        let input = vec![1.0; 8]; // need 16
        let mut output = vec![0.0; 16];
        assert!(apply_rope_yarn(&cfg, &input, &mut output).is_err());
    }

    // ── Batched ──────────────────────────────────────────────────

    #[test]
    fn test_batched_single_matches_unbatched() {
        let cfg = RoPEConfig::new(4, 2, 4).unwrap();
        let n = 2 * 4 * 4;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let mut out1 = vec![0.0; n];
        let mut out2 = vec![0.0; n];
        apply_rope_v2(&cfg, &input, &mut out1).unwrap();
        apply_rope_batched(&cfg, 1, &input, &mut out2).unwrap();
        for (a, b) in out1.iter().zip(&out2) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_batched_two_sequences() {
        let cfg = RoPEConfig::new(4, 1, 2).unwrap();
        let per_seq = 1 * 2 * 4;
        let batch = 2;
        let input: Vec<f32> = (0..batch * per_seq).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0; batch * per_seq];
        apply_rope_batched(&cfg, batch, &input, &mut output).unwrap();

        // Each batch element should be independently rotated.
        let mut ref_out = vec![0.0; per_seq];
        for b in 0..batch {
            apply_rope_v2(&cfg, &input[b * per_seq..(b + 1) * per_seq], &mut ref_out).unwrap();
            for (a, b) in output[b * per_seq..(b + 1) * per_seq].iter().zip(&ref_out) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_batched_shape_mismatch() {
        let cfg = RoPEConfig::new(4, 1, 4).unwrap();
        let input = vec![1.0; 8]; // need 2 * 16
        let mut output = vec![0.0; 32];
        assert!(apply_rope_batched(&cfg, 2, &input, &mut output).is_err());
    }

    // ── Context extension ────────────────────────────────────────

    #[test]
    fn test_context_ext_within_training_length() {
        let base = RoPEConfig::new(8, 4, 32).unwrap();
        let ext = rope_context_extension(&base, 16, 2048);
        // 16 ≤ 2048 → should return original config unchanged.
        assert_eq!(ext.scaling_type, base.scaling_type);
        assert_eq!(ext.scaling_factor, base.scaling_factor);
        assert_eq!(ext.max_seq_len, base.max_seq_len);
    }

    #[test]
    fn test_context_ext_preserves_short_context_behavior() {
        let base = RoPEConfig::new(8, 4, 128).unwrap();
        let ext = rope_context_extension(&base, 4, 2048);
        // Within training context → config should be identical to base.
        let table_base = compute_rope_frequencies_v2(&base);
        let table_ext = compute_rope_frequencies_v2(&ext);
        assert_eq!(
            table_base, table_ext,
            "within training context, frequencies should be identical"
        );
    }

    #[test]
    fn test_context_ext_beyond_training_length() {
        let base = RoPEConfig::new(8, 4, 4096).unwrap();
        let ext = rope_context_extension(&base, 4096, 2048);
        assert_eq!(ext.scaling_type, ScalingType::NtkAware);
        assert!((ext.scaling_factor - 2.0).abs() < 1e-6);
        assert_eq!(ext.max_seq_len, 4096);
    }

    #[test]
    fn test_context_ext_factor_value() {
        let base = RoPEConfig::new(8, 4, 8192).unwrap();
        let ext = rope_context_extension(&base, 8192, 2048);
        assert!((ext.scaling_factor - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_context_ext_at_boundary() {
        let base = RoPEConfig::new(8, 4, 2048).unwrap();
        let ext = rope_context_extension(&base, 2048, 2048);
        // Exactly at boundary → no extension.
        assert_eq!(ext.scaling_type, base.scaling_type);
    }

    // ── NTK internal ─────────────────────────────────────────────

    #[test]
    fn test_ntk_adjusted_base_factor_one() {
        // Factor=1.0: ntk_adjusted_base should still differ from base
        // because the formula applies ratio scaling.
        let adj = ntk_adjusted_base(10_000.0, 1.0, 64);
        assert!(adj > 10_000.0, "NTK base should increase");
    }

    #[test]
    fn test_ntk_adjusted_base_increases_with_factor() {
        let b1 = ntk_adjusted_base(10_000.0, 1.0, 64);
        let b4 = ntk_adjusted_base(10_000.0, 4.0, 64);
        assert!(b4 > b1, "higher factor → higher NTK base");
    }

    #[test]
    fn test_ntk_adjusted_base_small_head_dim() {
        // head_dim=2 → n=2.0 → denominator is 0 → guard returns base.
        let adj = ntk_adjusted_base(10_000.0, 4.0, 2);
        assert_eq!(adj, 10_000.0);
    }

    // ── Launch configs ───────────────────────────────────────────

    #[test]
    fn test_launch_rope_v2() {
        let cfg = RoPEConfig::new(64, 8, 128).unwrap();
        let lc = launch_rope_v2(&cfg).unwrap();
        assert_eq!(lc.grid, (128, 8, 1));
        assert_eq!(lc.block, (32, 1, 1));
    }

    #[test]
    fn test_launch_rope_v2_batched() {
        let cfg = RoPEConfig::new(64, 8, 128).unwrap();
        let lc = launch_rope_v2_batched(&cfg, 4).unwrap();
        assert_eq!(lc.grid, (128, 8, 4));
        assert_eq!(lc.block, (32, 1, 1));
    }

    #[test]
    fn test_launch_large_head_dim_capped() {
        let cfg = RoPEConfig::new(4096, 1, 1).unwrap();
        let lc = launch_rope_v2(&cfg).unwrap();
        assert_eq!(lc.block.0, 1024, "threads should be capped at 1024");
    }

    // ── CUDA kernel source ───────────────────────────────────────

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn test_cuda_kernel_src_not_empty() {
        assert!(!ROPE_V2_FORWARD_KERNEL_SRC.is_empty());
        assert!(!ROPE_V2_BATCHED_KERNEL_SRC.is_empty());
    }

    // ── Position offset ──────────────────────────────────────────

    #[test]
    fn test_freq_table_with_offset() {
        let cfg = RoPEConfig::new(4, 1, 1).unwrap().with_max_seq_len(1).with_position_offset(10);
        let table = compute_rope_frequencies_v2(&cfg);
        // Position 0 with offset 10 → angle = 10 * inv_freq[0].
        let angle = 10.0_f32;
        assert!((table[0] - angle.cos()).abs() < 1e-6);
        assert!((table[1] - angle.sin()).abs() < 1e-6);
    }

    // ── Multi-head ───────────────────────────────────────────────

    #[test]
    fn test_multi_head_independent() {
        let cfg = RoPEConfig::new(4, 2, 2).unwrap();
        let n = 2 * 2 * 4;
        let mut input = vec![0.0_f32; n];
        // Put signal in head 0 only.
        for i in 0..8 {
            input[i] = (i as f32) * 0.5;
        }
        let mut output = vec![0.0; n];
        apply_rope_v2(&cfg, &input, &mut output).unwrap();
        // Head 1 input is all zeros → output is all zeros.
        for &v in &output[8..16] {
            assert!(v.abs() < 1e-6, "head 1 should remain zero");
        }
    }

    // ── Regression: cos/sin pair sanity ──────────────────────────

    #[test]
    fn test_cos_sin_pair_unit_circle() {
        let cfg = RoPEConfig::new(4, 1, 8).unwrap();
        let table = compute_rope_frequencies_v2(&cfg);
        for pos in 0..8 {
            for i in 0..2 {
                let c = table[pos * 4 + 2 * i];
                let s = table[pos * 4 + 2 * i + 1];
                let r = c * c + s * s;
                assert!(
                    (r - 1.0).abs() < 1e-5,
                    "cos²+sin² should be 1 at pos={pos}, dim={i}: got {r}"
                );
            }
        }
    }

    // ── Linear scaling ───────────────────────────────────────────

    #[test]
    fn test_linear_scaling_doubles_angle() {
        let cfg1 = RoPEConfig::new(4, 1, 2).unwrap().with_max_seq_len(2);
        let cfg2 = RoPEConfig::new(4, 1, 2)
            .unwrap()
            .with_max_seq_len(2)
            .with_scaling(ScalingType::Linear, 2.0);
        let t1 = compute_rope_frequencies_v2(&cfg1);
        let t2 = compute_rope_frequencies_v2(&cfg2);
        // At pos=1 with Linear scaling factor=2, angle should double.
        // For dim pair 0: inv_freq=1, angle₁=1*1=1, angle₂=1*2=2.
        let hd = 4;
        let c1 = t1[hd]; // cos(1)
        let s1 = t1[hd + 1]; // sin(1)
        let c2 = t2[hd]; // cos(2)
        let s2 = t2[hd + 1]; // sin(2)
        assert!((c2 - (2.0_f32).cos()).abs() < 1e-5);
        assert!((s2 - (2.0_f32).sin()).abs() < 1e-5);
        assert!((c1 - (1.0_f32).cos()).abs() < 1e-5);
        assert!((s1 - (1.0_f32).sin()).abs() < 1e-5);
    }

    // ── Determinism ──────────────────────────────────────────────

    #[test]
    fn test_deterministic_output() {
        let cfg = RoPEConfig::new(8, 4, 16).unwrap();
        let n = 4 * 16 * 8;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let mut out1 = vec![0.0; n];
        let mut out2 = vec![0.0; n];
        apply_rope_v2(&cfg, &input, &mut out1).unwrap();
        apply_rope_v2(&cfg, &input, &mut out2).unwrap();
        assert_eq!(out1, out2, "output should be deterministic");
    }

    // ── Edge: very large head_dim ────────────────────────────────

    #[test]
    fn test_large_head_dim() {
        let cfg = RoPEConfig::new(256, 1, 2).unwrap();
        let n = 1 * 2 * 256;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let mut output = vec![0.0; n];
        apply_rope_v2(&cfg, &input, &mut output).unwrap();
        // Just verify no panic and output differs from input at pos 1.
        let diff: f32 = input[256..].iter().zip(&output[256..]).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3);
    }

    // ── Edge: seq_len = 1 ────────────────────────────────────────

    #[test]
    fn test_seq_len_one_pos_offset() {
        let cfg = RoPEConfig::new(4, 1, 1).unwrap().with_position_offset(100);
        let input = vec![1.0, 0.0, 0.0, 1.0];
        let mut output = vec![0.0; 4];
        apply_rope_v2(&cfg, &input, &mut output).unwrap();
        // With offset=100, angle for dim 0 = 100 * 1.0 = 100.
        let angle = 100.0_f32;
        let (sin, cos) = angle.sin_cos();
        assert!((output[0] - (1.0 * cos)).abs() < 1e-4);
        assert!((output[1] - (1.0 * sin)).abs() < 1e-4);
    }

    // ── Additional coverage ──────────────────────────────────────

    #[test]
    fn test_apply_rope_v2_all_zeros_input() {
        let cfg = RoPEConfig::new(4, 2, 4).unwrap();
        let n = 2 * 4 * 4;
        let input = vec![0.0_f32; n];
        let mut output = vec![1.0; n];
        apply_rope_v2(&cfg, &input, &mut output).unwrap();
        // Rotation of zero vector is zero.
        for &v in &output {
            assert!(v.abs() < 1e-10, "rotation of zero should be zero");
        }
    }

    #[test]
    fn test_batched_zero_batch_ok() {
        let cfg = RoPEConfig::new(4, 1, 2).unwrap();
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        apply_rope_batched(&cfg, 0, &input, &mut output).unwrap();
    }

    #[test]
    fn test_neox_shape_mismatch() {
        let cfg = RoPEConfig::new(4, 1, 4).unwrap();
        let input = vec![1.0; 4]; // need 16
        let mut output = vec![0.0; 16];
        assert!(apply_rope_neox_v2(&cfg, &input, &mut output).is_err());
    }

    #[test]
    fn test_context_ext_large_ratio() {
        let base = RoPEConfig::new(64, 8, 16384).unwrap();
        let ext = rope_context_extension(&base, 16384, 1024);
        // factor = 16384/1024 = 16.
        assert!((ext.scaling_factor - 16.0).abs() < 1e-4);
    }

    #[test]
    fn test_inv_freq_dim_zero_always_one() {
        // inv_freq[0] = base^0 = 1.0 for any base.
        let cfg = RoPEConfig::new(8, 1, 2).unwrap().with_base_freq(42.0);
        let inv = build_inv_freq(&cfg);
        assert!((inv[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_inv_freq_decreasing() {
        let cfg = RoPEConfig::new(16, 1, 2).unwrap();
        let inv = build_inv_freq(&cfg);
        for i in 1..inv.len() {
            assert!(
                inv[i] < inv[i - 1],
                "inv_freq should decrease: [{}]={} >= [{}]={}",
                i,
                inv[i],
                i - 1,
                inv[i - 1]
            );
        }
    }

    #[test]
    fn test_yarn_with_linear_base_config() {
        // apply_rope_yarn forces ScalingType::Yarn internally.
        let cfg = RoPEConfig::new(4, 1, 2).unwrap().with_scaling(ScalingType::Linear, 2.0);
        let input = vec![1.0, 0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.8];
        let mut output = vec![0.0; 8];
        apply_rope_yarn(&cfg, &input, &mut output).unwrap();
        // Should succeed and produce finite values.
        for &v in &output {
            assert!(v.is_finite(), "yarn output should be finite");
        }
    }

    #[test]
    fn test_rope_err_conversion_chain() {
        let err = rope_err(RoPEV2Error::InvalidSeqLen(0));
        let msg = err.to_string();
        assert!(msg.contains("seq_len"), "should carry through: {msg}");
    }

    #[test]
    fn test_config_clone_independence() {
        let cfg1 = RoPEConfig::new(8, 4, 32).unwrap().with_base_freq(5000.0);
        let mut cfg2 = cfg1.clone();
        cfg2.base_freq = 20_000.0;
        assert_eq!(cfg1.base_freq, 5000.0, "clone should be independent");
    }

    #[test]
    fn test_error_clone() {
        let e1 = RoPEV2Error::InvalidHeadDim(7);
        let e2 = e1.clone();
        assert_eq!(e1.to_string(), e2.to_string());
    }

    #[test]
    fn test_launch_config_debug() {
        let cfg = RoPEConfig::new(64, 8, 128).unwrap();
        let lc = launch_rope_v2(&cfg).unwrap();
        let debug = format!("{lc:?}");
        assert!(debug.contains("grid"));
    }
}
