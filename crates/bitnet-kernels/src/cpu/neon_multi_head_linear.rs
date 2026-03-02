//! ARM NEON multi-head linear projection kernels for Apple Silicon.
//!
//! Implements Q/K/V projections for multi-head attention using NEON
//! intrinsics. Supports:
//!
//! - Individual Q, K, V linear projections with NEON-accelerated GEMM
//! - Fused QKV projection (single weight matrix, one pass)
//! - Head splitting / reshaping from `[batch, seq, model_dim]` to
//!   `[batch, num_heads, seq, head_dim]`
//! - f32 dense weights, i8 quantized weights, and packed i2 ternary
//!   weights
//!
//! All NEON functions require `target_arch = "aarch64"` and use
//! `#[target_feature(enable = "neon")]`.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

// ── Configuration ──────────────────────────────────────────────────────

/// Configuration for a multi-head linear projection layer.
#[derive(Debug, Clone)]
pub struct MultiHeadLinearConfig {
    /// Model / embedding dimension (input width).
    pub model_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head (`model_dim / num_heads`).
    pub head_dim: usize,
    /// Batch size (number of input rows / tokens).
    pub batch_size: usize,
}

impl MultiHeadLinearConfig {
    /// Create a new configuration.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any dimension is zero or `model_dim` is not
    /// evenly divisible by `num_heads`.
    pub fn new(
        model_dim: usize,
        num_heads: usize,
        batch_size: usize,
    ) -> Result<Self, MultiHeadLinearError> {
        if model_dim == 0 || num_heads == 0 || batch_size == 0 {
            return Err(MultiHeadLinearError::InvalidDimension(
                "all dimensions must be non-zero".into(),
            ));
        }
        if model_dim % num_heads != 0 {
            return Err(MultiHeadLinearError::InvalidDimension(format!(
                "model_dim ({model_dim}) must be divisible by \
                     num_heads ({num_heads})"
            )));
        }
        Ok(Self { model_dim, num_heads, head_dim: model_dim / num_heads, batch_size })
    }
}

// ── Error type ─────────────────────────────────────────────────────────

/// Errors specific to multi-head linear projections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiHeadLinearError {
    /// A dimension was invalid (zero or not divisible).
    InvalidDimension(String),
    /// A buffer was too small for the requested operation.
    BufferTooSmall { name: &'static str, expected: usize, actual: usize },
}

impl std::fmt::Display for MultiHeadLinearError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimension(msg) => {
                write!(f, "invalid dimension: {msg}")
            }
            Self::BufferTooSmall { name, expected, actual } => {
                write!(f, "{name} buffer too small: need {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for MultiHeadLinearError {}

// ── Buffer validation ──────────────────────────────────────────────────

fn validate_projection_buffers(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &[f32],
    cfg: &MultiHeadLinearConfig,
) -> Result<(), MultiHeadLinearError> {
    let in_required = cfg.batch_size * cfg.model_dim;
    let w_required = cfg.model_dim * cfg.model_dim;
    let out_required = cfg.batch_size * cfg.model_dim;

    if input.len() < in_required {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "input",
            expected: in_required,
            actual: input.len(),
        });
    }
    if weight.len() < w_required {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "weight",
            expected: w_required,
            actual: weight.len(),
        });
    }
    if output.len() < out_required {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "output",
            expected: out_required,
            actual: output.len(),
        });
    }
    if let Some(b) = bias {
        if b.len() < cfg.model_dim {
            return Err(MultiHeadLinearError::BufferTooSmall {
                name: "bias",
                expected: cfg.model_dim,
                actual: b.len(),
            });
        }
    }
    Ok(())
}

// ── Scalar reference (for testing & fallback) ──────────────────────────

/// Scalar linear projection: `output = input · Wᵀ + bias`.
///
/// `weight` is row-major `[out_features, in_features]`.
pub fn multi_head_linear_scalar(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    cfg: &MultiHeadLinearConfig,
) -> Result<(), MultiHeadLinearError> {
    validate_projection_buffers(input, weight, bias, output, cfg)?;

    let d = cfg.model_dim;
    for b in 0..cfg.batch_size {
        let in_off = b * d;
        let out_off = b * d;
        for j in 0..d {
            let mut acc = 0.0f32;
            let w_off = j * d;
            for k in 0..d {
                acc += input[in_off + k] * weight[w_off + k];
            }
            if let Some(bias) = bias {
                acc += bias[j];
            }
            output[out_off + j] = acc;
        }
    }
    Ok(())
}

// ── NEON f32 projection ────────────────────────────────────────────────

/// NEON-accelerated linear projection with f32 weights.
///
/// Computes `output[b, :] = input[b, :] · Wᵀ + bias` for each row in
/// the batch using NEON `vfmaq_f32` for the inner dot product.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_multi_head_linear_f32(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    cfg: &MultiHeadLinearConfig,
) -> Result<(), MultiHeadLinearError> {
    validate_projection_buffers(input, weight, bias, output, cfg)?;

    let d = cfg.model_dim;
    let chunks = d / LANES;
    let tail = d % LANES;

    for b in 0..cfg.batch_size {
        let in_off = b * d;
        let out_off = b * d;
        let in_ptr = unsafe { input.as_ptr().add(in_off) };

        for j in 0..d {
            let w_ptr = unsafe { weight.as_ptr().add(j * d) };
            let mut acc = vdupq_n_f32(0.0);

            for c in 0..chunks {
                let offset = c * LANES;
                unsafe {
                    let vi = vld1q_f32(in_ptr.add(offset));
                    let vw = vld1q_f32(w_ptr.add(offset));
                    acc = vfmaq_f32(acc, vi, vw);
                }
            }

            let mut sum = vaddvq_f32(acc);

            // Scalar tail
            let tail_start = chunks * LANES;
            for t in 0..tail {
                sum += input[in_off + tail_start + t] * weight[j * d + tail_start + t];
            }

            if let Some(bias) = bias {
                sum += bias[j];
            }
            output[out_off + j] = sum;
        }
    }
    Ok(())
}

// ── NEON i8 quantized projection ───────────────────────────────────────

/// NEON-accelerated linear projection with i8 quantized weights.
///
/// The weight matrix is stored as `i8` with a uniform `scale` factor:
/// `effective_weight = (weight_i8 as f32) * scale`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_multi_head_linear_i8(
    input: &[f32],
    weight_i8: &[i8],
    scale: f32,
    bias: Option<&[f32]>,
    output: &mut [f32],
    cfg: &MultiHeadLinearConfig,
) -> Result<(), MultiHeadLinearError> {
    let d = cfg.model_dim;
    let w_required = d * d;
    if weight_i8.len() < w_required {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "weight_i8",
            expected: w_required,
            actual: weight_i8.len(),
        });
    }
    // Validate remaining buffers via the f32 path (weight size is
    // irrelevant there, so pass a dummy).
    let dummy_w = vec![0.0f32; w_required];
    validate_projection_buffers(input, &dummy_w, bias, output, cfg)?;

    let chunks = d / LANES;
    let tail = d % LANES;
    let scale_v = vdupq_n_f32(scale);

    for b in 0..cfg.batch_size {
        let in_off = b * d;
        let out_off = b * d;
        let in_ptr = unsafe { input.as_ptr().add(in_off) };

        for j in 0..d {
            let w_row = j * d;
            let mut acc = vdupq_n_f32(0.0);

            for c in 0..chunks {
                let offset = c * LANES;
                // Widen 4 × i8 → i32 → f32, then scale
                let w0 = weight_i8[w_row + offset] as f32;
                let w1 = weight_i8[w_row + offset + 1] as f32;
                let w2 = weight_i8[w_row + offset + 2] as f32;
                let w3 = weight_i8[w_row + offset + 3] as f32;
                let arr = [w0, w1, w2, w3];
                unsafe {
                    let vw_raw = vld1q_f32(arr.as_ptr());
                    let vw = vmulq_f32(vw_raw, scale_v);
                    let vi = vld1q_f32(in_ptr.add(offset));
                    acc = vfmaq_f32(acc, vi, vw);
                }
            }

            let mut sum = vaddvq_f32(acc);

            let tail_start = chunks * LANES;
            for t in 0..tail {
                let w_val = weight_i8[w_row + tail_start + t] as f32 * scale;
                sum += input[in_off + tail_start + t] * w_val;
            }

            if let Some(bias) = bias {
                sum += bias[j];
            }
            output[out_off + j] = sum;
        }
    }
    Ok(())
}

// ── I2 packed (ternary) projection ─────────────────────────────────────

/// Decode a 2-bit I2_S code to its signed float value.
#[inline(always)]
fn decode_i2(bits: u8) -> f32 {
    match bits & 0x03 {
        0b01 => 1.0,
        0b11 => -1.0,
        _ => 0.0,
    }
}

/// NEON-accelerated linear projection with I2 packed ternary weights.
///
/// Each byte of `weight_i2` holds 4 ternary values (2 bits each,
/// LSB-first). `scale` converts from ternary to the effective range.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_multi_head_linear_i2(
    input: &[f32],
    weight_i2: &[u8],
    scale: f32,
    bias: Option<&[f32]>,
    output: &mut [f32],
    cfg: &MultiHeadLinearConfig,
) -> Result<(), MultiHeadLinearError> {
    let d = cfg.model_dim;
    let packed_d = d.div_ceil(4);
    let w_required = d * packed_d;
    if weight_i2.len() < w_required {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "weight_i2",
            expected: w_required,
            actual: weight_i2.len(),
        });
    }
    let in_required = cfg.batch_size * d;
    let out_required = cfg.batch_size * d;
    if input.len() < in_required {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "input",
            expected: in_required,
            actual: input.len(),
        });
    }
    if output.len() < out_required {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "output",
            expected: out_required,
            actual: output.len(),
        });
    }
    if let Some(b) = bias {
        if b.len() < d {
            return Err(MultiHeadLinearError::BufferTooSmall {
                name: "bias",
                expected: d,
                actual: b.len(),
            });
        }
    }

    let lut: [f32; 4] = [0.0, 1.0, 0.0, -1.0];
    let full_bytes = d / 4;
    let tail_vals = d % 4;

    for b in 0..cfg.batch_size {
        let in_off = b * d;
        let out_off = b * d;
        let in_ptr = unsafe { input.as_ptr().add(in_off) };

        for j in 0..d {
            let row_start = j * packed_d;
            let mut acc = vdupq_n_f32(0.0);

            for byte_idx in 0..full_bytes {
                let byte = weight_i2[row_start + byte_idx];
                let c0 = (byte & 0x03) as usize;
                let c1 = ((byte >> 2) & 0x03) as usize;
                let c2 = ((byte >> 4) & 0x03) as usize;
                let c3 = ((byte >> 6) & 0x03) as usize;
                let w_arr = [lut[c0], lut[c1], lut[c2], lut[c3]];
                unsafe {
                    let vw = vld1q_f32(w_arr.as_ptr());
                    let vi = vld1q_f32(in_ptr.add(byte_idx * 4));
                    acc = vfmaq_f32(acc, vi, vw);
                }
            }

            let mut sum = vaddvq_f32(acc);

            // Scalar tail for non-aligned remainder
            if tail_vals > 0 && full_bytes < packed_d {
                let byte = weight_i2[row_start + full_bytes];
                for t in 0..tail_vals {
                    let bits = (byte >> (t * 2)) & 0x03;
                    sum += decode_i2(bits) * input[in_off + full_bytes * 4 + t];
                }
            }

            sum *= scale;
            if let Some(bias) = bias {
                sum += bias[j];
            }
            output[out_off + j] = sum;
        }
    }
    Ok(())
}

// ── Fused QKV projection ───────────────────────────────────────────────

/// Fused QKV projection: computes Q, K, and V in a single pass over
/// the input.
///
/// `weight_qkv` is a concatenated `[3 * model_dim, model_dim]` matrix
/// stored row-major (Q rows, then K rows, then V rows). Optional
/// `bias_qkv` is `[3 * model_dim]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_qkv_projection(
    input: &[f32],
    weight_qkv: &[f32],
    bias_qkv: Option<&[f32]>,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
    cfg: &MultiHeadLinearConfig,
) -> Result<(), MultiHeadLinearError> {
    let d = cfg.model_dim;
    let in_required = cfg.batch_size * d;
    let qkv_w_required = 3 * d * d;
    let out_required = cfg.batch_size * d;

    if input.len() < in_required {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "input",
            expected: in_required,
            actual: input.len(),
        });
    }
    if weight_qkv.len() < qkv_w_required {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "weight_qkv",
            expected: qkv_w_required,
            actual: weight_qkv.len(),
        });
    }
    if q_out.len() < out_required {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "q_out",
            expected: out_required,
            actual: q_out.len(),
        });
    }
    if k_out.len() < out_required {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "k_out",
            expected: out_required,
            actual: k_out.len(),
        });
    }
    if v_out.len() < out_required {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "v_out",
            expected: out_required,
            actual: v_out.len(),
        });
    }
    if let Some(b) = bias_qkv {
        if b.len() < 3 * d {
            return Err(MultiHeadLinearError::BufferTooSmall {
                name: "bias_qkv",
                expected: 3 * d,
                actual: b.len(),
            });
        }
    }

    let chunks = d / LANES;
    let tail = d % LANES;

    // Weight sub-matrices: Q=[0..d), K=[d..2d), V=[2d..3d) rows.
    let wq = &weight_qkv[..d * d];
    let wk = &weight_qkv[d * d..2 * d * d];
    let wv = &weight_qkv[2 * d * d..3 * d * d];

    for b in 0..cfg.batch_size {
        let in_off = b * d;
        let out_off = b * d;
        let in_ptr = unsafe { input.as_ptr().add(in_off) };

        for j in 0..d {
            let wq_ptr = unsafe { wq.as_ptr().add(j * d) };
            let wk_ptr = unsafe { wk.as_ptr().add(j * d) };
            let wv_ptr = unsafe { wv.as_ptr().add(j * d) };

            let mut acc_q = vdupq_n_f32(0.0);
            let mut acc_k = vdupq_n_f32(0.0);
            let mut acc_v = vdupq_n_f32(0.0);

            for c in 0..chunks {
                let offset = c * LANES;
                unsafe {
                    let vi = vld1q_f32(in_ptr.add(offset));
                    let vwq = vld1q_f32(wq_ptr.add(offset));
                    let vwk = vld1q_f32(wk_ptr.add(offset));
                    let vwv = vld1q_f32(wv_ptr.add(offset));
                    acc_q = vfmaq_f32(acc_q, vi, vwq);
                    acc_k = vfmaq_f32(acc_k, vi, vwk);
                    acc_v = vfmaq_f32(acc_v, vi, vwv);
                }
            }

            let mut sq = vaddvq_f32(acc_q);
            let mut sk = vaddvq_f32(acc_k);
            let mut sv = vaddvq_f32(acc_v);

            let tail_start = chunks * LANES;
            for t in 0..tail {
                let iv = input[in_off + tail_start + t];
                sq += iv * wq[j * d + tail_start + t];
                sk += iv * wk[j * d + tail_start + t];
                sv += iv * wv[j * d + tail_start + t];
            }

            if let Some(bias) = bias_qkv {
                sq += bias[j];
                sk += bias[d + j];
                sv += bias[2 * d + j];
            }

            q_out[out_off + j] = sq;
            k_out[out_off + j] = sk;
            v_out[out_off + j] = sv;
        }
    }
    Ok(())
}

// ── Head splitting / reshaping ─────────────────────────────────────────

/// Reshape a flat projection `[batch, model_dim]` into multi-head
/// layout `[batch, num_heads, head_dim]` (logically
/// `[batch, num_heads, 1, head_dim]` for single-token inference).
///
/// This is a pure data rearrangement — no arithmetic — so it is safe
/// on all architectures (no NEON required).
pub fn split_heads(
    input: &[f32],
    output: &mut [f32],
    cfg: &MultiHeadLinearConfig,
) -> Result<(), MultiHeadLinearError> {
    let total = cfg.batch_size * cfg.model_dim;
    if input.len() < total {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "input",
            expected: total,
            actual: input.len(),
        });
    }
    if output.len() < total {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "output",
            expected: total,
            actual: output.len(),
        });
    }

    // input  layout: [batch, model_dim]
    // output layout: [batch, num_heads, head_dim]
    // Since model_dim == num_heads * head_dim and both are contiguous,
    // the memory layout is identical — a simple copy suffices.
    output[..total].copy_from_slice(&input[..total]);
    Ok(())
}

/// Merge heads back from `[batch, num_heads, head_dim]` to
/// `[batch, model_dim]`.
///
/// Inverse of [`split_heads`]. Like `split_heads`, this is a pure copy
/// because the layouts are equivalent in memory.
pub fn merge_heads(
    input: &[f32],
    output: &mut [f32],
    cfg: &MultiHeadLinearConfig,
) -> Result<(), MultiHeadLinearError> {
    split_heads(input, output, cfg)
}

/// Reshape a flat projection `[batch, seq_len, model_dim]` into
/// multi-head layout `[batch, num_heads, seq_len, head_dim]`.
///
/// Unlike [`split_heads`], this handles an explicit sequence-length
/// dimension and must transpose the `seq` and `head` axes.
pub fn split_heads_seq(
    input: &[f32],
    output: &mut [f32],
    cfg: &MultiHeadLinearConfig,
    seq_len: usize,
) -> Result<(), MultiHeadLinearError> {
    let total = cfg.batch_size * seq_len * cfg.model_dim;
    if input.len() < total {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "input",
            expected: total,
            actual: input.len(),
        });
    }
    if output.len() < total {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "output",
            expected: total,
            actual: output.len(),
        });
    }

    let nh = cfg.num_heads;
    let hd = cfg.head_dim;

    // input:  [batch, seq_len, num_heads * head_dim]
    // output: [batch, num_heads, seq_len, head_dim]
    for b in 0..cfg.batch_size {
        for s in 0..seq_len {
            for h in 0..nh {
                let src = b * seq_len * cfg.model_dim + s * cfg.model_dim + h * hd;
                let dst = b * nh * seq_len * hd + h * seq_len * hd + s * hd;
                output[dst..dst + hd].copy_from_slice(&input[src..src + hd]);
            }
        }
    }
    Ok(())
}

/// Merge heads with a sequence dimension: transpose
/// `[batch, num_heads, seq_len, head_dim]` →
/// `[batch, seq_len, model_dim]`.
pub fn merge_heads_seq(
    input: &[f32],
    output: &mut [f32],
    cfg: &MultiHeadLinearConfig,
    seq_len: usize,
) -> Result<(), MultiHeadLinearError> {
    let total = cfg.batch_size * seq_len * cfg.model_dim;
    if input.len() < total {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "input",
            expected: total,
            actual: input.len(),
        });
    }
    if output.len() < total {
        return Err(MultiHeadLinearError::BufferTooSmall {
            name: "output",
            expected: total,
            actual: output.len(),
        });
    }

    let nh = cfg.num_heads;
    let hd = cfg.head_dim;

    // input:  [batch, num_heads, seq_len, head_dim]
    // output: [batch, seq_len, num_heads * head_dim]
    for b in 0..cfg.batch_size {
        for s in 0..seq_len {
            for h in 0..nh {
                let src = b * nh * seq_len * hd + h * seq_len * hd + s * hd;
                let dst = b * seq_len * cfg.model_dim + s * cfg.model_dim + h * hd;
                output[dst..dst + hd].copy_from_slice(&input[src..src + hd]);
            }
        }
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────────

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at [{i}]: {x} vs {y} (tol {tol})");
        }
    }

    /// Build an identity-like weight matrix `[dim, dim]`.
    fn identity_matrix(dim: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; dim * dim];
        for i in 0..dim {
            w[i * dim + i] = 1.0;
        }
        w
    }

    // ── Config tests ──────────────────────────────────────────────────

    #[test]
    fn test_config_valid() {
        let cfg = MultiHeadLinearConfig::new(64, 8, 2).unwrap();
        assert_eq!(cfg.head_dim, 8);
    }

    #[test]
    fn test_config_zero_dim() {
        assert!(MultiHeadLinearConfig::new(0, 8, 1).is_err());
        assert!(MultiHeadLinearConfig::new(64, 0, 1).is_err());
        assert!(MultiHeadLinearConfig::new(64, 8, 0).is_err());
    }

    #[test]
    fn test_config_not_divisible() {
        assert!(MultiHeadLinearConfig::new(65, 8, 1).is_err());
    }

    // ── Scalar projection tests ───────────────────────────────────────

    #[test]
    fn test_scalar_identity() {
        let cfg = MultiHeadLinearConfig::new(4, 2, 1).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = identity_matrix(4);
        let mut output = vec![0.0f32; 4];
        multi_head_linear_scalar(&input, &weight, None, &mut output, &cfg).unwrap();
        assert_close(&output, &input, 1e-6);
    }

    #[test]
    fn test_scalar_with_bias() {
        let cfg = MultiHeadLinearConfig::new(4, 2, 1).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = identity_matrix(4);
        let bias = [0.5, 0.5, 0.5, 0.5];
        let mut output = vec![0.0f32; 4];
        multi_head_linear_scalar(&input, &weight, Some(&bias), &mut output, &cfg).unwrap();
        assert_close(&output, &[1.5, 2.5, 3.5, 4.5], 1e-6);
    }

    #[test]
    fn test_scalar_batch() {
        let cfg = MultiHeadLinearConfig::new(4, 2, 2).unwrap();
        let input = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let weight = identity_matrix(4);
        let mut output = vec![0.0f32; 8];
        multi_head_linear_scalar(&input, &weight, None, &mut output, &cfg).unwrap();
        assert_close(&output, &input, 1e-6);
    }

    #[test]
    fn test_scalar_buffer_too_small() {
        let cfg = MultiHeadLinearConfig::new(4, 2, 1).unwrap();
        let input = [1.0; 4];
        let weight = identity_matrix(4);
        let mut output = vec![0.0f32; 2]; // too small
        let result = multi_head_linear_scalar(&input, &weight, None, &mut output, &cfg);
        assert!(result.is_err());
    }

    // ── NEON f32 projection tests ─────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_f32_identity() {
        let cfg = MultiHeadLinearConfig::new(8, 2, 1).unwrap();
        let input: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let weight = identity_matrix(8);
        let mut output = vec![0.0f32; 8];
        unsafe {
            neon_multi_head_linear_f32(&input, &weight, None, &mut output, &cfg).unwrap();
        }
        assert_close(&output, &input, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_f32_with_bias() {
        let cfg = MultiHeadLinearConfig::new(4, 2, 1).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = identity_matrix(4);
        let bias = [10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0f32; 4];
        unsafe {
            neon_multi_head_linear_f32(&input, &weight, Some(&bias), &mut output, &cfg).unwrap();
        }
        assert_close(&output, &[11.0, 22.0, 33.0, 44.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_f32_parity_with_scalar() {
        let cfg = MultiHeadLinearConfig::new(16, 4, 2).unwrap();
        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let weight: Vec<f32> = (0..256).map(|i| ((i as f64) * 0.03).sin() as f32).collect();
        let bias: Vec<f32> = (0..16).map(|i| i as f32 * 0.01).collect();
        let mut neon_out = vec![0.0f32; 32];
        let mut scalar_out = vec![0.0f32; 32];

        multi_head_linear_scalar(&input, &weight, Some(&bias), &mut scalar_out, &cfg).unwrap();
        unsafe {
            neon_multi_head_linear_f32(&input, &weight, Some(&bias), &mut neon_out, &cfg).unwrap();
        }
        assert_close(&neon_out, &scalar_out, 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_f32_non_aligned_dim() {
        // model_dim=6, not a multiple of LANES (4)
        let cfg = MultiHeadLinearConfig::new(6, 3, 1).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = identity_matrix(6);
        let mut output = vec![0.0f32; 6];
        unsafe {
            neon_multi_head_linear_f32(&input, &weight, None, &mut output, &cfg).unwrap();
        }
        assert_close(&output, &input, 1e-5);
    }

    // ── NEON i8 projection tests ──────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_i8_identity() {
        let cfg = MultiHeadLinearConfig::new(4, 2, 1).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        // Identity in i8 with scale = 1.0
        let mut weight_i8 = vec![0i8; 16];
        for i in 0..4 {
            weight_i8[i * 4 + i] = 1;
        }
        let mut output = vec![0.0f32; 4];
        unsafe {
            neon_multi_head_linear_i8(&input, &weight_i8, 1.0, None, &mut output, &cfg).unwrap();
        }
        assert_close(&output, &input, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_i8_scale() {
        let cfg = MultiHeadLinearConfig::new(4, 2, 1).unwrap();
        let input = [1.0; 4];
        let weight_i8 = vec![1i8; 16]; // all ones
        let mut output = vec![0.0f32; 4];
        unsafe {
            neon_multi_head_linear_i8(&input, &weight_i8, 0.5, None, &mut output, &cfg).unwrap();
        }
        // Each row: sum(1.0 * 1 * 0.5) * 4 = 2.0
        assert_close(&output, &[2.0, 2.0, 2.0, 2.0], 1e-5);
    }

    // ── NEON i2 (ternary) projection tests ────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_i2_identity_like() {
        let cfg = MultiHeadLinearConfig::new(4, 2, 1).unwrap();
        let input = [3.0, 7.0, 0.0, 0.0];
        // Pack a 4×4 identity in I2_S: each row has one +1 code
        let packed_d = 1usize; // 4 values / 4 per byte
        let mut weight_i2 = vec![0u8; 4 * packed_d];
        // Row 0: col 0 = +1 → bits 01 at position 0
        weight_i2[0 * packed_d] = 0b00_00_00_01;
        // Row 1: col 1 = +1 → bits 01 at position 2..3
        weight_i2[1 * packed_d] = 0b00_00_01_00;
        // Row 2: col 2 = +1 → bits 01 at position 4..5
        weight_i2[2 * packed_d] = 0b00_01_00_00;
        // Row 3: col 3 = +1 → bits 01 at position 6..7
        weight_i2[3 * packed_d] = 0b01_00_00_00;

        let mut output = vec![0.0f32; 4];
        unsafe {
            neon_multi_head_linear_i2(&input, &weight_i2, 1.0, None, &mut output, &cfg).unwrap();
        }
        assert_close(&output, &[3.0, 7.0, 0.0, 0.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_i2_all_ones() {
        let cfg = MultiHeadLinearConfig::new(4, 2, 1).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        // All +1: 0b01_01_01_01 per byte
        let weight_i2 = vec![0b01_01_01_01u8; 4];
        let mut output = vec![0.0f32; 4];
        unsafe {
            neon_multi_head_linear_i2(&input, &weight_i2, 1.0, None, &mut output, &cfg).unwrap();
        }
        // Each row: 1+2+3+4 = 10
        assert_close(&output, &[10.0, 10.0, 10.0, 10.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_i2_with_scale() {
        let cfg = MultiHeadLinearConfig::new(4, 2, 1).unwrap();
        let input = [1.0; 4];
        let weight_i2 = vec![0b01_01_01_01u8; 4]; // all +1
        let mut output = vec![0.0f32; 4];
        unsafe {
            neon_multi_head_linear_i2(&input, &weight_i2, 2.5, None, &mut output, &cfg).unwrap();
        }
        // Each row: 4 * 1.0 * 2.5 = 10.0
        assert_close(&output, &[10.0, 10.0, 10.0, 10.0], 1e-5);
    }

    // ── Fused QKV tests ───────────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_fused_qkv_identity() {
        let cfg = MultiHeadLinearConfig::new(4, 2, 1).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        let id = identity_matrix(4);
        // Concatenate 3 identity matrices
        let mut weight_qkv = Vec::with_capacity(3 * 16);
        weight_qkv.extend_from_slice(&id);
        weight_qkv.extend_from_slice(&id);
        weight_qkv.extend_from_slice(&id);

        let mut q = vec![0.0f32; 4];
        let mut k = vec![0.0f32; 4];
        let mut v = vec![0.0f32; 4];
        unsafe {
            neon_fused_qkv_projection(&input, &weight_qkv, None, &mut q, &mut k, &mut v, &cfg)
                .unwrap();
        }
        assert_close(&q, &input, 1e-5);
        assert_close(&k, &input, 1e-5);
        assert_close(&v, &input, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_fused_qkv_with_bias() {
        let cfg = MultiHeadLinearConfig::new(4, 2, 1).unwrap();
        let input = [1.0, 0.0, 0.0, 0.0];
        let id = identity_matrix(4);
        let mut weight_qkv = Vec::with_capacity(48);
        weight_qkv.extend_from_slice(&id);
        weight_qkv.extend_from_slice(&id);
        weight_qkv.extend_from_slice(&id);
        // bias: Q gets +1, K gets +2, V gets +3
        let mut bias_qkv = vec![0.0f32; 12];
        for i in 0..4 {
            bias_qkv[i] = 1.0;
            bias_qkv[4 + i] = 2.0;
            bias_qkv[8 + i] = 3.0;
        }

        let mut q = vec![0.0f32; 4];
        let mut k = vec![0.0f32; 4];
        let mut v = vec![0.0f32; 4];
        unsafe {
            neon_fused_qkv_projection(
                &input,
                &weight_qkv,
                Some(&bias_qkv),
                &mut q,
                &mut k,
                &mut v,
                &cfg,
            )
            .unwrap();
        }
        assert_close(&q, &[2.0, 1.0, 1.0, 1.0], 1e-5);
        assert_close(&k, &[3.0, 2.0, 2.0, 2.0], 1e-5);
        assert_close(&v, &[4.0, 3.0, 3.0, 3.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_fused_qkv_parity_with_separate() {
        let cfg = MultiHeadLinearConfig::new(8, 4, 2).unwrap();
        let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.2 - 1.0).collect();
        let weight: Vec<f32> = (0..192).map(|i| ((i as f64) * 0.05).sin() as f32).collect();
        let bias: Vec<f32> = (0..24).map(|i| i as f32 * 0.01).collect();

        // Separate projections
        let wq = &weight[..64];
        let wk = &weight[64..128];
        let wv = &weight[128..192];
        let bq = &bias[..8];
        let bk = &bias[8..16];
        let bv = &bias[16..24];

        let mut sq = vec![0.0f32; 16];
        let mut sk = vec![0.0f32; 16];
        let mut sv = vec![0.0f32; 16];
        multi_head_linear_scalar(&input, wq, Some(bq), &mut sq, &cfg).unwrap();
        multi_head_linear_scalar(&input, wk, Some(bk), &mut sk, &cfg).unwrap();
        multi_head_linear_scalar(&input, wv, Some(bv), &mut sv, &cfg).unwrap();

        // Fused projection
        let mut fq = vec![0.0f32; 16];
        let mut fk = vec![0.0f32; 16];
        let mut fv = vec![0.0f32; 16];
        unsafe {
            neon_fused_qkv_projection(
                &input,
                &weight,
                Some(&bias),
                &mut fq,
                &mut fk,
                &mut fv,
                &cfg,
            )
            .unwrap();
        }

        assert_close(&fq, &sq, 1e-4);
        assert_close(&fk, &sk, 1e-4);
        assert_close(&fv, &sv, 1e-4);
    }

    // ── Head split / merge tests ──────────────────────────────────────

    #[test]
    fn test_split_heads_roundtrip() {
        let cfg = MultiHeadLinearConfig::new(8, 4, 2).unwrap();
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut split = vec![0.0f32; 16];
        let mut merged = vec![0.0f32; 16];
        split_heads(&input, &mut split, &cfg).unwrap();
        merge_heads(&split, &mut merged, &cfg).unwrap();
        assert_close(&merged, &input, 0.0);
    }

    #[test]
    fn test_split_heads_seq_roundtrip() {
        let cfg = MultiHeadLinearConfig::new(8, 4, 1).unwrap();
        let seq_len = 3;
        let total = 1 * seq_len * 8;
        let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut split = vec![0.0f32; total];
        let mut merged = vec![0.0f32; total];
        split_heads_seq(&input, &mut split, &cfg, seq_len).unwrap();
        merge_heads_seq(&split, &mut merged, &cfg, seq_len).unwrap();
        assert_close(&merged, &input, 0.0);
    }

    #[test]
    fn test_split_heads_seq_layout() {
        // model_dim=4, num_heads=2, head_dim=2, batch=1, seq=2
        let cfg = MultiHeadLinearConfig::new(4, 2, 1).unwrap();
        // input [b=1, s=2, d=4]:
        //   s=0: [0, 1, 2, 3]  → head0=[0,1], head1=[2,3]
        //   s=1: [4, 5, 6, 7]  → head0=[4,5], head1=[6,7]
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 8];
        split_heads_seq(&input, &mut out, &cfg, 2).unwrap();
        // output [b=1, h=2, s=2, hd=2]:
        //   h=0: s=0:[0,1], s=1:[4,5]
        //   h=1: s=0:[2,3], s=1:[6,7]
        let expected = [0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0];
        assert_close(&out, &expected, 0.0);
    }

    #[test]
    fn test_split_heads_buffer_too_small() {
        let cfg = MultiHeadLinearConfig::new(4, 2, 1).unwrap();
        let input = [1.0; 4];
        let mut output = [0.0f32; 2]; // too small
        assert!(split_heads(&input, &mut output, &cfg).is_err());
    }

    // ── Ignored tests (resource-gated / slow) ─────────────────────────

    #[test]
    #[ignore = "benchmark: large model_dim projection timing"]
    fn test_large_projection_perf() {
        let cfg = MultiHeadLinearConfig::new(4096, 32, 1).unwrap();
        let input = vec![1.0f32; 4096];
        let weight = vec![0.01f32; 4096 * 4096];
        let mut output = vec![0.0f32; 4096];
        multi_head_linear_scalar(&input, &weight, None, &mut output, &cfg).unwrap();
        // Smoke check: at least one non-zero
        assert!(output.iter().any(|&v| v != 0.0));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "benchmark: NEON fused QKV for 2048-dim, \
                run manually for perf regression"]
    fn test_neon_fused_qkv_large() {
        let d = 2048;
        let cfg = MultiHeadLinearConfig::new(d, 32, 4).unwrap();
        let input = vec![0.01f32; 4 * d];
        let weight = vec![0.001f32; 3 * d * d];
        let mut q = vec![0.0f32; 4 * d];
        let mut k = vec![0.0f32; 4 * d];
        let mut v = vec![0.0f32; 4 * d];
        unsafe {
            neon_fused_qkv_projection(&input, &weight, None, &mut q, &mut k, &mut v, &cfg).unwrap();
        }
        assert!(q.iter().any(|&v| v != 0.0));
    }
}
