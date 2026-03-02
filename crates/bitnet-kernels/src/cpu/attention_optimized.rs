//! Optimized attention kernels for CPU inference.
//!
//! Provides memory-efficient (flash) attention, grouped-query attention (GQA),
//! multi-query attention (MQA), sliding-window attention, ALiBi positional
//! bias, and prefix-cache–aware attention.  All kernels operate on flat `f32`
//! slices laid out as `[num_heads, seq_len, head_dim]` (row-major).

use std::fmt;

// ── Error type ─────────────────────────────────────────────────────

/// Errors specific to the optimized attention kernels.
#[derive(Debug, Clone, PartialEq)]
pub enum FlashAttentionError {
    /// A dimensional or configuration argument is invalid.
    InvalidDimension(String),
    /// An input slice has an unexpected length.
    ShapeMismatch { expected: usize, actual: usize, context: String },
    /// The block size is invalid for the given sequence length.
    InvalidBlockSize(String),
    /// Head configuration is inconsistent (e.g. GQA divisibility).
    HeadConfigError(String),
    /// Window size must be positive.
    InvalidWindowSize(String),
}

impl fmt::Display for FlashAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension(msg) => write!(f, "invalid dimension: {msg}"),
            Self::ShapeMismatch { expected, actual, context } => {
                write!(f, "shape mismatch in {context}: expected {expected}, got {actual}")
            }
            Self::InvalidBlockSize(msg) => write!(f, "invalid block size: {msg}"),
            Self::HeadConfigError(msg) => write!(f, "head config error: {msg}"),
            Self::InvalidWindowSize(msg) => write!(f, "invalid window size: {msg}"),
        }
    }
}

impl std::error::Error for FlashAttentionError {}

/// Convenience alias used throughout this module.
type Result<T> = std::result::Result<T, FlashAttentionError>;

// ── Configuration ──────────────────────────────────────────────────

/// Configuration for the optimized attention kernels.
#[derive(Debug, Clone)]
pub struct OptimizedAttentionConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Per-head dimensionality.
    pub head_dim: usize,
    /// Maximum sequence length the kernel is prepared for.
    pub max_seq_len: usize,
    /// Apply causal (autoregressive) masking.
    pub causal: bool,
    /// Dropout probability (applied in training; ignored at inference).
    pub dropout_p: f32,
}

impl OptimizedAttentionConfig {
    /// Scaling factor `1 / √head_dim`.
    #[inline]
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.num_heads == 0 {
            return Err(FlashAttentionError::InvalidDimension("num_heads must be > 0".into()));
        }
        if self.head_dim == 0 {
            return Err(FlashAttentionError::InvalidDimension("head_dim must be > 0".into()));
        }
        if self.max_seq_len == 0 {
            return Err(FlashAttentionError::InvalidDimension("max_seq_len must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&self.dropout_p) {
            return Err(FlashAttentionError::InvalidDimension(
                "dropout_p must be in [0, 1]".into(),
            ));
        }
        Ok(())
    }
}

// ── Helpers ────────────────────────────────────────────────────────

/// Check that a flat slice has exactly `expected` elements.
#[inline]
fn check_len(slice: &[f32], expected: usize, name: &str) -> Result<()> {
    if slice.len() != expected {
        return Err(FlashAttentionError::ShapeMismatch {
            expected,
            actual: slice.len(),
            context: name.into(),
        });
    }
    Ok(())
}

/// Numerically-stable softmax in-place over `row[0..len]`.
#[inline]
fn softmax_inplace(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ── Flash Attention (tiled / O(N) memory) ──────────────────────────

/// Compute attention using a tiled (flash) algorithm that never
/// materialises the full `seq_len × seq_len` score matrix.
///
/// # Layout
/// *  `q`, `k`, `v` – `[num_heads, seq_len, head_dim]` (row-major)
///
/// # Algorithm
/// For each head, Q is split into blocks of `block_size` rows.  For each
/// Q-block the kernel iterates over all K/V-blocks, computing partial
/// scores and accumulating the weighted V output with an *online softmax*
/// (log-sum-exp running correction).  Peak additional memory is
/// `O(block_size × seq_len)` instead of `O(seq_len²)`.
pub fn flash_attention_forward(
    config: &OptimizedAttentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    block_size: usize,
) -> Result<Vec<f32>> {
    config.validate()?;
    if block_size == 0 {
        return Err(FlashAttentionError::InvalidBlockSize("block_size must be > 0".into()));
    }
    if seq_len > config.max_seq_len {
        return Err(FlashAttentionError::InvalidDimension(format!(
            "seq_len {seq_len} exceeds max_seq_len {}",
            config.max_seq_len
        )));
    }
    let nh = config.num_heads;
    let d = config.head_dim;
    let total = nh * seq_len * d;
    check_len(q, total, "Q")?;
    check_len(k, total, "K")?;
    check_len(v, total, "V")?;

    let scale = config.scale();
    let mut output = vec![0.0_f32; total];

    for h in 0..nh {
        let head_off = h * seq_len * d;

        // Process Q in blocks of `block_size` rows.
        let mut qi = 0;
        while qi < seq_len {
            let q_end = (qi + block_size).min(seq_len);
            let q_block_len = q_end - qi;

            // Per-row running online-softmax state.
            let mut row_max = vec![f32::NEG_INFINITY; q_block_len];
            let mut row_sum = vec![0.0_f32; q_block_len];
            // Accumulated output for this Q-block (q_block_len × d).
            let mut acc = vec![0.0_f32; q_block_len * d];

            // Iterate over K/V blocks.
            let mut kj = 0;
            while kj < seq_len {
                let k_end = (kj + block_size).min(seq_len);
                let kv_block_len = k_end - kj;

                for bi in 0..q_block_len {
                    let q_row = qi + bi;
                    let q_start = head_off + q_row * d;
                    let q_slice = &q[q_start..q_start + d];

                    for bj in 0..kv_block_len {
                        let k_col = kj + bj;
                        // Causal: skip future positions.
                        if config.causal && k_col > q_row {
                            continue;
                        }

                        let k_start = head_off + k_col * d;
                        let score = dot(q_slice, &k[k_start..k_start + d]) * scale;

                        // Online softmax update.
                        let prev_max = row_max[bi];
                        let new_max = prev_max.max(score);
                        let correction = (prev_max - new_max).exp();

                        // Rescale running sum & accumulator.
                        row_sum[bi] = row_sum[bi] * correction + (score - new_max).exp();

                        let acc_row = &mut acc[bi * d..(bi + 1) * d];
                        for a in acc_row.iter_mut() {
                            *a *= correction;
                        }

                        // Add weighted V.
                        let v_start = head_off + k_col * d;
                        let w = (score - new_max).exp();
                        for (a, &vv) in acc_row.iter_mut().zip(&v[v_start..v_start + d]) {
                            *a += w * vv;
                        }

                        row_max[bi] = new_max;
                    }
                }
                kj = k_end;
            }

            // Normalise accumulated output by row_sum.
            for bi in 0..q_block_len {
                let inv = if row_sum[bi] > 0.0 { 1.0 / row_sum[bi] } else { 0.0 };
                let out_start = head_off + (qi + bi) * d;
                let acc_row = &acc[bi * d..(bi + 1) * d];
                for (o, &a) in output[out_start..out_start + d].iter_mut().zip(acc_row) {
                    *o = a * inv;
                }
            }

            qi = q_end;
        }
    }

    Ok(output)
}

// ── Grouped-Query Attention (GQA) ──────────────────────────────────

/// Grouped-query attention: `num_kv_heads` key/value heads are shared
/// among `num_heads` query heads.  `num_heads` must be a multiple of
/// `num_kv_heads`.
///
/// # Layout
/// * `q` – `[num_heads, seq_len, head_dim]`
/// * `k`, `v` – `[num_kv_heads, seq_len, head_dim]`
pub fn group_query_attention(
    config: &OptimizedAttentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_kv_heads: usize,
) -> Result<Vec<f32>> {
    config.validate()?;
    let nh = config.num_heads;
    let d = config.head_dim;

    if num_kv_heads == 0 {
        return Err(FlashAttentionError::HeadConfigError("num_kv_heads must be > 0".into()));
    }
    if !nh.is_multiple_of(num_kv_heads) {
        return Err(FlashAttentionError::HeadConfigError(format!(
            "num_heads ({nh}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }

    let q_total = nh * seq_len * d;
    let kv_total = num_kv_heads * seq_len * d;
    check_len(q, q_total, "Q")?;
    check_len(k, kv_total, "K")?;
    check_len(v, kv_total, "V")?;

    let scale = config.scale();
    let group_size = nh / num_kv_heads;
    let mut output = vec![0.0_f32; q_total];

    for h in 0..nh {
        let kv_h = h / group_size;
        let q_head_off = h * seq_len * d;
        let kv_head_off = kv_h * seq_len * d;

        for qi in 0..seq_len {
            let q_start = q_head_off + qi * d;
            let q_slice = &q[q_start..q_start + d];

            // Compute scores for this query row against all valid keys.
            let kv_len = if config.causal { qi + 1 } else { seq_len };
            let mut scores = Vec::with_capacity(kv_len);
            for kj in 0..kv_len {
                let k_start = kv_head_off + kj * d;
                scores.push(dot(q_slice, &k[k_start..k_start + d]) * scale);
            }
            softmax_inplace(&mut scores);

            // Weighted sum of V.
            let out_start = q_head_off + qi * d;
            for (kj, &w) in scores.iter().enumerate() {
                let v_start = kv_head_off + kj * d;
                for dd in 0..d {
                    output[out_start + dd] += w * v[v_start + dd];
                }
            }
        }
    }

    Ok(output)
}

// ── Multi-Query Attention (MQA) ────────────────────────────────────

/// Multi-query attention: a single shared K/V head across all query heads.
///
/// # Layout
/// * `q` – `[num_heads, seq_len, head_dim]`
/// * `k`, `v` – `[1, seq_len, head_dim]`  (i.e. `[seq_len, head_dim]`)
pub fn multi_query_attention(
    config: &OptimizedAttentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
) -> Result<Vec<f32>> {
    group_query_attention(config, q, k, v, seq_len, 1)
}

// ── Sliding-Window Attention ───────────────────────────────────────

/// Sliding-window attention: each query only attends to the last
/// `window_size` key positions (plus itself).  Positions outside the
/// window receive `-∞` before softmax.
///
/// # Layout
/// * `q`, `k`, `v` – `[num_heads, seq_len, head_dim]`
pub fn sliding_window_attention(
    config: &OptimizedAttentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    window_size: usize,
) -> Result<Vec<f32>> {
    config.validate()?;
    if window_size == 0 {
        return Err(FlashAttentionError::InvalidWindowSize("window_size must be > 0".into()));
    }

    let nh = config.num_heads;
    let d = config.head_dim;
    let total = nh * seq_len * d;
    check_len(q, total, "Q")?;
    check_len(k, total, "K")?;
    check_len(v, total, "V")?;

    let scale = config.scale();
    let mut output = vec![0.0_f32; total];

    for h in 0..nh {
        let head_off = h * seq_len * d;

        for qi in 0..seq_len {
            let q_start = head_off + qi * d;
            let q_slice = &q[q_start..q_start + d];

            // Window bounds: attend to [win_start, win_end).
            let win_start = qi.saturating_sub(window_size.saturating_sub(1));
            let win_end = if config.causal { qi + 1 } else { (qi + window_size).min(seq_len) };

            let span = win_end - win_start;
            let mut scores = Vec::with_capacity(span);
            for kj in win_start..win_end {
                let k_start = head_off + kj * d;
                scores.push(dot(q_slice, &k[k_start..k_start + d]) * scale);
            }
            softmax_inplace(&mut scores);

            let out_start = head_off + qi * d;
            for (idx, &w) in scores.iter().enumerate() {
                let kj = win_start + idx;
                let v_start = head_off + kj * d;
                for dd in 0..d {
                    output[out_start + dd] += w * v[v_start + dd];
                }
            }
        }
    }

    Ok(output)
}

// ── ALiBi (Attention with Linear Biases) ───────────────────────────

/// Compute per-head ALiBi slopes.
///
/// For `n` heads the slopes are `2^{-8k/n}` for `k = 1..n`.
fn alibi_slopes(num_heads: usize) -> Vec<f32> {
    (1..=num_heads).map(|k| 2.0_f32.powf(-8.0 * k as f32 / num_heads as f32)).collect()
}

/// Scaled dot-product attention with ALiBi positional bias.
///
/// Instead of additive positional encodings, ALiBi adds a
/// head-specific linear bias `slope * |qi − kj|` to the pre-softmax
/// scores.
///
/// # Layout
/// * `q`, `k`, `v` – `[num_heads, seq_len, head_dim]`
pub fn attention_with_alibi(
    config: &OptimizedAttentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
) -> Result<Vec<f32>> {
    config.validate()?;
    let nh = config.num_heads;
    let d = config.head_dim;
    let total = nh * seq_len * d;
    check_len(q, total, "Q")?;
    check_len(k, total, "K")?;
    check_len(v, total, "V")?;

    let scale = config.scale();
    let slopes = alibi_slopes(nh);
    let mut output = vec![0.0_f32; total];

    for (h, &m) in slopes.iter().enumerate() {
        let head_off = h * seq_len * d;

        for qi in 0..seq_len {
            let q_start = head_off + qi * d;
            let q_slice = &q[q_start..q_start + d];

            let kv_len = if config.causal { qi + 1 } else { seq_len };
            let mut scores = Vec::with_capacity(kv_len);
            for kj in 0..kv_len {
                let k_start = head_off + kj * d;
                let s = dot(q_slice, &k[k_start..k_start + d]) * scale;
                // ALiBi bias: negative proportional to distance.
                let bias = -m * (qi as f32 - kj as f32).abs();
                scores.push(s + bias);
            }
            softmax_inplace(&mut scores);

            let out_start = head_off + qi * d;
            for (kj, &w) in scores.iter().enumerate() {
                let v_start = head_off + kj * d;
                for dd in 0..d {
                    output[out_start + dd] += w * v[v_start + dd];
                }
            }
        }
    }

    Ok(output)
}

// ── Prefix-Cache Attention ─────────────────────────────────────────

/// Attention with prefix caching.
///
/// Reuses a pre-computed K/V prefix (`prefix_k`, `prefix_v` of length
/// `prefix_len`) and only computes fresh scores for the new portion.
///
/// # Layout
/// * `q`           – `[num_heads, seq_len, head_dim]`  (full query)
/// * `prefix_k/v`  – `[num_heads, prefix_len, head_dim]`
/// * `new_k/v`     – `[num_heads, new_len, head_dim]`
///
/// where `new_len = seq_len - prefix_len`.
pub fn attention_with_prefix_cache(
    config: &OptimizedAttentionConfig,
    q: &[f32],
    prefix_k: &[f32],
    prefix_v: &[f32],
    new_k: &[f32],
    new_v: &[f32],
    seq_len: usize,
    prefix_len: usize,
) -> Result<Vec<f32>> {
    config.validate()?;
    if prefix_len > seq_len {
        return Err(FlashAttentionError::InvalidDimension(format!(
            "prefix_len ({prefix_len}) exceeds seq_len ({seq_len})"
        )));
    }

    let nh = config.num_heads;
    let d = config.head_dim;
    let new_len = seq_len - prefix_len;

    check_len(q, nh * seq_len * d, "Q")?;
    check_len(prefix_k, nh * prefix_len * d, "prefix_K")?;
    check_len(prefix_v, nh * prefix_len * d, "prefix_V")?;
    check_len(new_k, nh * new_len * d, "new_K")?;
    check_len(new_v, nh * new_len * d, "new_V")?;

    let scale = config.scale();
    let mut output = vec![0.0_f32; nh * seq_len * d];

    for h in 0..nh {
        let q_head_off = h * seq_len * d;
        let pk_head_off = h * prefix_len * d;
        let nk_head_off = h * new_len * d;

        for qi in 0..seq_len {
            let q_start = q_head_off + qi * d;
            let q_slice = &q[q_start..q_start + d];

            // Total key positions this query can attend to.
            let total_kv = if config.causal { qi + 1 } else { seq_len };

            let mut scores = Vec::with_capacity(total_kv);

            // Scores against prefix keys.
            let prefix_end = prefix_len.min(total_kv);
            for kj in 0..prefix_end {
                let k_start = pk_head_off + kj * d;
                scores.push(dot(q_slice, &prefix_k[k_start..k_start + d]) * scale);
            }
            // Scores against new keys.
            if total_kv > prefix_len {
                let new_end = total_kv - prefix_len;
                for nj in 0..new_end {
                    let k_start = nk_head_off + nj * d;
                    scores.push(dot(q_slice, &new_k[k_start..k_start + d]) * scale);
                }
            }

            softmax_inplace(&mut scores);

            // Weighted sum: prefix V then new V.
            let out_start = q_head_off + qi * d;
            for (idx, &w) in scores.iter().enumerate() {
                if idx < prefix_end {
                    let v_start = pk_head_off + idx * d;
                    for dd in 0..d {
                        output[out_start + dd] += w * prefix_v[v_start + dd];
                    }
                } else {
                    let nj = idx - prefix_len;
                    let v_start = nk_head_off + nj * d;
                    for dd in 0..d {
                        output[out_start + dd] += w * new_v[v_start + dd];
                    }
                }
            }
        }
    }

    Ok(output)
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────

    fn default_config(nh: usize, d: usize, max_seq: usize) -> OptimizedAttentionConfig {
        OptimizedAttentionConfig {
            num_heads: nh,
            head_dim: d,
            max_seq_len: max_seq,
            causal: false,
            dropout_p: 0.0,
        }
    }

    fn causal_config(nh: usize, d: usize, max_seq: usize) -> OptimizedAttentionConfig {
        OptimizedAttentionConfig {
            num_heads: nh,
            head_dim: d,
            max_seq_len: max_seq,
            causal: true,
            dropout_p: 0.0,
        }
    }

    /// Reference (naïve) non-causal attention for a single head.
    fn naive_attention(q: &[f32], k: &[f32], v: &[f32], seq: usize, d: usize) -> Vec<f32> {
        let scale = 1.0 / (d as f32).sqrt();
        let mut out = vec![0.0; seq * d];
        for i in 0..seq {
            let mut scores: Vec<f32> = (0..seq)
                .map(|j| dot(&q[i * d..(i + 1) * d], &k[j * d..(j + 1) * d]) * scale)
                .collect();
            softmax_inplace(&mut scores);
            for (j, &w) in scores.iter().enumerate() {
                for dd in 0..d {
                    out[i * d + dd] += w * v[j * d + dd];
                }
            }
        }
        out
    }

    /// Reference causal attention for a single head.
    fn naive_causal_attention(q: &[f32], k: &[f32], v: &[f32], seq: usize, d: usize) -> Vec<f32> {
        let scale = 1.0 / (d as f32).sqrt();
        let mut out = vec![0.0; seq * d];
        for i in 0..seq {
            let mut scores: Vec<f32> = (0..=i)
                .map(|j| dot(&q[i * d..(i + 1) * d], &k[j * d..(j + 1) * d]) * scale)
                .collect();
            softmax_inplace(&mut scores);
            for (j, &w) in scores.iter().enumerate() {
                for dd in 0..d {
                    out[i * d + dd] += w * v[j * d + dd];
                }
            }
        }
        out
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    fn assert_approx(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert!(
            approx_eq(a, b, tol),
            "{label}: max diff = {}",
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
        );
    }

    /// Simple deterministic data: values 0.01, 0.02, …
    fn make_data(len: usize) -> Vec<f32> {
        (0..len).map(|i| (i as f32 + 1.0) * 0.01).collect()
    }

    // ── OptimizedAttentionConfig validation ────────────────────────

    #[test]
    fn test_config_validate_ok() {
        default_config(4, 64, 128).validate().unwrap();
    }

    #[test]
    fn test_config_validate_zero_heads() {
        let r = default_config(0, 64, 128).validate();
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("num_heads"));
    }

    #[test]
    fn test_config_validate_zero_dim() {
        let r = default_config(4, 0, 128).validate();
        assert!(r.is_err());
    }

    #[test]
    fn test_config_validate_zero_seq() {
        let r = default_config(4, 64, 0).validate();
        assert!(r.is_err());
    }

    #[test]
    fn test_config_validate_bad_dropout() {
        let mut c = default_config(4, 64, 128);
        c.dropout_p = 1.5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_scale() {
        let c = default_config(1, 64, 8);
        let expected = 1.0 / 64.0_f32.sqrt();
        assert!((c.scale() - expected).abs() < 1e-7);
    }

    // ── FlashAttentionError Display ────────────────────────────────

    #[test]
    fn test_error_display_invalid_dim() {
        let e = FlashAttentionError::InvalidDimension("foo".into());
        assert_eq!(e.to_string(), "invalid dimension: foo");
    }

    #[test]
    fn test_error_display_shape_mismatch() {
        let e = FlashAttentionError::ShapeMismatch { expected: 10, actual: 5, context: "Q".into() };
        assert!(e.to_string().contains("shape mismatch"));
    }

    #[test]
    fn test_error_display_block_size() {
        let e = FlashAttentionError::InvalidBlockSize("bad".into());
        assert!(e.to_string().contains("block size"));
    }

    #[test]
    fn test_error_display_head_config() {
        let e = FlashAttentionError::HeadConfigError("oops".into());
        assert!(e.to_string().contains("head config"));
    }

    #[test]
    fn test_error_display_window() {
        let e = FlashAttentionError::InvalidWindowSize("zero".into());
        assert!(e.to_string().contains("window size"));
    }

    // ── flash_attention_forward ────────────────────────────────────

    #[test]
    fn test_flash_forward_single_head_non_causal() {
        let (nh, d, seq) = (1, 4, 3);
        let cfg = default_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = flash_attention_forward(&cfg, &data, &data, &data, seq, 2).unwrap();
        let ref_out = naive_attention(&data, &data, &data, seq, d);
        assert_approx(&out, &ref_out, 1e-5, "flash_non_causal_1h");
    }

    #[test]
    fn test_flash_forward_single_head_causal() {
        let (nh, d, seq) = (1, 4, 4);
        let cfg = causal_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = flash_attention_forward(&cfg, &data, &data, &data, seq, 2).unwrap();
        let ref_out = naive_causal_attention(&data, &data, &data, seq, d);
        assert_approx(&out, &ref_out, 1e-5, "flash_causal_1h");
    }

    #[test]
    fn test_flash_forward_multi_head() {
        let (nh, d, seq) = (2, 4, 3);
        let cfg = default_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = flash_attention_forward(&cfg, &data, &data, &data, seq, 2).unwrap();
        for h in 0..nh {
            let off = h * seq * d;
            let ref_out = naive_attention(
                &data[off..off + seq * d],
                &data[off..off + seq * d],
                &data[off..off + seq * d],
                seq,
                d,
            );
            assert_approx(
                &out[off..off + seq * d],
                &ref_out,
                1e-5,
                &format!("flash_multi_head_{h}"),
            );
        }
    }

    #[test]
    fn test_flash_forward_block_size_one() {
        let (nh, d, seq) = (1, 4, 5);
        let cfg = default_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = flash_attention_forward(&cfg, &data, &data, &data, seq, 1).unwrap();
        let ref_out = naive_attention(&data, &data, &data, seq, d);
        assert_approx(&out, &ref_out, 1e-5, "flash_block1");
    }

    #[test]
    fn test_flash_forward_block_ge_seq() {
        let (nh, d, seq) = (1, 4, 3);
        let cfg = default_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = flash_attention_forward(&cfg, &data, &data, &data, seq, 100).unwrap();
        let ref_out = naive_attention(&data, &data, &data, seq, d);
        assert_approx(&out, &ref_out, 1e-5, "flash_big_block");
    }

    #[test]
    fn test_flash_forward_seq_1() {
        let (nh, d, seq) = (1, 8, 1);
        let cfg = default_config(nh, d, seq);
        let q = make_data(d);
        let k = make_data(d);
        let v = make_data(d);
        let out = flash_attention_forward(&cfg, &q, &k, &v, seq, 1).unwrap();
        assert_approx(&out, &v, 1e-6, "flash_seq1");
    }

    #[test]
    fn test_flash_forward_zero_block_err() {
        let cfg = default_config(1, 4, 8);
        let data = make_data(4);
        let r = flash_attention_forward(&cfg, &data, &data, &data, 1, 0);
        assert!(r.is_err());
    }

    #[test]
    fn test_flash_forward_exceeds_max_seq() {
        let cfg = default_config(1, 4, 4);
        let data = make_data(1 * 5 * 4);
        let r = flash_attention_forward(&cfg, &data, &data, &data, 5, 2);
        assert!(r.is_err());
    }

    #[test]
    fn test_flash_forward_shape_mismatch_q() {
        let cfg = default_config(1, 4, 8);
        let good = make_data(1 * 3 * 4);
        let bad = make_data(5);
        let r = flash_attention_forward(&cfg, &bad, &good, &good, 3, 2);
        assert!(r.is_err());
    }

    #[test]
    fn test_flash_forward_shape_mismatch_k() {
        let cfg = default_config(1, 4, 8);
        let good = make_data(1 * 3 * 4);
        let bad = make_data(5);
        let r = flash_attention_forward(&cfg, &good, &bad, &good, 3, 2);
        assert!(r.is_err());
    }

    #[test]
    fn test_flash_forward_causal_multi_head() {
        let (nh, d, seq) = (3, 8, 6);
        let cfg = causal_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = flash_attention_forward(&cfg, &data, &data, &data, seq, 3).unwrap();
        for h in 0..nh {
            let off = h * seq * d;
            let ref_out = naive_causal_attention(
                &data[off..off + seq * d],
                &data[off..off + seq * d],
                &data[off..off + seq * d],
                seq,
                d,
            );
            assert_approx(
                &out[off..off + seq * d],
                &ref_out,
                1e-4,
                &format!("flash_causal_mh_{h}"),
            );
        }
    }

    // ── group_query_attention ──────────────────────────────────────

    #[test]
    fn test_gqa_equal_heads_matches_standard() {
        let (nh, d, seq) = (2, 4, 3);
        let cfg = default_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let gqa_out = group_query_attention(&cfg, &data, &data, &data, seq, nh).unwrap();
        let ref_out = flash_attention_forward(&cfg, &data, &data, &data, seq, seq).unwrap();
        assert_approx(&gqa_out, &ref_out, 1e-5, "gqa_eq_heads");
    }

    #[test]
    fn test_gqa_head_sharing() {
        let (nh, d, seq) = (4, 4, 3);
        let num_kv = 2;
        let cfg = default_config(nh, d, seq);
        let q = make_data(nh * seq * d);
        let kv = make_data(num_kv * seq * d);
        let out = group_query_attention(&cfg, &q, &kv, &kv, seq, num_kv).unwrap();
        // Heads 0 & 1 share kv_head 0; heads 2 & 3 share kv_head 1.
        // With identical Q within each pair the output would match,
        // but here Q differs per head so we just check finiteness.
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gqa_causal() {
        let (nh, d, seq) = (2, 4, 4);
        let cfg = causal_config(nh, d, seq);
        let q = make_data(nh * seq * d);
        let kv = make_data(1 * seq * d);
        let out = group_query_attention(&cfg, &q, &kv, &kv, seq, 1).unwrap();
        // First query row of head 0 must only depend on position 0.
        assert_approx(&out[0..d], &kv[0..d], 1e-6, "gqa_causal_first_row");
    }

    #[test]
    fn test_gqa_indivisible_heads_error() {
        let cfg = default_config(3, 4, 4);
        let q = make_data(3 * 4 * 4);
        let kv = make_data(2 * 4 * 4);
        let r = group_query_attention(&cfg, &q, &kv, &kv, 4, 2);
        assert!(r.is_err());
    }

    #[test]
    fn test_gqa_zero_kv_heads_error() {
        let cfg = default_config(2, 4, 4);
        let data = make_data(2 * 4 * 4);
        let r = group_query_attention(&cfg, &data, &data, &data, 4, 0);
        assert!(r.is_err());
    }

    #[test]
    fn test_gqa_shape_mismatch_q() {
        let cfg = default_config(2, 4, 4);
        let kv = make_data(1 * 4 * 4);
        let bad_q = make_data(5);
        let r = group_query_attention(&cfg, &bad_q, &kv, &kv, 4, 1);
        assert!(r.is_err());
    }

    #[test]
    fn test_gqa_shape_mismatch_kv() {
        let cfg = default_config(2, 4, 4);
        let q = make_data(2 * 4 * 4);
        let bad_kv = make_data(5);
        let r = group_query_attention(&cfg, &q, &bad_kv, &bad_kv, 4, 1);
        assert!(r.is_err());
    }

    #[test]
    fn test_gqa_non_causal_attends_all() {
        let (nh, d, seq) = (2, 4, 3);
        let cfg = default_config(nh, d, seq);
        let q = make_data(nh * seq * d);
        let kv = make_data(1 * seq * d);
        let out = group_query_attention(&cfg, &q, &kv, &kv, seq, 1).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── multi_query_attention ──────────────────────────────────────

    #[test]
    fn test_mqa_basic() {
        let (nh, d, seq) = (4, 4, 3);
        let cfg = default_config(nh, d, seq);
        let q = make_data(nh * seq * d);
        let kv = make_data(1 * seq * d);
        let out = multi_query_attention(&cfg, &q, &kv, &kv, seq).unwrap();
        assert_eq!(out.len(), nh * seq * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_mqa_matches_gqa_with_1() {
        let (nh, d, seq) = (4, 4, 3);
        let cfg = default_config(nh, d, seq);
        let q = make_data(nh * seq * d);
        let kv = make_data(seq * d);
        let mqa = multi_query_attention(&cfg, &q, &kv, &kv, seq).unwrap();
        let gqa = group_query_attention(&cfg, &q, &kv, &kv, seq, 1).unwrap();
        assert_approx(&mqa, &gqa, 1e-7, "mqa_vs_gqa");
    }

    #[test]
    fn test_mqa_causal() {
        let (nh, d, seq) = (2, 4, 4);
        let cfg = causal_config(nh, d, seq);
        let q = make_data(nh * seq * d);
        let kv = make_data(seq * d);
        let out = multi_query_attention(&cfg, &q, &kv, &kv, seq).unwrap();
        for h in 0..nh {
            let off = h * seq * d;
            assert_approx(&out[off..off + d], &kv[0..d], 1e-6, "mqa_causal_row0");
        }
    }

    // ── sliding_window_attention ───────────────────────────────────

    #[test]
    fn test_sw_window_covers_all_matches_standard() {
        let (nh, d, seq) = (1, 4, 4);
        let cfg = default_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let sw_out = sliding_window_attention(&cfg, &data, &data, &data, seq, seq + 10).unwrap();
        let ref_out = naive_attention(&data, &data, &data, seq, d);
        assert_approx(&sw_out, &ref_out, 1e-5, "sw_full_window");
    }

    #[test]
    fn test_sw_window_1_causal() {
        let (nh, d, seq) = (1, 4, 4);
        let cfg = causal_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = sliding_window_attention(&cfg, &data, &data, &data, seq, 1).unwrap();
        for qi in 0..seq {
            assert_approx(
                &out[qi * d..(qi + 1) * d],
                &data[qi * d..(qi + 1) * d],
                1e-6,
                &format!("sw_w1_row{qi}"),
            );
        }
    }

    #[test]
    fn test_sw_window_2_causal() {
        let (nh, d, seq) = (1, 4, 5);
        let cfg = causal_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = sliding_window_attention(&cfg, &data, &data, &data, seq, 2).unwrap();
        assert_approx(&out[0..d], &data[0..d], 1e-6, "sw_w2_row0");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sw_window_zero_error() {
        let cfg = default_config(1, 4, 8);
        let data = make_data(1 * 4 * 4);
        let r = sliding_window_attention(&cfg, &data, &data, &data, 4, 0);
        assert!(r.is_err());
    }

    #[test]
    fn test_sw_multi_head() {
        let (nh, d, seq) = (2, 4, 4);
        let cfg = default_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = sliding_window_attention(&cfg, &data, &data, &data, seq, 2).unwrap();
        assert_eq!(out.len(), nh * seq * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sw_non_causal() {
        let (nh, d, seq) = (1, 4, 5);
        let cfg = default_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = sliding_window_attention(&cfg, &data, &data, &data, seq, 3).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sw_shape_mismatch() {
        let cfg = default_config(1, 4, 8);
        let good = make_data(4 * 4);
        let bad = make_data(5);
        let r = sliding_window_attention(&cfg, &bad, &good, &good, 4, 2);
        assert!(r.is_err());
    }

    // ── attention_with_alibi ───────────────────────────────────────

    #[test]
    fn test_alibi_slopes_basic() {
        let s = alibi_slopes(4);
        assert_eq!(s.len(), 4);
        for i in 1..s.len() {
            assert!(s[i] < s[i - 1], "slopes not decreasing at {i}");
        }
    }

    #[test]
    fn test_alibi_slopes_single() {
        let s = alibi_slopes(1);
        assert_eq!(s.len(), 1);
        assert!((s[0] - (1.0 / 256.0)).abs() < 1e-7);
    }

    #[test]
    fn test_alibi_non_causal() {
        let (nh, d, seq) = (2, 4, 4);
        let cfg = default_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = attention_with_alibi(&cfg, &data, &data, &data, seq).unwrap();
        assert_eq!(out.len(), nh * seq * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_alibi_causal() {
        let (nh, d, seq) = (2, 4, 4);
        let cfg = causal_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = attention_with_alibi(&cfg, &data, &data, &data, seq).unwrap();
        for h in 0..nh {
            let off = h * seq * d;
            assert_approx(
                &out[off..off + d],
                &data[off..off + d],
                1e-6,
                &format!("alibi_causal_h{h}_row0"),
            );
        }
    }

    #[test]
    fn test_alibi_differs_from_standard() {
        let (nh, d, seq) = (1, 4, 4);
        let cfg = default_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let std_out = flash_attention_forward(&cfg, &data, &data, &data, seq, seq).unwrap();
        let alibi_out = attention_with_alibi(&cfg, &data, &data, &data, seq).unwrap();
        assert!(!approx_eq(&std_out, &alibi_out, 1e-6), "ALiBi output should differ from standard");
    }

    #[test]
    fn test_alibi_seq_1() {
        let (nh, d, seq) = (1, 8, 1);
        let cfg = default_config(nh, d, seq);
        let data = make_data(d);
        let out = attention_with_alibi(&cfg, &data, &data, &data, seq).unwrap();
        assert_approx(&out, &data, 1e-6, "alibi_seq1");
    }

    #[test]
    fn test_alibi_shape_mismatch() {
        let cfg = default_config(1, 4, 8);
        let good = make_data(1 * 4 * 4);
        let bad = make_data(5);
        let r = attention_with_alibi(&cfg, &bad, &good, &good, 4);
        assert!(r.is_err());
    }

    // ── attention_with_prefix_cache ────────────────────────────────

    #[test]
    fn test_prefix_cache_no_prefix() {
        let (nh, d, seq) = (1, 4, 4);
        let cfg = default_config(nh, d, seq);
        let q = make_data(nh * seq * d);
        let new_k = make_data(nh * seq * d);
        let new_v = make_data(nh * seq * d);
        let prefix_k: Vec<f32> = vec![];
        let prefix_v: Vec<f32> = vec![];
        let out =
            attention_with_prefix_cache(&cfg, &q, &prefix_k, &prefix_v, &new_k, &new_v, seq, 0)
                .unwrap();
        let ref_out = naive_attention(&q, &new_k, &new_v, seq, d);
        assert_approx(&out, &ref_out, 1e-5, "prefix_none");
    }

    #[test]
    fn test_prefix_cache_full_prefix() {
        let (nh, d, seq) = (1, 4, 4);
        let cfg = default_config(nh, d, seq);
        let q = make_data(nh * seq * d);
        let prefix_k = make_data(nh * seq * d);
        let prefix_v = make_data(nh * seq * d);
        let new_k: Vec<f32> = vec![];
        let new_v: Vec<f32> = vec![];
        let out =
            attention_with_prefix_cache(&cfg, &q, &prefix_k, &prefix_v, &new_k, &new_v, seq, seq)
                .unwrap();
        let ref_out = naive_attention(&q, &prefix_k, &prefix_v, seq, d);
        assert_approx(&out, &ref_out, 1e-5, "prefix_full");
    }

    #[test]
    fn test_prefix_cache_half() {
        let (nh, d, seq) = (1, 4, 4);
        let prefix_len = 2;
        let new_len = 2;
        let cfg = default_config(nh, d, seq);
        let q = make_data(nh * seq * d);
        let prefix_k = make_data(nh * prefix_len * d);
        let prefix_v = make_data(nh * prefix_len * d);
        let new_k = make_data(nh * new_len * d);
        let new_v = make_data(nh * new_len * d);
        let out = attention_with_prefix_cache(
            &cfg, &q, &prefix_k, &prefix_v, &new_k, &new_v, seq, prefix_len,
        )
        .unwrap();

        let mut full_k = prefix_k.clone();
        full_k.extend_from_slice(&new_k);
        let mut full_v = prefix_v.clone();
        full_v.extend_from_slice(&new_v);
        let ref_out = naive_attention(&q, &full_k, &full_v, seq, d);
        assert_approx(&out, &ref_out, 1e-5, "prefix_half");
    }

    #[test]
    fn test_prefix_cache_causal() {
        let (nh, d, seq) = (1, 4, 4);
        let prefix_len = 2;
        let new_len = 2;
        let cfg = causal_config(nh, d, seq);
        let q = make_data(nh * seq * d);
        let prefix_k = make_data(nh * prefix_len * d);
        let prefix_v = make_data(nh * prefix_len * d);
        let new_k = make_data(nh * new_len * d);
        let new_v = make_data(nh * new_len * d);
        let out = attention_with_prefix_cache(
            &cfg, &q, &prefix_k, &prefix_v, &new_k, &new_v, seq, prefix_len,
        )
        .unwrap();

        let mut full_k = prefix_k.clone();
        full_k.extend_from_slice(&new_k);
        let mut full_v = prefix_v.clone();
        full_v.extend_from_slice(&new_v);
        let ref_out = naive_causal_attention(&q, &full_k, &full_v, seq, d);
        assert_approx(&out, &ref_out, 1e-5, "prefix_causal");
    }

    #[test]
    fn test_prefix_cache_exceeds_seq_error() {
        let cfg = default_config(1, 4, 8);
        let q = make_data(1 * 4 * 4);
        let pk = make_data(1 * 5 * 4);
        let pv = make_data(1 * 5 * 4);
        let nk: Vec<f32> = vec![];
        let nv: Vec<f32> = vec![];
        let r = attention_with_prefix_cache(&cfg, &q, &pk, &pv, &nk, &nv, 4, 5);
        assert!(r.is_err());
    }

    #[test]
    fn test_prefix_cache_multi_head() {
        let (nh, d, seq) = (2, 4, 4);
        let prefix_len = 2;
        let new_len = 2;
        let cfg = default_config(nh, d, seq);
        let q = make_data(nh * seq * d);
        let pk = make_data(nh * prefix_len * d);
        let pv = make_data(nh * prefix_len * d);
        let nk = make_data(nh * new_len * d);
        let nv = make_data(nh * new_len * d);
        let out =
            attention_with_prefix_cache(&cfg, &q, &pk, &pv, &nk, &nv, seq, prefix_len).unwrap();
        assert_eq!(out.len(), nh * seq * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Numerical stability / edge cases ───────────────────────────

    #[test]
    fn test_softmax_stability_large_values() {
        let mut row = vec![1000.0, 1001.0, 1000.5];
        softmax_inplace(&mut row);
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "softmax should sum to 1");
        assert!(row.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_softmax_stability_negative_infinity() {
        let mut row = vec![0.0, f32::NEG_INFINITY, 0.0];
        softmax_inplace(&mut row);
        assert!((row[1]).abs() < 1e-7, "masked position should be ~0");
        assert!((row[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_single_element() {
        let mut row = vec![42.0];
        softmax_inplace(&mut row);
        assert!((row[0] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_flash_attention_output_finite() {
        let (nh, d, seq) = (4, 16, 8);
        let cfg = causal_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = flash_attention_forward(&cfg, &data, &data, &data, seq, 4).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "all values must be finite");
    }

    #[test]
    fn test_output_sums_nonzero() {
        let (nh, d, seq) = (2, 4, 3);
        let cfg = default_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = flash_attention_forward(&cfg, &data, &data, &data, seq, 2).unwrap();
        let s: f32 = out.iter().sum();
        assert!(s.abs() > 1e-6, "output should not be all zeros");
    }

    #[test]
    fn test_causal_first_row_equals_v() {
        let (nh, d, seq) = (2, 4, 5);
        let cfg = causal_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let out = flash_attention_forward(&cfg, &data, &data, &data, seq, 2).unwrap();
        for h in 0..nh {
            let off = h * seq * d;
            assert_approx(
                &out[off..off + d],
                &data[off..off + d],
                1e-6,
                &format!("causal_first_h{h}"),
            );
        }
    }

    #[test]
    fn test_different_qkv_flash() {
        let (nh, d, seq) = (1, 4, 3);
        let cfg = default_config(nh, d, seq);
        let n = nh * seq * d;
        let q = make_data(n);
        let k: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5) * 0.02).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03).collect();
        let out = flash_attention_forward(&cfg, &q, &k, &v, seq, 2).unwrap();
        let ref_out = naive_attention(&q, &k, &v, seq, d);
        assert_approx(&out, &ref_out, 1e-5, "diff_qkv_flash");
    }

    #[test]
    fn test_different_qkv_gqa() {
        let (nh, d, seq) = (2, 4, 3);
        let num_kv = 1;
        let cfg = default_config(nh, d, seq);
        let q = make_data(nh * seq * d);
        let k: Vec<f32> = (0..num_kv * seq * d).map(|i| (i as f32 + 0.5) * 0.02).collect();
        let v: Vec<f32> = (0..num_kv * seq * d).map(|i| (i as f32) * 0.03).collect();
        let out = group_query_attention(&cfg, &q, &k, &v, seq, num_kv).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Additional coverage ────────────────────────────────────────

    #[test]
    fn test_flash_causal_last_row_attends_all() {
        let (nh, d, seq) = (1, 4, 4);
        let cfg = causal_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let causal_out = flash_attention_forward(&cfg, &data, &data, &data, seq, 2).unwrap();
        let non_causal_cfg = default_config(nh, d, seq);
        let nc_out = flash_attention_forward(&non_causal_cfg, &data, &data, &data, seq, 2).unwrap();
        let last = (seq - 1) * d;
        assert_approx(
            &causal_out[last..last + d],
            &nc_out[last..last + d],
            1e-5,
            "causal_last_row",
        );
    }

    #[test]
    fn test_sw_causal_large_window_matches_causal() {
        let (nh, d, seq) = (1, 4, 4);
        let cfg = causal_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let sw = sliding_window_attention(&cfg, &data, &data, &data, seq, 100).unwrap();
        let ref_out = naive_causal_attention(&data, &data, &data, seq, d);
        assert_approx(&sw, &ref_out, 1e-5, "sw_large_win_causal");
    }

    #[test]
    fn test_mqa_shape_mismatch() {
        let cfg = default_config(2, 4, 8);
        let q = make_data(2 * 4 * 4);
        let bad_kv = make_data(5);
        let r = multi_query_attention(&cfg, &q, &bad_kv, &bad_kv, 4);
        assert!(r.is_err());
    }

    #[test]
    fn test_prefix_cache_shape_mismatch_new() {
        let cfg = default_config(1, 4, 8);
        let q = make_data(1 * 4 * 4);
        let pk = make_data(1 * 2 * 4);
        let pv = make_data(1 * 2 * 4);
        let bad = make_data(5);
        let r = attention_with_prefix_cache(&cfg, &q, &pk, &pv, &bad, &bad, 4, 2);
        assert!(r.is_err());
    }

    #[test]
    fn test_flash_block_size_variations() {
        let (nh, d, seq) = (1, 4, 6);
        let cfg = default_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let ref_out = naive_attention(&data, &data, &data, seq, d);
        for bs in [1, 2, 3, 4, 5, 6, 7, 10] {
            let out = flash_attention_forward(&cfg, &data, &data, &data, seq, bs).unwrap();
            assert_approx(&out, &ref_out, 1e-5, &format!("flash_bs{bs}"));
        }
    }

    #[test]
    fn test_flash_causal_block_size_variations() {
        let (nh, d, seq) = (1, 4, 6);
        let cfg = causal_config(nh, d, seq);
        let data = make_data(nh * seq * d);
        let ref_out = naive_causal_attention(&data, &data, &data, seq, d);
        for bs in [1, 2, 3, 4, 5, 6, 10] {
            let out = flash_attention_forward(&cfg, &data, &data, &data, seq, bs).unwrap();
            assert_approx(&out, &ref_out, 1e-4, &format!("flash_causal_bs{bs}"));
        }
    }

    #[test]
    fn test_gqa_multi_head_consistency() {
        let (nh, d, seq) = (8, 4, 3);
        let num_kv = 2;
        let cfg = default_config(nh, d, seq);
        let kv = make_data(num_kv * seq * d);
        let group_size = nh / num_kv;
        // Build Q identical within each group so grouped heads must match.
        let mut q_same = vec![0.0_f32; nh * seq * d];
        for h in 0..nh {
            let kv_h = h / group_size;
            let src = &kv[kv_h * seq * d..(kv_h + 1) * seq * d];
            q_same[h * seq * d..(h + 1) * seq * d].copy_from_slice(src);
        }
        let out = group_query_attention(&cfg, &q_same, &kv, &kv, seq, num_kv).unwrap();
        for g in 0..num_kv {
            let base = g * group_size * seq * d;
            for member in 1..group_size {
                let off = base + member * seq * d;
                assert_approx(
                    &out[base..base + seq * d],
                    &out[off..off + seq * d],
                    1e-6,
                    &format!("gqa_group{g}_member{member}"),
                );
            }
        }
    }
}
