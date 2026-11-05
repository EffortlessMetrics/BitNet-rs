//! CPU SIMD-optimized 1D convolution kernels.
//!
//! Provides standard, depthwise, pointwise, grouped, and transposed 1D
//! convolution on contiguous `f32` slices in NCL (batch, channels, length)
//! layout, with an im2col transform for GEMM-based convolution and runtime
//! AVX2 dispatch.
//!
//! # Layout conventions
//!
//! * Input:  `[batch_size, in_channels,  in_length]`  — row-major
//! * Weight: `[out_channels, in_channels/groups, kernel_size]` — row-major
//! * Bias:   `[out_channels]`
//! * Output: `[batch_size, out_channels, out_length]` — row-major
#![allow(unsafe_op_in_unsafe_fn)]

use bitnet_common::{BitNetError, KernelError, Result};
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── Configuration ──────────────────────────────────────────────────────

/// Padding mode for 1D convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaddingMode {
    /// Pad with zeros (default).
    Zero,
    /// Reflect at boundaries: `[a,b,c,d] → [c,b, | a,b,c,d | ,c,b]`.
    Reflect,
    /// Replicate edge values: `[a,b,c,d] → [a,a, | a,b,c,d | ,d,d]`.
    Replicate,
    /// Circular (wrap-around): `[a,b,c,d] → [c,d, | a,b,c,d | ,a,b]`.
    Circular,
}

/// Configuration for a 1D convolution operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conv1dConfig {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels (filters).
    pub out_channels: usize,
    /// Kernel (filter) width.
    pub kernel_size: usize,
    /// Stride between successive windows.
    pub stride: usize,
    /// Zero-padding added to each side of the input.
    pub padding: usize,
    /// Spacing between kernel elements.
    pub dilation: usize,
    /// Number of groups for grouped convolution.
    pub groups: usize,
}

impl Conv1dConfig {
    /// Create a simple config with default stride=1, padding=0, dilation=1, groups=1.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.in_channels == 0 {
            return Err(invalid_args("in_channels must be > 0"));
        }
        if self.out_channels == 0 {
            return Err(invalid_args("out_channels must be > 0"));
        }
        if self.kernel_size == 0 {
            return Err(invalid_args("kernel_size must be > 0"));
        }
        if self.stride == 0 {
            return Err(invalid_args("stride must be > 0"));
        }
        if self.dilation == 0 {
            return Err(invalid_args("dilation must be > 0"));
        }
        if self.groups == 0 {
            return Err(invalid_args("groups must be > 0"));
        }
        if !self.in_channels.is_multiple_of(self.groups) {
            return Err(invalid_args("in_channels must be divisible by groups"));
        }
        if !self.out_channels.is_multiple_of(self.groups) {
            return Err(invalid_args("out_channels must be divisible by groups"));
        }
        Ok(())
    }
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self::new(1, 1, 1)
    }
}

// ── Output size ────────────────────────────────────────────────────────

/// Compute the output length for a 1D convolution.
///
/// Formula: `(in_len + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
pub fn compute_output_length(
    in_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let effective_kernel = dilation * (kernel_size - 1) + 1;
    let padded = in_len + 2 * padding;
    if padded < effective_kernel {
        return 0;
    }
    (padded - effective_kernel) / stride + 1
}

// ── Padding ────────────────────────────────────────────────────────────

/// Apply padding to a 1D signal of length `in_len`.
///
/// Returns a new vector of length `in_len + 2 * pad`.
pub fn apply_padding(input: &[f32], pad: usize, mode: PaddingMode) -> Vec<f32> {
    let n = input.len();
    if pad == 0 || n == 0 {
        return input.to_vec();
    }
    let out_len = n + 2 * pad;
    let mut out = vec![0.0f32; out_len];

    // Copy centre.
    out[pad..pad + n].copy_from_slice(input);

    match mode {
        PaddingMode::Zero => { /* already zero-filled */ }
        PaddingMode::Reflect => {
            for i in 0..pad {
                let src = (i + 1).min(n - 1);
                out[pad - 1 - i] = input[src];
            }
            for i in 0..pad {
                let src = n.saturating_sub(2).saturating_sub(i);
                out[pad + n + i] = input[src];
            }
        }
        PaddingMode::Replicate => {
            for i in 0..pad {
                out[i] = input[0];
                out[pad + n + i] = input[n - 1];
            }
        }
        PaddingMode::Circular => {
            for i in 0..pad {
                out[pad - 1 - i] = input[n - 1 - (i % n)];
                out[pad + n + i] = input[i % n];
            }
        }
    }
    out
}

// ── im2col / col2im ───────────────────────────────────────────────────

/// im2col transform for 1D: rearrange input patches into columns for
/// GEMM-based convolution.
///
/// Input is a single channel-group image: `[ic_per_group, in_len]`.
/// Returns matrix `[col_h, col_w]` row-major where
/// `col_h = ic_per_group * kernel_size` and `col_w = out_len`.
pub fn im2col(
    input: &[f32],
    config: &Conv1dConfig,
    in_len: usize,
    group: usize,
) -> Result<Vec<f32>> {
    config.validate()?;
    let ic_per_group = config.in_channels / config.groups;
    let out_len = compute_output_length(
        in_len,
        config.kernel_size,
        config.stride,
        config.padding,
        config.dilation,
    );
    if out_len == 0 {
        return Err(invalid_args("output length is zero for im2col"));
    }
    let expected = config.in_channels * in_len;
    if input.len() != expected {
        return Err(invalid_args(&format!(
            "im2col input length {} != expected {expected}",
            input.len(),
        )));
    }
    if group >= config.groups {
        return Err(invalid_args(&format!("group index {group} >= groups {}", config.groups)));
    }

    let col_h = ic_per_group * config.kernel_size;
    let col_w = out_len;
    let mut columns = vec![0.0f32; col_h * col_w];

    for ic in 0..ic_per_group {
        let abs_ic = group * ic_per_group + ic;
        for k in 0..config.kernel_size {
            let row = ic * config.kernel_size + k;
            for o in 0..out_len {
                let i_pos =
                    (o * config.stride + k * config.dilation) as isize - config.padding as isize;
                let val = if i_pos >= 0 && (i_pos as usize) < in_len {
                    input[abs_ic * in_len + i_pos as usize]
                } else {
                    0.0
                };
                columns[row * col_w + o] = val;
            }
        }
    }
    Ok(columns)
}

/// col2im transform for 1D: accumulate column patches back into a signal.
///
/// `cols` has shape `[ic_per_group * kernel_size, out_len]` row-major.
/// Produces `[ic_per_group, in_len]`.
pub fn col2im(
    cols: &[f32],
    config: &Conv1dConfig,
    in_len: usize,
    out_len: usize,
    group: usize,
) -> Result<Vec<f32>> {
    config.validate()?;
    let ic_per_group = config.in_channels / config.groups;
    let col_h = ic_per_group * config.kernel_size;
    if cols.len() != col_h * out_len {
        return Err(invalid_args(&format!(
            "col2im cols length {} != expected {}",
            cols.len(),
            col_h * out_len,
        )));
    }
    if group >= config.groups {
        return Err(invalid_args(&format!("group index {group} >= groups {}", config.groups)));
    }

    let mut output = vec![0.0f32; ic_per_group * in_len];

    for ic in 0..ic_per_group {
        for k in 0..config.kernel_size {
            let row = ic * config.kernel_size + k;
            for o in 0..out_len {
                let i_pos =
                    (o * config.stride + k * config.dilation) as isize - config.padding as isize;
                if i_pos >= 0 && (i_pos as usize) < in_len {
                    output[ic * in_len + i_pos as usize] += cols[row * out_len + o];
                }
            }
        }
    }
    Ok(output)
}

// ── Runtime dispatch ──────────────────────────────────────────────────

/// Returns `true` when AVX2 + FMA are available at runtime.
#[inline]
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ── Scalar helpers ────────────────────────────────────────────────────

/// Scalar dot product of two slices.
#[inline]
fn dot_scalar(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

// ── AVX2 helpers (x86_64 only) ────────────────────────────────────────

/// Horizontal sum of an `__m256` register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

/// AVX2 FMA dot product of two contiguous f32 slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn dot_avx2(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    let mut sum = hsum_avx2(acc);
    // Scalar tail.
    while i < len {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
}

/// Dispatch a dot product to AVX2 or scalar.
#[inline]
fn dot_dispatch(a: &[f32], b: &[f32], len: usize) -> f32 {
    if has_avx2() {
        #[cfg(target_arch = "x86_64")]
        // Safety: guarded by runtime AVX2 check.
        unsafe {
            return dot_avx2(a, b, len);
        }
    }
    dot_scalar(a, b, len)
}

// ── Standard 1D convolution (scalar) ──────────────────────────────────

/// Standard 1D convolution (NCL layout, scalar implementation).
///
/// - `input`:  `[batch_size, in_channels, in_len]` flattened row-major.
/// - `weight`: `[out_channels, in_channels/groups, kernel_size]` flattened.
/// - `bias`:   optional `[out_channels]`.
///
/// Returns `[batch_size, out_channels, out_len]` flattened.
pub fn conv1d_f32(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv1dConfig,
    batch_size: usize,
    in_len: usize,
) -> Result<Vec<f32>> {
    config.validate()?;
    let out_len = compute_output_length(
        in_len,
        config.kernel_size,
        config.stride,
        config.padding,
        config.dilation,
    );
    if out_len == 0 {
        return Err(invalid_args("output length is zero; check kernel/padding/dilation"));
    }
    validate_conv1d_buffers(input, weight, bias, config, batch_size, in_len)?;

    let ic_per_group = config.in_channels / config.groups;
    let oc_per_group = config.out_channels / config.groups;
    let mut output = vec![0.0f32; batch_size * config.out_channels * out_len];

    for n in 0..batch_size {
        for g in 0..config.groups {
            for oc in 0..oc_per_group {
                let abs_oc = g * oc_per_group + oc;
                let bias_val = bias.map_or(0.0, |b| b[abs_oc]);
                for o in 0..out_len {
                    let mut sum = bias_val;
                    for ic in 0..ic_per_group {
                        let abs_ic = g * ic_per_group + ic;
                        for k in 0..config.kernel_size {
                            let i_pos = (o * config.stride + k * config.dilation) as isize
                                - config.padding as isize;
                            if i_pos >= 0 && (i_pos as usize) < in_len {
                                let in_idx =
                                    (n * config.in_channels + abs_ic) * in_len + i_pos as usize;
                                let w_idx = (abs_oc * ic_per_group + ic) * config.kernel_size + k;
                                sum += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                    let out_idx = (n * config.out_channels + abs_oc) * out_len + o;
                    output[out_idx] = sum;
                }
            }
        }
    }
    Ok(output)
}

// ── AVX2 1D convolution with im2col ───────────────────────────────────

/// SIMD-accelerated 1D convolution using im2col + GEMM with runtime
/// AVX2/FMA dispatch.
///
/// Same interface as [`conv1d_f32`] but uses the im2col transform to
/// convert the convolution into a matrix multiplication, which is then
/// computed with AVX2 FMA dot products when available.
pub fn conv1d_avx2(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv1dConfig,
    batch_size: usize,
    in_len: usize,
) -> Result<Vec<f32>> {
    config.validate()?;
    let out_len = compute_output_length(
        in_len,
        config.kernel_size,
        config.stride,
        config.padding,
        config.dilation,
    );
    if out_len == 0 {
        return Err(invalid_args("output length is zero; check kernel/padding/dilation"));
    }
    validate_conv1d_buffers(input, weight, bias, config, batch_size, in_len)?;

    let ic_per_group = config.in_channels / config.groups;
    let oc_per_group = config.out_channels / config.groups;
    let col_h = ic_per_group * config.kernel_size;
    let mut output = vec![0.0f32; batch_size * config.out_channels * out_len];

    for n in 0..batch_size {
        let img = &input[n * config.in_channels * in_len..(n + 1) * config.in_channels * in_len];
        for g in 0..config.groups {
            let cols = im2col(img, config, in_len, g)?;

            // GEMM: weight[oc_per_group, col_h] × cols[col_h, out_len]
            for oc in 0..oc_per_group {
                let abs_oc = g * oc_per_group + oc;
                let bias_val = bias.map_or(0.0, |b| b[abs_oc]);
                let w_row = &weight[abs_oc * col_h..(abs_oc + 1) * col_h];
                for o in 0..out_len {
                    // Gather the column vector for this output position.
                    // cols is [col_h, out_len] row-major, so column o
                    // has stride out_len.
                    let mut col_vec = vec![0.0f32; col_h];
                    for r in 0..col_h {
                        col_vec[r] = cols[r * out_len + o];
                    }
                    let sum = bias_val + dot_dispatch(w_row, &col_vec, col_h);
                    let out_idx = (n * config.out_channels + abs_oc) * out_len + o;
                    output[out_idx] = sum;
                }
            }
        }
    }
    Ok(output)
}

// ── Depthwise 1D convolution ──────────────────────────────────────────

/// Depthwise separable 1D convolution (groups == in_channels == out_channels).
///
/// - `input`:  `[batch_size, channels, in_len]` flattened.
/// - `weight`: `[channels, 1, kernel_size]` flattened.
/// - `bias`:   optional `[channels]`.
pub fn conv1d_depthwise(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv1dConfig,
    batch_size: usize,
    in_len: usize,
) -> Result<Vec<f32>> {
    if config.groups != config.in_channels || config.in_channels != config.out_channels {
        return Err(invalid_args(
            "conv1d_depthwise requires groups == in_channels == out_channels",
        ));
    }
    config.validate()?;

    let channels = config.in_channels;
    let out_len = compute_output_length(
        in_len,
        config.kernel_size,
        config.stride,
        config.padding,
        config.dilation,
    );
    if out_len == 0 {
        return Err(invalid_args("output length is zero"));
    }
    let expected_input = batch_size * channels * in_len;
    if input.len() != expected_input {
        return Err(invalid_args(&format!(
            "input length {} != expected {expected_input}",
            input.len(),
        )));
    }
    let expected_weight = channels * config.kernel_size;
    if weight.len() != expected_weight {
        return Err(invalid_args(&format!(
            "weight length {} != expected {expected_weight}",
            weight.len(),
        )));
    }
    if let Some(b) = bias
        && b.len() != channels
    {
        return Err(invalid_args(&format!("bias length {} != channels {channels}", b.len())));
    }

    let mut output = vec![0.0f32; batch_size * channels * out_len];

    for n in 0..batch_size {
        for c in 0..channels {
            let bias_val = bias.map_or(0.0, |b| b[c]);
            for o in 0..out_len {
                let mut sum = bias_val;
                for k in 0..config.kernel_size {
                    let i_pos = (o * config.stride + k * config.dilation) as isize
                        - config.padding as isize;
                    if i_pos >= 0 && (i_pos as usize) < in_len {
                        let in_idx = (n * channels + c) * in_len + i_pos as usize;
                        let w_idx = c * config.kernel_size + k;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
                let out_idx = (n * channels + c) * out_len + o;
                output[out_idx] = sum;
            }
        }
    }
    Ok(output)
}

// ── Pointwise (1×1) convolution ───────────────────────────────────────

/// Pointwise (1×1) 1D convolution — optimised as a matrix multiply.
///
/// - `input`:  `[batch_size, in_channels, length]` flattened.
/// - `weight`: `[out_channels, in_channels]` flattened.
/// - `bias`:   optional `[out_channels]`.
///
/// Equivalent to `conv1d` with kernel_size=1, stride=1, padding=0,
/// dilation=1, groups=1, but avoids the im2col overhead.
pub fn conv1d_pointwise(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
) -> Result<Vec<f32>> {
    if in_channels == 0 || out_channels == 0 || length == 0 {
        return Err(invalid_args("dimensions must be > 0"));
    }
    let expected_input = batch_size * in_channels * length;
    if input.len() != expected_input {
        return Err(invalid_args(&format!(
            "input length {} != expected {expected_input}",
            input.len(),
        )));
    }
    if weight.len() != out_channels * in_channels {
        return Err(invalid_args(&format!(
            "weight length {} != expected {}",
            weight.len(),
            out_channels * in_channels,
        )));
    }
    if let Some(b) = bias
        && b.len() != out_channels
    {
        return Err(invalid_args(&format!(
            "bias length {} != out_channels {out_channels}",
            b.len(),
        )));
    }

    let mut output = vec![0.0f32; batch_size * out_channels * length];

    for n in 0..batch_size {
        for oc in 0..out_channels {
            let bias_val = bias.map_or(0.0, |b| b[oc]);
            let w_row = &weight[oc * in_channels..(oc + 1) * in_channels];
            for l in 0..length {
                // Gather input vector at position l across all channels.
                let mut in_vec = vec![0.0f32; in_channels];
                for ic in 0..in_channels {
                    in_vec[ic] = input[(n * in_channels + ic) * length + l];
                }
                let sum = bias_val + dot_dispatch(w_row, &in_vec, in_channels);
                output[(n * out_channels + oc) * length + l] = sum;
            }
        }
    }
    Ok(output)
}

// ── Transposed 1D convolution ─────────────────────────────────────────

/// Transposed (fractionally-strided) 1D convolution.
///
/// - `input`:  `[batch_size, in_channels, in_len]` flattened.
/// - `weight`: `[in_channels, out_channels/groups, kernel_size]` flattened.
///   (note: weight layout is transposed relative to forward conv)
/// - `bias`:   optional `[out_channels]`.
///
/// Output length: `(in_len - 1) * stride - 2*padding + dilation*(kernel_size-1) + 1`
pub fn conv1d_transposed(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv1dConfig,
    batch_size: usize,
    in_len: usize,
) -> Result<Vec<f32>> {
    config.validate()?;

    let ic_per_group = config.in_channels / config.groups;
    let oc_per_group = config.out_channels / config.groups;

    // Transposed conv output length.
    let effective_kernel = config.dilation * (config.kernel_size - 1) + 1;
    let out_len = if in_len == 0 {
        0
    } else {
        (in_len - 1) * config.stride + effective_kernel - 2 * config.padding
    };
    if out_len == 0 {
        return Err(invalid_args("transposed output length is zero"));
    }

    let expected_input = batch_size * config.in_channels * in_len;
    if input.len() != expected_input {
        return Err(invalid_args(&format!(
            "input length {} != expected {expected_input}",
            input.len(),
        )));
    }
    // Weight: [in_channels, oc_per_group, kernel_size]
    let expected_weight = config.in_channels * oc_per_group * config.kernel_size;
    if weight.len() != expected_weight {
        return Err(invalid_args(&format!(
            "weight length {} != expected {expected_weight}",
            weight.len(),
        )));
    }
    if let Some(b) = bias
        && b.len() != config.out_channels
    {
        return Err(invalid_args(&format!(
            "bias length {} != out_channels {}",
            b.len(),
            config.out_channels,
        )));
    }

    let mut output = vec![0.0f32; batch_size * config.out_channels * out_len];

    // Add bias first.
    if let Some(b) = bias {
        for n in 0..batch_size {
            for (oc, &bias_val) in b.iter().enumerate() {
                let base = (n * config.out_channels + oc) * out_len;
                for o in 0..out_len {
                    output[base + o] = bias_val;
                }
            }
        }
    }

    for n in 0..batch_size {
        for g in 0..config.groups {
            for ic in 0..ic_per_group {
                let abs_ic = g * ic_per_group + ic;
                for i in 0..in_len {
                    let in_val = input[(n * config.in_channels + abs_ic) * in_len + i];
                    for oc in 0..oc_per_group {
                        let abs_oc = g * oc_per_group + oc;
                        for k in 0..config.kernel_size {
                            let o_pos = i * config.stride + k * config.dilation;
                            if o_pos >= config.padding && (o_pos - config.padding) < out_len {
                                let o_idx = o_pos - config.padding;
                                let w_idx = (abs_ic * oc_per_group + oc) * config.kernel_size + k;
                                let out_idx = (n * config.out_channels + abs_oc) * out_len + o_idx;
                                output[out_idx] += in_val * weight[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

// ── Grouped 1D convolution ────────────────────────────────────────────

/// Grouped 1D convolution — dispatches to the fastest available path.
///
/// Identical semantics to [`conv1d_f32`] but auto-selects:
/// - depthwise path when `groups == in_channels == out_channels`
/// - AVX2 im2col path when AVX2 is available
/// - scalar fallback otherwise
pub fn conv1d_grouped(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv1dConfig,
    batch_size: usize,
    in_len: usize,
) -> Result<Vec<f32>> {
    if config.groups == config.in_channels && config.in_channels == config.out_channels {
        return conv1d_depthwise(input, weight, bias, config, batch_size, in_len);
    }
    if has_avx2() {
        return conv1d_avx2(input, weight, bias, config, batch_size, in_len);
    }
    conv1d_f32(input, weight, bias, config, batch_size, in_len)
}

// ── Auto-dispatch entry point ─────────────────────────────────────────

/// Runtime-dispatched 1D convolution — auto-selects AVX2 vs scalar.
///
/// This is the recommended entry point. It selects:
/// 1. Depthwise path when `groups == in_channels == out_channels`
/// 2. AVX2 im2col+GEMM when AVX2/FMA available
/// 3. Scalar fallback otherwise
pub fn conv1d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv1dConfig,
    batch_size: usize,
    in_len: usize,
) -> Result<Vec<f32>> {
    conv1d_grouped(input, weight, bias, config, batch_size, in_len)
}

// ── Validation helper ─────────────────────────────────────────────────

fn validate_conv1d_buffers(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv1dConfig,
    batch_size: usize,
    in_len: usize,
) -> Result<()> {
    let ic_per_group = config.in_channels / config.groups;
    let expected_input = batch_size * config.in_channels * in_len;
    if input.len() != expected_input {
        return Err(invalid_args(&format!(
            "input length {} != expected {} (batch={batch_size}, C={}, L={in_len})",
            input.len(),
            expected_input,
            config.in_channels,
        )));
    }
    let expected_weight = config.out_channels * ic_per_group * config.kernel_size;
    if weight.len() != expected_weight {
        return Err(invalid_args(&format!(
            "weight length {} != expected {expected_weight}",
            weight.len(),
        )));
    }
    if let Some(b) = bias
        && b.len() != config.out_channels
    {
        return Err(invalid_args(&format!(
            "bias length {} != out_channels {}",
            b.len(),
            config.out_channels,
        )));
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn approx_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= TOL)
    }

    /// Naive reference 1D convolution for verification.
    fn naive_conv1d(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        config: &Conv1dConfig,
        batch_size: usize,
        in_len: usize,
    ) -> Vec<f32> {
        let out_len = compute_output_length(
            in_len,
            config.kernel_size,
            config.stride,
            config.padding,
            config.dilation,
        );
        let ic_per_group = config.in_channels / config.groups;
        let oc_per_group = config.out_channels / config.groups;
        let mut output = vec![0.0f32; batch_size * config.out_channels * out_len];

        for n in 0..batch_size {
            for g in 0..config.groups {
                for oc in 0..oc_per_group {
                    let abs_oc = g * oc_per_group + oc;
                    let bias_val = bias.map_or(0.0, |b| b[abs_oc]);
                    for o in 0..out_len {
                        let mut sum = bias_val;
                        for ic in 0..ic_per_group {
                            let abs_ic = g * ic_per_group + ic;
                            for k in 0..config.kernel_size {
                                let i_pos = (o * config.stride + k * config.dilation) as isize
                                    - config.padding as isize;
                                if i_pos >= 0 && (i_pos as usize) < in_len {
                                    let in_idx =
                                        (n * config.in_channels + abs_ic) * in_len + i_pos as usize;
                                    let w_idx =
                                        (abs_oc * ic_per_group + ic) * config.kernel_size + k;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                        let out_idx = (n * config.out_channels + abs_oc) * out_len + o;
                        output[out_idx] = sum;
                    }
                }
            }
        }
        output
    }

    // ── compute_output_length ─────────────────────────────────

    #[test]
    fn output_length_no_padding() {
        assert_eq!(compute_output_length(10, 3, 1, 0, 1), 8);
    }

    #[test]
    fn output_length_with_padding() {
        assert_eq!(compute_output_length(10, 3, 1, 1, 1), 10);
    }

    #[test]
    fn output_length_with_stride() {
        assert_eq!(compute_output_length(10, 3, 2, 0, 1), 4);
    }

    #[test]
    fn output_length_with_dilation() {
        // effective kernel = 1 + 2*(3-1) = 5
        assert_eq!(compute_output_length(10, 3, 1, 0, 2), 6);
    }

    #[test]
    fn output_length_kernel_larger_than_input() {
        assert_eq!(compute_output_length(2, 5, 1, 0, 1), 0);
    }

    #[test]
    fn output_length_same_padding() {
        assert_eq!(compute_output_length(8, 3, 1, 1, 1), 8);
    }

    #[test]
    fn output_length_kernel_one() {
        assert_eq!(compute_output_length(10, 1, 1, 0, 1), 10);
    }

    #[test]
    fn output_length_stride_equals_kernel() {
        assert_eq!(compute_output_length(9, 3, 3, 0, 1), 3);
    }

    // ── Config ────────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let c = Conv1dConfig::default();
        assert_eq!(c.in_channels, 1);
        assert_eq!(c.out_channels, 1);
        assert_eq!(c.kernel_size, 1);
        assert_eq!(c.stride, 1);
        assert_eq!(c.padding, 0);
        assert_eq!(c.dilation, 1);
        assert_eq!(c.groups, 1);
    }

    #[test]
    fn config_new_basic() {
        let c = Conv1dConfig::new(4, 8, 3);
        assert_eq!(c.in_channels, 4);
        assert_eq!(c.out_channels, 8);
        assert_eq!(c.kernel_size, 3);
    }

    #[test]
    fn config_validate_zero_in_channels() {
        let c = Conv1dConfig::new(0, 1, 3);
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_zero_out_channels() {
        let c = Conv1dConfig::new(1, 0, 3);
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_zero_kernel() {
        let c = Conv1dConfig::new(1, 1, 0);
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_zero_stride() {
        let mut c = Conv1dConfig::new(1, 1, 3);
        c.stride = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_zero_dilation() {
        let mut c = Conv1dConfig::new(1, 1, 3);
        c.dilation = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_zero_groups() {
        let mut c = Conv1dConfig::new(4, 4, 3);
        c.groups = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_groups_not_divisor_of_in() {
        let mut c = Conv1dConfig::new(3, 6, 3);
        c.groups = 2;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_groups_not_divisor_of_out() {
        let mut c = Conv1dConfig::new(4, 6, 3);
        c.groups = 4;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_ok() {
        let mut c = Conv1dConfig::new(4, 8, 3);
        c.groups = 2;
        assert!(c.validate().is_ok());
    }

    // ── Padding modes ─────────────────────────────────────────

    #[test]
    fn padding_zero() {
        let input = [1.0, 2.0, 3.0];
        let padded = apply_padding(&input, 2, PaddingMode::Zero);
        assert_eq!(padded, vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn padding_reflect() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let padded = apply_padding(&input, 2, PaddingMode::Reflect);
        assert_eq!(padded[0..2], [3.0, 2.0]); // reflected left
        assert_eq!(padded[2..6], [1.0, 2.0, 3.0, 4.0]); // centre
        assert_eq!(padded[6..8], [3.0, 2.0]); // reflected right
    }

    #[test]
    fn padding_replicate() {
        let input = [1.0, 2.0, 3.0];
        let padded = apply_padding(&input, 2, PaddingMode::Replicate);
        assert_eq!(padded, vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn padding_circular() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let padded = apply_padding(&input, 2, PaddingMode::Circular);
        assert_eq!(padded[0..2], [3.0, 4.0]); // wrapped left
        assert_eq!(padded[2..6], [1.0, 2.0, 3.0, 4.0]); // centre
        assert_eq!(padded[6..8], [1.0, 2.0]); // wrapped right
    }

    #[test]
    fn padding_zero_amount() {
        let input = [1.0, 2.0, 3.0];
        let padded = apply_padding(&input, 0, PaddingMode::Reflect);
        assert_eq!(padded, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn padding_empty_input() {
        let input: [f32; 0] = [];
        let padded = apply_padding(&input, 2, PaddingMode::Zero);
        assert!(padded.is_empty());
    }

    // ── Basic conv1d_f32 ──────────────────────────────────────

    #[test]
    fn conv1d_f32_identity_kernel() {
        // kernel_size=1, should be pass-through scaled by weight.
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![2.0];
        let config = Conv1dConfig::new(1, 1, 1);
        let out = conv1d_f32(&input, &weight, None, &config, 1, 4).unwrap();
        assert!(approx_eq(&out, &[2.0, 4.0, 6.0, 8.0]));
    }

    #[test]
    fn conv1d_f32_kernel3_no_padding() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0, 1.0, 1.0];
        let config = Conv1dConfig::new(1, 1, 3);
        let out = conv1d_f32(&input, &weight, None, &config, 1, 5).unwrap();
        // [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
        assert!(approx_eq(&out, &[6.0, 9.0, 12.0]));
    }

    #[test]
    fn conv1d_f32_kernel3_with_padding() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 1.0, 1.0];
        let mut config = Conv1dConfig::new(1, 1, 3);
        config.padding = 1;
        let out = conv1d_f32(&input, &weight, None, &config, 1, 3).unwrap();
        // [0+1+2, 1+2+3, 2+3+0] = [3, 6, 5]
        assert!(approx_eq(&out, &[3.0, 6.0, 5.0]));
    }

    #[test]
    fn conv1d_f32_with_bias() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 0.0, 0.0];
        let bias = vec![10.0];
        let config = Conv1dConfig::new(1, 1, 3);
        let out = conv1d_f32(&input, &weight, Some(&bias), &config, 1, 3).unwrap();
        assert!(approx_eq(&out, &[11.0]));
    }

    #[test]
    fn conv1d_f32_kernel5() {
        let input: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let weight = vec![1.0; 5];
        let config = Conv1dConfig::new(1, 1, 5);
        let out = conv1d_f32(&input, &weight, None, &config, 1, 10).unwrap();
        // Window sums: [15, 20, 25, 30, 35, 40]
        assert!(approx_eq(&out, &[15.0, 20.0, 25.0, 30.0, 35.0, 40.0]));
    }

    #[test]
    fn conv1d_f32_kernel7() {
        let input: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let weight = vec![1.0; 7];
        let config = Conv1dConfig::new(1, 1, 7);
        let out = conv1d_f32(&input, &weight, None, &config, 1, 10).unwrap();
        // [28, 35, 42, 49]
        assert!(approx_eq(&out, &[28.0, 35.0, 42.0, 49.0]));
    }

    #[test]
    fn conv1d_f32_stride2() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let weight = vec![1.0, 1.0, 1.0];
        let mut config = Conv1dConfig::new(1, 1, 3);
        config.stride = 2;
        let out = conv1d_f32(&input, &weight, None, &config, 1, 7).unwrap();
        // positions: 0→[1+2+3]=6, 2→[3+4+5]=12, 4→[5+6+7]=18
        assert!(approx_eq(&out, &[6.0, 12.0, 18.0]));
    }

    #[test]
    fn conv1d_f32_dilation2() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0, 1.0, 1.0];
        let mut config = Conv1dConfig::new(1, 1, 3);
        config.dilation = 2;
        // effective kernel = 5, out_len = 1
        let out = conv1d_f32(&input, &weight, None, &config, 1, 5).unwrap();
        // [1+3+5] = 9
        assert!(approx_eq(&out, &[9.0]));
    }

    #[test]
    fn conv1d_f32_multi_channel() {
        // 2 input channels, 1 output channel, kernel=2
        let input = vec![
            1.0, 2.0, 3.0, // ch0
            4.0, 5.0, 6.0, // ch1
        ];
        let weight = vec![
            1.0, 1.0, // oc0, ic0
            1.0, 1.0, // oc0, ic1
        ];
        let config = Conv1dConfig::new(2, 1, 2);
        let out = conv1d_f32(&input, &weight, None, &config, 1, 3).unwrap();
        // o[0] = (1+2) + (4+5) = 12
        // o[1] = (2+3) + (5+6) = 16
        assert!(approx_eq(&out, &[12.0, 16.0]));
    }

    #[test]
    fn conv1d_f32_multi_output_channel() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![
            1.0, 1.0, 1.0, // oc0
            2.0, 2.0, 2.0, // oc1
        ];
        let config = Conv1dConfig::new(1, 2, 3);
        let out = conv1d_f32(&input, &weight, None, &config, 1, 3).unwrap();
        // oc0: 1+2+3=6, oc1: 2+4+6=12
        assert!(approx_eq(&out, &[6.0, 12.0]));
    }

    #[test]
    fn conv1d_f32_batch() {
        let input = vec![
            1.0, 2.0, 3.0, // batch 0
            4.0, 5.0, 6.0, // batch 1
        ];
        let weight = vec![1.0, 1.0, 1.0];
        let config = Conv1dConfig::new(1, 1, 3);
        let out = conv1d_f32(&input, &weight, None, &config, 2, 3).unwrap();
        assert!(approx_eq(&out, &[6.0, 15.0]));
    }

    #[test]
    fn conv1d_f32_vs_naive() {
        // Multi-channel, multi-output, batched, with padding and stride.
        let input: Vec<f32> = (0..2 * 4 * 8).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..6 * 2 * 3).map(|i| (i as f32) * 0.05 - 0.5).collect();
        let bias: Vec<f32> = (0..6).map(|i| i as f32 * 0.1).collect();
        let mut config = Conv1dConfig::new(4, 6, 3);
        config.stride = 2;
        config.padding = 1;
        config.groups = 2;

        let expected = naive_conv1d(&input, &weight, Some(&bias), &config, 2, 8);
        let actual = conv1d_f32(&input, &weight, Some(&bias), &config, 2, 8).unwrap();
        assert!(approx_eq(&expected, &actual), "conv1d_f32 vs naive mismatch");
    }

    #[test]
    fn conv1d_f32_no_bias() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![0.5, 0.5];
        let config = Conv1dConfig::new(1, 1, 2);
        let out = conv1d_f32(&input, &weight, None, &config, 1, 4).unwrap();
        assert!(approx_eq(&out, &[1.5, 2.5, 3.5]));
    }

    #[test]
    fn conv1d_f32_batch_size_one() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0];
        let config = Conv1dConfig::new(1, 1, 1);
        let out = conv1d_f32(&input, &weight, None, &config, 1, 3).unwrap();
        assert!(approx_eq(&out, &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn conv1d_f32_kernel_larger_than_input_errors() {
        let input = vec![1.0, 2.0];
        let weight = vec![1.0; 5];
        let config = Conv1dConfig::new(1, 1, 5);
        assert!(conv1d_f32(&input, &weight, None, &config, 1, 2).is_err());
    }

    // ── conv1d_avx2 (im2col + GEMM) ──────────────────────────

    #[test]
    fn conv1d_avx2_matches_scalar() {
        let input: Vec<f32> = (0..3 * 16).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..4 * 3 * 3).map(|i| (i as f32) * 0.05 - 0.5).collect();
        let bias: Vec<f32> = vec![0.1, -0.2, 0.3, 0.0];
        let mut config = Conv1dConfig::new(3, 4, 3);
        config.padding = 1;

        let scalar = conv1d_f32(&input, &weight, Some(&bias), &config, 1, 16).unwrap();
        let avx2 = conv1d_avx2(&input, &weight, Some(&bias), &config, 1, 16).unwrap();
        assert!(approx_eq(&scalar, &avx2), "avx2 vs scalar mismatch");
    }

    #[test]
    fn conv1d_avx2_batched() {
        let input: Vec<f32> = (0..2 * 2 * 10).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..3 * 2 * 3).map(|i| (i as f32) * 0.05).collect();
        let config = Conv1dConfig::new(2, 3, 3);

        let scalar = conv1d_f32(&input, &weight, None, &config, 2, 10).unwrap();
        let avx2 = conv1d_avx2(&input, &weight, None, &config, 2, 10).unwrap();
        assert!(approx_eq(&scalar, &avx2), "avx2 batched mismatch");
    }

    #[test]
    fn conv1d_avx2_grouped() {
        let input: Vec<f32> = (0..4 * 8).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..4 * 2 * 3).map(|i| (i as f32) * 0.05 - 0.3).collect();
        let mut config = Conv1dConfig::new(4, 4, 3);
        config.groups = 2;

        let scalar = conv1d_f32(&input, &weight, None, &config, 1, 8).unwrap();
        let avx2 = conv1d_avx2(&input, &weight, None, &config, 1, 8).unwrap();
        assert!(approx_eq(&scalar, &avx2), "avx2 grouped mismatch");
    }

    #[test]
    fn conv1d_avx2_stride_and_dilation() {
        let input: Vec<f32> = (0..2 * 20).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..3 * 2 * 3).map(|i| (i as f32) * 0.05).collect();
        let mut config = Conv1dConfig::new(2, 3, 3);
        config.stride = 2;
        config.dilation = 2;
        config.padding = 2;

        let scalar = conv1d_f32(&input, &weight, None, &config, 1, 20).unwrap();
        let avx2 = conv1d_avx2(&input, &weight, None, &config, 1, 20).unwrap();
        assert!(approx_eq(&scalar, &avx2), "avx2 stride+dilation mismatch");
    }

    #[test]
    fn conv1d_avx2_large_input() {
        // Ensure AVX2 vectorization has enough data to exercise SIMD lanes.
        let in_ch = 8;
        let out_ch = 16;
        let in_len = 64;
        let k = 5;
        let input: Vec<f32> = (0..in_ch * in_len).map(|i| (i as f32) * 0.01).collect();
        let weight: Vec<f32> = (0..out_ch * in_ch * k).map(|i| (i as f32) * 0.002 - 0.4).collect();
        let bias: Vec<f32> = (0..out_ch).map(|i| (i as f32) * 0.05).collect();
        let mut config = Conv1dConfig::new(in_ch, out_ch, k);
        config.padding = 2;

        let scalar = conv1d_f32(&input, &weight, Some(&bias), &config, 1, in_len).unwrap();
        let avx2 = conv1d_avx2(&input, &weight, Some(&bias), &config, 1, in_len).unwrap();
        assert!(approx_eq(&scalar, &avx2), "avx2 large input mismatch");
    }

    // ── Depthwise ─────────────────────────────────────────────

    #[test]
    fn depthwise_basic() {
        // 3 channels, kernel=3
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // ch0
            5.0, 6.0, 7.0, 8.0, // ch1
            9.0, 10.0, 11.0, 12.0, // ch2
        ];
        let weight = vec![
            1.0, 0.0, 0.0, // ch0
            0.0, 1.0, 0.0, // ch1
            0.0, 0.0, 1.0, // ch2
        ];
        let mut config = Conv1dConfig::new(3, 3, 3);
        config.groups = 3;
        let out = conv1d_depthwise(&input, &weight, None, &config, 1, 4).unwrap();
        // ch0: [1, 2], ch1: [6, 7], ch2: [11, 12]
        assert!(approx_eq(&out, &[1.0, 2.0, 6.0, 7.0, 11.0, 12.0]));
    }

    #[test]
    fn depthwise_with_bias() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 1.0, 1.0];
        let bias = vec![10.0];
        let mut config = Conv1dConfig::new(1, 1, 3);
        config.groups = 1;
        let out = conv1d_depthwise(&input, &weight, Some(&bias), &config, 1, 3).unwrap();
        assert!(approx_eq(&out, &[16.0]));
    }

    #[test]
    fn depthwise_matches_grouped_conv() {
        // Depthwise is a special case of grouped conv (groups=channels).
        let input: Vec<f32> = (0..4 * 10).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..4 * 3).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let bias: Vec<f32> = vec![0.1, -0.2, 0.3, 0.0];
        let mut config = Conv1dConfig::new(4, 4, 3);
        config.groups = 4;
        config.padding = 1;

        let depthwise = conv1d_depthwise(&input, &weight, Some(&bias), &config, 1, 10).unwrap();
        let grouped = conv1d_f32(&input, &weight, Some(&bias), &config, 1, 10).unwrap();
        assert!(approx_eq(&depthwise, &grouped), "depthwise vs grouped mismatch");
    }

    #[test]
    fn depthwise_rejects_non_depthwise() {
        let config = Conv1dConfig::new(4, 8, 3);
        let input = vec![0.0; 4 * 10];
        let weight = vec![0.0; 8 * 4 * 3];
        assert!(conv1d_depthwise(&input, &weight, None, &config, 1, 10).is_err());
    }

    #[test]
    fn depthwise_stride() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0, 1.0, 1.0];
        let mut config = Conv1dConfig::new(1, 1, 3);
        config.groups = 1;
        config.stride = 2;
        let out = conv1d_depthwise(&input, &weight, None, &config, 1, 5).unwrap();
        // positions: 0→[1+2+3]=6, 2→[3+4+5]=12
        assert!(approx_eq(&out, &[6.0, 12.0]));
    }

    #[test]
    fn depthwise_dilation() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0, 1.0, 1.0];
        let mut config = Conv1dConfig::new(1, 1, 3);
        config.groups = 1;
        config.dilation = 2;
        let out = conv1d_depthwise(&input, &weight, None, &config, 1, 5).unwrap();
        assert!(approx_eq(&out, &[9.0]));
    }

    // ── Pointwise ─────────────────────────────────────────────

    #[test]
    fn pointwise_basic() {
        let input = vec![
            1.0, 2.0, 3.0, // ch0
            4.0, 5.0, 6.0, // ch1
        ];
        let weight = vec![
            1.0, 1.0, // oc0: sum of channels
            1.0, -1.0, // oc1: difference
        ];
        let out = conv1d_pointwise(&input, &weight, None, 1, 2, 2, 3).unwrap();
        // oc0: [5, 7, 9], oc1: [-3, -3, -3]
        assert!(approx_eq(&out, &[5.0, 7.0, 9.0, -3.0, -3.0, -3.0]));
    }

    #[test]
    fn pointwise_with_bias() {
        let input = vec![1.0, 2.0];
        let weight = vec![2.0];
        let bias = vec![5.0];
        let out = conv1d_pointwise(&input, &weight, Some(&bias), 1, 1, 1, 2).unwrap();
        assert!(approx_eq(&out, &[7.0, 9.0]));
    }

    #[test]
    fn pointwise_matches_conv1d_k1() {
        // Pointwise should match conv1d with kernel_size=1.
        let input: Vec<f32> = (0..4 * 8).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..6 * 4).map(|i| (i as f32) * 0.05 - 0.5).collect();
        let bias: Vec<f32> = (0..6).map(|i| i as f32 * 0.1).collect();

        let config = Conv1dConfig::new(4, 6, 1);
        let conv_out = conv1d_f32(&input, &weight, Some(&bias), &config, 1, 8).unwrap();
        let pw_out = conv1d_pointwise(&input, &weight, Some(&bias), 1, 4, 6, 8).unwrap();
        assert!(approx_eq(&conv_out, &pw_out), "pointwise vs conv1d k=1 mismatch");
    }

    #[test]
    fn pointwise_batch() {
        let input = vec![
            1.0, 2.0, // batch0, ch0
            3.0, 4.0, // batch0, ch1
            5.0, 6.0, // batch1, ch0
            7.0, 8.0, // batch1, ch1
        ];
        let weight = vec![1.0, 1.0]; // sum channels
        let out = conv1d_pointwise(&input, &weight, None, 2, 2, 1, 2).unwrap();
        // batch0: [4, 6], batch1: [12, 14]
        assert!(approx_eq(&out, &[4.0, 6.0, 12.0, 14.0]));
    }

    // ── Transposed convolution ────────────────────────────────

    #[test]
    fn transposed_basic() {
        // 1 in-channel, 1 out-channel, kernel=3, stride=1, no padding.
        // Input: [2, 3], Weight: [1, 1, 1]
        // Output length = (2-1)*1 + 3 - 0 = 4
        let input = vec![2.0, 3.0];
        let weight = vec![1.0, 1.0, 1.0];
        let config = Conv1dConfig::new(1, 1, 3);
        let out = conv1d_transposed(&input, &weight, None, &config, 1, 2).unwrap();
        // out[0]=2, out[1]=2+3=5, out[2]=2+3=5, out[3]=3
        assert!(approx_eq(&out, &[2.0, 5.0, 5.0, 3.0]));
    }

    #[test]
    fn transposed_with_stride() {
        // stride=2 upsamples the signal.
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 1.0];
        let mut config = Conv1dConfig::new(1, 1, 2);
        config.stride = 2;
        // output length = (3-1)*2 + 2 - 0 = 6
        let out = conv1d_transposed(&input, &weight, None, &config, 1, 3).unwrap();
        // Scatter: pos0=1, pos1=1, pos2=2, pos3=2, pos4=3, pos5=3
        assert!(approx_eq(&out, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]));
    }

    #[test]
    fn transposed_with_bias() {
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 1.0];
        let bias = vec![10.0];
        let config = Conv1dConfig::new(1, 1, 2);
        let out = conv1d_transposed(&input, &weight, Some(&bias), &config, 1, 2).unwrap();
        // out_len = 3, bias fills: [10, 10, 10], then +[1, 1+2, 2] = [11, 13, 12]
        assert!(approx_eq(&out, &[11.0, 13.0, 12.0]));
    }

    #[test]
    fn transposed_multi_channel() {
        // in_ch=1, out_ch=2, kernel=2, stride=1.
        // Weight: [1, 2, 1, 2] → ic0→(oc0: [1,1], oc1: [2,2])
        // Wait, weight layout for transposed: [in_channels, oc_per_group, kernel_size]
        // = [1, 2, 2] flattened = [oc0_k0, oc0_k1, oc1_k0, oc1_k1]
        let input = vec![1.0, 2.0, 3.0]; // 1 channel, len=3
        let weight = vec![1.0, 1.0, 2.0, 2.0]; // ic0→oc0:[1,1], ic0→oc1:[2,2]
        let config = Conv1dConfig::new(1, 2, 2);
        // out_len = (3-1)*1 + 2 = 4
        let out = conv1d_transposed(&input, &weight, None, &config, 1, 3).unwrap();
        // oc0: [1, 1+2, 2+3, 3] = [1, 3, 5, 3]
        // oc1: [2, 2+4, 4+6, 6] = [2, 6, 10, 6]
        assert!(approx_eq(&out, &[1.0, 3.0, 5.0, 3.0, 2.0, 6.0, 10.0, 6.0]));
    }

    #[test]
    fn transposed_with_padding() {
        // Padding trims the output from both sides.
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 1.0, 1.0];
        let mut config = Conv1dConfig::new(1, 1, 3);
        config.padding = 1;
        // out_len = (3-1)*1 + 3 - 2 = 3
        let out = conv1d_transposed(&input, &weight, None, &config, 1, 3).unwrap();
        // Full (no pad): [1, 3, 6, 5, 3], trimmed by 1 each side: [3, 6, 5]
        assert!(approx_eq(&out, &[3.0, 6.0, 5.0]));
    }

    // ── Grouped convolution ───────────────────────────────────

    #[test]
    fn grouped_groups1_matches_ungrouped() {
        let input: Vec<f32> = (0..2 * 8).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..4 * 2 * 3).map(|i| (i as f32) * 0.05 - 0.3).collect();
        let bias: Vec<f32> = vec![0.1, -0.1, 0.2, 0.0];
        let config = Conv1dConfig::new(2, 4, 3);

        let ungrouped = conv1d_f32(&input, &weight, Some(&bias), &config, 1, 8).unwrap();
        let grouped = conv1d_grouped(&input, &weight, Some(&bias), &config, 1, 8).unwrap();
        assert!(approx_eq(&ungrouped, &grouped), "grouped(g=1) vs ungrouped mismatch");
    }

    #[test]
    fn grouped_groups2() {
        let input: Vec<f32> = (0..4 * 10).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..6 * 2 * 3).map(|i| (i as f32) * 0.05 - 0.4).collect();
        let mut config = Conv1dConfig::new(4, 6, 3);
        config.groups = 2;

        let reference = conv1d_f32(&input, &weight, None, &config, 1, 10).unwrap();
        let grouped = conv1d_grouped(&input, &weight, None, &config, 1, 10).unwrap();
        assert!(approx_eq(&reference, &grouped), "grouped(g=2) mismatch");
    }

    #[test]
    fn grouped_depthwise_dispatch() {
        // When groups == in_channels == out_channels, grouped should use depthwise.
        let input: Vec<f32> = (0..4 * 8).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..4 * 3).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let mut config = Conv1dConfig::new(4, 4, 3);
        config.groups = 4;

        let depthwise = conv1d_depthwise(&input, &weight, None, &config, 1, 8).unwrap();
        let grouped = conv1d_grouped(&input, &weight, None, &config, 1, 8).unwrap();
        assert!(approx_eq(&depthwise, &grouped), "grouped depthwise dispatch mismatch");
    }

    // ── Auto-dispatch (conv1d) ────────────────────────────────

    #[test]
    fn conv1d_dispatch_matches_scalar() {
        let input: Vec<f32> = (0..3 * 12).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..4 * 3 * 3).map(|i| (i as f32) * 0.02 - 0.3).collect();
        let config = Conv1dConfig::new(3, 4, 3);

        let scalar = conv1d_f32(&input, &weight, None, &config, 1, 12).unwrap();
        let dispatch = conv1d(&input, &weight, None, &config, 1, 12).unwrap();
        assert!(approx_eq(&scalar, &dispatch), "dispatch vs scalar mismatch");
    }

    // ── im2col / col2im ───────────────────────────────────────

    #[test]
    fn im2col_basic() {
        // 1 channel, in_len=5, kernel=3, no padding.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = Conv1dConfig::new(1, 1, 3);
        let cols = im2col(&input, &config, 5, 0).unwrap();
        // col_h = 3, col_w = 3 (output positions)
        // row0: [1, 2, 3]
        // row1: [2, 3, 4]
        // row2: [3, 4, 5]
        assert_eq!(cols.len(), 9);
        assert!(approx_eq(&cols, &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]));
    }

    #[test]
    fn im2col_with_padding() {
        let input = vec![1.0, 2.0, 3.0];
        let mut config = Conv1dConfig::new(1, 1, 3);
        config.padding = 1;
        let cols = im2col(&input, &config, 3, 0).unwrap();
        // out_len=3, col_h=3
        // row0: [0, 1, 2], row1: [1, 2, 3], row2: [2, 3, 0]
        assert!(approx_eq(&cols, &[0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 0.0]));
    }

    #[test]
    fn im2col_multi_channel() {
        let input = vec![
            1.0, 2.0, 3.0, // ch0
            4.0, 5.0, 6.0, // ch1
        ];
        let config = Conv1dConfig::new(2, 1, 2);
        let cols = im2col(&input, &config, 3, 0).unwrap();
        // col_h = 2*2 = 4, col_w = 2
        // ic0_k0: [1, 2], ic0_k1: [2, 3], ic1_k0: [4, 5], ic1_k1: [5, 6]
        assert!(approx_eq(&cols, &[1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0]));
    }

    #[test]
    fn col2im_roundtrip() {
        // im2col then col2im should accumulate overlapping patches.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = Conv1dConfig::new(1, 1, 3);
        let cols = im2col(&input, &config, 5, 0).unwrap();
        let out_len = compute_output_length(5, 3, 1, 0, 1);
        let reconstructed = col2im(&cols, &config, 5, out_len, 0).unwrap();
        // Each position is accumulated: [1, 2+2, 3+3+3, 4+4, 5] = [1, 4, 9, 8, 5]
        assert!(approx_eq(&reconstructed, &[1.0, 4.0, 9.0, 8.0, 5.0]));
    }

    #[test]
    fn col2im_validates_length() {
        let config = Conv1dConfig::new(1, 1, 3);
        let bad_cols = vec![0.0; 5]; // wrong length
        assert!(col2im(&bad_cols, &config, 5, 3, 0).is_err());
    }

    // ── Edge cases ────────────────────────────────────────────

    #[test]
    fn conv1d_f32_all_zeros() {
        let input = vec![0.0; 10];
        let weight = vec![1.0, 2.0, 3.0];
        let config = Conv1dConfig::new(1, 1, 3);
        let out = conv1d_f32(&input, &weight, None, &config, 1, 10).unwrap();
        assert!(out.iter().all(|&x| x.abs() < TOL));
    }

    #[test]
    fn conv1d_f32_negative_weights() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![-1.0, -1.0, -1.0];
        let config = Conv1dConfig::new(1, 1, 3);
        let out = conv1d_f32(&input, &weight, None, &config, 1, 5).unwrap();
        assert!(approx_eq(&out, &[-6.0, -9.0, -12.0]));
    }

    #[test]
    fn conv1d_f32_wrong_input_length() {
        let input = vec![1.0, 2.0]; // too short
        let weight = vec![1.0, 1.0, 1.0];
        let config = Conv1dConfig::new(1, 1, 3);
        assert!(conv1d_f32(&input, &weight, None, &config, 1, 5).is_err());
    }

    #[test]
    fn conv1d_f32_wrong_weight_length() {
        let input = vec![1.0; 5];
        let weight = vec![1.0, 2.0]; // wrong
        let config = Conv1dConfig::new(1, 1, 3);
        assert!(conv1d_f32(&input, &weight, None, &config, 1, 5).is_err());
    }

    #[test]
    fn conv1d_f32_wrong_bias_length() {
        let input = vec![1.0; 5];
        let weight = vec![1.0, 1.0, 1.0];
        let bias = vec![1.0, 2.0]; // too many for out_channels=1
        let config = Conv1dConfig::new(1, 1, 3);
        assert!(conv1d_f32(&input, &weight, Some(&bias), &config, 1, 5).is_err());
    }

    #[test]
    fn conv1d_f32_single_element() {
        let input = vec![5.0];
        let weight = vec![3.0];
        let config = Conv1dConfig::new(1, 1, 1);
        let out = conv1d_f32(&input, &weight, None, &config, 1, 1).unwrap();
        assert!(approx_eq(&out, &[15.0]));
    }

    #[test]
    fn conv1d_stride3() {
        let input: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let weight = vec![1.0, 1.0, 1.0];
        let mut config = Conv1dConfig::new(1, 1, 3);
        config.stride = 3;
        let out = conv1d_f32(&input, &weight, None, &config, 1, 12).unwrap();
        // positions: 0→6, 3→15, 6→24, 9→33
        assert!(approx_eq(&out, &[6.0, 15.0, 24.0, 33.0]));
    }

    #[test]
    fn conv1d_large_dilation() {
        // dilation=3, kernel=3 → effective kernel = 7
        let input: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let weight = vec![1.0, 1.0, 1.0];
        let mut config = Conv1dConfig::new(1, 1, 3);
        config.dilation = 3;
        // out_len = (10 - 7) / 1 + 1 = 4
        let out = conv1d_f32(&input, &weight, None, &config, 1, 10).unwrap();
        // [1+4+7, 2+5+8, 3+6+9, 4+7+10] = [12, 15, 18, 21]
        assert!(approx_eq(&out, &[12.0, 15.0, 18.0, 21.0]));
    }

    #[test]
    fn pointwise_zero_dim_errors() {
        assert!(conv1d_pointwise(&[], &[], None, 1, 0, 1, 1).is_err());
        assert!(conv1d_pointwise(&[], &[], None, 1, 1, 0, 1).is_err());
        assert!(conv1d_pointwise(&[], &[], None, 1, 1, 1, 0).is_err());
    }

    #[test]
    fn transposed_conv_then_forward_approx() {
        // For valid (no padding) conv: conv_transpose(conv(x)) preserves structure
        // at the centre. We verify shapes and that the round-trip is non-degenerate.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight_fwd = vec![1.0, 0.5, 0.25]; // [oc=1, ic=1, k=3]
        let config_fwd = Conv1dConfig::new(1, 1, 3);

        let fwd = conv1d_f32(&input, &weight_fwd, None, &config_fwd, 1, 5).unwrap();
        assert_eq!(fwd.len(), 3); // out_len = 3

        // Transposed with same kernel recovers a 5-element signal.
        let weight_t = vec![1.0, 0.5, 0.25]; // [ic=1, oc=1, k=3]
        let config_t = Conv1dConfig::new(1, 1, 3);
        let back = conv1d_transposed(&fwd, &weight_t, None, &config_t, 1, 3).unwrap();
        assert_eq!(back.len(), 5);
        // Verify non-trivial output.
        assert!(back.iter().any(|&x| x.abs() > TOL));
    }

    #[test]
    fn im2col_validates_input_length() {
        let config = Conv1dConfig::new(2, 1, 3);
        let bad_input = vec![0.0; 5]; // expected 2*3=6
        assert!(im2col(&bad_input, &config, 3, 0).is_err());
    }

    #[test]
    fn im2col_validates_group_index() {
        let config = Conv1dConfig::new(2, 1, 3);
        let input = vec![0.0; 6];
        assert!(im2col(&input, &config, 3, 1).is_err()); // groups=1, index=1 invalid
    }

    #[test]
    fn conv1d_avx2_kernel1() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![2.0];
        let config = Conv1dConfig::new(1, 1, 1);
        let out = conv1d_avx2(&input, &weight, None, &config, 1, 4).unwrap();
        assert!(approx_eq(&out, &[2.0, 4.0, 6.0, 8.0]));
    }

    #[test]
    fn conv1d_grouped_with_padding_stride_dilation() {
        let input: Vec<f32> = (0..6 * 16).map(|i| (i as f32) * 0.05).collect();
        let weight: Vec<f32> = (0..9 * 2 * 5).map(|i| (i as f32) * 0.01 - 0.2).collect();
        let bias: Vec<f32> = (0..9).map(|i| (i as f32) * 0.1).collect();
        let mut config = Conv1dConfig::new(6, 9, 5);
        config.groups = 3;
        config.stride = 2;
        config.padding = 2;
        config.dilation = 1;

        let reference = conv1d_f32(&input, &weight, Some(&bias), &config, 1, 16).unwrap();
        let grouped = conv1d_grouped(&input, &weight, Some(&bias), &config, 1, 16).unwrap();
        assert!(approx_eq(&reference, &grouped), "grouped complex mismatch");
    }
}
