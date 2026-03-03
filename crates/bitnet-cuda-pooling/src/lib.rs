//! CUDA pooling kernels for `BitNet` inference.
//!
//! Provides max, average, global average, and adaptive pooling operations
//! in 1D and 2D variants with configurable stride, padding, and dilation.
//!
//! All kernel functions accept a pooling config that describes the window
//! geometry and return a newly-allocated output buffer.  When compiled
//! without the `gpu`/`cuda` feature the implementations run on the CPU so
//! that the public API can still be tested and used as a reference.

use std::fmt;

// Re-export the error type from bitnet-common for downstream consumers.
pub use bitnet_common::BitNetError;

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Padding mode applied before the pooling window slides over the input.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    /// No padding -- the output shrinks according to kernel/stride/dilation.
    #[default]
    Valid,
    /// Zero-pad so the output spatial size equals `ceil(input_size / stride)`.
    Same,
    /// Explicit per-side padding amounts.
    Explicit(usize),
}

impl fmt::Display for PaddingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Valid => write!(f, "valid"),
            Self::Same => write!(f, "same"),
            Self::Explicit(p) => write!(f, "explicit({p})"),
        }
    }
}

/// Configuration for a 1-D pooling operation.
#[derive(Debug, Clone, Copy)]
pub struct PoolingConfig1D {
    /// Width of the pooling window.
    pub kernel_size: usize,
    /// Step between successive window positions.
    pub stride: usize,
    /// Padding mode.
    pub padding: PaddingMode,
    /// Dilation (spacing between kernel elements).
    pub dilation: usize,
}

impl PoolingConfig1D {
    /// Create a config with the given kernel size and stride, no padding, dilation 1.
    #[must_use]
    pub const fn new(kernel_size: usize, stride: usize) -> Self {
        Self { kernel_size, stride, padding: PaddingMode::Valid, dilation: 1 }
    }
}

impl fmt::Display for PoolingConfig1D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pool1D(k={}, s={}, pad={}, d={})",
            self.kernel_size, self.stride, self.padding, self.dilation
        )
    }
}

/// Configuration for a 2-D pooling operation.
#[derive(Debug, Clone, Copy)]
pub struct PoolingConfig2D {
    /// Kernel height and width.
    pub kernel_size: [usize; 2],
    /// Stride along height and width.
    pub stride: [usize; 2],
    /// Padding mode.
    pub padding: PaddingMode,
    /// Dilation along height and width.
    pub dilation: [usize; 2],
}

impl PoolingConfig2D {
    /// Create a config with the given kernel size and stride, no padding, dilation 1.
    #[must_use]
    pub const fn new(kernel_size: [usize; 2], stride: [usize; 2]) -> Self {
        Self { kernel_size, stride, padding: PaddingMode::Valid, dilation: [1, 1] }
    }
}

impl fmt::Display for PoolingConfig2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pool2D(k={:?}, s={:?}, pad={}, d={:?})",
            self.kernel_size, self.stride, self.padding, self.dilation
        )
    }
}

/// Configuration for adaptive 1-D pooling.
#[derive(Debug, Clone, Copy)]
pub struct AdaptivePoolingConfig1D {
    /// Desired output length.
    pub output_size: usize,
}

/// Configuration for adaptive 2-D pooling.
#[derive(Debug, Clone, Copy)]
pub struct AdaptivePoolingConfig2D {
    /// Desired output (height, width).
    pub output_size: [usize; 2],
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Effective kernel size accounting for dilation.
#[must_use]
const fn effective_kernel(kernel: usize, dilation: usize) -> usize {
    (kernel - 1) * dilation + 1
}

/// Total padding (both sides) for the `Same` mode.
#[must_use]
const fn total_pad_same(input: usize, stride: usize, eff_kernel: usize) -> usize {
    let out = input.div_ceil(stride);
    let needed = (out - 1) * stride + eff_kernel;
    needed.saturating_sub(input)
}

/// Total padding (both sides combined) given the padding mode.
///
/// For `Explicit(p)`, returns `2 * p` (symmetric).
#[must_use]
const fn resolve_total_padding(
    mode: PaddingMode,
    input: usize,
    stride: usize,
    eff_kernel: usize,
) -> usize {
    match mode {
        PaddingMode::Valid => 0,
        PaddingMode::Same => total_pad_same(input, stride, eff_kernel),
        PaddingMode::Explicit(p) => 2 * p,
    }
}

/// Compute the output length for a 1-D pooling dimension.
#[must_use]
pub const fn output_length(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
) -> usize {
    let eff = effective_kernel(kernel, dilation);
    let total_pad = resolve_total_padding(padding, input, stride, eff);
    let padded = input + total_pad;
    if padded < eff {
        return 0;
    }
    (padded - eff) / stride + 1
}

/// Safe indexing: returns `Some(idx)` when the signed position is in bounds.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::cast_possible_truncation)]
const fn checked_index(base: i64, offset: i64, limit: usize) -> Option<usize> {
    let pos = base + offset;
    if pos < 0 {
        return None;
    }
    let idx = pos as u64;
    if idx < limit as u64 { Some(idx as usize) } else { None }
}

// ---------------------------------------------------------------------------
// 1-D pooling kernels
// ---------------------------------------------------------------------------

/// 1-D max pooling.
#[allow(clippy::cast_possible_wrap)]
#[must_use]
pub fn max_pool_1d(input: &[f32], cfg: &PoolingConfig1D) -> Vec<f32> {
    let input_len = input.len();
    let eff = effective_kernel(cfg.kernel_size, cfg.dilation);
    let pad = resolve_total_padding(cfg.padding, input_len, cfg.stride, eff) / 2;
    let out_len = output_length(input_len, cfg.kernel_size, cfg.stride, cfg.padding, cfg.dilation);
    let mut output = Vec::with_capacity(out_len);
    for out_idx in 0..out_len {
        let base = (out_idx * cfg.stride) as i64 - pad as i64;
        let mut val = f32::NEG_INFINITY;
        for k_idx in 0..cfg.kernel_size {
            if let Some(src) = checked_index(base, (k_idx * cfg.dilation) as i64, input_len) {
                val = val.max(input[src]);
            }
        }
        output.push(val);
    }
    output
}

/// 1-D average pooling.
#[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
#[must_use]
pub fn avg_pool_1d(input: &[f32], cfg: &PoolingConfig1D) -> Vec<f32> {
    let input_len = input.len();
    let eff = effective_kernel(cfg.kernel_size, cfg.dilation);
    let pad = resolve_total_padding(cfg.padding, input_len, cfg.stride, eff) / 2;
    let out_len = output_length(input_len, cfg.kernel_size, cfg.stride, cfg.padding, cfg.dilation);
    let mut output = Vec::with_capacity(out_len);
    for out_idx in 0..out_len {
        let base = (out_idx * cfg.stride) as i64 - pad as i64;
        let mut sum = 0.0_f32;
        let mut count = 0_u32;
        for k_idx in 0..cfg.kernel_size {
            if let Some(src) = checked_index(base, (k_idx * cfg.dilation) as i64, input_len) {
                sum += input[src];
                count += 1;
            }
        }
        output.push(if count == 0 { 0.0 } else { sum / count as f32 });
    }
    output
}

// ---------------------------------------------------------------------------
// 2-D pooling kernels
// ---------------------------------------------------------------------------

/// 2-D max pooling (row-major layout: height * width).
#[allow(clippy::cast_possible_wrap)]
#[must_use]
pub fn max_pool_2d(input: &[f32], height: usize, width: usize, cfg: &PoolingConfig2D) -> Vec<f32> {
    let eff_h = effective_kernel(cfg.kernel_size[0], cfg.dilation[0]);
    let eff_w = effective_kernel(cfg.kernel_size[1], cfg.dilation[1]);
    let pad_h = resolve_total_padding(cfg.padding, height, cfg.stride[0], eff_h) / 2;
    let pad_w = resolve_total_padding(cfg.padding, width, cfg.stride[1], eff_w) / 2;
    let out_h =
        output_length(height, cfg.kernel_size[0], cfg.stride[0], cfg.padding, cfg.dilation[0]);
    let out_w =
        output_length(width, cfg.kernel_size[1], cfg.stride[1], cfg.padding, cfg.dilation[1]);
    let mut output = Vec::with_capacity(out_h * out_w);
    for row_idx in 0..out_h {
        for col_idx in 0..out_w {
            let base_h = (row_idx * cfg.stride[0]) as i64 - pad_h as i64;
            let base_w = (col_idx * cfg.stride[1]) as i64 - pad_w as i64;
            let mut val = f32::NEG_INFINITY;
            for kh in 0..cfg.kernel_size[0] {
                for kw in 0..cfg.kernel_size[1] {
                    if let Some(row) = checked_index(base_h, (kh * cfg.dilation[0]) as i64, height)
                        && let Some(col) =
                            checked_index(base_w, (kw * cfg.dilation[1]) as i64, width)
                    {
                        val = val.max(input[row * width + col]);
                    }
                }
            }
            output.push(val);
        }
    }
    output
}

/// 2-D average pooling (row-major layout).
#[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
#[must_use]
pub fn avg_pool_2d(input: &[f32], height: usize, width: usize, cfg: &PoolingConfig2D) -> Vec<f32> {
    let eff_h = effective_kernel(cfg.kernel_size[0], cfg.dilation[0]);
    let eff_w = effective_kernel(cfg.kernel_size[1], cfg.dilation[1]);
    let pad_h = resolve_total_padding(cfg.padding, height, cfg.stride[0], eff_h) / 2;
    let pad_w = resolve_total_padding(cfg.padding, width, cfg.stride[1], eff_w) / 2;
    let out_h =
        output_length(height, cfg.kernel_size[0], cfg.stride[0], cfg.padding, cfg.dilation[0]);
    let out_w =
        output_length(width, cfg.kernel_size[1], cfg.stride[1], cfg.padding, cfg.dilation[1]);
    let mut output = Vec::with_capacity(out_h * out_w);
    for row_idx in 0..out_h {
        for col_idx in 0..out_w {
            let base_h = (row_idx * cfg.stride[0]) as i64 - pad_h as i64;
            let base_w = (col_idx * cfg.stride[1]) as i64 - pad_w as i64;
            let mut sum = 0.0_f32;
            let mut count = 0_u32;
            for kh in 0..cfg.kernel_size[0] {
                for kw in 0..cfg.kernel_size[1] {
                    if let Some(row) = checked_index(base_h, (kh * cfg.dilation[0]) as i64, height)
                        && let Some(col) =
                            checked_index(base_w, (kw * cfg.dilation[1]) as i64, width)
                    {
                        sum += input[row * width + col];
                        count += 1;
                    }
                }
            }
            output.push(if count == 0 { 0.0 } else { sum / count as f32 });
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Global pooling
// ---------------------------------------------------------------------------

/// Global average pooling over a 1-D slice.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn global_avg_pool_1d(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }
    let sum: f32 = input.iter().sum();
    sum / input.len() as f32
}

/// Global average pooling over a 2-D plane (row-major).
#[must_use]
pub fn global_avg_pool_2d(input: &[f32], _height: usize, _width: usize) -> f32 {
    global_avg_pool_1d(input)
}

/// Batched global average pooling -- each contiguous chunk of `spatial_len`
/// elements is independently averaged.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn global_avg_pool_batched(input: &[f32], spatial_len: usize) -> Vec<f32> {
    if spatial_len == 0 {
        return Vec::new();
    }
    input
        .chunks(spatial_len)
        .map(|chunk| {
            let sum: f32 = chunk.iter().sum();
            sum / chunk.len() as f32
        })
        .collect()
}

/// Global max pooling over a 1-D slice.
#[must_use]
pub fn global_max_pool_1d(input: &[f32]) -> f32 {
    input.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Global max pooling over a 2-D plane (row-major).
#[must_use]
pub fn global_max_pool_2d(input: &[f32], _height: usize, _width: usize) -> f32 {
    global_max_pool_1d(input)
}

// ---------------------------------------------------------------------------
// Adaptive pooling
// ---------------------------------------------------------------------------

/// Adaptive average pooling 1-D.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn adaptive_avg_pool_1d(input: &[f32], cfg: &AdaptivePoolingConfig1D) -> Vec<f32> {
    let input_len = input.len();
    let out_len = cfg.output_size;
    if out_len == 0 {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(out_len);
    for out_idx in 0..out_len {
        let start = out_idx * input_len / out_len;
        let end = (out_idx + 1) * input_len / out_len;
        let mut sum = 0.0_f32;
        let mut count = 0_u32;
        for &elem in &input[start..end] {
            sum += elem;
            count += 1;
        }
        output.push(if count == 0 { 0.0 } else { sum / count as f32 });
    }
    output
}

/// Adaptive max pooling 1-D.
#[must_use]
pub fn adaptive_max_pool_1d(input: &[f32], cfg: &AdaptivePoolingConfig1D) -> Vec<f32> {
    let input_len = input.len();
    let out_len = cfg.output_size;
    if out_len == 0 {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(out_len);
    for out_idx in 0..out_len {
        let start = out_idx * input_len / out_len;
        let end = (out_idx + 1) * input_len / out_len;
        let mut val = f32::NEG_INFINITY;
        for &elem in &input[start..end] {
            val = val.max(elem);
        }
        output.push(val);
    }
    output
}

/// Adaptive average pooling 2-D (row-major).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn adaptive_avg_pool_2d(
    input: &[f32],
    height: usize,
    width: usize,
    cfg: &AdaptivePoolingConfig2D,
) -> Vec<f32> {
    let out_h = cfg.output_size[0];
    let out_w = cfg.output_size[1];
    if out_h == 0 || out_w == 0 {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(out_h * out_w);
    for row_out in 0..out_h {
        let row_start = row_out * height / out_h;
        let row_end = (row_out + 1) * height / out_h;
        for col_out in 0..out_w {
            let col_start = col_out * width / out_w;
            let col_end = (col_out + 1) * width / out_w;
            let mut sum = 0.0_f32;
            let mut count = 0_u32;
            for row in row_start..row_end {
                for col in col_start..col_end {
                    sum += input[row * width + col];
                    count += 1;
                }
            }
            output.push(if count == 0 { 0.0 } else { sum / count as f32 });
        }
    }
    output
}

/// Adaptive max pooling 2-D (row-major).
#[must_use]
pub fn adaptive_max_pool_2d(
    input: &[f32],
    height: usize,
    width: usize,
    cfg: &AdaptivePoolingConfig2D,
) -> Vec<f32> {
    let out_h = cfg.output_size[0];
    let out_w = cfg.output_size[1];
    if out_h == 0 || out_w == 0 {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(out_h * out_w);
    for row_out in 0..out_h {
        let row_start = row_out * height / out_h;
        let row_end = (row_out + 1) * height / out_h;
        for col_out in 0..out_w {
            let col_start = col_out * width / out_w;
            let col_end = (col_out + 1) * width / out_w;
            let mut val = f32::NEG_INFINITY;
            for row in row_start..row_end {
                for col in col_start..col_end {
                    val = val.max(input[row * width + col]);
                }
            }
            output.push(val);
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers / config defaults --

    #[test]
    fn padding_mode_default_is_valid() {
        assert_eq!(PaddingMode::default(), PaddingMode::Valid);
    }

    #[test]
    fn padding_mode_display() {
        assert_eq!(PaddingMode::Valid.to_string(), "valid");
        assert_eq!(PaddingMode::Same.to_string(), "same");
        assert_eq!(PaddingMode::Explicit(3).to_string(), "explicit(3)");
    }

    #[test]
    fn config_1d_display() {
        let cfg = PoolingConfig1D::new(3, 1);
        let display = cfg.to_string();
        assert!(display.contains("k=3"));
        assert!(display.contains("s=1"));
    }

    #[test]
    fn config_2d_display() {
        let cfg = PoolingConfig2D::new([2, 2], [1, 1]);
        let display = cfg.to_string();
        assert!(display.contains("[2, 2]"));
    }

    #[test]
    fn config_1d_defaults() {
        let cfg = PoolingConfig1D::new(3, 2);
        assert_eq!(cfg.kernel_size, 3);
        assert_eq!(cfg.stride, 2);
        assert_eq!(cfg.padding, PaddingMode::Valid);
        assert_eq!(cfg.dilation, 1);
    }

    #[test]
    fn config_2d_defaults() {
        let cfg = PoolingConfig2D::new([3, 3], [2, 2]);
        assert_eq!(cfg.kernel_size, [3, 3]);
        assert_eq!(cfg.stride, [2, 2]);
        assert_eq!(cfg.padding, PaddingMode::Valid);
        assert_eq!(cfg.dilation, [1, 1]);
    }

    // -- output_length --

    #[test]
    fn output_length_basic() {
        // input=10, kernel=3, stride=1, valid, dilation=1 => 8
        assert_eq!(output_length(10, 3, 1, PaddingMode::Valid, 1), 8);
    }

    #[test]
    fn output_length_stride() {
        // input=10, kernel=3, stride=2 => (10-3)/2+1 = 4
        assert_eq!(output_length(10, 3, 2, PaddingMode::Valid, 1), 4);
    }

    #[test]
    fn output_length_same_padding() {
        // Same padding keeps output = ceil(10/1) = 10
        assert_eq!(output_length(10, 3, 1, PaddingMode::Same, 1), 10);
    }

    #[test]
    fn output_length_same_padding_stride2() {
        assert_eq!(output_length(10, 3, 2, PaddingMode::Same, 1), 5);
    }

    #[test]
    fn output_length_explicit_padding() {
        // pad=1 on each side => padded=12, (12-3)/1+1=10
        assert_eq!(output_length(10, 3, 1, PaddingMode::Explicit(1), 1), 10);
    }

    #[test]
    fn output_length_dilation() {
        // eff_kernel = (3-1)*2+1 = 5, (10-5)/1+1 = 6
        assert_eq!(output_length(10, 3, 1, PaddingMode::Valid, 2), 6);
    }

    #[test]
    fn output_length_zero_when_too_small() {
        // input=2, kernel=5, stride=1 => padded=2 < eff=5 => 0
        assert_eq!(output_length(2, 5, 1, PaddingMode::Valid, 1), 0);
    }

    // -- effective_kernel --

    #[test]
    fn effective_kernel_no_dilation() {
        assert_eq!(effective_kernel(3, 1), 3);
    }

    #[test]
    fn effective_kernel_with_dilation() {
        assert_eq!(effective_kernel(3, 2), 5);
        assert_eq!(effective_kernel(5, 3), 13);
    }

    // -- checked_index --

    #[test]
    fn checked_index_in_bounds() {
        assert_eq!(checked_index(2, 3, 10), Some(5));
    }

    #[test]
    fn checked_index_negative() {
        assert_eq!(checked_index(-5, 2, 10), None);
    }

    #[test]
    fn checked_index_out_of_bounds() {
        assert_eq!(checked_index(8, 5, 10), None);
    }

    #[test]
    fn checked_index_zero() {
        assert_eq!(checked_index(0, 0, 1), Some(0));
    }

    // -- 1-D max pooling --

    #[test]
    fn max_pool_1d_basic() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let cfg = PoolingConfig1D::new(2, 1);
        let out = max_pool_1d(&input, &cfg);
        assert_eq!(out, vec![3.0, 3.0, 5.0, 5.0]);
    }

    #[test]
    fn max_pool_1d_stride2() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
        let cfg = PoolingConfig1D::new(2, 2);
        let out = max_pool_1d(&input, &cfg);
        assert_eq!(out, vec![3.0, 5.0, 6.0]);
    }

    #[test]
    fn max_pool_1d_same_padding() {
        let input = vec![1.0, 2.0, 3.0];
        let mut cfg = PoolingConfig1D::new(3, 1);
        cfg.padding = PaddingMode::Same;
        let out = max_pool_1d(&input, &cfg);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn max_pool_1d_dilation() {
        // input[0]=1, input[2]=3 => max=3; input[1]=2, input[3]=4 => max=4
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut cfg = PoolingConfig1D::new(2, 1);
        cfg.dilation = 2;
        let out = max_pool_1d(&input, &cfg);
        assert_eq!(out, vec![3.0, 4.0]);
    }

    #[test]
    fn max_pool_1d_empty() {
        let out = max_pool_1d(&[], &PoolingConfig1D::new(2, 1));
        assert!(out.is_empty());
    }

    #[test]
    fn max_pool_1d_single_element() {
        let input = vec![42.0];
        let cfg = PoolingConfig1D::new(1, 1);
        assert_eq!(max_pool_1d(&input, &cfg), vec![42.0]);
    }

    #[test]
    fn max_pool_1d_all_negative() {
        let input = vec![-3.0, -1.0, -4.0, -2.0];
        let cfg = PoolingConfig1D::new(2, 1);
        let out = max_pool_1d(&input, &cfg);
        assert_eq!(out, vec![-1.0, -1.0, -2.0]);
    }

    #[test]
    fn max_pool_1d_kernel_equals_input() {
        let input = vec![1.0, 5.0, 3.0];
        let cfg = PoolingConfig1D::new(3, 1);
        assert_eq!(max_pool_1d(&input, &cfg), vec![5.0]);
    }

    // -- 1-D avg pooling --

    #[test]
    fn avg_pool_1d_basic() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let cfg = PoolingConfig1D::new(2, 1);
        let out = avg_pool_1d(&input, &cfg);
        assert_eq!(out, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn avg_pool_1d_stride2() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let cfg = PoolingConfig1D::new(2, 2);
        let out = avg_pool_1d(&input, &cfg);
        assert_eq!(out, vec![3.0, 7.0]);
    }

    #[test]
    fn avg_pool_1d_same_padding() {
        let input = vec![1.0, 2.0, 3.0];
        let mut cfg = PoolingConfig1D::new(3, 1);
        cfg.padding = PaddingMode::Same;
        let out = avg_pool_1d(&input, &cfg);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn avg_pool_1d_empty() {
        let out = avg_pool_1d(&[], &PoolingConfig1D::new(2, 1));
        assert!(out.is_empty());
    }

    #[test]
    fn avg_pool_1d_kernel_equals_input() {
        let input = vec![10.0, 20.0, 30.0];
        let cfg = PoolingConfig1D::new(3, 1);
        let out = avg_pool_1d(&input, &cfg);
        assert_eq!(out, vec![20.0]);
    }

    // -- 2-D max pooling --

    #[test]
    fn max_pool_2d_basic() {
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let cfg = PoolingConfig2D::new([2, 2], [2, 2]);
        let out = max_pool_2d(&input, 4, 4, &cfg);
        assert_eq!(out, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn max_pool_2d_stride1() {
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let cfg = PoolingConfig2D::new([2, 2], [1, 1]);
        let out = max_pool_2d(&input, 3, 3, &cfg);
        assert_eq!(out, vec![5.0, 6.0, 8.0, 9.0]);
    }

    #[test]
    fn max_pool_2d_same_padding() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut cfg = PoolingConfig2D::new([2, 2], [1, 1]);
        cfg.padding = PaddingMode::Same;
        let out = max_pool_2d(&input, 2, 2, &cfg);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn max_pool_2d_empty() {
        let out = max_pool_2d(&[], 0, 0, &PoolingConfig2D::new([2, 2], [1, 1]));
        assert!(out.is_empty());
    }

    // -- 2-D avg pooling --

    #[test]
    fn avg_pool_2d_basic() {
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let cfg = PoolingConfig2D::new([2, 2], [2, 2]);
        let out = avg_pool_2d(&input, 4, 4, &cfg);
        assert_eq!(out, vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn avg_pool_2d_stride1() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = PoolingConfig2D::new([2, 2], [1, 1]);
        let out = avg_pool_2d(&input, 2, 2, &cfg);
        assert_eq!(out, vec![2.5]);
    }

    #[test]
    fn avg_pool_2d_empty() {
        let out = avg_pool_2d(&[], 0, 0, &PoolingConfig2D::new([2, 2], [1, 1]));
        assert!(out.is_empty());
    }

    // -- Global average pooling --

    #[test]
    fn global_avg_pool_1d_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        assert!((global_avg_pool_1d(&input) - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn global_avg_pool_1d_empty() {
        assert!((global_avg_pool_1d(&[]) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn global_avg_pool_1d_single() {
        assert!((global_avg_pool_1d(&[7.0]) - 7.0).abs() < f32::EPSILON);
    }

    #[test]
    fn global_avg_pool_2d_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        assert!((global_avg_pool_2d(&input, 2, 2) - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn global_avg_pool_batched_basic() {
        let input = vec![1.0, 3.0, 2.0, 4.0];
        let out = global_avg_pool_batched(&input, 2);
        assert_eq!(out, vec![2.0, 3.0]);
    }

    #[test]
    fn global_avg_pool_batched_empty_spatial() {
        let out = global_avg_pool_batched(&[1.0, 2.0], 0);
        assert!(out.is_empty());
    }

    #[test]
    fn global_avg_pool_batched_single_batch() {
        let input = vec![10.0, 20.0, 30.0];
        let out = global_avg_pool_batched(&input, 3);
        assert_eq!(out, vec![20.0]);
    }

    // -- Global max pooling --

    #[test]
    fn global_max_pool_1d_basic() {
        let input = vec![1.0, 5.0, 3.0, 2.0];
        assert!((global_max_pool_1d(&input) - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn global_max_pool_1d_all_negative() {
        let input = vec![-3.0, -1.0, -4.0];
        assert!((global_max_pool_1d(&input) - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn global_max_pool_2d_basic() {
        let input = vec![1.0, 9.0, 3.0, 4.0];
        assert!((global_max_pool_2d(&input, 2, 2) - 9.0).abs() < f32::EPSILON);
    }

    // -- Adaptive avg pooling 1-D --

    #[test]
    fn adaptive_avg_pool_1d_halve() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let cfg = AdaptivePoolingConfig1D { output_size: 2 };
        let out = adaptive_avg_pool_1d(&input, &cfg);
        assert_eq!(out, vec![3.0, 7.0]);
    }

    #[test]
    fn adaptive_avg_pool_1d_identity() {
        let input = vec![1.0, 2.0, 3.0];
        let cfg = AdaptivePoolingConfig1D { output_size: 3 };
        let out = adaptive_avg_pool_1d(&input, &cfg);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn adaptive_avg_pool_1d_single_output() {
        let input = vec![10.0, 20.0, 30.0, 40.0];
        let cfg = AdaptivePoolingConfig1D { output_size: 1 };
        let out = adaptive_avg_pool_1d(&input, &cfg);
        assert_eq!(out, vec![25.0]);
    }

    #[test]
    fn adaptive_avg_pool_1d_zero_output() {
        let input = vec![1.0, 2.0];
        let cfg = AdaptivePoolingConfig1D { output_size: 0 };
        assert!(adaptive_avg_pool_1d(&input, &cfg).is_empty());
    }

    // -- Adaptive max pooling 1-D --

    #[test]
    fn adaptive_max_pool_1d_halve() {
        let input = vec![1.0, 4.0, 2.0, 8.0];
        let cfg = AdaptivePoolingConfig1D { output_size: 2 };
        let out = adaptive_max_pool_1d(&input, &cfg);
        assert_eq!(out, vec![4.0, 8.0]);
    }

    #[test]
    fn adaptive_max_pool_1d_identity() {
        let input = vec![5.0, 3.0, 7.0];
        let cfg = AdaptivePoolingConfig1D { output_size: 3 };
        let out = adaptive_max_pool_1d(&input, &cfg);
        assert_eq!(out, vec![5.0, 3.0, 7.0]);
    }

    #[test]
    fn adaptive_max_pool_1d_single_output() {
        let input = vec![1.0, 9.0, 3.0];
        let cfg = AdaptivePoolingConfig1D { output_size: 1 };
        let out = adaptive_max_pool_1d(&input, &cfg);
        assert_eq!(out, vec![9.0]);
    }

    // -- Adaptive avg pooling 2-D --

    #[test]
    fn adaptive_avg_pool_2d_halve() {
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let cfg = AdaptivePoolingConfig2D { output_size: [2, 2] };
        let out = adaptive_avg_pool_2d(&input, 4, 4, &cfg);
        assert_eq!(out, vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn adaptive_avg_pool_2d_single() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = AdaptivePoolingConfig2D { output_size: [1, 1] };
        let out = adaptive_avg_pool_2d(&input, 2, 2, &cfg);
        assert_eq!(out, vec![2.5]);
    }

    #[test]
    fn adaptive_avg_pool_2d_zero_output() {
        let cfg = AdaptivePoolingConfig2D { output_size: [0, 0] };
        assert!(adaptive_avg_pool_2d(&[1.0], 1, 1, &cfg).is_empty());
    }

    // -- Adaptive max pooling 2-D --

    #[test]
    fn adaptive_max_pool_2d_halve() {
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let cfg = AdaptivePoolingConfig2D { output_size: [2, 2] };
        let out = adaptive_max_pool_2d(&input, 4, 4, &cfg);
        assert_eq!(out, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn adaptive_max_pool_2d_single() {
        let input = vec![1.0, 9.0, 3.0, 4.0];
        let cfg = AdaptivePoolingConfig2D { output_size: [1, 1] };
        let out = adaptive_max_pool_2d(&input, 2, 2, &cfg);
        assert_eq!(out, vec![9.0]);
    }

    // -- Determinism --

    #[test]
    fn max_pool_1d_deterministic() {
        let input: Vec<f32> = (0_i16..100).map(|idx| f32::from(idx).sin()).collect();
        let cfg = PoolingConfig1D::new(5, 2);
        let run_a = max_pool_1d(&input, &cfg);
        let run_b = max_pool_1d(&input, &cfg);
        assert_eq!(run_a, run_b);
    }

    #[test]
    fn avg_pool_1d_deterministic() {
        let input: Vec<f32> = (0_i16..100).map(|idx| f32::from(idx).sin()).collect();
        let cfg = PoolingConfig1D::new(5, 2);
        let run_a = avg_pool_1d(&input, &cfg);
        let run_b = avg_pool_1d(&input, &cfg);
        assert_eq!(run_a, run_b);
    }

    #[test]
    fn max_pool_2d_deterministic() {
        let input: Vec<f32> = (0_i16..100).map(|idx| f32::from(idx).sin()).collect();
        let cfg = PoolingConfig2D::new([3, 3], [1, 1]);
        let run_a = max_pool_2d(&input, 10, 10, &cfg);
        let run_b = max_pool_2d(&input, 10, 10, &cfg);
        assert_eq!(run_a, run_b);
    }

    #[test]
    fn avg_pool_2d_deterministic() {
        let input: Vec<f32> = (0_i16..100).map(|idx| f32::from(idx).sin()).collect();
        let cfg = PoolingConfig2D::new([3, 3], [1, 1]);
        let run_a = avg_pool_2d(&input, 10, 10, &cfg);
        let run_b = avg_pool_2d(&input, 10, 10, &cfg);
        assert_eq!(run_a, run_b);
    }

    // -- Stride + padding + dilation combos --

    #[test]
    fn max_pool_1d_explicit_padding() {
        let input = vec![1.0, 2.0, 3.0];
        let mut cfg = PoolingConfig1D::new(3, 1);
        cfg.padding = PaddingMode::Explicit(1);
        let out = max_pool_1d(&input, &cfg);
        // padded = [0,1,2,3,0], windows: [0,1,2]=2, [1,2,3]=3, [2,3,0]=3
        assert_eq!(out, vec![2.0, 3.0, 3.0]);
    }

    #[test]
    fn avg_pool_1d_explicit_padding() {
        let input = vec![2.0, 4.0, 6.0];
        let mut cfg = PoolingConfig1D::new(3, 1);
        cfg.padding = PaddingMode::Explicit(1);
        let out = avg_pool_1d(&input, &cfg);
        // Only in-bounds elements are counted (exclude pads).
        // Window [0,2,4] => in-bounds: [2,4] => avg=3, [2,4,6] => avg=4, [4,6,0] => in-bounds: [4,6] => avg=5
        assert_eq!(out, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn max_pool_1d_large_dilation() {
        let input = vec![1.0, 0.0, 0.0, 0.0, 5.0];
        let mut cfg = PoolingConfig1D::new(2, 1);
        cfg.dilation = 4;
        // eff_kernel = (2-1)*4+1 = 5, only one position fits
        let out = max_pool_1d(&input, &cfg);
        assert_eq!(out, vec![5.0]);
    }

    #[test]
    fn max_pool_2d_dilation() {
        #[rustfmt::skip]
        let input = vec![
            9.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let mut cfg = PoolingConfig2D::new([2, 2], [1, 1]);
        cfg.dilation = [2, 2];
        // eff=3x3, only (0,0) fits; samples (0,0),(0,2),(2,0),(2,2) = max(9,0,0,1) = 9
        let out = max_pool_2d(&input, 3, 3, &cfg);
        assert_eq!(out, vec![9.0]);
    }

    #[test]
    fn avg_pool_2d_explicit_padding() {
        let input = vec![4.0, 8.0, 12.0, 16.0];
        let mut cfg = PoolingConfig2D::new([2, 2], [1, 1]);
        cfg.padding = PaddingMode::Explicit(1);
        let out = avg_pool_2d(&input, 2, 2, &cfg);
        // With pad=1: output is 3x3
        assert_eq!(out.len(), 9);
    }

    // -- Edge cases for adaptive --

    #[test]
    fn adaptive_avg_pool_1d_large_output() {
        // output_size > input_len: some bins will be empty
        let input = vec![1.0, 2.0];
        let cfg = AdaptivePoolingConfig1D { output_size: 4 };
        let out = adaptive_avg_pool_1d(&input, &cfg);
        // bins: 0*2/4..1*2/4=0..0 (empty), 1*2/4..2*2/4=0..1, 2*2/4..3*2/4=1..1 (empty), 3*2/4..4*2/4=1..2
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn adaptive_max_pool_2d_identity() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = AdaptivePoolingConfig2D { output_size: [2, 2] };
        let out = adaptive_max_pool_2d(&input, 2, 2, &cfg);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn adaptive_avg_pool_2d_identity() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = AdaptivePoolingConfig2D { output_size: [2, 2] };
        let out = adaptive_avg_pool_2d(&input, 2, 2, &cfg);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // -- Additional stress / coverage --

    #[test]
    fn max_pool_1d_large_kernel() {
        let input = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let cfg = PoolingConfig1D::new(5, 1);
        let out = max_pool_1d(&input, &cfg);
        assert_eq!(out, vec![3.0]);
    }

    #[test]
    fn avg_pool_1d_uniform() {
        let input = vec![5.0; 10];
        let cfg = PoolingConfig1D::new(3, 1);
        let out = avg_pool_1d(&input, &cfg);
        for val in &out {
            assert!((val - 5.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn global_avg_pool_batched_uneven() {
        // 5 elements, spatial_len=2 => chunks [1,2], [3,4], [5]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = global_avg_pool_batched(&input, 2);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.5).abs() < f32::EPSILON);
        assert!((out[1] - 3.5).abs() < f32::EPSILON);
        assert!((out[2] - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn max_pool_2d_non_square() {
        #[rustfmt::skip]
        let input = vec![
            1.0, 5.0, 3.0,
            4.0, 2.0, 6.0,
        ];
        let cfg = PoolingConfig2D::new([2, 2], [1, 1]);
        let out = max_pool_2d(&input, 2, 3, &cfg);
        // out_h=1, out_w=2: windows (0:2,0:2)=[1,5,4,2]=>5, (0:2,1:3)=[5,3,2,6]=>6
        assert_eq!(out, vec![5.0, 6.0]);
    }

    #[test]
    fn avg_pool_2d_non_square() {
        #[rustfmt::skip]
        let input = vec![
            2.0, 4.0,
            6.0, 8.0,
            10.0, 12.0,
        ];
        let cfg = PoolingConfig2D::new([2, 2], [1, 1]);
        let out = avg_pool_2d(&input, 3, 2, &cfg);
        // out_h=2, out_w=1: window [2,4,6,8]=>5, [6,8,10,12]=>9
        assert_eq!(out, vec![5.0, 9.0]);
    }
}
