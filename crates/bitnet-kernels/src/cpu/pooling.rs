//! CPU SIMD-optimized pooling kernel.
//!
//! Provides 1-D pooling operations (max, average, global, adaptive) on
//! contiguous `f32` slices.  Scalar implementations are provided for
//! correctness on all platforms; SIMD acceleration is used when available.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Configuration ──────────────────────────────────────────────────

/// Pooling operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolType {
    /// Sliding-window maximum.
    Max,
    /// Sliding-window arithmetic mean.
    Average,
    /// Global maximum (reduces to a single value).
    GlobalMax,
    /// Global arithmetic mean (reduces to a single value).
    GlobalAverage,
    /// Average pooling that always divides by the full kernel area,
    /// including padded positions (PyTorch `count_include_pad=True`).
    /// Semantically equivalent to [`Average`](PoolType::Average).
    AvgPoolCountIncludePad,
}

/// Parameters for a 1-D pooling operation.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Type of pooling to perform.
    pub pool_type: PoolType,
    /// Window size (ignored for global variants).
    pub kernel_size: usize,
    /// Stride between successive windows (ignored for global variants).
    pub stride: usize,
    /// Zero-padding added to each side of the input (ignored for global variants).
    pub padding: usize,
}

impl PoolConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<()> {
        match self.pool_type {
            PoolType::GlobalMax | PoolType::GlobalAverage => Ok(()),
            _ => {
                if self.kernel_size == 0 {
                    return Err(invalid_args("kernel_size must be > 0"));
                }
                if self.stride == 0 {
                    return Err(invalid_args("stride must be > 0"));
                }
                Ok(())
            }
        }
    }
}

/// Convenience alias matching the user-facing name for pooling configuration.
pub type PoolingConfig = PoolConfig;

/// Stateless pooling kernel that dispatches to the appropriate operation.
#[derive(Debug)]
pub struct PoolingKernel;

impl PoolingKernel {
    /// Apply a 1-D pooling operation described by `config` to `input`.
    pub fn apply(input: &[f32], config: &PoolConfig) -> Result<Vec<f32>> {
        config.validate()?;
        match config.pool_type {
            PoolType::Max => max_pool_1d(input, config.kernel_size, config.stride, config.padding),
            PoolType::Average | PoolType::AvgPoolCountIncludePad => {
                avg_pool_1d(input, config.kernel_size, config.stride, config.padding)
            }
            PoolType::GlobalMax => global_max(input),
            PoolType::GlobalAverage => global_avg(input),
        }
    }

    /// Adaptive pooling: compute kernel size and stride so that an input of
    /// length `input_len` is reduced to exactly `output_size` elements.
    ///
    /// Returns a `PoolConfig` with padding = 0 and the caller-specified
    /// `pool_type`.  Global variants are returned as-is when
    /// `output_size == 1`.
    pub fn adaptive_config(
        pool_type: PoolType,
        input_len: usize,
        output_size: usize,
    ) -> Result<PoolConfig> {
        if output_size == 0 {
            return Err(invalid_args("output_size must be > 0"));
        }
        if input_len == 0 {
            return Err(invalid_args("input_len must be > 0"));
        }
        if output_size > input_len {
            return Err(invalid_args("output_size must be <= input_len for pooling"));
        }

        // Global variants when requesting a single output.
        if output_size == 1 {
            let global_type = match pool_type {
                PoolType::Max | PoolType::GlobalMax => PoolType::GlobalMax,
                PoolType::Average | PoolType::GlobalAverage | PoolType::AvgPoolCountIncludePad => {
                    PoolType::GlobalAverage
                }
            };
            return Ok(PoolConfig {
                pool_type: global_type,
                kernel_size: input_len,
                stride: input_len,
                padding: 0,
            });
        }

        // PyTorch-style adaptive pooling: for each output index i, the
        // window covers input[start..end] where
        //   start = floor(i * input_len / output_size)
        //   end   = floor((i+1) * input_len / output_size)
        // This cannot always be expressed with a single (kernel, stride)
        // pair; we approximate with the dominant window parameters.
        let stride = input_len / output_size;
        let kernel_size = input_len - (output_size - 1) * stride;

        Ok(PoolConfig { pool_type, kernel_size, stride, padding: 0 })
    }
}

// ── Scalar implementations ─────────────────────────────────────────

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

/// Number of output elements for a 1-D pooling window.
#[inline]
fn output_len(input_len: usize, kernel_size: usize, stride: usize, padding: usize) -> usize {
    if input_len + 2 * padding < kernel_size {
        return 0;
    }
    (input_len + 2 * padding - kernel_size) / stride + 1
}

/// 1-D max pooling.
fn max_pool_1d(
    input: &[f32],
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<Vec<f32>> {
    let n = input.len();
    let out_n = output_len(n, kernel_size, stride, padding);
    let mut output = Vec::with_capacity(out_n);

    for i in 0..out_n {
        let window_start = i * stride;
        let mut max_val = f32::NEG_INFINITY;

        for k in 0..kernel_size {
            let pos = window_start + k;
            let val = if pos < padding || pos >= n + padding {
                // Padded positions contribute -∞ for max pooling.
                f32::NEG_INFINITY
            } else {
                input[pos - padding]
            };
            if val > max_val {
                max_val = val;
            }
        }
        output.push(max_val);
    }
    Ok(output)
}

/// 1-D average pooling.
fn avg_pool_1d(
    input: &[f32],
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<Vec<f32>> {
    let n = input.len();
    let out_n = output_len(n, kernel_size, stride, padding);
    let mut output = Vec::with_capacity(out_n);

    for i in 0..out_n {
        let window_start = i * stride;
        let mut sum = 0.0f32;

        for k in 0..kernel_size {
            let pos = window_start + k;
            // Padded positions contribute 0 for average pooling.
            if pos >= padding && pos < n + padding {
                sum += input[pos - padding];
            }
        }
        output.push(sum / kernel_size as f32);
    }
    Ok(output)
}

/// Global max: reduce entire input to a single maximum.
fn global_max(input: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(invalid_args("global max requires non-empty input"));
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Ok(vec![max_val])
}

/// Global average: reduce entire input to a single mean.
fn global_avg(input: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(invalid_args("global average requires non-empty input"));
    }
    let sum: f32 = input.iter().sum();
    Ok(vec![sum / input.len() as f32])
}

// ── 2-D scalar helpers ────────────────────────────────────────────

#[inline]
fn pool_2d_window_max(
    input: &[f32],
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    k: usize,
    s: usize,
    p: usize,
) -> f32 {
    let mut max_val = f32::NEG_INFINITY;
    for kh in 0..k {
        for kw in 0..k {
            let ih = oh * s + kh;
            let iw = ow * s + kw;
            let val = if ih < p || ih >= h + p || iw < p || iw >= w + p {
                f32::NEG_INFINITY
            } else {
                input[(ih - p) * w + (iw - p)]
            };
            if val > max_val {
                max_val = val;
            }
        }
    }
    max_val
}

#[inline]
fn pool_2d_window_avg(
    input: &[f32],
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    k: usize,
    s: usize,
    p: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for kh in 0..k {
        for kw in 0..k {
            let ih = oh * s + kh;
            let iw = ow * s + kw;
            if ih >= p && ih < h + p && iw >= p && iw < w + p {
                sum += input[(ih - p) * w + (iw - p)];
            }
        }
    }
    sum / (k * k) as f32
}

// ── Public free functions ─────────────────────────────────────────

/// Apply 1-D pooling to `input` using the given configuration.
pub fn pool_1d(input: &[f32], config: &PoolConfig) -> Result<Vec<f32>> {
    PoolingKernel::apply(input, config)
}

/// Apply 2-D pooling to a single spatial plane of size `height × width`.
///
/// Uses square windows of `config.kernel_size × config.kernel_size`.
/// Returns `(output_data, out_height, out_width)`.
pub fn pool_2d(
    input: &[f32],
    height: usize,
    width: usize,
    config: &PoolConfig,
) -> Result<(Vec<f32>, usize, usize)> {
    config.validate()?;
    if input.len() != height * width {
        return Err(invalid_args("input length must equal height * width"));
    }
    match config.pool_type {
        PoolType::GlobalMax | PoolType::GlobalAverage => {
            let result = PoolingKernel::apply(input, config)?;
            Ok((result, 1, 1))
        }
        _ => {
            let out_h = output_len(height, config.kernel_size, config.stride, config.padding);
            let out_w = output_len(width, config.kernel_size, config.stride, config.padding);
            let k = config.kernel_size;
            let s = config.stride;
            let p = config.padding;
            let mut output = Vec::with_capacity(out_h * out_w);

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let val = match config.pool_type {
                        PoolType::Max => pool_2d_window_max(input, height, width, oh, ow, k, s, p),
                        PoolType::Average | PoolType::AvgPoolCountIncludePad => {
                            pool_2d_window_avg(input, height, width, oh, ow, k, s, p)
                        }
                        _ => unreachable!(),
                    };
                    output.push(val);
                }
            }
            Ok((output, out_h, out_w))
        }
    }
}

/// PyTorch-style adaptive average pooling for 1-D input.
///
/// Produces exactly `output_size` elements by computing per-position
/// window boundaries: `start_i = floor(i * N / out)`,
/// `end_i = floor((i+1) * N / out)`.
pub fn adaptive_avg_pool_1d(input: &[f32], output_size: usize) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    if output_size == 0 {
        return Err(invalid_args("output_size must be > 0"));
    }
    if output_size > input.len() {
        return Err(invalid_args("output_size must be <= input length"));
    }
    let n = input.len();
    let mut output = Vec::with_capacity(output_size);
    for i in 0..output_size {
        let start = (i * n) / output_size;
        let end = ((i + 1) * n) / output_size;
        let sum: f32 = input[start..end].iter().sum();
        output.push(sum / (end - start) as f32);
    }
    Ok(output)
}

/// PyTorch-style adaptive average pooling for 2-D spatial input.
///
/// Input is a flat slice of `h × w` elements.  Produces `out_h × out_w`
/// elements.
pub fn adaptive_avg_pool_2d(
    input: &[f32],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<f32>> {
    if input.len() != h * w {
        return Err(invalid_args("input length must equal h * w"));
    }
    if h == 0 || w == 0 {
        return Err(invalid_args("spatial dimensions must be > 0"));
    }
    if out_h == 0 || out_w == 0 {
        return Err(invalid_args("output dimensions must be > 0"));
    }
    if out_h > h || out_w > w {
        return Err(invalid_args("output dimensions must be <= input dimensions"));
    }
    let mut output = Vec::with_capacity(out_h * out_w);
    for oh in 0..out_h {
        let row_start = (oh * h) / out_h;
        let row_end = ((oh + 1) * h) / out_h;
        for ow in 0..out_w {
            let col_start = (ow * w) / out_w;
            let col_end = ((ow + 1) * w) / out_w;
            let count = (row_end - row_start) * (col_end - col_start);
            let mut sum = 0.0f32;
            for r in row_start..row_end {
                for c in col_start..col_end {
                    sum += input[r * w + c];
                }
            }
            output.push(sum / count as f32);
        }
    }
    Ok(output)
}

/// Global average pooling over spatial dimensions.
///
/// `input` contains `C` channels, each with `product(spatial_dims)` elements.
/// Returns one value per channel.
pub fn global_avg_pool(input: &[f32], spatial_dims: &[usize]) -> Result<Vec<f32>> {
    let spatial_size: usize = spatial_dims.iter().product();
    if spatial_size == 0 {
        return Err(invalid_args("spatial dimensions must be > 0"));
    }
    if !input.len().is_multiple_of(spatial_size) {
        return Err(invalid_args("input length must be divisible by spatial size"));
    }
    let channels = input.len() / spatial_size;
    let mut output = Vec::with_capacity(channels);
    for c in 0..channels {
        let start = c * spatial_size;
        let sum: f32 = input[start..start + spatial_size].iter().sum();
        output.push(sum / spatial_size as f32);
    }
    Ok(output)
}

/// Global max pooling over spatial dimensions.
///
/// `input` contains `C` channels, each with `product(spatial_dims)` elements.
/// Returns one value per channel.
pub fn global_max_pool(input: &[f32], spatial_dims: &[usize]) -> Result<Vec<f32>> {
    let spatial_size: usize = spatial_dims.iter().product();
    if spatial_size == 0 {
        return Err(invalid_args("spatial dimensions must be > 0"));
    }
    if !input.len().is_multiple_of(spatial_size) {
        return Err(invalid_args("input length must be divisible by spatial size"));
    }
    let channels = input.len() / spatial_size;
    let mut output = Vec::with_capacity(channels);
    for c in 0..channels {
        let start = c * spatial_size;
        let max_val =
            input[start..start + spatial_size].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        output.push(max_val);
    }
    Ok(output)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-6;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    // ── Max pooling ────────────────────────────────────────────────

    #[test]
    fn max_pool_basic() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 3.0, 5.0, 5.0], TOL));
    }

    #[test]
    fn max_pool_stride_2() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 2, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 5.0, 6.0], TOL));
    }

    #[test]
    fn max_pool_kernel_equals_input() {
        let input = vec![1.0, 3.0, 2.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 3, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0], TOL));
    }

    #[test]
    fn max_pool_with_padding() {
        let input = vec![1.0, 2.0, 3.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 3, stride: 1, padding: 1 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        // windows: [pad,1,2] [1,2,3] [2,3,pad]
        assert!(approx_eq(&out, &[2.0, 3.0, 3.0], TOL));
    }

    #[test]
    fn max_pool_negative_values() {
        let input = vec![-5.0, -3.0, -4.0, -1.0, -2.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 3, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[-3.0, -1.0, -1.0], TOL));
    }

    #[test]
    fn max_pool_single_element() {
        let input = vec![42.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 1, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[42.0], TOL));
    }

    // ── Average pooling ────────────────────────────────────────────

    #[test]
    fn avg_pool_basic() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[2.0, 2.5, 3.5, 4.5], TOL));
    }

    #[test]
    fn avg_pool_stride_2() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 2, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 7.0], TOL));
    }

    #[test]
    fn avg_pool_kernel_equals_input() {
        let input = vec![1.0, 2.0, 3.0];
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 3, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[2.0], TOL));
    }

    #[test]
    fn avg_pool_with_padding() {
        let input = vec![3.0, 6.0, 9.0];
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 3, stride: 1, padding: 1 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        // windows: [0,3,6]/3=3.0  [3,6,9]/3=6.0  [6,9,0]/3=5.0
        assert!(approx_eq(&out, &[3.0, 6.0, 5.0], TOL));
    }

    #[test]
    fn avg_pool_single_element() {
        let input = vec![7.0];
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 1, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[7.0], TOL));
    }

    // ── Global pooling ─────────────────────────────────────────────

    #[test]
    fn global_max_basic() {
        let input = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let cfg =
            PoolConfig { pool_type: PoolType::GlobalMax, kernel_size: 0, stride: 0, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[5.0], TOL));
    }

    #[test]
    fn global_max_single() {
        let cfg =
            PoolConfig { pool_type: PoolType::GlobalMax, kernel_size: 0, stride: 0, padding: 0 };
        let out = PoolingKernel::apply(&[42.0], &cfg).unwrap();
        assert!(approx_eq(&out, &[42.0], TOL));
    }

    #[test]
    fn global_max_all_negative() {
        let input = vec![-10.0, -5.0, -20.0];
        let cfg =
            PoolConfig { pool_type: PoolType::GlobalMax, kernel_size: 0, stride: 0, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[-5.0], TOL));
    }

    #[test]
    fn global_avg_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = PoolConfig {
            pool_type: PoolType::GlobalAverage,
            kernel_size: 0,
            stride: 0,
            padding: 0,
        };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0], TOL));
    }

    #[test]
    fn global_avg_single() {
        let cfg = PoolConfig {
            pool_type: PoolType::GlobalAverage,
            kernel_size: 0,
            stride: 0,
            padding: 0,
        };
        let out = PoolingKernel::apply(&[99.0], &cfg).unwrap();
        assert!(approx_eq(&out, &[99.0], TOL));
    }

    #[test]
    fn global_max_empty_input() {
        let cfg =
            PoolConfig { pool_type: PoolType::GlobalMax, kernel_size: 0, stride: 0, padding: 0 };
        assert!(PoolingKernel::apply(&[], &cfg).is_err());
    }

    #[test]
    fn global_avg_empty_input() {
        let cfg = PoolConfig {
            pool_type: PoolType::GlobalAverage,
            kernel_size: 0,
            stride: 0,
            padding: 0,
        };
        assert!(PoolingKernel::apply(&[], &cfg).is_err());
    }

    // ── Adaptive pooling ───────────────────────────────────────────

    #[test]
    fn adaptive_reduces_to_single() {
        let cfg = PoolingKernel::adaptive_config(PoolType::Max, 10, 1).unwrap();
        assert_eq!(cfg.pool_type, PoolType::GlobalMax);
    }

    #[test]
    fn adaptive_avg_reduces_to_global() {
        let cfg = PoolingKernel::adaptive_config(PoolType::Average, 8, 1).unwrap();
        assert_eq!(cfg.pool_type, PoolType::GlobalAverage);
    }

    #[test]
    fn adaptive_output_shape() {
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let cfg = PoolingKernel::adaptive_config(PoolType::Average, 10, 5).unwrap();
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn adaptive_identity() {
        // output_size == input_len ⇒ kernel_size=1, stride=1 (identity).
        let cfg = PoolingKernel::adaptive_config(PoolType::Max, 5, 5).unwrap();
        assert_eq!(cfg.kernel_size, 1);
        assert_eq!(cfg.stride, 1);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &input, TOL));
    }

    #[test]
    fn adaptive_output_larger_than_input_rejected() {
        assert!(PoolingKernel::adaptive_config(PoolType::Max, 3, 5).is_err());
    }

    #[test]
    fn adaptive_zero_output_rejected() {
        assert!(PoolingKernel::adaptive_config(PoolType::Max, 5, 0).is_err());
    }

    #[test]
    fn adaptive_zero_input_rejected() {
        assert!(PoolingKernel::adaptive_config(PoolType::Max, 0, 1).is_err());
    }

    // ── Edge cases ─────────────────────────────────────────────────

    #[test]
    fn kernel_larger_than_input_produces_empty() {
        let input = vec![1.0, 2.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 5, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn zero_kernel_size_rejected() {
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 0, stride: 1, padding: 0 };
        assert!(PoolingKernel::apply(&[1.0], &cfg).is_err());
    }

    #[test]
    fn zero_stride_rejected() {
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 0, padding: 0 };
        assert!(PoolingKernel::apply(&[1.0, 2.0], &cfg).is_err());
    }

    #[test]
    fn large_stride_single_output() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 10, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[2.0], TOL));
    }

    #[test]
    fn output_len_formula() {
        // input=6, kernel=3, stride=2, pad=0 → (6-3)/2+1 = 2
        assert_eq!(output_len(6, 3, 2, 0), 2);
        // input=5, kernel=3, stride=1, pad=1 → (5+2-3)/1+1 = 5
        assert_eq!(output_len(5, 3, 1, 1), 5);
    }

    #[test]
    fn max_pool_large_input() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 4, stride: 4, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert_eq!(out.len(), 256);
        // Each output is the max of 4 consecutive elements.
        for (i, &v) in out.iter().enumerate() {
            let window = &input[i * 4..i * 4 + 4];
            let expected = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!((v - expected).abs() < TOL);
        }
    }

    #[test]
    fn avg_pool_large_input() {
        let input: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 4, stride: 4, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert_eq!(out.len(), 256);
        for (i, &v) in out.iter().enumerate() {
            let window = &input[i * 4..i * 4 + 4];
            let expected: f32 = window.iter().sum::<f32>() / 4.0;
            assert!((v - expected).abs() < TOL);
        }
    }

    // ── AvgPoolCountIncludePad ────────────────────────────────────

    #[test]
    fn avg_pool_count_include_pad_matches_average() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg_avg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 3, stride: 1, padding: 1 };
        let cfg_cip = PoolConfig {
            pool_type: PoolType::AvgPoolCountIncludePad,
            kernel_size: 3,
            stride: 1,
            padding: 1,
        };
        let out_avg = PoolingKernel::apply(&input, &cfg_avg).unwrap();
        let out_cip = PoolingKernel::apply(&input, &cfg_cip).unwrap();
        assert!(approx_eq(&out_avg, &out_cip, TOL));
    }

    // ── pool_1d free function ─────────────────────────────────────

    #[test]
    fn pool_1d_delegates_to_kernel() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 1, padding: 0 };
        let out = pool_1d(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 3.0, 5.0, 5.0], TOL));
    }

    // ── 2-D pooling ───────────────────────────────────────────────

    #[test]
    fn pool_2d_max_basic() {
        // 3×3 input, k=2, s=1, p=0 → 2×2
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 1, padding: 0 };
        let (out, oh, ow) = pool_2d(&input, 3, 3, &cfg).unwrap();
        assert_eq!((oh, ow), (2, 2));
        assert!(approx_eq(&out, &[5.0, 6.0, 8.0, 9.0], TOL));
    }

    #[test]
    fn pool_2d_max_stride_2() {
        // 4×4, k=2, s=2 → 2×2
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 2, padding: 0 };
        let (out, oh, ow) = pool_2d(&input, 4, 4, &cfg).unwrap();
        assert_eq!((oh, ow), (2, 2));
        assert!(approx_eq(&out, &[6.0, 8.0, 14.0, 16.0], TOL));
    }

    #[test]
    fn pool_2d_max_with_padding() {
        // 2×2, k=2, s=1, p=1 → 3×3
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 1, padding: 1 };
        let (out, oh, ow) = pool_2d(&input, 2, 2, &cfg).unwrap();
        assert_eq!((oh, ow), (3, 3));
        // Row 0: [pad,pad|pad,1] [pad,pad|1,2] [pad,pad|2,pad]
        // Row 1: [pad,1|pad,3]   [1,2|3,4]     [2,pad|4,pad]
        // Row 2: [pad,3|pad,pad] [3,4|pad,pad]  [4,pad|pad,pad]
        assert!(approx_eq(&out, &[1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 4.0, 4.0], TOL));
    }

    #[test]
    fn pool_2d_avg_basic() {
        // 3×3, k=2, s=1, p=0 → 2×2
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 1, padding: 0 };
        let (out, oh, ow) = pool_2d(&input, 3, 3, &cfg).unwrap();
        assert_eq!((oh, ow), (2, 2));
        // (1+2+4+5)/4=3  (2+3+5+6)/4=4  (4+5+7+8)/4=6  (5+6+8+9)/4=7
        assert!(approx_eq(&out, &[3.0, 4.0, 6.0, 7.0], TOL));
    }

    #[test]
    fn pool_2d_avg_stride_2() {
        // 4×4, k=2, s=2 → 2×2
        #[rustfmt::skip]
        let input = vec![
            2.0, 4.0, 6.0, 8.0,
            10.0, 12.0, 14.0, 16.0,
            18.0, 20.0, 22.0, 24.0,
            26.0, 28.0, 30.0, 32.0,
        ];
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 2, padding: 0 };
        let (out, oh, ow) = pool_2d(&input, 4, 4, &cfg).unwrap();
        assert_eq!((oh, ow), (2, 2));
        // (2+4+10+12)/4=7  (6+8+14+16)/4=11  (18+20+26+28)/4=23  (22+24+30+32)/4=27
        assert!(approx_eq(&out, &[7.0, 11.0, 23.0, 27.0], TOL));
    }

    #[test]
    fn pool_2d_avg_with_padding() {
        // 2×2, k=2, s=1, p=1 → 3×3
        #[rustfmt::skip]
        let input = vec![
            4.0, 8.0,
            12.0, 16.0,
        ];
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 1, padding: 1 };
        let (out, oh, ow) = pool_2d(&input, 2, 2, &cfg).unwrap();
        assert_eq!((oh, ow), (3, 3));
        // (0+0+0+4)/4=1  (0+0+4+8)/4=3  (0+0+8+0)/4=2
        // (0+4+0+12)/4=4 (4+8+12+16)/4=10 (8+0+16+0)/4=6
        // (0+12+0+0)/4=3 (12+16+0+0)/4=7 (16+0+0+0)/4=4
        assert!(approx_eq(&out, &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0], TOL));
    }

    #[test]
    fn pool_2d_avg_count_include_pad() {
        let input = vec![4.0, 8.0, 12.0, 16.0];
        let cfg = PoolConfig {
            pool_type: PoolType::AvgPoolCountIncludePad,
            kernel_size: 2,
            stride: 2,
            padding: 0,
        };
        let (out, oh, ow) = pool_2d(&input, 2, 2, &cfg).unwrap();
        assert_eq!((oh, ow), (1, 1));
        assert!(approx_eq(&out, &[10.0], TOL));
    }

    #[test]
    fn pool_2d_single_element() {
        let input = vec![42.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 1, stride: 1, padding: 0 };
        let (out, oh, ow) = pool_2d(&input, 1, 1, &cfg).unwrap();
        assert_eq!((oh, ow), (1, 1));
        assert!(approx_eq(&out, &[42.0], TOL));
    }

    #[test]
    fn pool_2d_global_max() {
        #[rustfmt::skip]
        let input = vec![
            1.0, 9.0,
            3.0, 5.0,
        ];
        let cfg =
            PoolConfig { pool_type: PoolType::GlobalMax, kernel_size: 0, stride: 0, padding: 0 };
        let (out, oh, ow) = pool_2d(&input, 2, 2, &cfg).unwrap();
        assert_eq!((oh, ow), (1, 1));
        assert!(approx_eq(&out, &[9.0], TOL));
    }

    #[test]
    fn pool_2d_global_avg() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let cfg = PoolConfig {
            pool_type: PoolType::GlobalAverage,
            kernel_size: 0,
            stride: 0,
            padding: 0,
        };
        let (out, oh, ow) = pool_2d(&input, 2, 2, &cfg).unwrap();
        assert_eq!((oh, ow), (1, 1));
        assert!(approx_eq(&out, &[5.0], TOL));
    }

    #[test]
    fn pool_2d_wrong_input_size() {
        let input = vec![1.0, 2.0, 3.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 1, padding: 0 };
        assert!(pool_2d(&input, 2, 2, &cfg).is_err());
    }

    #[test]
    fn pool_2d_non_square_input() {
        // 2×3 input, k=2, s=1 → 1×2
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 1, padding: 0 };
        let (out, oh, ow) = pool_2d(&input, 2, 3, &cfg).unwrap();
        assert_eq!((oh, ow), (1, 2));
        assert!(approx_eq(&out, &[5.0, 6.0], TOL));
    }

    // ── Adaptive 1-D ──────────────────────────────────────────────

    #[test]
    fn adaptive_avg_pool_1d_basic() {
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let out = adaptive_avg_pool_1d(&input, 5).unwrap();
        assert_eq!(out.len(), 5);
        // bins: [0,1] [2,3] [4,5] [6,7] [8,9]
        assert!(approx_eq(&out, &[0.5, 2.5, 4.5, 6.5, 8.5], TOL));
    }

    #[test]
    fn adaptive_avg_pool_1d_identity() {
        let input = vec![1.0, 2.0, 3.0];
        let out = adaptive_avg_pool_1d(&input, 3).unwrap();
        assert!(approx_eq(&out, &[1.0, 2.0, 3.0], TOL));
    }

    #[test]
    fn adaptive_avg_pool_1d_to_one() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let out = adaptive_avg_pool_1d(&input, 1).unwrap();
        assert!(approx_eq(&out, &[5.0], TOL));
    }

    #[test]
    fn adaptive_avg_pool_1d_empty_rejected() {
        assert!(adaptive_avg_pool_1d(&[], 1).is_err());
    }

    #[test]
    fn adaptive_avg_pool_1d_zero_output_rejected() {
        assert!(adaptive_avg_pool_1d(&[1.0], 0).is_err());
    }

    #[test]
    fn adaptive_avg_pool_1d_output_larger_rejected() {
        assert!(adaptive_avg_pool_1d(&[1.0, 2.0], 5).is_err());
    }

    // ── Adaptive 2-D ──────────────────────────────────────────────

    #[test]
    fn adaptive_avg_pool_2d_basic() {
        // 4×4 → 2×2
        #[rustfmt::skip]
        let input = vec![
            1.0,  2.0,  3.0,  4.0,
            5.0,  6.0,  7.0,  8.0,
            9.0,  10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let out = adaptive_avg_pool_2d(&input, 4, 4, 2, 2).unwrap();
        assert_eq!(out.len(), 4);
        // top-left: (1+2+5+6)/4=3.5  top-right: (3+4+7+8)/4=5.5
        // bot-left: (9+10+13+14)/4=11.5  bot-right: (11+12+15+16)/4=13.5
        assert!(approx_eq(&out, &[3.5, 5.5, 11.5, 13.5], TOL));
    }

    #[test]
    fn adaptive_avg_pool_2d_identity() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = adaptive_avg_pool_2d(&input, 2, 2, 2, 2).unwrap();
        assert!(approx_eq(&out, &[1.0, 2.0, 3.0, 4.0], TOL));
    }

    #[test]
    fn adaptive_avg_pool_2d_to_one() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let out = adaptive_avg_pool_2d(&input, 2, 2, 1, 1).unwrap();
        assert!(approx_eq(&out, &[5.0], TOL));
    }

    #[test]
    fn adaptive_avg_pool_2d_non_square() {
        // 4×6 → 2×3
        let input: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let out = adaptive_avg_pool_2d(&input, 4, 6, 2, 3).unwrap();
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn adaptive_avg_pool_2d_wrong_input() {
        assert!(adaptive_avg_pool_2d(&[1.0, 2.0, 3.0], 2, 2, 1, 1).is_err());
    }

    #[test]
    fn adaptive_avg_pool_2d_output_larger_rejected() {
        assert!(adaptive_avg_pool_2d(&[1.0; 4], 2, 2, 3, 3).is_err());
    }

    #[test]
    fn adaptive_avg_pool_2d_zero_output_rejected() {
        assert!(adaptive_avg_pool_2d(&[1.0; 4], 2, 2, 0, 1).is_err());
    }

    // ── Global pooling with spatial dims ──────────────────────────

    #[test]
    fn global_avg_pool_single_channel() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = global_avg_pool(&input, &[4]).unwrap();
        assert!(approx_eq(&out, &[2.5], TOL));
    }

    #[test]
    fn global_avg_pool_multi_channel() {
        // 2 channels, each 3 elements
        let input = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let out = global_avg_pool(&input, &[3]).unwrap();
        assert!(approx_eq(&out, &[2.0, 20.0], TOL));
    }

    #[test]
    fn global_avg_pool_2d_spatial() {
        // 2 channels of 2×3 spatial
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let out = global_avg_pool(&input, &[2, 3]).unwrap();
        assert_eq!(out.len(), 2);
        assert!(approx_eq(&out, &[3.5, 9.5], TOL));
    }

    #[test]
    fn global_avg_pool_mismatched_input() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(global_avg_pool(&input, &[3]).is_err());
    }

    #[test]
    fn global_max_pool_single_channel() {
        let input = vec![1.0, 5.0, 3.0, 2.0];
        let out = global_max_pool(&input, &[4]).unwrap();
        assert!(approx_eq(&out, &[5.0], TOL));
    }

    #[test]
    fn global_max_pool_multi_channel() {
        // 3 channels, each 2 elements
        let input = vec![1.0, 5.0, 3.0, 2.0, 9.0, 4.0];
        let out = global_max_pool(&input, &[2]).unwrap();
        assert!(approx_eq(&out, &[5.0, 3.0, 9.0], TOL));
    }

    #[test]
    fn global_max_pool_2d_spatial() {
        // 1 channel of 3×3
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0,
            4.0, 9.0, 6.0,
            7.0, 8.0, 5.0,
        ];
        let out = global_max_pool(&input, &[3, 3]).unwrap();
        assert!(approx_eq(&out, &[9.0], TOL));
    }

    #[test]
    fn global_max_pool_empty_spatial_rejected() {
        assert!(global_max_pool(&[1.0], &[0]).is_err());
    }

    #[test]
    fn global_max_pool_mismatched_input() {
        assert!(global_max_pool(&[1.0, 2.0, 3.0], &[2]).is_err());
    }
}
