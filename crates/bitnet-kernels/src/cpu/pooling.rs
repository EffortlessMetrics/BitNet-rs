//! CPU SIMD-optimized pooling kernels.
//!
//! Provides 1-D and 2-D pooling operations (max, average, global, adaptive,
//! Lp-norm) on contiguous `f32` slices with optional index tracking.
//!
//! Scalar implementations are provided for correctness on all platforms;
//! AVX2 SIMD acceleration is auto-selected at runtime when available on
//! x86-64 targets.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Configuration ──────────────────────────────────────────────────

/// Pooling operation type.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    AvgPoolCountIncludePad,
    /// Lp-norm pooling: `(sum |x_i|^p)^(1/p)` over each window.
    Lp(f32),
    /// Adaptive pooling marker (used with adaptive_* functions).
    Adaptive,
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
    /// Spacing between kernel elements (atrous/dilated pooling). Default: 1.
    pub dilation: usize,
    /// When true, use `ceil` instead of `floor` for output length. Default: false.
    pub ceil_mode: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            pool_type: PoolType::Max,
            kernel_size: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            ceil_mode: false,
        }
    }
}

impl PoolConfig {
    /// Create a new config with defaults for dilation (1) and ceil_mode (false).
    pub fn new(pool_type: PoolType, kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self { pool_type, kernel_size, stride, padding, dilation: 1, ceil_mode: false }
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<()> {
        match self.pool_type {
            PoolType::GlobalMax | PoolType::GlobalAverage | PoolType::Adaptive => Ok(()),
            _ => {
                if self.kernel_size == 0 {
                    return Err(invalid_args("kernel_size must be > 0"));
                }
                if self.stride == 0 {
                    return Err(invalid_args("stride must be > 0"));
                }
                if self.dilation == 0 {
                    return Err(invalid_args("dilation must be > 0"));
                }
                Ok(())
            }
        }
    }

    /// Effective kernel size accounting for dilation.
    #[inline]
    pub fn effective_kernel_size(&self) -> usize {
        self.dilation * (self.kernel_size - 1) + 1
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
            PoolType::Max => {
                let (out, _) = max_pool_1d_scalar(input, config)?;
                Ok(out)
            }
            PoolType::Average | PoolType::AvgPoolCountIncludePad => {
                avg_pool_1d_scalar(input, config)
            }
            PoolType::GlobalMax => global_max(input),
            PoolType::GlobalAverage => global_avg(input),
            PoolType::Lp(p) => lp_pool_1d_scalar(input, p, config),
            PoolType::Adaptive => Err(invalid_args(
                "use adaptive_avg_pool_1d or adaptive_max_pool_1d for adaptive pooling",
            )),
        }
    }

    /// Adaptive pooling: compute kernel size and stride so that an input of
    /// length `input_len` is reduced to exactly `output_size` elements.
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

        if output_size == 1 {
            let global_type = match pool_type {
                PoolType::Max | PoolType::GlobalMax => PoolType::GlobalMax,
                _ => PoolType::GlobalAverage,
            };
            return Ok(PoolConfig {
                pool_type: global_type,
                kernel_size: input_len,
                stride: input_len,
                padding: 0,
                dilation: 1,
                ceil_mode: false,
            });
        }

        let stride = input_len / output_size;
        let kernel_size = input_len - (output_size - 1) * stride;

        Ok(PoolConfig { pool_type, kernel_size, stride, padding: 0, dilation: 1, ceil_mode: false })
    }
}

// ── Helpers ────────────────────────────────────────────────────────

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

/// Number of output elements for a 1-D pooling window.
#[inline]
fn output_len(
    input_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> usize {
    let effective_k = dilation * (kernel_size - 1) + 1;
    let numerator = input_len + 2 * padding;
    if numerator < effective_k {
        return 0;
    }
    let diff = numerator - effective_k;
    if ceil_mode { diff.div_ceil(stride) + 1 } else { diff / stride + 1 }
}

/// Legacy output_len without dilation/ceil_mode (dilation=1, ceil_mode=false).
#[inline]
fn output_len_simple(input_len: usize, kernel_size: usize, stride: usize, padding: usize) -> usize {
    output_len(input_len, kernel_size, stride, padding, 1, false)
}

// ── Scalar implementations ─────────────────────────────────────────

/// 1-D max pooling (scalar) returning (output, indices).
fn max_pool_1d_scalar(input: &[f32], config: &PoolConfig) -> Result<(Vec<f32>, Vec<usize>)> {
    let n = input.len();
    let out_n = output_len(
        n,
        config.kernel_size,
        config.stride,
        config.padding,
        config.dilation,
        config.ceil_mode,
    );
    let mut output = Vec::with_capacity(out_n);
    let mut indices = Vec::with_capacity(out_n);

    for i in 0..out_n {
        let window_start = i * config.stride;
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0usize;

        for k in 0..config.kernel_size {
            let pos = window_start + k * config.dilation;
            let (val, real_idx) = if pos < config.padding || pos >= n + config.padding {
                (f32::NEG_INFINITY, 0)
            } else {
                let idx = pos - config.padding;
                (input[idx], idx)
            };
            if val > max_val {
                max_val = val;
                max_idx = real_idx;
            }
        }
        output.push(max_val);
        indices.push(max_idx);
    }
    Ok((output, indices))
}

/// 1-D average pooling (scalar).
fn avg_pool_1d_scalar(input: &[f32], config: &PoolConfig) -> Result<Vec<f32>> {
    let n = input.len();
    let out_n = output_len(
        n,
        config.kernel_size,
        config.stride,
        config.padding,
        config.dilation,
        config.ceil_mode,
    );
    let mut output = Vec::with_capacity(out_n);

    for i in 0..out_n {
        let window_start = i * config.stride;
        let mut sum = 0.0f32;

        for k in 0..config.kernel_size {
            let pos = window_start + k * config.dilation;
            if pos >= config.padding && pos < n + config.padding {
                sum += input[pos - config.padding];
            }
        }
        output.push(sum / config.kernel_size as f32);
    }
    Ok(output)
}

/// 1-D Lp-norm pooling (scalar): `(sum |x_i|^p)^(1/p)` over each window.
fn lp_pool_1d_scalar(input: &[f32], p: f32, config: &PoolConfig) -> Result<Vec<f32>> {
    if p <= 0.0 {
        return Err(invalid_args("Lp norm p must be > 0"));
    }
    let n = input.len();
    let out_n = output_len(
        n,
        config.kernel_size,
        config.stride,
        config.padding,
        config.dilation,
        config.ceil_mode,
    );
    let mut output = Vec::with_capacity(out_n);

    for i in 0..out_n {
        let window_start = i * config.stride;
        let mut acc = 0.0f32;

        for k in 0..config.kernel_size {
            let pos = window_start + k * config.dilation;
            if pos >= config.padding && pos < n + config.padding {
                acc += input[pos - config.padding].abs().powf(p);
            }
        }
        output.push(acc.powf(1.0 / p));
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

// ── AVX2 implementations ──────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// AVX2-optimized 1-D max pooling returning (output, indices).
    ///
    /// # Safety
    /// Caller must verify AVX2 is available via `is_x86_feature_detected!("avx2")`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn max_pool_1d_avx2_inner(
        input: &[f32],
        config: &PoolConfig,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let n = input.len();
        let out_n = super::output_len(
            n,
            config.kernel_size,
            config.stride,
            config.padding,
            config.dilation,
            config.ceil_mode,
        );
        let mut output = Vec::with_capacity(out_n);
        let mut indices = Vec::with_capacity(out_n);

        // Process groups of 8 output positions at a time when possible.
        let mut i = 0;
        while i + 8 <= out_n {
            let mut max_vals = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut max_idxs = [0usize; 8];

            for k in 0..config.kernel_size {
                let mut vals = [f32::NEG_INFINITY; 8];
                let mut idxs = [0usize; 8];

                for j in 0..8 {
                    let pos = (i + j) * config.stride + k * config.dilation;
                    if pos >= config.padding && pos < n + config.padding {
                        let idx = pos - config.padding;
                        vals[j] = input[idx];
                        idxs[j] = idx;
                    }
                }

                let v = unsafe { _mm256_loadu_ps(vals.as_ptr()) };
                let cmp = _mm256_cmp_ps(v, max_vals, _CMP_GT_OQ);
                let mask = _mm256_movemask_ps(cmp) as u32;

                max_vals = _mm256_max_ps(max_vals, v);

                // Update indices for lanes where new value is greater.
                for j in 0..8 {
                    if mask & (1 << j) != 0 {
                        max_idxs[j] = idxs[j];
                    }
                }
            }

            let mut out_vals = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(out_vals.as_mut_ptr(), max_vals) };
            output.extend_from_slice(&out_vals);
            indices.extend_from_slice(&max_idxs);
            i += 8;
        }

        // Scalar tail for remaining positions.
        for idx in i..out_n {
            let window_start = idx * config.stride;
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0usize;

            for k in 0..config.kernel_size {
                let pos = window_start + k * config.dilation;
                if pos >= config.padding && pos < n + config.padding {
                    let real_idx = pos - config.padding;
                    let val = input[real_idx];
                    if val > max_val {
                        max_val = val;
                        max_idx = real_idx;
                    }
                } else if f32::NEG_INFINITY > max_val {
                    max_val = f32::NEG_INFINITY;
                }
            }
            output.push(max_val);
            indices.push(max_idx);
        }

        Ok((output, indices))
    }

    /// AVX2-optimized 1-D average pooling.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available via `is_x86_feature_detected!("avx2")`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn avg_pool_1d_avx2_inner(input: &[f32], config: &PoolConfig) -> Result<Vec<f32>> {
        let n = input.len();
        let out_n = super::output_len(
            n,
            config.kernel_size,
            config.stride,
            config.padding,
            config.dilation,
            config.ceil_mode,
        );
        let mut output = Vec::with_capacity(out_n);
        let inv_k = _mm256_set1_ps(1.0 / config.kernel_size as f32);

        let mut i = 0;
        while i + 8 <= out_n {
            let mut sums = _mm256_setzero_ps();

            for k in 0..config.kernel_size {
                let mut vals = [0.0f32; 8];
                for (j, v) in vals.iter_mut().enumerate() {
                    let pos = (i + j) * config.stride + k * config.dilation;
                    if pos >= config.padding && pos < n + config.padding {
                        *v = input[pos - config.padding];
                    }
                }
                let v = unsafe { _mm256_loadu_ps(vals.as_ptr()) };
                sums = _mm256_add_ps(sums, v);
            }

            let mut out_vals = [0.0f32; 8];
            let avgs = _mm256_mul_ps(sums, inv_k);
            unsafe { _mm256_storeu_ps(out_vals.as_mut_ptr(), avgs) };
            output.extend_from_slice(&out_vals);
            i += 8;
        }

        // Scalar tail.
        for idx in i..out_n {
            let window_start = idx * config.stride;
            let mut sum = 0.0f32;
            for k in 0..config.kernel_size {
                let pos = window_start + k * config.dilation;
                if pos >= config.padding && pos < n + config.padding {
                    sum += input[pos - config.padding];
                }
            }
            output.push(sum / config.kernel_size as f32);
        }

        Ok(output)
    }
}

// ── Runtime dispatch ──────────────────────────────────────────────

/// 1-D max pooling with runtime SIMD dispatch.
/// Returns `(output_values, max_indices)`.
pub fn max_pool1d(input: &[f32], config: &PoolConfig) -> Result<(Vec<f32>, Vec<usize>)> {
    config.validate()?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection verified above.
            return unsafe { avx2::max_pool_1d_avx2_inner(input, config) };
        }
    }

    max_pool_1d_scalar(input, config)
}

/// 1-D average pooling with runtime SIMD dispatch.
pub fn avg_pool1d(input: &[f32], config: &PoolConfig) -> Result<Vec<f32>> {
    config.validate()?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::avg_pool_1d_avx2_inner(input, config) };
        }
    }

    avg_pool_1d_scalar(input, config)
}

/// Explicit AVX2 max pooling entry point.
///
/// Falls back to scalar on non-x86_64 or when AVX2 is unavailable.
pub fn max_pool1d_avx2(input: &[f32], config: &PoolConfig) -> Result<(Vec<f32>, Vec<usize>)> {
    config.validate()?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::max_pool_1d_avx2_inner(input, config) };
        }
    }

    max_pool_1d_scalar(input, config)
}

/// Explicit AVX2 average pooling entry point.
///
/// Falls back to scalar on non-x86_64 or when AVX2 is unavailable.
pub fn avg_pool1d_avx2(input: &[f32], config: &PoolConfig) -> Result<Vec<f32>> {
    config.validate()?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::avg_pool_1d_avx2_inner(input, config) };
        }
    }

    avg_pool_1d_scalar(input, config)
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
            let out_h =
                output_len_simple(height, config.kernel_size, config.stride, config.padding);
            let out_w = output_len_simple(width, config.kernel_size, config.stride, config.padding);
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
                        _ => {
                            return Err(invalid_args(
                                "pool_2d supports only Max, Average, or Global variants",
                            ));
                        }
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
/// window boundaries.
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

/// PyTorch-style adaptive max pooling for 1-D input.
///
/// Returns `(output_values, max_indices)`.
pub fn adaptive_max_pool1d(input: &[f32], output_size: usize) -> Result<(Vec<f32>, Vec<usize>)> {
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
    let mut indices = Vec::with_capacity(output_size);
    for i in 0..output_size {
        let start = (i * n) / output_size;
        let end = ((i + 1) * n) / output_size;
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = start;
        for (j, &val) in input[start..end].iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = start + j;
            }
        }
        output.push(max_val);
        indices.push(max_idx);
    }
    Ok((output, indices))
}

/// PyTorch-style adaptive average pooling for 2-D spatial input.
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
/// Returns `(output_values, max_indices_per_channel)`.
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

/// Lp-norm pooling: `(sum |x_i|^p)^(1/p)` over each window.
pub fn lp_pool1d(input: &[f32], p: f32, config: &PoolConfig) -> Result<Vec<f32>> {
    config.validate()?;
    lp_pool_1d_scalar(input, p, config)
}

/// Inverse max pooling: scatter `input` values back to their original
/// positions using `indices`, producing a tensor of `output_size`.
///
/// Positions not covered by any index are filled with zero.
pub fn max_unpool1d(input: &[f32], indices: &[usize], output_size: usize) -> Result<Vec<f32>> {
    if input.len() != indices.len() {
        return Err(invalid_args("input and indices must have the same length"));
    }
    for &idx in indices {
        if idx >= output_size {
            return Err(invalid_args("index out of bounds for output_size"));
        }
    }
    let mut output = vec![0.0f32; output_size];
    for (&val, &idx) in input.iter().zip(indices.iter()) {
        output[idx] = val;
    }
    Ok(output)
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

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    // ── Max pooling (scalar) ──────────────────────────────────────

    #[test]
    fn max_pool_basic() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 3.0, 5.0, 5.0], TOL));
    }

    #[test]
    fn max_pool_stride_2() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 2, 0);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 5.0, 6.0], TOL));
    }

    #[test]
    fn max_pool_kernel_equals_input() {
        let input = vec![1.0, 3.0, 2.0];
        let cfg = PoolConfig::new(PoolType::Max, 3, 1, 0);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0], TOL));
    }

    #[test]
    fn max_pool_with_padding() {
        let input = vec![1.0, 2.0, 3.0];
        let cfg = PoolConfig::new(PoolType::Max, 3, 1, 1);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[2.0, 3.0, 3.0], TOL));
    }

    #[test]
    fn max_pool_negative_values() {
        let input = vec![-5.0, -3.0, -4.0, -1.0, -2.0];
        let cfg = PoolConfig::new(PoolType::Max, 3, 1, 0);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[-3.0, -1.0, -1.0], TOL));
    }

    #[test]
    fn max_pool_single_element() {
        let input = vec![42.0];
        let cfg = PoolConfig::new(PoolType::Max, 1, 1, 0);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[42.0], TOL));
    }

    // ── Average pooling ────────────────────────────────────────────

    #[test]
    fn avg_pool_basic() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let cfg = PoolConfig::new(PoolType::Average, 2, 1, 0);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[2.0, 2.5, 3.5, 4.5], TOL));
    }

    #[test]
    fn avg_pool_stride_2() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let cfg = PoolConfig::new(PoolType::Average, 2, 2, 0);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 7.0], TOL));
    }

    #[test]
    fn avg_pool_kernel_equals_input() {
        let input = vec![1.0, 2.0, 3.0];
        let cfg = PoolConfig::new(PoolType::Average, 3, 1, 0);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[2.0], TOL));
    }

    #[test]
    fn avg_pool_with_padding() {
        let input = vec![3.0, 6.0, 9.0];
        let cfg = PoolConfig::new(PoolType::Average, 3, 1, 1);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 6.0, 5.0], TOL));
    }

    #[test]
    fn avg_pool_single_element() {
        let input = vec![7.0];
        let cfg = PoolConfig::new(PoolType::Average, 1, 1, 0);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[7.0], TOL));
    }

    // ── Global pooling ─────────────────────────────────────────────

    #[test]
    fn global_max_basic() {
        let input = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let cfg = PoolConfig { pool_type: PoolType::GlobalMax, ..PoolConfig::default() };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[5.0], TOL));
    }

    #[test]
    fn global_max_single() {
        let cfg = PoolConfig { pool_type: PoolType::GlobalMax, ..PoolConfig::default() };
        let out = PoolingKernel::apply(&[42.0], &cfg).unwrap();
        assert!(approx_eq(&out, &[42.0], TOL));
    }

    #[test]
    fn global_max_all_negative() {
        let input = vec![-10.0, -5.0, -20.0];
        let cfg = PoolConfig { pool_type: PoolType::GlobalMax, ..PoolConfig::default() };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[-5.0], TOL));
    }

    #[test]
    fn global_avg_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = PoolConfig { pool_type: PoolType::GlobalAverage, ..PoolConfig::default() };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0], TOL));
    }

    #[test]
    fn global_avg_single() {
        let cfg = PoolConfig { pool_type: PoolType::GlobalAverage, ..PoolConfig::default() };
        let out = PoolingKernel::apply(&[99.0], &cfg).unwrap();
        assert!(approx_eq(&out, &[99.0], TOL));
    }

    #[test]
    fn global_max_empty_input() {
        let cfg = PoolConfig { pool_type: PoolType::GlobalMax, ..PoolConfig::default() };
        assert!(PoolingKernel::apply(&[], &cfg).is_err());
    }

    #[test]
    fn global_avg_empty_input() {
        let cfg = PoolConfig { pool_type: PoolType::GlobalAverage, ..PoolConfig::default() };
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
        let cfg = PoolConfig::new(PoolType::Max, 5, 1, 0);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn zero_kernel_size_rejected() {
        let cfg = PoolConfig::new(PoolType::Max, 0, 1, 0);
        assert!(PoolingKernel::apply(&[1.0], &cfg).is_err());
    }

    #[test]
    fn zero_stride_rejected() {
        let cfg = PoolConfig::new(PoolType::Average, 2, 0, 0);
        assert!(PoolingKernel::apply(&[1.0, 2.0], &cfg).is_err());
    }

    #[test]
    fn large_stride_single_output() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 10, 0);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[2.0], TOL));
    }

    #[test]
    fn output_len_formula() {
        assert_eq!(output_len_simple(6, 3, 2, 0), 2);
        assert_eq!(output_len_simple(5, 3, 1, 1), 5);
    }

    #[test]
    fn max_pool_large_input() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
        let cfg = PoolConfig::new(PoolType::Max, 4, 4, 0);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert_eq!(out.len(), 256);
        for (i, &v) in out.iter().enumerate() {
            let window = &input[i * 4..i * 4 + 4];
            let expected = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!((v - expected).abs() < TOL);
        }
    }

    #[test]
    fn avg_pool_large_input() {
        let input: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let cfg = PoolConfig::new(PoolType::Average, 4, 4, 0);
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
        let cfg_avg = PoolConfig::new(PoolType::Average, 3, 1, 1);
        let cfg_cip = PoolConfig::new(PoolType::AvgPoolCountIncludePad, 3, 1, 1);
        let out_avg = PoolingKernel::apply(&input, &cfg_avg).unwrap();
        let out_cip = PoolingKernel::apply(&input, &cfg_cip).unwrap();
        assert!(approx_eq(&out_avg, &out_cip, TOL));
    }

    // ── pool_1d free function ─────────────────────────────────────

    #[test]
    fn pool_1d_delegates_to_kernel() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0);
        let out = pool_1d(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 3.0, 5.0, 5.0], TOL));
    }

    // ── 2-D pooling ───────────────────────────────────────────────

    #[test]
    fn pool_2d_max_basic() {
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0);
        let (out, oh, ow) = pool_2d(&input, 3, 3, &cfg).unwrap();
        assert_eq!((oh, ow), (2, 2));
        assert!(approx_eq(&out, &[5.0, 6.0, 8.0, 9.0], TOL));
    }

    #[test]
    fn pool_2d_max_stride_2() {
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let cfg = PoolConfig::new(PoolType::Max, 2, 2, 0);
        let (out, oh, ow) = pool_2d(&input, 4, 4, &cfg).unwrap();
        assert_eq!((oh, ow), (2, 2));
        assert!(approx_eq(&out, &[6.0, 8.0, 14.0, 16.0], TOL));
    }

    #[test]
    fn pool_2d_max_with_padding() {
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 1);
        let (out, oh, ow) = pool_2d(&input, 2, 2, &cfg).unwrap();
        assert_eq!((oh, ow), (3, 3));
        assert!(approx_eq(&out, &[1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 4.0, 4.0], TOL));
    }

    #[test]
    fn pool_2d_avg_basic() {
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let cfg = PoolConfig::new(PoolType::Average, 2, 1, 0);
        let (out, oh, ow) = pool_2d(&input, 3, 3, &cfg).unwrap();
        assert_eq!((oh, ow), (2, 2));
        assert!(approx_eq(&out, &[3.0, 4.0, 6.0, 7.0], TOL));
    }

    #[test]
    fn pool_2d_avg_stride_2() {
        #[rustfmt::skip]
        let input = vec![
            2.0, 4.0, 6.0, 8.0,
            10.0, 12.0, 14.0, 16.0,
            18.0, 20.0, 22.0, 24.0,
            26.0, 28.0, 30.0, 32.0,
        ];
        let cfg = PoolConfig::new(PoolType::Average, 2, 2, 0);
        let (out, oh, ow) = pool_2d(&input, 4, 4, &cfg).unwrap();
        assert_eq!((oh, ow), (2, 2));
        assert!(approx_eq(&out, &[7.0, 11.0, 23.0, 27.0], TOL));
    }

    #[test]
    fn pool_2d_avg_with_padding() {
        #[rustfmt::skip]
        let input = vec![
            4.0, 8.0,
            12.0, 16.0,
        ];
        let cfg = PoolConfig::new(PoolType::Average, 2, 1, 1);
        let (out, oh, ow) = pool_2d(&input, 2, 2, &cfg).unwrap();
        assert_eq!((oh, ow), (3, 3));
        assert!(approx_eq(&out, &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0], TOL));
    }

    #[test]
    fn pool_2d_avg_count_include_pad() {
        let input = vec![4.0, 8.0, 12.0, 16.0];
        let cfg = PoolConfig::new(PoolType::AvgPoolCountIncludePad, 2, 2, 0);
        let (out, oh, ow) = pool_2d(&input, 2, 2, &cfg).unwrap();
        assert_eq!((oh, ow), (1, 1));
        assert!(approx_eq(&out, &[10.0], TOL));
    }

    #[test]
    fn pool_2d_single_element() {
        let input = vec![42.0];
        let cfg = PoolConfig::new(PoolType::Max, 1, 1, 0);
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
        let cfg = PoolConfig { pool_type: PoolType::GlobalMax, ..PoolConfig::default() };
        let (out, oh, ow) = pool_2d(&input, 2, 2, &cfg).unwrap();
        assert_eq!((oh, ow), (1, 1));
        assert!(approx_eq(&out, &[9.0], TOL));
    }

    #[test]
    fn pool_2d_global_avg() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let cfg = PoolConfig { pool_type: PoolType::GlobalAverage, ..PoolConfig::default() };
        let (out, oh, ow) = pool_2d(&input, 2, 2, &cfg).unwrap();
        assert_eq!((oh, ow), (1, 1));
        assert!(approx_eq(&out, &[5.0], TOL));
    }

    #[test]
    fn pool_2d_wrong_input_size() {
        let input = vec![1.0, 2.0, 3.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0);
        assert!(pool_2d(&input, 2, 2, &cfg).is_err());
    }

    #[test]
    fn pool_2d_non_square_input() {
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0);
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
        #[rustfmt::skip]
        let input = vec![
            1.0,  2.0,  3.0,  4.0,
            5.0,  6.0,  7.0,  8.0,
            9.0,  10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let out = adaptive_avg_pool_2d(&input, 4, 4, 2, 2).unwrap();
        assert_eq!(out.len(), 4);
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
        let input = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let out = global_avg_pool(&input, &[3]).unwrap();
        assert!(approx_eq(&out, &[2.0, 20.0], TOL));
    }

    #[test]
    fn global_avg_pool_2d_spatial() {
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
        let input = vec![1.0, 5.0, 3.0, 2.0, 9.0, 4.0];
        let out = global_max_pool(&input, &[2]).unwrap();
        assert!(approx_eq(&out, &[5.0, 3.0, 9.0], TOL));
    }

    #[test]
    fn global_max_pool_2d_spatial() {
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

    // ═══════════════════════════════════════════════════════════════
    // New tests: max_pool1d with indices
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn max_pool1d_returns_correct_indices() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0);
        let (vals, idxs) = max_pool1d(&input, &cfg).unwrap();
        assert!(approx_eq(&vals, &[3.0, 3.0, 5.0, 5.0], TOL));
        assert_eq!(idxs, vec![1, 1, 3, 3]);
    }

    #[test]
    fn max_pool1d_stride2_indices() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 2, 0);
        let (vals, idxs) = max_pool1d(&input, &cfg).unwrap();
        assert!(approx_eq(&vals, &[3.0, 5.0, 6.0], TOL));
        assert_eq!(idxs, vec![1, 3, 5]);
    }

    #[test]
    fn max_pool1d_single_element_index() {
        let input = vec![42.0];
        let cfg = PoolConfig::new(PoolType::Max, 1, 1, 0);
        let (vals, idxs) = max_pool1d(&input, &cfg).unwrap();
        assert!(approx_eq(&vals, &[42.0], TOL));
        assert_eq!(idxs, vec![0]);
    }

    #[test]
    fn max_pool1d_all_same_values() {
        let input = vec![5.0; 8];
        let cfg = PoolConfig::new(PoolType::Max, 3, 1, 0);
        let (vals, idxs) = max_pool1d(&input, &cfg).unwrap();
        assert_eq!(vals.len(), 6);
        // All values equal; first occurrence in each window wins.
        for &v in &vals {
            assert!((v - 5.0).abs() < TOL);
        }
        for (i, &idx) in idxs.iter().enumerate() {
            assert_eq!(idx, i); // first element in window
        }
    }

    #[test]
    fn max_pool1d_with_padding_indices() {
        let input = vec![1.0, 2.0, 3.0];
        let cfg = PoolConfig::new(PoolType::Max, 3, 1, 1);
        let (vals, idxs) = max_pool1d(&input, &cfg).unwrap();
        assert!(approx_eq(&vals, &[2.0, 3.0, 3.0], TOL));
        assert_eq!(idxs, vec![1, 2, 2]);
    }

    #[test]
    fn max_pool1d_negative_values_indices() {
        let input = vec![-5.0, -3.0, -4.0, -1.0, -2.0];
        let cfg = PoolConfig::new(PoolType::Max, 3, 1, 0);
        let (vals, idxs) = max_pool1d(&input, &cfg).unwrap();
        assert!(approx_eq(&vals, &[-3.0, -1.0, -1.0], TOL));
        assert_eq!(idxs, vec![1, 3, 3]);
    }

    // ═══════════════════════════════════════════════════════════════
    // New tests: avg_pool1d
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn avg_pool1d_basic() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let cfg = PoolConfig::new(PoolType::Average, 2, 1, 0);
        let out = avg_pool1d(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[2.0, 2.5, 3.5, 4.5], TOL));
    }

    #[test]
    fn avg_pool1d_stride_2() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let cfg = PoolConfig::new(PoolType::Average, 2, 2, 0);
        let out = avg_pool1d(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 7.0], TOL));
    }

    #[test]
    fn avg_pool1d_with_padding() {
        let input = vec![3.0, 6.0, 9.0];
        let cfg = PoolConfig::new(PoolType::Average, 3, 1, 1);
        let out = avg_pool1d(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 6.0, 5.0], TOL));
    }

    // ═══════════════════════════════════════════════════════════════
    // New tests: dilation
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn max_pool1d_dilation_2() {
        // input: [1, 2, 3, 4, 5], k=2, stride=1, dilation=2
        // effective kernel size = 2*1+1 = 3
        // window 0: positions 0, 2 → max(1, 3) = 3
        // window 1: positions 1, 3 → max(2, 4) = 4
        // window 2: positions 2, 4 → max(3, 5) = 5
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size: 2,
            stride: 1,
            padding: 0,
            dilation: 2,
            ceil_mode: false,
        };
        let (vals, idxs) = max_pool1d(&input, &cfg).unwrap();
        assert!(approx_eq(&vals, &[3.0, 4.0, 5.0], TOL));
        assert_eq!(idxs, vec![2, 3, 4]);
    }

    #[test]
    fn avg_pool1d_dilation_2() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = PoolConfig {
            pool_type: PoolType::Average,
            kernel_size: 2,
            stride: 1,
            padding: 0,
            dilation: 2,
            ceil_mode: false,
        };
        let out = avg_pool1d(&input, &cfg).unwrap();
        // window 0: (1+3)/2=2, window 1: (2+4)/2=3, window 2: (3+5)/2=4
        assert!(approx_eq(&out, &[2.0, 3.0, 4.0], TOL));
    }

    #[test]
    fn dilation_zero_rejected() {
        let cfg = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size: 2,
            stride: 1,
            padding: 0,
            dilation: 0,
            ceil_mode: false,
        };
        assert!(max_pool1d(&[1.0, 2.0], &cfg).is_err());
    }

    #[test]
    fn dilation_3_max_pool() {
        // input: [0, 1, 2, 3, 4, 5, 6], k=3, stride=1, dilation=3
        // effective kernel size = 3*2+1 = 7, only 1 output position
        // window 0: positions 0, 3, 6 → max(0, 3, 6) = 6
        let input: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let cfg = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size: 3,
            stride: 1,
            padding: 0,
            dilation: 3,
            ceil_mode: false,
        };
        let (vals, _) = max_pool1d(&input, &cfg).unwrap();
        assert!(approx_eq(&vals, &[6.0], TOL));
    }

    // ═══════════════════════════════════════════════════════════════
    // New tests: ceil_mode
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn ceil_mode_adds_extra_output() {
        // input len 5, k=3, stride=2, pad=0
        // floor: (5-3)/2+1 = 2
        // ceil:  ceil((5-3)/2)+1 = 2  (same here)
        // input len 6, k=3, stride=2, pad=0
        // floor: (6-3)/2+1 = 2
        // ceil:  ceil((6-3)/2)+1 = 3
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cfg_floor = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size: 3,
            stride: 2,
            padding: 0,
            dilation: 1,
            ceil_mode: false,
        };
        let cfg_ceil = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size: 3,
            stride: 2,
            padding: 0,
            dilation: 1,
            ceil_mode: true,
        };
        let (vals_f, _) = max_pool1d(&input, &cfg_floor).unwrap();
        let (vals_c, _) = max_pool1d(&input, &cfg_ceil).unwrap();
        assert_eq!(vals_f.len(), 2); // floor
        assert_eq!(vals_c.len(), 3); // ceil gives one extra
    }

    #[test]
    fn ceil_mode_false_matches_default() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size: 2,
            stride: 2,
            padding: 0,
            dilation: 1,
            ceil_mode: false,
        };
        let (vals, _) = max_pool1d(&input, &cfg).unwrap();
        let default_cfg = PoolConfig::new(PoolType::Max, 2, 2, 0);
        let (vals_d, _) = max_pool1d(&input, &default_cfg).unwrap();
        assert!(approx_eq(&vals, &vals_d, TOL));
    }

    // ═══════════════════════════════════════════════════════════════
    // New tests: Lp-norm pooling
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn lp_pool1d_l2_norm() {
        // L2 pooling: sqrt(sum(x^2)) per window
        let input = vec![3.0, 4.0, 0.0, 5.0, 12.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0); // pool_type ignored for lp
        let out = lp_pool1d(&input, 2.0, &cfg).unwrap();
        // window 0: sqrt(9+16) = 5
        // window 1: sqrt(16+0) = 4
        // window 2: sqrt(0+25) = 5
        // window 3: sqrt(25+144) = 13
        assert!(approx_eq(&out, &[5.0, 4.0, 5.0, 13.0], TOL));
    }

    #[test]
    fn lp_pool1d_l1_norm() {
        // L1 pooling: sum(|x|) per window
        let input = vec![-1.0, 2.0, -3.0, 4.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0);
        let out = lp_pool1d(&input, 1.0, &cfg).unwrap();
        // window 0: 1+2 = 3
        // window 1: 2+3 = 5
        // window 2: 3+4 = 7
        assert!(approx_eq(&out, &[3.0, 5.0, 7.0], TOL));
    }

    #[test]
    fn lp_pool1d_p_zero_rejected() {
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0);
        assert!(lp_pool1d(&[1.0, 2.0], 0.0, &cfg).is_err());
    }

    #[test]
    fn lp_pool1d_negative_p_rejected() {
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0);
        assert!(lp_pool1d(&[1.0, 2.0], -1.0, &cfg).is_err());
    }

    #[test]
    fn lp_pool1d_stride_2() {
        let input = vec![3.0, 4.0, 5.0, 12.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 2, 0);
        let out = lp_pool1d(&input, 2.0, &cfg).unwrap();
        // window 0: sqrt(9+16) = 5
        // window 1: sqrt(25+144) = 13
        assert!(approx_eq(&out, &[5.0, 13.0], TOL));
    }

    #[test]
    fn lp_pool1d_via_pool_type() {
        let input = vec![3.0, 4.0, 0.0, 5.0];
        let cfg = PoolConfig::new(PoolType::Lp(2.0), 2, 1, 0);
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[5.0, 4.0, 5.0], TOL));
    }

    #[test]
    fn lp_pool1d_large_p() {
        // As p → ∞, Lp norm → max(|x|). Use p=20 to stay within f32 range.
        let input = vec![1.0, 10.0, 2.0];
        let cfg = PoolConfig::new(PoolType::Max, 3, 1, 0);
        let out = lp_pool1d(&input, 20.0, &cfg).unwrap();
        assert!((out[0] - 10.0).abs() < 0.01);
    }

    // ═══════════════════════════════════════════════════════════════
    // New tests: max_unpool1d
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn max_unpool1d_basic() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0);
        let (pooled, indices) = max_pool1d(&input, &cfg).unwrap();
        let unpooled = max_unpool1d(&pooled, &indices, 5).unwrap();
        // Indices: [1, 1, 3, 3] → output[1] = 3, output[3] = 5 (last write wins)
        assert_eq!(unpooled.len(), 5);
        assert!((unpooled[1] - 3.0).abs() < TOL);
        assert!((unpooled[3] - 5.0).abs() < TOL);
    }

    #[test]
    fn max_unpool1d_round_trip_preserves_maxima() {
        let input = vec![10.0, 20.0, 30.0, 40.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 2, 0);
        let (pooled, indices) = max_pool1d(&input, &cfg).unwrap();
        let unpooled = max_unpool1d(&pooled, &indices, 4).unwrap();
        assert!((unpooled[1] - 20.0).abs() < TOL);
        assert!((unpooled[3] - 40.0).abs() < TOL);
        // Non-max positions are zero.
        assert!((unpooled[0]).abs() < TOL);
        assert!((unpooled[2]).abs() < TOL);
    }

    #[test]
    fn max_unpool1d_length_mismatch_rejected() {
        assert!(max_unpool1d(&[1.0, 2.0], &[0], 5).is_err());
    }

    #[test]
    fn max_unpool1d_index_out_of_bounds_rejected() {
        assert!(max_unpool1d(&[1.0], &[10], 5).is_err());
    }

    #[test]
    fn max_unpool1d_empty() {
        let out = max_unpool1d(&[], &[], 5).unwrap();
        assert_eq!(out, vec![0.0; 5]);
    }

    // ═══════════════════════════════════════════════════════════════
    // New tests: adaptive_max_pool1d
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn adaptive_max_pool1d_basic() {
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let (vals, idxs) = adaptive_max_pool1d(&input, 5).unwrap();
        assert_eq!(vals.len(), 5);
        // bins: [0,1] [2,3] [4,5] [6,7] [8,9]
        assert!(approx_eq(&vals, &[1.0, 3.0, 5.0, 7.0, 9.0], TOL));
        assert_eq!(idxs, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn adaptive_max_pool1d_identity() {
        let input = vec![5.0, 3.0, 7.0];
        let (vals, idxs) = adaptive_max_pool1d(&input, 3).unwrap();
        assert!(approx_eq(&vals, &[5.0, 3.0, 7.0], TOL));
        assert_eq!(idxs, vec![0, 1, 2]);
    }

    #[test]
    fn adaptive_max_pool1d_to_one() {
        let input = vec![2.0, 8.0, 4.0, 6.0];
        let (vals, idxs) = adaptive_max_pool1d(&input, 1).unwrap();
        assert!(approx_eq(&vals, &[8.0], TOL));
        assert_eq!(idxs, vec![1]);
    }

    #[test]
    fn adaptive_max_pool1d_empty_rejected() {
        assert!(adaptive_max_pool1d(&[], 1).is_err());
    }

    #[test]
    fn adaptive_max_pool1d_zero_output_rejected() {
        assert!(adaptive_max_pool1d(&[1.0], 0).is_err());
    }

    #[test]
    fn adaptive_max_pool1d_output_larger_rejected() {
        assert!(adaptive_max_pool1d(&[1.0], 5).is_err());
    }

    // ═══════════════════════════════════════════════════════════════
    // New tests: AVX2 vs scalar parity
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn avx2_max_pool1d_matches_scalar() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let cfg = PoolConfig::new(PoolType::Max, 3, 1, 0);
        let (vals_dispatch, idxs_dispatch) = max_pool1d(&input, &cfg).unwrap();
        let (vals_scalar, idxs_scalar) = max_pool_1d_scalar(&input, &cfg).unwrap();
        assert!(approx_eq(&vals_dispatch, &vals_scalar, TOL));
        assert_eq!(idxs_dispatch, idxs_scalar);
    }

    #[test]
    fn avx2_avg_pool1d_matches_scalar() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32).cos()).collect();
        let cfg = PoolConfig::new(PoolType::Average, 4, 2, 0);
        let out_dispatch = avg_pool1d(&input, &cfg).unwrap();
        let out_scalar = avg_pool_1d_scalar(&input, &cfg).unwrap();
        assert!(approx_eq(&out_dispatch, &out_scalar, TOL));
    }

    #[test]
    fn avx2_max_pool1d_explicit_matches_dispatch() {
        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let cfg = PoolConfig::new(PoolType::Max, 4, 2, 1);
        let (v1, i1) = max_pool1d(&input, &cfg).unwrap();
        let (v2, i2) = max_pool1d_avx2(&input, &cfg).unwrap();
        assert!(approx_eq(&v1, &v2, TOL));
        assert_eq!(i1, i2);
    }

    #[test]
    fn avx2_avg_pool1d_explicit_matches_dispatch() {
        let input: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let cfg = PoolConfig::new(PoolType::Average, 4, 4, 0);
        let o1 = avg_pool1d(&input, &cfg).unwrap();
        let o2 = avg_pool1d_avx2(&input, &cfg).unwrap();
        assert!(approx_eq(&o1, &o2, TOL));
    }

    #[test]
    fn avx2_max_pool1d_small_input() {
        // Fewer than 8 output positions → scalar tail only.
        let input = vec![5.0, 3.0, 7.0, 1.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0);
        let (vals, idxs) = max_pool1d_avx2(&input, &cfg).unwrap();
        assert!(approx_eq(&vals, &[5.0, 7.0, 7.0], TOL));
        assert_eq!(idxs, vec![0, 2, 2]);
    }

    #[test]
    fn avx2_avg_pool1d_small_input() {
        let input = vec![2.0, 4.0, 6.0];
        let cfg = PoolConfig::new(PoolType::Average, 2, 1, 0);
        let out = avg_pool1d_avx2(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 5.0], TOL));
    }

    #[test]
    fn avx2_max_pool1d_large_stride() {
        let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let cfg = PoolConfig::new(PoolType::Max, 4, 8, 0);
        let (v1, i1) = max_pool1d(&input, &cfg).unwrap();
        let (v2, i2) = max_pool1d_avx2(&input, &cfg).unwrap();
        assert!(approx_eq(&v1, &v2, TOL));
        assert_eq!(i1, i2);
    }

    #[test]
    fn avx2_max_pool1d_with_dilation() {
        let input: Vec<f32> = (0..32).map(|i| (i as f32).sin()).collect();
        let cfg = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size: 3,
            stride: 1,
            padding: 0,
            dilation: 2,
            ceil_mode: false,
        };
        let (v1, i1) = max_pool1d(&input, &cfg).unwrap();
        let (v2, i2) = max_pool_1d_scalar(&input, &cfg).unwrap();
        assert!(approx_eq(&v1, &v2, TOL));
        assert_eq!(i1, i2);
    }

    // ═══════════════════════════════════════════════════════════════
    // New tests: runtime dispatch
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn dispatch_max_pool1d_basic() {
        let input = vec![1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 9.0, 0.0];
        let cfg = PoolConfig::new(PoolType::Max, 3, 1, 0);
        let (vals, idxs) = max_pool1d(&input, &cfg).unwrap();
        assert_eq!(vals.len(), 8);
        // Verify each window manually.
        assert!(approx_eq(&vals, &[5.0, 7.0, 7.0, 8.0, 8.0, 8.0, 9.0, 9.0], TOL));
        // Indices point to the correct source positions.
        for (i, &idx) in idxs.iter().enumerate() {
            assert!((input[idx] - vals[i]).abs() < TOL);
        }
    }

    #[test]
    fn dispatch_avg_pool1d_basic() {
        let input = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let cfg = PoolConfig::new(PoolType::Average, 3, 1, 0);
        let out = avg_pool1d(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[4.0, 6.0, 8.0], TOL));
    }

    // ═══════════════════════════════════════════════════════════════
    // New tests: PoolConfig validation and construction
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn pool_config_default() {
        let cfg = PoolConfig::default();
        assert_eq!(cfg.pool_type, PoolType::Max);
        assert_eq!(cfg.kernel_size, 1);
        assert_eq!(cfg.stride, 1);
        assert_eq!(cfg.padding, 0);
        assert_eq!(cfg.dilation, 1);
        assert!(!cfg.ceil_mode);
    }

    #[test]
    fn pool_config_new_sets_defaults() {
        let cfg = PoolConfig::new(PoolType::Average, 3, 2, 1);
        assert_eq!(cfg.dilation, 1);
        assert!(!cfg.ceil_mode);
    }

    #[test]
    fn effective_kernel_size_no_dilation() {
        let cfg = PoolConfig::new(PoolType::Max, 3, 1, 0);
        assert_eq!(cfg.effective_kernel_size(), 3);
    }

    #[test]
    fn effective_kernel_size_with_dilation() {
        let cfg = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size: 3,
            stride: 1,
            padding: 0,
            dilation: 2,
            ceil_mode: false,
        };
        assert_eq!(cfg.effective_kernel_size(), 5); // 2*(3-1)+1 = 5
    }

    #[test]
    fn adaptive_type_variant() {
        let cfg = PoolConfig { pool_type: PoolType::Adaptive, ..PoolConfig::default() };
        assert!(cfg.validate().is_ok());
    }

    // ═══════════════════════════════════════════════════════════════
    // New tests: output_len with dilation and ceil_mode
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn output_len_with_dilation() {
        // input=7, k=3, stride=1, pad=0, dilation=2
        // effective_k = 2*(3-1)+1 = 5
        // out = (7-5)/1+1 = 3
        assert_eq!(output_len(7, 3, 1, 0, 2, false), 3);
    }

    #[test]
    fn output_len_with_ceil_mode() {
        // input=6, k=3, stride=2, pad=0, dilation=1
        // floor: (6-3)/2+1 = 2
        // ceil: ceil((6-3)/2)+1 = ceil(1.5)+1 = 3
        assert_eq!(output_len(6, 3, 2, 0, 1, false), 2);
        assert_eq!(output_len(6, 3, 2, 0, 1, true), 3);
    }

    #[test]
    fn output_len_exact_division_ceil_matches_floor() {
        // When division is exact, ceil == floor.
        assert_eq!(output_len(5, 3, 1, 0, 1, false), 3);
        assert_eq!(output_len(5, 3, 1, 0, 1, true), 3);
    }

    #[test]
    fn output_len_dilation_and_ceil() {
        // input=10, k=3, stride=2, pad=0, dilation=2
        // effective_k = 5, diff = 10-5 = 5
        // floor: 5/2+1 = 3
        // ceil: (5+1)/2+1 = 4
        assert_eq!(output_len(10, 3, 2, 0, 2, false), 3);
        assert_eq!(output_len(10, 3, 2, 0, 2, true), 4);
    }

    // ═══════════════════════════════════════════════════════════════
    // Stress tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn max_pool1d_large_input_stress() {
        let input: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();
        let cfg = PoolConfig::new(PoolType::Max, 8, 4, 0);
        let (vals, idxs) = max_pool1d(&input, &cfg).unwrap();
        assert_eq!(vals.len(), (4096 - 8) / 4 + 1);
        for (i, &idx) in idxs.iter().enumerate() {
            assert!((input[idx] - vals[i]).abs() < TOL);
        }
    }

    #[test]
    fn avg_pool1d_large_input_stress() {
        let input: Vec<f32> = (0..4096).map(|i| i as f32).collect();
        let cfg = PoolConfig::new(PoolType::Average, 16, 8, 0);
        let out = avg_pool1d(&input, &cfg).unwrap();
        let expected_len = (4096 - 16) / 8 + 1;
        assert_eq!(out.len(), expected_len);
    }

    #[test]
    fn avx2_scalar_parity_large_with_padding() {
        let input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).cos()).collect();
        let cfg = PoolConfig::new(PoolType::Max, 5, 2, 2);
        let (v1, i1) = max_pool1d(&input, &cfg).unwrap();
        let (v2, i2) = max_pool_1d_scalar(&input, &cfg).unwrap();
        assert!(approx_eq(&v1, &v2, TOL));
        assert_eq!(i1, i2);
    }

    #[test]
    fn max_pool1d_monotonic_increasing() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let cfg = PoolConfig::new(PoolType::Max, 3, 1, 0);
        let (vals, _) = max_pool1d(&input, &cfg).unwrap();
        // Max of each window of monotonic increasing = last element of window.
        for (i, &v) in vals.iter().enumerate() {
            assert!((v - (i + 2) as f32).abs() < TOL);
        }
    }

    #[test]
    fn max_pool1d_monotonic_decreasing() {
        let input: Vec<f32> = (0..100).rev().map(|i| i as f32).collect();
        let cfg = PoolConfig::new(PoolType::Max, 3, 1, 0);
        let (vals, _) = max_pool1d(&input, &cfg).unwrap();
        // Max of each window of monotonic decreasing = first element of window.
        for (i, &v) in vals.iter().enumerate() {
            assert!((v - (99 - i) as f32).abs() < TOL);
        }
    }

    #[test]
    fn lp_pool1d_with_dilation() {
        let input = vec![1.0, 0.0, 2.0, 0.0, 3.0];
        let cfg = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size: 2,
            stride: 1,
            padding: 0,
            dilation: 2,
            ceil_mode: false,
        };
        let out = lp_pool1d(&input, 2.0, &cfg).unwrap();
        // window 0: positions 0,2 → sqrt(1+4) = sqrt(5) ≈ 2.236
        // window 1: positions 1,3 → sqrt(0+0) = 0
        // window 2: positions 2,4 → sqrt(4+9) = sqrt(13) ≈ 3.606
        assert!((out[0] - 5.0f32.sqrt()).abs() < TOL);
        assert!((out[1]).abs() < TOL);
        assert!((out[2] - 13.0f32.sqrt()).abs() < TOL);
    }

    #[test]
    fn max_unpool1d_large_output() {
        let pooled = vec![10.0, 20.0, 30.0];
        let indices = vec![5, 15, 25];
        let out = max_unpool1d(&pooled, &indices, 30).unwrap();
        assert_eq!(out.len(), 30);
        assert!((out[5] - 10.0).abs() < TOL);
        assert!((out[15] - 20.0).abs() < TOL);
        assert!((out[25] - 30.0).abs() < TOL);
        // All other positions are zero.
        for (i, &v) in out.iter().enumerate() {
            if i != 5 && i != 15 && i != 25 {
                assert!(v.abs() < TOL);
            }
        }
    }

    #[test]
    fn adaptive_max_pool1d_non_uniform_bins() {
        // 7 elements → 3 bins: [0,1] [2,3,4] [5,6]  (floor division)
        let input = vec![1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 4.0];
        let (vals, idxs) = adaptive_max_pool1d(&input, 3).unwrap();
        assert_eq!(vals.len(), 3);
        // bin 0: floor(0*7/3)=0..floor(1*7/3)=2 → [1,5] → max=5 at idx=1
        // bin 1: 2..floor(2*7/3)=4 → [2,8] → max=8 at idx=3
        // bin 2: 4..7 → [3,7,4] → max=7 at idx=5
        assert!(approx_eq(&vals, &[5.0, 8.0, 7.0], TOL));
        assert_eq!(idxs, vec![1, 3, 5]);
    }

    #[test]
    fn global_avg_pool_three_channels() {
        // 3 channels, spatial=4
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // ch0: mean=2.5
            10.0, 20.0, 30.0, 40.0, // ch1: mean=25
            -1.0, -2.0, -3.0, -4.0, // ch2: mean=-2.5
        ];
        let out = global_avg_pool(&input, &[4]).unwrap();
        assert!(approx_eq(&out, &[2.5, 25.0, -2.5], TOL));
    }

    #[test]
    fn global_max_pool_three_channels() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, -1.0, -2.0, -3.0, -4.0];
        let out = global_max_pool(&input, &[4]).unwrap();
        assert!(approx_eq(&out, &[4.0, 40.0, -1.0], TOL));
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration-style tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn pool_then_unpool_round_trip() {
        let input = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 2, 0);
        let (pooled, indices) = max_pool1d(&input, &cfg).unwrap();
        let unpooled = max_unpool1d(&pooled, &indices, 6).unwrap();
        // Only max positions should have values.
        for &idx in &indices {
            assert!(unpooled[idx] > 0.0);
        }
    }

    #[test]
    fn adaptive_then_unpool() {
        let input: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let (pooled, indices) = adaptive_max_pool1d(&input, 5).unwrap();
        let unpooled = max_unpool1d(&pooled, &indices, 20).unwrap();
        for (&val, &idx) in pooled.iter().zip(indices.iter()) {
            assert!((unpooled[idx] - val).abs() < TOL);
        }
    }

    #[test]
    fn max_pool1d_kernel_size_1_is_identity() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let cfg = PoolConfig::new(PoolType::Max, 1, 1, 0);
        let (vals, idxs) = max_pool1d(&input, &cfg).unwrap();
        assert!(approx_eq(&vals, &input, TOL));
        let expected_idxs: Vec<usize> = (0..16).collect();
        assert_eq!(idxs, expected_idxs);
    }

    #[test]
    fn avg_pool1d_kernel_size_1_is_identity() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let cfg = PoolConfig::new(PoolType::Average, 1, 1, 0);
        let out = avg_pool1d(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &input, TOL));
    }

    #[test]
    fn max_pool1d_all_inf() {
        let input = vec![f32::NEG_INFINITY; 4];
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0);
        let (vals, _) = max_pool1d(&input, &cfg).unwrap();
        for &v in &vals {
            assert!(v == f32::NEG_INFINITY);
        }
    }

    #[test]
    fn avg_pool1d_zeros() {
        let input = vec![0.0; 10];
        let cfg = PoolConfig::new(PoolType::Average, 3, 1, 0);
        let out = avg_pool1d(&input, &cfg).unwrap();
        for &v in &out {
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn lp_pool1d_all_zeros() {
        let input = vec![0.0; 8];
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0);
        let out = lp_pool1d(&input, 2.0, &cfg).unwrap();
        for &v in &out {
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn max_pool1d_nan_propagation() {
        let input = vec![1.0, f32::NAN, 3.0, 4.0];
        let cfg = PoolConfig::new(PoolType::Max, 2, 1, 0);
        let (vals, _) = max_pool1d(&input, &cfg).unwrap();
        // NaN comparison: NaN > x is false, so max stays at non-NaN.
        // Window [1, NaN] → max stays 1 (NAN > 1.0 is false)
        // Window [NaN, 3] → 3 > NaN is false; max starts -inf, then NaN doesn't win, then 3 wins
        // Actually: max starts at -inf. 1.0 > -inf → max=1. NaN > 1 → false. So window 0 → 1.
        // Window 1: -inf, NaN > -inf → false. 3 > -inf → true, max=3. So window 1 → 3.
        assert!((vals[0] - 1.0).abs() < TOL || vals[0].is_nan());
    }
}
