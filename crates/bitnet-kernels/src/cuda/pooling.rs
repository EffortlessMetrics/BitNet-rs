//! CUDA pooling kernels with CPU fallback.
//!
//! # Kernel strategy
//!
//! Provides both 1-D and 2-D pooling operations over contiguous `f32` slices:
//!
//! - **1-D pooling** — [`PoolingConfig`] / [`pooling_cpu`] for sliding-window
//!   max and average with configurable kernel size, stride, and padding.
//! - **2-D pooling** — [`Pool2dConfig`] / [`max_pool2d_cpu`] / [`avg_pool2d_cpu`] /
//!   [`adaptive_avg_pool2d_cpu`] for spatial pooling across (H, W) dimensions
//!   with support for batched multi-channel inputs.
//!
//! Each CUDA kernel launches one thread per output element. Grid dimensions
//! are computed from the output tensor shape.
//!
//! # CPU fallback
//!
//! Pure-Rust implementations are provided for correctness testing and non-GPU
//! environments. The `*_forward` functions automatically dispatch to GPU when
//! available, falling back to CPU otherwise.

use bitnet_common::{KernelError, Result};

// ===================================================================
// 1-D Pooling (existing)
// ===================================================================

/// Pooling operation variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaPoolType {
    /// Sliding-window maximum.
    Max,
    /// Sliding-window arithmetic mean.
    Average,
}

/// Launch / shape configuration for a 1-D pooling operation.
#[derive(Debug, Clone)]
pub struct PoolingConfig {
    /// Type of pooling to perform.
    pub pool_type: CudaPoolType,
    /// Number of elements in the input.
    pub input_len: usize,
    /// Window (kernel) size.
    pub kernel_size: usize,
    /// Stride between successive windows.
    pub stride: usize,
    /// Zero-padding added to each side of the input.
    pub padding: usize,
    /// Threads per block — typically `min(output_len, 1024)`.
    pub threads_per_block: u32,
}

impl PoolingConfig {
    /// Create a validated configuration for the given parameters.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] when any dimension is
    /// zero or the configuration would produce no output elements.
    pub fn new(
        pool_type: CudaPoolType,
        input_len: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        if input_len == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "input_len must be > 0".into() }.into()
            );
        }
        if kernel_size == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "kernel_size must be > 0".into() }.into()
            );
        }
        if stride == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "stride must be > 0".into() }.into()
            );
        }

        let out_len = output_len_1d(input_len, kernel_size, stride, padding);
        if out_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "pooling produces 0 outputs: input_len={input_len}, \
                     kernel_size={kernel_size}, stride={stride}, \
                     padding={padding}"
                ),
            }
            .into());
        }

        let threads_per_block = (out_len as u32).min(1024);
        Ok(Self { pool_type, input_len, kernel_size, stride, padding, threads_per_block })
    }

    /// Number of output elements this configuration produces.
    #[inline]
    pub fn output_len(&self) -> usize {
        output_len_1d(self.input_len, self.kernel_size, self.stride, self.padding)
    }

    /// CUDA grid dimensions `(ceil(output_len / threads_per_block), 1, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let n = self.output_len() as u32;
        let tpb = self.threads_per_block;
        (n.div_ceil(tpb), 1, 1)
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

#[inline]
fn output_len_1d(input_len: usize, kernel_size: usize, stride: usize, padding: usize) -> usize {
    let padded = input_len + 2 * padding;
    if padded < kernel_size {
        return 0;
    }
    (padded - kernel_size) / stride + 1
}

// -------------------------------------------------------------------
// 1-D CPU fallback
// -------------------------------------------------------------------

/// CPU fallback for 1-D pooling.
pub fn pooling_cpu(input: &[f32], output: &mut [f32], config: &PoolingConfig) -> Result<()> {
    let out_n = config.output_len();
    if input.len() < config.input_len {
        return Err(KernelError::InvalidArguments {
            reason: format!("pooling input length {} < expected {}", input.len(), config.input_len),
        }
        .into());
    }
    if output.len() < out_n {
        return Err(KernelError::InvalidArguments {
            reason: format!("pooling output length {} < expected {}", output.len(), out_n),
        }
        .into());
    }

    match config.pool_type {
        CudaPoolType::Max => max_pool_1d_cpu(input, output, config, out_n),
        CudaPoolType::Average => avg_pool_1d_cpu(input, output, config, out_n),
    }
    Ok(())
}

fn max_pool_1d_cpu(input: &[f32], output: &mut [f32], config: &PoolingConfig, out_n: usize) {
    let n = config.input_len;
    let pad = config.padding;
    for (i, out_val) in output.iter_mut().enumerate().take(out_n) {
        let ws = i * config.stride;
        let mut max_val = f32::NEG_INFINITY;
        for k in 0..config.kernel_size {
            let pos = ws + k;
            let val =
                if pos < pad || pos >= n + pad { f32::NEG_INFINITY } else { input[pos - pad] };
            if val > max_val {
                max_val = val;
            }
        }
        *out_val = max_val;
    }
}

fn avg_pool_1d_cpu(input: &[f32], output: &mut [f32], config: &PoolingConfig, out_n: usize) {
    let n = config.input_len;
    let pad = config.padding;
    for (i, out_val) in output.iter_mut().enumerate().take(out_n) {
        let ws = i * config.stride;
        let mut sum = 0.0_f32;
        for k in 0..config.kernel_size {
            let pos = ws + k;
            if pos >= pad && pos < n + pad {
                sum += input[pos - pad];
            }
        }
        *out_val = sum / config.kernel_size as f32;
    }
}

/// Launch stub for the 1-D pooling CUDA kernel.
pub fn launch_pooling(_input: &[f32], _output: &mut [f32], config: &PoolingConfig) -> Result<()> {
    log::debug!(
        "pooling stub: type={:?}, input_len={}, kernel={}, \
         stride={}, padding={}, grid={:?}",
        config.pool_type,
        config.input_len,
        config.kernel_size,
        config.stride,
        config.padding,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "pooling CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Apply 1-D pooling with automatic dispatch: GPU if available, else CPU.
pub fn pooling_forward(input: &[f32], output: &mut [f32], config: &PoolingConfig) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_pooling(input, output, config)
        {
            return Ok(());
        }
    }
    pooling_cpu(input, output, config)
}

// ===================================================================
// 2-D Pooling
// ===================================================================

// -------------------------------------------------------------------
// CUDA kernel source strings
// -------------------------------------------------------------------

/// CUDA kernel source for 2-D max pooling.
///
/// Each thread computes one output element. The grid covers the full
/// `(batch * channels * out_h * out_w)` output tensor.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const MAX_POOL2D_KERNEL_SRC: &str = r#"
extern "C" __global__ void max_pool2d_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w, int out_h, int out_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c  = (idx / (out_w * out_h)) % channels;
    int b  = idx / (out_w * out_h * channels);

    float max_val = -1e30f;
    for (int kh = 0; kh < kernel_h; ++kh) {
        int ih = oh * stride_h - pad_h + kh * dilation_h;
        if (ih < 0 || ih >= in_h) continue;
        for (int kw = 0; kw < kernel_w; ++kw) {
            int iw = ow * stride_w - pad_w + kw * dilation_w;
            if (iw < 0 || iw >= in_w) continue;
            int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
            float val = input[in_idx];
            if (val > max_val) max_val = val;
        }
    }
    output[idx] = max_val;
}
"#;

/// CUDA kernel source for 2-D average pooling (count_include_pad semantics).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const AVG_POOL2D_KERNEL_SRC: &str = r#"
extern "C" __global__ void avg_pool2d_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w, int out_h, int out_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c  = (idx / (out_w * out_h)) % channels;
    int b  = idx / (out_w * out_h * channels);

    float sum = 0.0f;
    int count = kernel_h * kernel_w;
    for (int kh = 0; kh < kernel_h; ++kh) {
        int ih = oh * stride_h - pad_h + kh * dilation_h;
        if (ih < 0 || ih >= in_h) continue;
        for (int kw = 0; kw < kernel_w; ++kw) {
            int iw = ow * stride_w - pad_w + kw * dilation_w;
            if (iw < 0 || iw >= in_w) continue;
            int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
            sum += input[in_idx];
        }
    }
    output[idx] = sum / (float)count;
}
"#;

/// CUDA kernel source for adaptive average pooling (PyTorch-style boundaries).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ADAPTIVE_AVG_POOL2D_KERNEL_SRC: &str = r#"
extern "C" __global__ void adaptive_avg_pool2d_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c  = (idx / (out_w * out_h)) % channels;
    int b  = idx / (out_w * out_h * channels);

    int h_start = (oh * in_h) / out_h;
    int h_end   = ((oh + 1) * in_h) / out_h;
    int w_start = (ow * in_w) / out_w;
    int w_end   = ((ow + 1) * in_w) / out_w;

    float sum = 0.0f;
    int count = (h_end - h_start) * (w_end - w_start);
    for (int ih = h_start; ih < h_end; ++ih) {
        for (int iw = w_start; iw < w_end; ++iw) {
            int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
            sum += input[in_idx];
        }
    }
    output[idx] = (count > 0) ? (sum / (float)count) : 0.0f;
}
"#;

// -------------------------------------------------------------------
// 2-D configuration
// -------------------------------------------------------------------

/// Configuration for a 2-D pooling operation.
#[derive(Debug, Clone)]
pub struct Pool2dConfig {
    /// Batch size.
    pub batch: usize,
    /// Number of channels.
    pub channels: usize,
    /// Input height.
    pub in_h: usize,
    /// Input width.
    pub in_w: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Stride in the height dimension.
    pub stride_h: usize,
    /// Stride in the width dimension.
    pub stride_w: usize,
    /// Padding in the height dimension.
    pub pad_h: usize,
    /// Padding in the width dimension.
    pub pad_w: usize,
    /// Dilation in the height dimension.
    pub dilation_h: usize,
    /// Dilation in the width dimension.
    pub dilation_w: usize,
}

impl Pool2dConfig {
    /// Create a new config with the given spatial parameters and unit dilation.
    pub fn new(
        batch: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Result<Self> {
        Self::with_dilation(
            batch, channels, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, 1, 1,
        )
    }

    /// Create a new config with explicit dilation parameters.
    pub fn with_dilation(
        batch: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dilation_h: usize,
        dilation_w: usize,
    ) -> Result<Self> {
        if batch == 0 || channels == 0 || in_h == 0 || in_w == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "batch, channels, in_h, in_w must all be > 0".into(),
            }
            .into());
        }
        if kernel_h == 0 || kernel_w == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "kernel_h and kernel_w must be > 0".into(),
            }
            .into());
        }
        if stride_h == 0 || stride_w == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "stride_h and stride_w must be > 0".into(),
            }
            .into());
        }
        if dilation_h == 0 || dilation_w == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "dilation_h and dilation_w must be > 0".into(),
            }
            .into());
        }
        let cfg = Self {
            batch,
            channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
        };
        if cfg.out_h() == 0 || cfg.out_w() == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "pool2d produces 0 outputs: in=({in_h},{in_w}), \
                     kernel=({kernel_h},{kernel_w}), stride=({stride_h},{stride_w}), \
                     pad=({pad_h},{pad_w}), dilation=({dilation_h},{dilation_w})"
                ),
            }
            .into());
        }
        Ok(cfg)
    }

    /// Output height.
    #[inline]
    pub fn out_h(&self) -> usize {
        let effective_k = (self.kernel_h - 1) * self.dilation_h + 1;
        let padded = self.in_h + 2 * self.pad_h;
        if padded < effective_k {
            return 0;
        }
        (padded - effective_k) / self.stride_h + 1
    }

    /// Output width.
    #[inline]
    pub fn out_w(&self) -> usize {
        let effective_k = (self.kernel_w - 1) * self.dilation_w + 1;
        let padded = self.in_w + 2 * self.pad_w;
        if padded < effective_k {
            return 0;
        }
        (padded - effective_k) / self.stride_w + 1
    }

    /// Total number of elements in the input tensor.
    #[inline]
    pub fn input_numel(&self) -> usize {
        self.batch * self.channels * self.in_h * self.in_w
    }

    /// Total number of elements in the output tensor.
    #[inline]
    pub fn output_numel(&self) -> usize {
        self.batch * self.channels * self.out_h() * self.out_w()
    }
}

/// Configuration for adaptive average pooling (target output size only).
#[derive(Debug, Clone)]
pub struct AdaptivePool2dConfig {
    /// Batch size.
    pub batch: usize,
    /// Number of channels.
    pub channels: usize,
    /// Input height.
    pub in_h: usize,
    /// Input width.
    pub in_w: usize,
    /// Target output height.
    pub out_h: usize,
    /// Target output width.
    pub out_w: usize,
}

impl AdaptivePool2dConfig {
    /// Create a validated adaptive pool config.
    pub fn new(
        batch: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
    ) -> Result<Self> {
        if batch == 0 || channels == 0 || in_h == 0 || in_w == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "batch, channels, in_h, in_w must all be > 0".into(),
            }
            .into());
        }
        if out_h == 0 || out_w == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "target out_h and out_w must be > 0".into(),
            }
            .into());
        }
        if out_h > in_h || out_w > in_w {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "adaptive pool target ({out_h},{out_w}) must be <= input ({in_h},{in_w})"
                ),
            }
            .into());
        }
        Ok(Self { batch, channels, in_h, in_w, out_h, out_w })
    }

    /// Total number of elements in the input tensor.
    #[inline]
    pub fn input_numel(&self) -> usize {
        self.batch * self.channels * self.in_h * self.in_w
    }

    /// Total number of elements in the output tensor.
    #[inline]
    pub fn output_numel(&self) -> usize {
        self.batch * self.channels * self.out_h * self.out_w
    }
}

// -------------------------------------------------------------------
// 2-D CPU fallback implementations
// -------------------------------------------------------------------

/// CPU fallback for 2-D max pooling.
///
/// Input/output layout: `[batch, channels, height, width]` (NCHW, row-major).
pub fn max_pool2d_cpu(input: &[f32], output: &mut [f32], config: &Pool2dConfig) -> Result<()> {
    check_pool2d_slices(input, output, config)?;
    let (oh, ow) = (config.out_h(), config.out_w());
    let in_plane = config.in_h * config.in_w;
    let out_plane = oh * ow;

    for b in 0..config.batch {
        for c in 0..config.channels {
            let in_off = (b * config.channels + c) * in_plane;
            let out_off = (b * config.channels + c) * out_plane;
            for i in 0..oh {
                for j in 0..ow {
                    let mut max_val = f32::NEG_INFINITY;
                    for kh in 0..config.kernel_h {
                        let ih = (i * config.stride_h + kh * config.dilation_h)
                            .wrapping_sub(config.pad_h);
                        if ih >= config.in_h {
                            continue;
                        }
                        for kw in 0..config.kernel_w {
                            let iw = (j * config.stride_w + kw * config.dilation_w)
                                .wrapping_sub(config.pad_w);
                            if iw >= config.in_w {
                                continue;
                            }
                            let val = input[in_off + ih * config.in_w + iw];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }
                    output[out_off + i * ow + j] = max_val;
                }
            }
        }
    }
    Ok(())
}

/// CPU fallback for 2-D average pooling (count_include_pad semantics).
///
/// Input/output layout: `[batch, channels, height, width]` (NCHW, row-major).
pub fn avg_pool2d_cpu(input: &[f32], output: &mut [f32], config: &Pool2dConfig) -> Result<()> {
    check_pool2d_slices(input, output, config)?;
    let (oh, ow) = (config.out_h(), config.out_w());
    let in_plane = config.in_h * config.in_w;
    let out_plane = oh * ow;
    let kernel_area = (config.kernel_h * config.kernel_w) as f32;

    for b in 0..config.batch {
        for c in 0..config.channels {
            let in_off = (b * config.channels + c) * in_plane;
            let out_off = (b * config.channels + c) * out_plane;
            for i in 0..oh {
                for j in 0..ow {
                    let mut sum = 0.0_f32;
                    for kh in 0..config.kernel_h {
                        let ih = (i * config.stride_h + kh * config.dilation_h)
                            .wrapping_sub(config.pad_h);
                        if ih >= config.in_h {
                            continue;
                        }
                        for kw in 0..config.kernel_w {
                            let iw = (j * config.stride_w + kw * config.dilation_w)
                                .wrapping_sub(config.pad_w);
                            if iw >= config.in_w {
                                continue;
                            }
                            sum += input[in_off + ih * config.in_w + iw];
                        }
                    }
                    output[out_off + i * ow + j] = sum / kernel_area;
                }
            }
        }
    }
    Ok(())
}

/// CPU fallback for adaptive average pooling 2-D.
///
/// Uses PyTorch-style window boundaries:
///   `start = floor(i * in_size / out_size)`
///   `end   = floor((i+1) * in_size / out_size)`
pub fn adaptive_avg_pool2d_cpu(
    input: &[f32],
    output: &mut [f32],
    config: &AdaptivePool2dConfig,
) -> Result<()> {
    if input.len() < config.input_numel() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "adaptive pool input length {} < expected {}",
                input.len(),
                config.input_numel()
            ),
        }
        .into());
    }
    if output.len() < config.output_numel() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "adaptive pool output length {} < expected {}",
                output.len(),
                config.output_numel()
            ),
        }
        .into());
    }

    let in_plane = config.in_h * config.in_w;
    let out_plane = config.out_h * config.out_w;

    for b in 0..config.batch {
        for c in 0..config.channels {
            let in_off = (b * config.channels + c) * in_plane;
            let out_off = (b * config.channels + c) * out_plane;
            for i in 0..config.out_h {
                let h_start = (i * config.in_h) / config.out_h;
                let h_end = ((i + 1) * config.in_h) / config.out_h;
                for j in 0..config.out_w {
                    let w_start = (j * config.in_w) / config.out_w;
                    let w_end = ((j + 1) * config.in_w) / config.out_w;
                    let count = ((h_end - h_start) * (w_end - w_start)) as f32;
                    let mut sum = 0.0_f32;
                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            sum += input[in_off + ih * config.in_w + iw];
                        }
                    }
                    output[out_off + i * config.out_w + j] =
                        if count > 0.0 { sum / count } else { 0.0 };
                }
            }
        }
    }
    Ok(())
}

fn check_pool2d_slices(input: &[f32], output: &[f32], config: &Pool2dConfig) -> Result<()> {
    if input.len() < config.input_numel() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "pool2d input length {} < expected {}",
                input.len(),
                config.input_numel()
            ),
        }
        .into());
    }
    if output.len() < config.output_numel() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "pool2d output length {} < expected {}",
                output.len(),
                config.output_numel()
            ),
        }
        .into());
    }
    Ok(())
}

// -------------------------------------------------------------------
// 2-D CUDA launch stubs
// -------------------------------------------------------------------

/// Launch stub for 2-D max pooling CUDA kernel.
pub fn launch_max_pool2d(_input: &[f32], _output: &mut [f32], config: &Pool2dConfig) -> Result<()> {
    log::debug!(
        "max_pool2d stub: in=({},{},{},{}), kernel=({},{}), stride=({},{})",
        config.batch,
        config.channels,
        config.in_h,
        config.in_w,
        config.kernel_h,
        config.kernel_w,
        config.stride_h,
        config.stride_w,
    );
    Err(KernelError::GpuError {
        reason: "max_pool2d CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch stub for 2-D average pooling CUDA kernel.
pub fn launch_avg_pool2d(_input: &[f32], _output: &mut [f32], config: &Pool2dConfig) -> Result<()> {
    log::debug!(
        "avg_pool2d stub: in=({},{},{},{}), kernel=({},{})",
        config.batch,
        config.channels,
        config.in_h,
        config.in_w,
        config.kernel_h,
        config.kernel_w,
    );
    Err(KernelError::GpuError {
        reason: "avg_pool2d CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch stub for adaptive average pooling 2-D CUDA kernel.
pub fn launch_adaptive_avg_pool2d(
    _input: &[f32],
    _output: &mut [f32],
    config: &AdaptivePool2dConfig,
) -> Result<()> {
    log::debug!(
        "adaptive_avg_pool2d stub: in=({},{},{},{}), out=({},{})",
        config.batch,
        config.channels,
        config.in_h,
        config.in_w,
        config.out_h,
        config.out_w,
    );
    Err(KernelError::GpuError {
        reason: "adaptive_avg_pool2d CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// -------------------------------------------------------------------
// 2-D unified dispatch
// -------------------------------------------------------------------

/// Apply 2-D max pooling with automatic GPU/CPU dispatch.
pub fn max_pool2d_forward(input: &[f32], output: &mut [f32], config: &Pool2dConfig) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_max_pool2d(input, output, config)
        {
            return Ok(());
        }
    }
    max_pool2d_cpu(input, output, config)
}

/// Apply 2-D average pooling with automatic GPU/CPU dispatch.
pub fn avg_pool2d_forward(input: &[f32], output: &mut [f32], config: &Pool2dConfig) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_avg_pool2d(input, output, config)
        {
            return Ok(());
        }
    }
    avg_pool2d_cpu(input, output, config)
}

/// Apply adaptive average pooling 2-D with automatic GPU/CPU dispatch.
pub fn adaptive_avg_pool2d_forward(
    input: &[f32],
    output: &mut [f32],
    config: &AdaptivePool2dConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_adaptive_avg_pool2d(input, output, config)
        {
            return Ok(());
        }
    }
    adaptive_avg_pool2d_cpu(input, output, config)
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-6;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    // -- 1-D config tests -----------------------------------------------

    #[test]
    fn config_basic() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 10, 3, 1, 0).unwrap();
        assert_eq!(cfg.output_len(), 8);
        assert_eq!(cfg.threads_per_block, 8);
    }

    #[test]
    fn config_with_padding() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 5, 3, 1, 1).unwrap();
        assert_eq!(cfg.output_len(), 5);
    }

    #[test]
    fn config_rejects_zero_input() {
        assert!(PoolingConfig::new(CudaPoolType::Max, 0, 3, 1, 0).is_err());
    }

    #[test]
    fn config_rejects_zero_kernel() {
        assert!(PoolingConfig::new(CudaPoolType::Max, 10, 0, 1, 0).is_err());
    }

    #[test]
    fn config_rejects_zero_stride() {
        assert!(PoolingConfig::new(CudaPoolType::Max, 10, 3, 0, 0).is_err());
    }

    #[test]
    fn config_rejects_zero_output() {
        assert!(PoolingConfig::new(CudaPoolType::Max, 2, 10, 1, 0).is_err());
    }

    #[test]
    fn config_grid_dim() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 2048, 2, 2, 0).unwrap();
        assert_eq!(cfg.output_len(), 1024);
        assert_eq!(cfg.grid_dim(), (1, 1, 1));
        assert_eq!(cfg.block_dim(), (1024, 1, 1));
    }

    #[test]
    fn config_grid_dim_large() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 4096, 2, 1, 0).unwrap();
        let (gx, _, _) = cfg.grid_dim();
        assert_eq!(gx, 4);
    }

    // -- 1-D CPU max pooling --------------------------------------------

    #[test]
    fn cpu_max_pool_basic() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 5, 2, 1, 0).unwrap();
        let input = [1.0, 3.0, 2.0, 5.0, 4.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[3.0, 3.0, 5.0, 5.0], TOL));
    }

    #[test]
    fn cpu_max_pool_stride_2() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 6, 2, 2, 0).unwrap();
        let input = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[3.0, 5.0, 6.0], TOL));
    }

    #[test]
    fn cpu_max_pool_with_padding() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 3, 3, 1, 1).unwrap();
        let input = [1.0, 2.0, 3.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[2.0, 3.0, 3.0], TOL));
    }

    #[test]
    fn cpu_max_pool_negative_values() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 5, 3, 1, 0).unwrap();
        let input = [-5.0, -3.0, -4.0, -1.0, -2.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[-3.0, -1.0, -1.0], TOL));
    }

    #[test]
    fn cpu_max_pool_single_element() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 1, 1, 1, 0).unwrap();
        let input = [42.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[42.0], TOL));
    }

    // -- 1-D CPU average pooling ----------------------------------------

    #[test]
    fn cpu_avg_pool_basic() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 5, 2, 1, 0).unwrap();
        let input = [1.0, 3.0, 2.0, 5.0, 4.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[2.0, 2.5, 3.5, 4.5], TOL));
    }

    #[test]
    fn cpu_avg_pool_stride_2() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 4, 2, 2, 0).unwrap();
        let input = [2.0, 4.0, 6.0, 8.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[3.0, 7.0], TOL));
    }

    #[test]
    fn cpu_avg_pool_with_padding() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 3, 3, 1, 1).unwrap();
        let input = [3.0, 6.0, 9.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[3.0, 6.0, 5.0], TOL));
    }

    #[test]
    fn cpu_avg_pool_single_element() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 1, 1, 1, 0).unwrap();
        let input = [7.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[7.0], TOL));
    }

    // -- 1-D error handling ---------------------------------------------

    #[test]
    fn cpu_rejects_short_input() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 10, 3, 1, 0).unwrap();
        let input = [1.0; 5];
        let mut output = vec![0.0; cfg.output_len()];
        assert!(pooling_cpu(&input, &mut output, &cfg).is_err());
    }

    #[test]
    fn cpu_rejects_short_output() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 5, 2, 1, 0).unwrap();
        let input = [1.0; 5];
        let mut output = [0.0; 1];
        assert!(pooling_cpu(&input, &mut output, &cfg).is_err());
    }

    // -- 1-D unified dispatch -------------------------------------------

    #[test]
    fn forward_dispatches_cpu() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 5, 2, 1, 0).unwrap();
        let input = [1.0, 3.0, 2.0, 5.0, 4.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_forward(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[3.0, 3.0, 5.0, 5.0], TOL));
    }

    #[test]
    fn forward_matches_cpu_avg() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 6, 3, 2, 0).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_fwd = vec![0.0; cfg.output_len()];
        let mut out_cpu = vec![0.0; cfg.output_len()];
        pooling_forward(&input, &mut out_fwd, &cfg).unwrap();
        pooling_cpu(&input, &mut out_cpu, &cfg).unwrap();
        for (i, (&f, &c)) in out_fwd.iter().zip(out_cpu.iter()).enumerate() {
            assert!((f - c).abs() < TOL, "mismatch at {i}: forward={f}, cpu={c}");
        }
    }

    #[test]
    fn forward_large_input() {
        let n = 1024;
        let cfg = PoolingConfig::new(CudaPoolType::Max, n, 4, 4, 0).unwrap();
        let input: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let mut output = vec![0.0; cfg.output_len()];
        pooling_forward(&input, &mut output, &cfg).unwrap();
        assert_eq!(output.len(), 256);
        for (i, &v) in output.iter().enumerate() {
            let window = &input[i * 4..i * 4 + 4];
            let expected = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!((v - expected).abs() < TOL, "mismatch at {i}");
        }
    }

    // -- 2-D max pooling ------------------------------------------------

    #[test]
    fn max_pool2d_basic() {
        let cfg = Pool2dConfig::new(1, 1, 4, 4, 2, 2, 2, 2, 0, 0).unwrap();
        #[rustfmt::skip]
        let input = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let mut output = vec![0.0; cfg.output_numel()];
        max_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[6.0, 8.0, 14.0, 16.0], TOL));
    }

    #[test]
    fn max_pool2d_1x1_kernel() {
        let cfg = Pool2dConfig::new(1, 1, 3, 3, 1, 1, 1, 1, 0, 0).unwrap();
        let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let mut output = vec![0.0; cfg.output_numel()];
        max_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &input, TOL));
    }

    #[test]
    fn max_pool2d_stride_gt_kernel() {
        let cfg = Pool2dConfig::new(1, 1, 6, 6, 2, 2, 3, 3, 0, 0).unwrap();
        assert_eq!(cfg.out_h(), 2);
        assert_eq!(cfg.out_w(), 2);
        #[rustfmt::skip]
        let input = [
            1.0, 2.0, 0.0, 3.0, 4.0, 0.0,
            5.0, 6.0, 0.0, 7.0, 8.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            9.0, 10.0, 0.0, 11.0, 12.0, 0.0,
            13.0, 14.0, 0.0, 15.0, 16.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut output = vec![0.0; cfg.output_numel()];
        max_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[6.0, 8.0, 14.0, 16.0], TOL));
    }

    #[test]
    fn max_pool2d_with_padding() {
        let cfg = Pool2dConfig::new(1, 1, 3, 3, 3, 3, 1, 1, 1, 1).unwrap();
        assert_eq!(cfg.out_h(), 3);
        #[rustfmt::skip]
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut output = vec![0.0; cfg.output_numel()];
        max_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert_eq!(output[0], 5.0);
        assert_eq!(output[4], 9.0);
        assert_eq!(output[8], 9.0);
    }

    #[test]
    fn max_pool2d_multi_channel() {
        let cfg = Pool2dConfig::new(1, 2, 2, 2, 2, 2, 1, 1, 0, 0).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; cfg.output_numel()];
        max_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[4.0, 8.0], TOL));
    }

    #[test]
    fn max_pool2d_batched() {
        let cfg = Pool2dConfig::new(2, 1, 2, 2, 2, 2, 1, 1, 0, 0).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0; cfg.output_numel()];
        max_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[4.0, 40.0], TOL));
    }

    #[test]
    fn max_pool2d_with_dilation() {
        let cfg = Pool2dConfig::with_dilation(1, 1, 5, 5, 2, 2, 1, 1, 0, 0, 2, 2).unwrap();
        assert_eq!(cfg.out_h(), 3);
        #[rustfmt::skip]
        let input = [
            1.0,  2.0,  3.0,  4.0,  5.0,
            6.0,  7.0,  8.0,  9.0,  10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0,
        ];
        let mut output = vec![0.0; cfg.output_numel()];
        max_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert_eq!(output[0], 13.0);
        assert_eq!(output[8], 25.0);
    }

    #[test]
    fn max_pool2d_negative_values() {
        let cfg = Pool2dConfig::new(1, 1, 2, 2, 2, 2, 1, 1, 0, 0).unwrap();
        let input = [-4.0, -3.0, -2.0, -1.0];
        let mut output = vec![0.0; cfg.output_numel()];
        max_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[-1.0], TOL));
    }

    // -- 2-D average pooling --------------------------------------------

    #[test]
    fn avg_pool2d_basic() {
        let cfg = Pool2dConfig::new(1, 1, 4, 4, 2, 2, 2, 2, 0, 0).unwrap();
        #[rustfmt::skip]
        let input = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let mut output = vec![0.0; cfg.output_numel()];
        avg_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[3.5, 5.5, 11.5, 13.5], TOL));
    }

    #[test]
    fn avg_pool2d_with_padding() {
        let cfg = Pool2dConfig::new(1, 1, 2, 2, 2, 2, 1, 1, 1, 1).unwrap();
        assert_eq!(cfg.out_h(), 3);
        let input = [4.0, 4.0, 4.0, 4.0];
        let mut output = vec![0.0; cfg.output_numel()];
        avg_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert_eq!(output[0], 1.0);
        assert_eq!(output[4], 4.0);
    }

    #[test]
    fn avg_pool2d_multi_channel() {
        let cfg = Pool2dConfig::new(1, 2, 2, 2, 2, 2, 1, 1, 0, 0).unwrap();
        let input = [2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0; cfg.output_numel()];
        avg_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[5.0, 25.0], TOL));
    }

    #[test]
    fn avg_pool2d_numerical_precision() {
        let cfg = Pool2dConfig::new(1, 1, 4, 4, 4, 4, 1, 1, 0, 0).unwrap();
        let input = vec![0.1_f32; 16];
        let mut output = vec![0.0; cfg.output_numel()];
        avg_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert!((output[0] - 0.1).abs() < 1e-5);
    }

    // -- Adaptive average pooling 2-D -----------------------------------

    #[test]
    fn adaptive_avg_pool2d_to_target() {
        let cfg = AdaptivePool2dConfig::new(1, 1, 4, 4, 2, 2).unwrap();
        #[rustfmt::skip]
        let input = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let mut output = vec![0.0; cfg.output_numel()];
        adaptive_avg_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[3.5, 5.5, 11.5, 13.5], TOL));
    }

    #[test]
    fn adaptive_avg_pool2d_global() {
        let cfg = AdaptivePool2dConfig::new(1, 1, 4, 4, 1, 1).unwrap();
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let mut output = vec![0.0; 1];
        adaptive_avg_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert!((output[0] - 8.5).abs() < TOL);
    }

    #[test]
    fn adaptive_avg_pool2d_identity() {
        let cfg = AdaptivePool2dConfig::new(1, 1, 3, 3, 3, 3).unwrap();
        let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let mut output = vec![0.0; 9];
        adaptive_avg_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &input, TOL));
    }

    #[test]
    fn adaptive_avg_pool2d_multi_channel_batched() {
        let cfg = AdaptivePool2dConfig::new(2, 2, 4, 4, 1, 1).unwrap();
        let mut input = vec![0.0_f32; 2 * 2 * 4 * 4];
        for b in 0..2 {
            for c in 0..2 {
                let val = (b * 2 + c + 1) as f32 * 10.0;
                let off = (b * 2 + c) * 16;
                for x in &mut input[off..off + 16] {
                    *x = val;
                }
            }
        }
        let mut output = vec![0.0; cfg.output_numel()];
        adaptive_avg_pool2d_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[10.0, 20.0, 30.0, 40.0], TOL));
    }

    #[test]
    fn adaptive_avg_pool2d_rejects_upsampling() {
        assert!(AdaptivePool2dConfig::new(1, 1, 3, 3, 4, 4).is_err());
    }

    #[test]
    fn adaptive_avg_pool2d_rejects_zero_target() {
        assert!(AdaptivePool2dConfig::new(1, 1, 4, 4, 0, 0).is_err());
    }

    // -- 2-D unified dispatch -------------------------------------------

    #[test]
    fn max_pool2d_forward_dispatches_cpu() {
        let cfg = Pool2dConfig::new(1, 1, 4, 4, 2, 2, 2, 2, 0, 0).unwrap();
        #[rustfmt::skip]
        let input = [
            1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,  13.0, 14.0, 15.0, 16.0,
        ];
        let mut output = vec![0.0; cfg.output_numel()];
        max_pool2d_forward(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[6.0, 8.0, 14.0, 16.0], TOL));
    }

    #[test]
    fn avg_pool2d_forward_dispatches_cpu() {
        let cfg = Pool2dConfig::new(1, 1, 4, 4, 2, 2, 2, 2, 0, 0).unwrap();
        #[rustfmt::skip]
        let input = [
            1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,  13.0, 14.0, 15.0, 16.0,
        ];
        let mut output = vec![0.0; cfg.output_numel()];
        avg_pool2d_forward(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[3.5, 5.5, 11.5, 13.5], TOL));
    }

    #[test]
    fn adaptive_avg_pool2d_forward_dispatches_cpu() {
        let cfg = AdaptivePool2dConfig::new(1, 1, 4, 4, 2, 2).unwrap();
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let mut output = vec![0.0; cfg.output_numel()];
        adaptive_avg_pool2d_forward(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[3.5, 5.5, 11.5, 13.5], TOL));
    }

    // -- 2-D error handling ---------------------------------------------

    #[test]
    fn pool2d_rejects_zero_dims() {
        assert!(Pool2dConfig::new(0, 1, 4, 4, 2, 2, 2, 2, 0, 0).is_err());
        assert!(Pool2dConfig::new(1, 0, 4, 4, 2, 2, 2, 2, 0, 0).is_err());
        assert!(Pool2dConfig::new(1, 1, 0, 4, 2, 2, 2, 2, 0, 0).is_err());
    }

    #[test]
    fn pool2d_rejects_zero_kernel() {
        assert!(Pool2dConfig::new(1, 1, 4, 4, 0, 2, 1, 1, 0, 0).is_err());
    }

    #[test]
    fn pool2d_rejects_zero_stride() {
        assert!(Pool2dConfig::new(1, 1, 4, 4, 2, 2, 0, 1, 0, 0).is_err());
    }

    #[test]
    fn pool2d_rejects_zero_dilation() {
        assert!(Pool2dConfig::with_dilation(1, 1, 4, 4, 2, 2, 1, 1, 0, 0, 0, 1).is_err());
    }

    #[test]
    fn pool2d_rejects_oversized_kernel() {
        assert!(Pool2dConfig::new(1, 1, 2, 2, 5, 5, 1, 1, 0, 0).is_err());
    }

    #[test]
    fn max_pool2d_rejects_short_input() {
        let cfg = Pool2dConfig::new(1, 1, 4, 4, 2, 2, 2, 2, 0, 0).unwrap();
        let input = [1.0; 8];
        let mut output = vec![0.0; cfg.output_numel()];
        assert!(max_pool2d_cpu(&input, &mut output, &cfg).is_err());
    }

    #[test]
    fn max_pool2d_rejects_short_output() {
        let cfg = Pool2dConfig::new(1, 1, 4, 4, 2, 2, 2, 2, 0, 0).unwrap();
        let input = [1.0; 16];
        let mut output = [0.0; 1];
        assert!(max_pool2d_cpu(&input, &mut output, &cfg).is_err());
    }

    // -- CUDA kernel source sanity checks -------------------------------

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn kernel_src_max_pool2d_not_empty() {
        assert!(!MAX_POOL2D_KERNEL_SRC.is_empty());
        assert!(MAX_POOL2D_KERNEL_SRC.contains("max_pool2d_f32"));
    }

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn kernel_src_avg_pool2d_not_empty() {
        assert!(!AVG_POOL2D_KERNEL_SRC.is_empty());
        assert!(AVG_POOL2D_KERNEL_SRC.contains("avg_pool2d_f32"));
    }

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn kernel_src_adaptive_not_empty() {
        assert!(!ADAPTIVE_AVG_POOL2D_KERNEL_SRC.is_empty());
        assert!(ADAPTIVE_AVG_POOL2D_KERNEL_SRC.contains("adaptive_avg_pool2d_f32"));
    }

    // -- GPU launch stub tests ------------------------------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn cuda_pooling_max_launch() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 1024, 4, 4, 0).unwrap();
        let input = vec![1.0_f32; 1024];
        let mut output = vec![0.0_f32; cfg.output_len()];
        let result = launch_pooling(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA max pooling launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn cuda_pooling_avg_launch() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 1024, 4, 4, 0).unwrap();
        let input = vec![1.0_f32; 1024];
        let mut output = vec![0.0_f32; cfg.output_len()];
        let result = launch_pooling(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA avg pooling launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn cuda_max_pool2d_launch() {
        let cfg = Pool2dConfig::new(1, 1, 8, 8, 2, 2, 2, 2, 0, 0).unwrap();
        let input = vec![1.0_f32; 64];
        let mut output = vec![0.0_f32; cfg.output_numel()];
        let result = launch_max_pool2d(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA max_pool2d launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn cuda_avg_pool2d_launch() {
        let cfg = Pool2dConfig::new(1, 1, 8, 8, 2, 2, 2, 2, 0, 0).unwrap();
        let input = vec![1.0_f32; 64];
        let mut output = vec![0.0_f32; cfg.output_numel()];
        let result = launch_avg_pool2d(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA avg_pool2d launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn cuda_adaptive_avg_pool2d_launch() {
        let cfg = AdaptivePool2dConfig::new(1, 1, 8, 8, 2, 2).unwrap();
        let input = vec![1.0_f32; 64];
        let mut output = vec![0.0_f32; cfg.output_numel()];
        let result = launch_adaptive_avg_pool2d(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA adaptive launch failed: {result:?}");
    }
}
