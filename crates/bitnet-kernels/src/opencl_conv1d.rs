//! OpenCL-optimized 1D convolution operations for sequence models.
//!
//! # Overview
//!
//! This module provides 1D convolution primitives used in sequence-based neural
//! network architectures. All operations ship with **CPU reference
//! implementations** that require no OpenCL runtime, plus embedded OpenCL C
//! kernel source for GPU dispatch on Intel / AMD / other OpenCL-capable devices.
//!
//! # Variants
//!
//! - [`Conv1dKernel`] — standard 1D convolution with configurable stride,
//!   padding, dilation, and groups.
//! - [`DepthwiseConv1d`] — depthwise separable convolution (`groups ==
//!   in_channels`).
//! - [`CausalConv1d`] — left-padded causal convolution that prevents future
//!   token leakage.
//! - [`Conv1dTranspose`] — transposed (deconvolution) for upsampling.
//! - [`WinoGrad1d`] — Winograd F(2,3) transform for accelerating small-kernel
//!   convolutions.
//! - [`ConvStats`] — performance bookkeeping (GFLOP/s, bandwidth, timing).
//!
//! # OpenCL kernels
//!
//! [`CONV1D_CL`] contains naive and tiled OpenCL C implementations that mirror
//! the CPU reference logic. They are compiled at runtime when an OpenCL context
//! is available.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Parameters for a 1D convolution operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1dConfig {
    /// Width of the convolution kernel.
    pub kernel_size: usize,
    /// Step size between successive kernel applications.
    pub stride: usize,
    /// Zero-padding added to both sides of the input.
    pub padding: usize,
    /// Spacing between kernel elements (atrous convolution).
    pub dilation: usize,
    /// Number of blocked connections from input to output channels.
    /// `1` = standard conv, `in_channels` = depthwise.
    pub groups: usize,
}

impl Conv1dConfig {
    /// Create a configuration with explicit parameters.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] when `kernel_size`, `stride`,
    /// `dilation`, or `groups` is zero.
    pub fn new(
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        if kernel_size == 0 || stride == 0 || dilation == 0 || groups == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "conv1d params must be non-zero: kernel_size={kernel_size}, \
                     stride={stride}, dilation={dilation}, groups={groups}"
                ),
            }
            .into());
        }
        Ok(Self { kernel_size, stride, padding, dilation, groups })
    }

    /// Shorthand for a simple kernel with stride=1, no padding, no dilation,
    /// groups=1.
    pub fn simple(kernel_size: usize) -> Result<Self> {
        Self::new(kernel_size, 1, 0, 1, 1)
    }

    /// Compute the output length for a given input length.
    pub fn output_length(&self, input_length: usize) -> usize {
        let effective_kernel = self.dilation * (self.kernel_size - 1) + 1;
        let padded = input_length + 2 * self.padding;
        if padded < effective_kernel {
            return 0;
        }
        (padded - effective_kernel) / self.stride + 1
    }
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self { kernel_size: 3, stride: 1, padding: 0, dilation: 1, groups: 1 }
    }
}

// ---------------------------------------------------------------------------
// Performance tracking
// ---------------------------------------------------------------------------

/// Tracks performance statistics for convolution kernels.
#[derive(Debug, Clone, Default)]
pub struct ConvStats {
    /// Total floating-point operations performed.
    pub flops: u64,
    /// Achieved GFLOP/s (set after timing).
    pub gflops: f64,
    /// Bytes transferred to/from memory.
    pub bytes_transferred: u64,
    /// Achieved bandwidth in GB/s (set after timing).
    pub bandwidth_gbs: f64,
    /// Wall-clock kernel time in seconds.
    pub kernel_time_secs: f64,
}

impl ConvStats {
    /// Compute stats for a conv1d operation.
    pub fn for_conv1d(
        batch: usize,
        out_channels: usize,
        in_channels: usize,
        output_length: usize,
        kernel_size: usize,
        groups: usize,
    ) -> Self {
        // MACs per output element: (in_channels / groups) * kernel_size
        // FLOPs = 2 * MACs (multiply + add)
        let macs_per_elem = (in_channels / groups.max(1)) as u64 * kernel_size as u64;
        let total_elems = batch as u64 * out_channels as u64 * output_length as u64;
        let flops = 2 * macs_per_elem * total_elems;

        // Read: input + weight; Write: output  (all f32)
        let input_bytes = (batch * in_channels * (output_length + kernel_size - 1)) as u64 * 4;
        let weight_bytes = (out_channels * (in_channels / groups.max(1)) * kernel_size) as u64 * 4;
        let output_bytes = total_elems * 4;
        let bytes_transferred = input_bytes + weight_bytes + output_bytes;

        Self { flops, gflops: 0.0, bytes_transferred, bandwidth_gbs: 0.0, kernel_time_secs: 0.0 }
    }

    /// Finalise after measuring wall-clock time.
    pub fn finalise(&mut self, elapsed_secs: f64) {
        self.kernel_time_secs = elapsed_secs;
        if elapsed_secs > 0.0 {
            self.gflops = self.flops as f64 / elapsed_secs / 1e9;
            self.bandwidth_gbs = self.bytes_transferred as f64 / elapsed_secs / 1e9;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference: standard 1D convolution
// ---------------------------------------------------------------------------

/// Standard 1D convolution kernel with CPU reference implementation.
///
/// Supports grouped convolution, stride, padding, and dilation.
pub struct Conv1dKernel;

impl Conv1dKernel {
    /// Forward pass for 1D convolution.
    ///
    /// # Layout
    ///
    /// - `input`:  `[batch, in_channels, in_length]`
    /// - `weight`: `[out_channels, in_channels / groups, kernel_size]`
    /// - `bias`:   `[out_channels]` (optional)
    /// - `output`: `[batch, out_channels, out_length]`
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] on dimension mismatch.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
        cfg: &Conv1dConfig,
        batch: usize,
        in_channels: usize,
        in_length: usize,
        out_channels: usize,
    ) -> Result<()> {
        validate_conv1d_args(
            input,
            weight,
            bias,
            output,
            cfg,
            batch,
            in_channels,
            in_length,
            out_channels,
        )?;

        let out_len = cfg.output_length(in_length);
        let ch_per_group = in_channels / cfg.groups;
        let oc_per_group = out_channels / cfg.groups;

        output.fill(0.0);

        // Apply bias
        if let Some(b) = bias {
            for n in 0..batch {
                #[allow(clippy::needless_range_loop)]
                for oc in 0..out_channels {
                    let base = n * out_channels * out_len + oc * out_len;
                    for o in 0..out_len {
                        output[base + o] = b[oc];
                    }
                }
            }
        }

        // Convolution
        for n in 0..batch {
            for g in 0..cfg.groups {
                for oc_local in 0..oc_per_group {
                    let oc = g * oc_per_group + oc_local;
                    for ic_local in 0..ch_per_group {
                        let ic = g * ch_per_group + ic_local;
                        for o in 0..out_len {
                            let out_idx = n * out_channels * out_len + oc * out_len + o;
                            for k in 0..cfg.kernel_size {
                                let i_pos = o * cfg.stride + k * cfg.dilation;
                                if i_pos >= cfg.padding && i_pos - cfg.padding < in_length {
                                    let i_actual = i_pos - cfg.padding;
                                    let in_idx =
                                        n * in_channels * in_length + ic * in_length + i_actual;
                                    let w_idx = oc * ch_per_group * cfg.kernel_size
                                        + ic_local * cfg.kernel_size
                                        + k;
                                    output[out_idx] += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CPU reference: depthwise separable convolution
// ---------------------------------------------------------------------------

/// Depthwise separable 1D convolution (`groups == in_channels`).
///
/// Each input channel is convolved independently with its own filter.
pub struct DepthwiseConv1d;

impl DepthwiseConv1d {
    /// Forward pass.
    ///
    /// # Layout
    ///
    /// - `input`:  `[batch, channels, in_length]`
    /// - `weight`: `[channels, 1, kernel_size]`
    /// - `bias`:   `[channels]` (optional)
    /// - `output`: `[batch, channels, out_length]`
    pub fn forward(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
        cfg: &Conv1dConfig,
        batch: usize,
        channels: usize,
        in_length: usize,
    ) -> Result<()> {
        if cfg.groups != channels {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "depthwise conv requires groups == channels: groups={}, channels={channels}",
                    cfg.groups
                ),
            }
            .into());
        }

        // Delegate to the generic grouped conv1d with groups == channels
        Conv1dKernel::forward(
            input, weight, bias, output, cfg, batch, channels, in_length, channels,
        )
    }
}

// ---------------------------------------------------------------------------
// CPU reference: causal convolution
// ---------------------------------------------------------------------------

/// Left-padded causal 1D convolution that prevents future-token leakage.
///
/// Automatically applies `padding = dilation * (kernel_size - 1)` on the left
/// and trims excess output so that `out_length == in_length` (when stride=1).
pub struct CausalConv1d;

impl CausalConv1d {
    /// Forward pass.
    ///
    /// # Layout
    ///
    /// - `input`:  `[batch, in_channels, in_length]`
    /// - `weight`: `[out_channels, in_channels / groups, kernel_size]`
    /// - `bias`:   `[out_channels]` (optional)
    /// - `output`: `[batch, out_channels, out_length]`
    ///
    /// The output length equals `in_length` when `stride == 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
        cfg: &Conv1dConfig,
        batch: usize,
        in_channels: usize,
        in_length: usize,
        out_channels: usize,
    ) -> Result<()> {
        // Causal padding = dilation * (kernel_size - 1) on the left only.
        let causal_pad = cfg.dilation * (cfg.kernel_size - 1);

        // Build a padded config with symmetric padding for the generic conv,
        // then trim the right side from the output.
        let padded_cfg = Conv1dConfig {
            kernel_size: cfg.kernel_size,
            stride: cfg.stride,
            padding: causal_pad,
            dilation: cfg.dilation,
            groups: cfg.groups,
        };

        let padded_out_len = padded_cfg.output_length(in_length);
        let causal_out_len = causal_output_length(in_length, cfg);

        if padded_out_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "causal conv produces zero-length output".into(),
            }
            .into());
        }

        let expected_output = batch * out_channels * causal_out_len;
        if output.len() != expected_output {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "causal conv1d output size mismatch: expected {expected_output}, got {}",
                    output.len()
                ),
            }
            .into());
        }

        // Run the generic conv with the left-only padding.
        // We allocate a temporary buffer for the full (symmetrically padded) output,
        // then copy only the first `causal_out_len` positions per channel.
        let tmp_len = batch * out_channels * padded_out_len;
        let mut tmp = vec![0.0_f32; tmp_len];

        Conv1dKernel::forward(
            input,
            weight,
            bias,
            &mut tmp,
            &padded_cfg,
            batch,
            in_channels,
            in_length,
            out_channels,
        )?;

        // Copy the first `causal_out_len` positions (trim the right).
        for n in 0..batch {
            for oc in 0..out_channels {
                let src_base = n * out_channels * padded_out_len + oc * padded_out_len;
                let dst_base = n * out_channels * causal_out_len + oc * causal_out_len;
                output[dst_base..dst_base + causal_out_len]
                    .copy_from_slice(&tmp[src_base..src_base + causal_out_len]);
            }
        }

        Ok(())
    }
}

/// Compute the output length of a causal convolution.
pub fn causal_output_length(in_length: usize, cfg: &Conv1dConfig) -> usize {
    // With full left-padding and stride, output = ceil(in_length / stride)
    in_length.div_ceil(cfg.stride)
}

// ---------------------------------------------------------------------------
// CPU reference: transposed convolution
// ---------------------------------------------------------------------------

/// Transposed (deconvolution) 1D convolution for upsampling.
pub struct Conv1dTranspose;

impl Conv1dTranspose {
    /// Forward pass.
    ///
    /// # Layout
    ///
    /// - `input`:  `[batch, in_channels, in_length]`
    /// - `weight`: `[in_channels, out_channels / groups, kernel_size]`
    /// - `bias`:   `[out_channels]` (optional)
    /// - `output`: `[batch, out_channels, out_length]`
    ///
    /// Output length = `(in_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
        cfg: &Conv1dConfig,
        batch: usize,
        in_channels: usize,
        in_length: usize,
        out_channels: usize,
    ) -> Result<()> {
        let out_len = transpose_output_length(in_length, cfg);
        let ch_per_group = in_channels / cfg.groups;
        let oc_per_group = out_channels / cfg.groups;

        validate_transpose_args(
            input,
            weight,
            bias,
            output,
            cfg,
            batch,
            in_channels,
            in_length,
            out_channels,
            out_len,
        )?;

        output.fill(0.0);

        // Apply bias
        if let Some(b) = bias {
            for n in 0..batch {
                #[allow(clippy::needless_range_loop)]
                for oc in 0..out_channels {
                    let base = n * out_channels * out_len + oc * out_len;
                    for o in 0..out_len {
                        output[base + o] = b[oc];
                    }
                }
            }
        }

        // Transposed convolution: scatter input into output
        for n in 0..batch {
            for g in 0..cfg.groups {
                for ic_local in 0..ch_per_group {
                    let ic = g * ch_per_group + ic_local;
                    for oc_local in 0..oc_per_group {
                        let oc = g * oc_per_group + oc_local;
                        for i in 0..in_length {
                            let in_idx = n * in_channels * in_length + ic * in_length + i;
                            let in_val = input[in_idx];
                            for k in 0..cfg.kernel_size {
                                let o_pos_raw = i * cfg.stride + k * cfg.dilation;
                                if o_pos_raw >= cfg.padding {
                                    let o_pos = o_pos_raw - cfg.padding;
                                    if o_pos < out_len {
                                        let w_idx = ic * oc_per_group * cfg.kernel_size
                                            + oc_local * cfg.kernel_size
                                            + k;
                                        let out_idx =
                                            n * out_channels * out_len + oc * out_len + o_pos;
                                        output[out_idx] += in_val * weight[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Compute the output length of a transposed 1D convolution.
pub fn transpose_output_length(in_length: usize, cfg: &Conv1dConfig) -> usize {
    if in_length == 0 {
        return 0;
    }
    (in_length - 1) * cfg.stride + cfg.dilation * (cfg.kernel_size - 1) + 1 - 2 * cfg.padding
}

// ---------------------------------------------------------------------------
// Winograd F(2,3) for kernel_size == 3
// ---------------------------------------------------------------------------

/// Winograd F(2,3) transformed convolution for `kernel_size == 3`.
///
/// Reduces multiplications from 6 to 4 per two output elements at the cost of
/// extra additions and a small transform overhead.
pub struct WinoGrad1d;

impl WinoGrad1d {
    /// Forward pass using the Winograd F(2,3) algorithm.
    ///
    /// Only supports `kernel_size == 3`, `stride == 1`, `dilation == 1`.
    ///
    /// # Layout
    ///
    /// - `input`:  `[batch, in_channels, in_length]`
    /// - `weight`: `[out_channels, in_channels / groups, 3]`
    /// - `bias`:   `[out_channels]` (optional)
    /// - `output`: `[batch, out_channels, out_length]`
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
        cfg: &Conv1dConfig,
        batch: usize,
        in_channels: usize,
        in_length: usize,
        out_channels: usize,
    ) -> Result<()> {
        if cfg.kernel_size != 3 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "WinoGrad1d F(2,3) requires kernel_size=3, got {}",
                    cfg.kernel_size
                ),
            }
            .into());
        }
        if cfg.stride != 1 || cfg.dilation != 1 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "WinoGrad1d F(2,3) requires stride=1 and dilation=1, \
                     got stride={}, dilation={}",
                    cfg.stride, cfg.dilation
                ),
            }
            .into());
        }

        validate_conv1d_args(
            input,
            weight,
            bias,
            output,
            cfg,
            batch,
            in_channels,
            in_length,
            out_channels,
        )?;

        let out_len = cfg.output_length(in_length);
        let ch_per_group = in_channels / cfg.groups;
        let oc_per_group = out_channels / cfg.groups;

        output.fill(0.0);

        if let Some(b) = bias {
            for n in 0..batch {
                #[allow(clippy::needless_range_loop)]
                for oc in 0..out_channels {
                    let base = n * out_channels * out_len + oc * out_len;
                    for o in 0..out_len {
                        output[base + o] = b[oc];
                    }
                }
            }
        }

        // Process tiles of 2 output elements at a time.
        // F(2,3): input tile = 4 elements, kernel = 3 elements, output = 2 elements.
        let num_tiles = out_len.div_ceil(2);

        for n in 0..batch {
            for g in 0..cfg.groups {
                for oc_local in 0..oc_per_group {
                    let oc = g * oc_per_group + oc_local;
                    for ic_local in 0..ch_per_group {
                        let ic = g * ch_per_group + ic_local;

                        // Filter transform: G * g  (done once per filter)
                        let w_base = oc * ch_per_group * 3 + ic_local * 3;
                        let g0 = weight[w_base];
                        let g1 = weight[w_base + 1];
                        let g2 = weight[w_base + 2];

                        // Winograd filter transform for F(2,3):
                        // u0 = g0, u1 = (g0+g1+g2)/2, u2 = (g0-g1+g2)/2, u3 = g2
                        let u0 = g0;
                        let u1 = (g0 + g1 + g2) * 0.5;
                        let u2 = (g0 - g1 + g2) * 0.5;
                        let u3 = g2;

                        for tile in 0..num_tiles {
                            let o_base = tile * 2;
                            // Input tile: 4 consecutive elements (with padding awareness)
                            let in_start = o_base; // stride=1, padding handled by cfg
                            let d = [
                                get_padded(
                                    input,
                                    n,
                                    ic,
                                    in_channels,
                                    in_length,
                                    in_start,
                                    cfg.padding,
                                ),
                                get_padded(
                                    input,
                                    n,
                                    ic,
                                    in_channels,
                                    in_length,
                                    in_start + 1,
                                    cfg.padding,
                                ),
                                get_padded(
                                    input,
                                    n,
                                    ic,
                                    in_channels,
                                    in_length,
                                    in_start + 2,
                                    cfg.padding,
                                ),
                                get_padded(
                                    input,
                                    n,
                                    ic,
                                    in_channels,
                                    in_length,
                                    in_start + 3,
                                    cfg.padding,
                                ),
                            ];

                            // Input transform: B^T * d
                            let v0 = d[0] - d[2];
                            let v1 = d[1] + d[2];
                            let v2 = -d[1] + d[2];
                            let v3 = d[1] - d[3];

                            // Element-wise multiply
                            let m0 = u0 * v0;
                            let m1 = u1 * v1;
                            let m2 = u2 * v2;
                            let m3 = u3 * v3;

                            // Output transform: A^T * m
                            let y0 = m0 + m1 + m2;
                            let y1 = m1 - m2 - m3;

                            let out_base = n * out_channels * out_len + oc * out_len;
                            if o_base < out_len {
                                output[out_base + o_base] += y0;
                            }
                            if o_base + 1 < out_len {
                                output[out_base + o_base + 1] += y1;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Read from the input tensor with padding support.
#[inline]
fn get_padded(
    input: &[f32],
    batch: usize,
    channel: usize,
    channels: usize,
    length: usize,
    pos: usize,
    padding: usize,
) -> f32 {
    if pos >= padding && pos - padding < length {
        let actual = pos - padding;
        input[batch * channels * length + channel * length + actual]
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C source for 1D convolution (naive + tiled implementations).
///
/// # Kernels
///
/// - `conv1d_naive` — one work-item per output element, suitable for
///   correctness testing and small tensors.
/// - `conv1d_tiled` — uses local memory to cache input tiles, reducing
///   global memory traffic for larger workloads.
pub const CONV1D_CL: &str = r#"
// ---------------------------------------------------------------------------
// Naive 1D convolution
// ---------------------------------------------------------------------------
// Work-item layout: global_id(0) = output position, global_id(1) = output
//   channel, global_id(2) = batch index.
__kernel void conv1d_naive(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    const int in_channels,
    const int in_length,
    const int out_channels,
    const int out_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int has_bias)
{
    int o   = get_global_id(0);
    int oc  = get_global_id(1);
    int n   = get_global_id(2);

    if (o >= out_length || oc >= out_channels || n >= get_global_size(2))
        return;

    int g           = oc / (out_channels / groups);
    int ch_per_grp  = in_channels / groups;
    int oc_local    = oc - g * (out_channels / groups);

    float acc = (has_bias != 0) ? bias[oc] : 0.0f;

    for (int ic_l = 0; ic_l < ch_per_grp; ic_l++) {
        int ic = g * ch_per_grp + ic_l;
        for (int k = 0; k < kernel_size; k++) {
            int i_pos = o * stride + k * dilation;
            int i_actual = i_pos - padding;
            if (i_actual >= 0 && i_actual < in_length) {
                int in_idx = n * in_channels * in_length + ic * in_length + i_actual;
                int w_idx  = oc * ch_per_grp * kernel_size + ic_l * kernel_size + k;
                acc += input[in_idx] * weight[w_idx];
            }
        }
    }

    int out_idx = n * out_channels * out_length + oc * out_length + o;
    output[out_idx] = acc;
}

// ---------------------------------------------------------------------------
// Tiled 1D convolution — uses local memory for input caching
// ---------------------------------------------------------------------------
// Each work-group processes TILE_SIZE output elements for one (batch, oc) pair.
// Local memory holds the needed input region for the tile.
#ifndef TILE_SIZE
#define TILE_SIZE 64
#endif

__kernel void conv1d_tiled(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    const int in_channels,
    const int in_length,
    const int out_channels,
    const int out_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int has_bias)
{
    int local_id  = get_local_id(0);
    int tile_base = get_group_id(0) * TILE_SIZE;
    int oc        = get_global_id(1);
    int n         = get_global_id(2);

    if (oc >= out_channels || n >= get_global_size(2))
        return;

    int g           = oc / (out_channels / groups);
    int ch_per_grp  = in_channels / groups;
    int eff_kernel  = dilation * (kernel_size - 1) + 1;

    // Input region needed for this tile
    int in_start = tile_base * stride - padding;
    int in_tile_len = (TILE_SIZE - 1) * stride + eff_kernel;

    __local float tile_buf[1024]; // max: TILE_SIZE * stride + eff_kernel

    float acc = 0.0f;

    for (int ic_l = 0; ic_l < ch_per_grp; ic_l++) {
        int ic = g * ch_per_grp + ic_l;

        // Cooperative load of the input tile into local memory
        for (int i = local_id; i < in_tile_len; i += TILE_SIZE) {
            int global_pos = in_start + i;
            if (global_pos >= 0 && global_pos < in_length) {
                tile_buf[i] = input[n * in_channels * in_length + ic * in_length + global_pos];
            } else {
                tile_buf[i] = 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int o = tile_base + local_id;
        if (o < out_length) {
            int local_start = local_id * stride;
            for (int k = 0; k < kernel_size; k++) {
                int local_pos = local_start + k * dilation;
                int w_idx = oc * ch_per_grp * kernel_size + ic_l * kernel_size + k;
                acc += tile_buf[local_pos] * weight[w_idx];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int o = tile_base + local_id;
    if (o < out_length) {
        if (has_bias != 0) acc += bias[oc];
        int out_idx = n * out_channels * out_length + oc * out_length + o;
        output[out_idx] = acc;
    }
}
"#;

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn validate_conv1d_args(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &[f32],
    cfg: &Conv1dConfig,
    batch: usize,
    in_channels: usize,
    in_length: usize,
    out_channels: usize,
) -> Result<()> {
    if batch == 0 || in_channels == 0 || in_length == 0 || out_channels == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dimensions must be non-zero: batch={batch}, in_ch={in_channels}, \
                 in_len={in_length}, out_ch={out_channels}"
            ),
        }
        .into());
    }

    if !in_channels.is_multiple_of(cfg.groups) || !out_channels.is_multiple_of(cfg.groups) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "channels must be divisible by groups: in_ch={in_channels}, \
                 out_ch={out_channels}, groups={}",
                cfg.groups
            ),
        }
        .into());
    }

    let out_len = cfg.output_length(in_length);
    if out_len == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "convolution produces zero-length output".into(),
        }
        .into());
    }

    let expected_input = batch * in_channels * in_length;
    let ch_per_group = in_channels / cfg.groups;
    let expected_weight = out_channels * ch_per_group * cfg.kernel_size;
    let expected_output = batch * out_channels * out_len;

    if input.len() != expected_input {
        return Err(KernelError::InvalidArguments {
            reason: format!("input size mismatch: expected {expected_input}, got {}", input.len()),
        }
        .into());
    }
    if weight.len() != expected_weight {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "weight size mismatch: expected {expected_weight}, got {}",
                weight.len()
            ),
        }
        .into());
    }
    if output.len() != expected_output {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "output size mismatch: expected {expected_output}, got {}",
                output.len()
            ),
        }
        .into());
    }

    if let Some(b) = bias
        && b.len() != out_channels
    {
        return Err(KernelError::InvalidArguments {
            reason: format!("bias size mismatch: expected {out_channels}, got {}", b.len()),
        }
        .into());
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_transpose_args(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &[f32],
    cfg: &Conv1dConfig,
    batch: usize,
    in_channels: usize,
    in_length: usize,
    out_channels: usize,
    out_len: usize,
) -> Result<()> {
    if batch == 0 || in_channels == 0 || in_length == 0 || out_channels == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dimensions must be non-zero: batch={batch}, in_ch={in_channels}, \
                 in_len={in_length}, out_ch={out_channels}"
            ),
        }
        .into());
    }

    if !in_channels.is_multiple_of(cfg.groups) || !out_channels.is_multiple_of(cfg.groups) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "channels must be divisible by groups: in_ch={in_channels}, \
                 out_ch={out_channels}, groups={}",
                cfg.groups
            ),
        }
        .into());
    }

    if out_len == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "transposed conv produces zero-length output".into(),
        }
        .into());
    }

    let expected_input = batch * in_channels * in_length;
    let oc_per_group = out_channels / cfg.groups;
    let expected_weight = in_channels * oc_per_group * cfg.kernel_size;
    let expected_output = batch * out_channels * out_len;

    if input.len() != expected_input {
        return Err(KernelError::InvalidArguments {
            reason: format!("input size mismatch: expected {expected_input}, got {}", input.len()),
        }
        .into());
    }
    if weight.len() != expected_weight {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "weight size mismatch: expected {expected_weight}, got {}",
                weight.len()
            ),
        }
        .into());
    }
    if output.len() != expected_output {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "output size mismatch: expected {expected_output}, got {}",
                output.len()
            ),
        }
        .into());
    }

    if let Some(b) = bias
        && b.len() != out_channels
    {
        return Err(KernelError::InvalidArguments {
            reason: format!("bias size mismatch: expected {out_channels}, got {}", b.len()),
        }
        .into());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at index {i}: {x} vs {y} (tol={tol})");
        }
    }

    /// Brute-force reference conv1d for cross-checking.
    fn naive_conv1d_ref(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Vec<f32> {
        let eff_k = dilation * (kernel_size - 1) + 1;
        let out_len = (in_len + 2 * padding - eff_k) / stride + 1;
        let ch_per_g = in_ch / groups;
        let oc_per_g = out_ch / groups;
        let mut out = vec![0.0_f32; batch * out_ch * out_len];

        for n in 0..batch {
            for g in 0..groups {
                for oc_l in 0..oc_per_g {
                    let oc = g * oc_per_g + oc_l;
                    for o in 0..out_len {
                        let mut acc = bias.map_or(0.0, |b| b[oc]);
                        for ic_l in 0..ch_per_g {
                            let ic = g * ch_per_g + ic_l;
                            for k in 0..kernel_size {
                                let i_pos = o * stride + k * dilation;
                                if i_pos >= padding && i_pos - padding < in_len {
                                    let idx = n * in_ch * in_len + ic * in_len + (i_pos - padding);
                                    let w_idx =
                                        oc * ch_per_g * kernel_size + ic_l * kernel_size + k;
                                    acc += input[idx] * weight[w_idx];
                                }
                            }
                        }
                        out[n * out_ch * out_len + oc * out_len + o] = acc;
                    }
                }
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // Config tests
    // -----------------------------------------------------------------------

    #[test]
    fn config_output_length_no_padding() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        assert_eq!(cfg.output_length(5), 3); // (5 - 3) / 1 + 1
        assert_eq!(cfg.output_length(10), 8);
    }

    #[test]
    fn config_output_length_with_padding() {
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 1).unwrap();
        assert_eq!(cfg.output_length(5), 5); // same-padding
    }

    #[test]
    fn config_output_length_with_stride() {
        let cfg = Conv1dConfig::new(3, 2, 1, 1, 1).unwrap();
        assert_eq!(cfg.output_length(7), 4); // (7 + 2 - 3) / 2 + 1 = 4
    }

    #[test]
    fn config_output_length_with_dilation() {
        let cfg = Conv1dConfig::new(3, 1, 0, 2, 1).unwrap();
        // effective_kernel = 2*(3-1)+1 = 5
        assert_eq!(cfg.output_length(7), 3); // (7 - 5) / 1 + 1 = 3
    }

    #[test]
    fn config_output_length_zero_when_kernel_too_large() {
        let cfg = Conv1dConfig::simple(10).unwrap();
        assert_eq!(cfg.output_length(3), 0);
    }

    #[test]
    fn config_default() {
        let cfg = Conv1dConfig::default();
        assert_eq!(cfg.kernel_size, 3);
        assert_eq!(cfg.stride, 1);
        assert_eq!(cfg.padding, 0);
        assert_eq!(cfg.dilation, 1);
        assert_eq!(cfg.groups, 1);
    }

    #[test]
    fn config_rejects_zero_kernel_size() {
        assert!(Conv1dConfig::new(0, 1, 0, 1, 1).is_err());
    }

    #[test]
    fn config_rejects_zero_stride() {
        assert!(Conv1dConfig::new(3, 0, 0, 1, 1).is_err());
    }

    #[test]
    fn config_rejects_zero_dilation() {
        assert!(Conv1dConfig::new(3, 1, 0, 0, 1).is_err());
    }

    #[test]
    fn config_rejects_zero_groups() {
        assert!(Conv1dConfig::new(3, 1, 0, 1, 0).is_err());
    }

    // -----------------------------------------------------------------------
    // Forward pass — kernel_size = 1
    // -----------------------------------------------------------------------

    #[test]
    fn conv1d_kernel1_identity() {
        // kernel_size=1, identity weight => output == input
        let cfg = Conv1dConfig::simple(1).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0]; // [1, 1, 4]
        let weight = vec![1.0]; // [1, 1, 1]
        let mut output = vec![0.0; 4];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 4, 1).unwrap();
        assert_close(&output, &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn conv1d_kernel1_with_bias() {
        let cfg = Conv1dConfig::simple(1).unwrap();
        let input = vec![1.0, 2.0, 3.0]; // [1, 1, 3]
        let weight = vec![2.0]; // [1, 1, 1]
        let bias = vec![0.5];
        let mut output = vec![0.0; 3];
        Conv1dKernel::forward(&input, &weight, Some(&bias), &mut output, &cfg, 1, 1, 3, 1).unwrap();
        assert_close(&output, &[2.5, 4.5, 6.5], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Forward pass — kernel_size = 3
    // -----------------------------------------------------------------------

    #[test]
    fn conv1d_kernel3_no_padding() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // [1, 1, 5]
        let weight = vec![1.0, 1.0, 1.0]; // [1, 1, 3]
        let mut output = vec![0.0; 3]; // out_len = 3
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 5, 1).unwrap();
        assert_close(&output, &[6.0, 9.0, 12.0], 1e-6);
    }

    #[test]
    fn conv1d_kernel3_with_padding() {
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 1).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // [1, 1, 5]
        let weight = vec![1.0, 1.0, 1.0]; // [1, 1, 3]
        let mut output = vec![0.0; 5]; // same-padding
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 5, 1).unwrap();
        // pad(0,1,2,3,4,5,0)
        assert_close(&output, &[3.0, 6.0, 9.0, 12.0, 9.0], 1e-6);
    }

    #[test]
    fn conv1d_kernel3_cross_check_with_ref() {
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 1).unwrap();
        let input: Vec<f32> = (0..12).map(|i| i as f32).collect(); // [1, 3, 4]
        // weight: [out_ch=2, in_ch=3, ks=3] = 18 elements
        let weight: Vec<f32> = (0..18).map(|i| (i as f32) * 0.1).collect();
        let bias = vec![0.1, -0.1];
        let out_len = cfg.output_length(4);
        let mut output = vec![0.0; 2 * out_len]; // [1, 2, out_len]

        Conv1dKernel::forward(&input, &weight, Some(&bias), &mut output, &cfg, 1, 3, 4, 2).unwrap();

        let expected = naive_conv1d_ref(&input, &weight, Some(&bias), 1, 3, 4, 2, 3, 1, 1, 1, 1);
        assert_close(&output, &expected, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Forward pass — kernel_size = 5
    // -----------------------------------------------------------------------

    #[test]
    fn conv1d_kernel5_no_padding() {
        let cfg = Conv1dConfig::simple(5).unwrap();
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect(); // [1, 1, 8]
        let weight = vec![1.0; 5]; // [1, 1, 5]
        let mut output = vec![0.0; 4]; // out_len = 4
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 8, 1).unwrap();
        // [1+2+3+4+5, 2+3+4+5+6, 3+4+5+6+7, 4+5+6+7+8]
        assert_close(&output, &[15.0, 20.0, 25.0, 30.0], 1e-6);
    }

    #[test]
    fn conv1d_kernel5_with_padding() {
        let cfg = Conv1dConfig::new(5, 1, 2, 1, 1).unwrap();
        let input: Vec<f32> = (1..=5).map(|i| i as f32).collect(); // [1, 1, 5]
        let weight = vec![1.0; 5];
        let mut output = vec![0.0; 5]; // same-padding
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 5, 1).unwrap();
        let expected = naive_conv1d_ref(&input, &weight, None, 1, 1, 5, 1, 5, 1, 2, 1, 1);
        assert_close(&output, &expected, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Forward pass — kernel_size = 7
    // -----------------------------------------------------------------------

    #[test]
    fn conv1d_kernel7_no_padding() {
        let cfg = Conv1dConfig::simple(7).unwrap();
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let weight = vec![1.0; 7];
        let mut output = vec![0.0; 4]; // out_len = 10-7+1 = 4
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 10, 1).unwrap();
        let expected = naive_conv1d_ref(&input, &weight, None, 1, 1, 10, 1, 7, 1, 0, 1, 1);
        assert_close(&output, &expected, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Stride / padding / dilation combinations
    // -----------------------------------------------------------------------

    #[test]
    fn conv1d_stride2() {
        let cfg = Conv1dConfig::new(3, 2, 0, 1, 1).unwrap();
        let input: Vec<f32> = (1..=6).map(|i| i as f32).collect(); // [1,1,6]
        let weight = vec![1.0, 0.0, -1.0];
        let out_len = cfg.output_length(6); // (6-3)/2+1 = 2
        let mut output = vec![0.0; out_len];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 6, 1).unwrap();
        let expected = naive_conv1d_ref(&input, &weight, None, 1, 1, 6, 1, 3, 2, 0, 1, 1);
        assert_close(&output, &expected, 1e-6);
    }

    #[test]
    fn conv1d_stride3_padding1() {
        let cfg = Conv1dConfig::new(3, 3, 1, 1, 1).unwrap();
        let input: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let weight = vec![0.5; 3];
        let out_len = cfg.output_length(9);
        let mut output = vec![0.0; out_len];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 9, 1).unwrap();
        let expected = naive_conv1d_ref(&input, &weight, None, 1, 1, 9, 1, 3, 3, 1, 1, 1);
        assert_close(&output, &expected, 1e-6);
    }

    #[test]
    fn conv1d_dilation2() {
        let cfg = Conv1dConfig::new(3, 1, 0, 2, 1).unwrap();
        let input: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let weight = vec![1.0, 0.0, 1.0];
        // effective kernel = 5, out_len = 3
        let out_len = cfg.output_length(7);
        let mut output = vec![0.0; out_len];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 7, 1).unwrap();
        // pos0: inp[0]*1 + inp[2]*0 + inp[4]*1 = 0+4=4
        // pos1: inp[1]*1 + inp[3]*0 + inp[5]*1 = 1+5=6
        // pos2: inp[2]*1 + inp[4]*0 + inp[6]*1 = 2+6=8
        assert_close(&output, &[4.0, 6.0, 8.0], 1e-6);
    }

    #[test]
    fn conv1d_dilation3_padding2() {
        let cfg = Conv1dConfig::new(3, 1, 2, 3, 1).unwrap();
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let weight = vec![1.0; 3];
        let out_len = cfg.output_length(10);
        let mut output = vec![0.0; out_len];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 10, 1).unwrap();
        let expected = naive_conv1d_ref(&input, &weight, None, 1, 1, 10, 1, 3, 1, 2, 3, 1);
        assert_close(&output, &expected, 1e-6);
    }

    #[test]
    fn conv1d_stride2_dilation2_padding2() {
        let cfg = Conv1dConfig::new(3, 2, 2, 2, 1).unwrap();
        let input: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let weight = vec![1.0, -1.0, 1.0];
        let out_len = cfg.output_length(12);
        let mut output = vec![0.0; out_len];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 12, 1).unwrap();
        let expected = naive_conv1d_ref(&input, &weight, None, 1, 1, 12, 1, 3, 2, 2, 2, 1);
        assert_close(&output, &expected, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Multi-channel / multi-batch
    // -----------------------------------------------------------------------

    #[test]
    fn conv1d_multi_channel() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        // [1, 2, 5] input, [3, 2, 3] weight
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let weight: Vec<f32> = (0..18).map(|i| (i as f32) * 0.1).collect();
        let out_len = cfg.output_length(5); // 3
        let mut output = vec![0.0; 3 * out_len]; // [1, 3, 3]
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 2, 5, 3).unwrap();
        let expected = naive_conv1d_ref(&input, &weight, None, 1, 2, 5, 3, 3, 1, 0, 1, 1);
        assert_close(&output, &expected, 1e-5);
    }

    #[test]
    fn conv1d_batch2() {
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 1).unwrap();
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect(); // [2, 2, 4]
        let weight: Vec<f32> = (0..6).map(|i| (i as f32) * 0.1).collect(); // [1, 2, 3]
        let out_len = cfg.output_length(4); // 4
        let mut output = vec![0.0; 2 * 1 * out_len]; // [2, 1, 4]
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 2, 2, 4, 1).unwrap();
        let expected = naive_conv1d_ref(&input, &weight, None, 2, 2, 4, 1, 3, 1, 1, 1, 1);
        assert_close(&output, &expected, 1e-5);
    }

    #[test]
    fn conv1d_batch3_multichannel() {
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 1).unwrap();
        let batch = 3;
        let in_ch = 4;
        let in_len = 6;
        let out_ch = 2;
        let input: Vec<f32> = (0..(batch * in_ch * in_len)).map(|i| (i as f32) * 0.01).collect();
        let weight: Vec<f32> = (0..(out_ch * in_ch * 3)).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let bias = vec![0.5, -0.5];
        let out_len = cfg.output_length(in_len);
        let mut output = vec![0.0; batch * out_ch * out_len];
        Conv1dKernel::forward(
            &input,
            &weight,
            Some(&bias),
            &mut output,
            &cfg,
            batch,
            in_ch,
            in_len,
            out_ch,
        )
        .unwrap();
        let expected = naive_conv1d_ref(
            &input,
            &weight,
            Some(&bias),
            batch,
            in_ch,
            in_len,
            out_ch,
            3,
            1,
            1,
            1,
            1,
        );
        assert_close(&output, &expected, 1e-4);
    }

    // -----------------------------------------------------------------------
    // Grouped convolution
    // -----------------------------------------------------------------------

    #[test]
    fn conv1d_groups2() {
        let cfg = Conv1dConfig::new(3, 1, 0, 1, 2).unwrap();
        // [1, 4, 5] input, groups=2 => 2 ch/group
        // [4, 2, 3] weight (4 out_ch, 2 in_ch per group)
        let input: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let weight: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
        let out_len = cfg.output_length(5);
        let mut output = vec![0.0; 4 * out_len];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 4, 5, 4).unwrap();
        let expected = naive_conv1d_ref(&input, &weight, None, 1, 4, 5, 4, 3, 1, 0, 1, 2);
        assert_close(&output, &expected, 1e-5);
    }

    #[test]
    fn conv1d_groups_equal_channels() {
        // groups == in_channels == out_channels => depthwise
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 3).unwrap();
        let input: Vec<f32> = (0..15).map(|i| i as f32).collect(); // [1, 3, 5]
        let weight = vec![1.0; 9]; // [3, 1, 3]
        let out_len = cfg.output_length(5);
        let mut output = vec![0.0; 3 * out_len]; // [1, 3, 5]
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 3, 5, 3).unwrap();
        let expected = naive_conv1d_ref(&input, &weight, None, 1, 3, 5, 3, 3, 1, 1, 1, 3);
        assert_close(&output, &expected, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Depthwise convolution
    // -----------------------------------------------------------------------

    #[test]
    fn depthwise_basic() {
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 4).unwrap();
        let input: Vec<f32> = (0..20).map(|i| i as f32).collect(); // [1, 4, 5]
        let weight: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1).collect(); // [4, 1, 3]
        let out_len = cfg.output_length(5);
        let mut output = vec![0.0; 4 * out_len];
        DepthwiseConv1d::forward(&input, &weight, None, &mut output, &cfg, 1, 4, 5).unwrap();
        let expected = naive_conv1d_ref(&input, &weight, None, 1, 4, 5, 4, 3, 1, 1, 1, 4);
        assert_close(&output, &expected, 1e-5);
    }

    #[test]
    fn depthwise_rejects_wrong_groups() {
        let cfg = Conv1dConfig::new(3, 1, 0, 1, 2).unwrap();
        let input = vec![0.0; 20];
        let weight = vec![0.0; 6];
        let mut output = vec![0.0; 12];
        assert!(
            DepthwiseConv1d::forward(&input, &weight, None, &mut output, &cfg, 1, 4, 5).is_err()
        );
    }

    #[test]
    fn depthwise_with_bias() {
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 2).unwrap();
        let input: Vec<f32> = vec![1.0; 10]; // [1, 2, 5]
        let weight = vec![1.0; 6]; // [2, 1, 3]
        let bias = vec![10.0, -10.0];
        let out_len = cfg.output_length(5);
        let mut output = vec![0.0; 2 * out_len];
        DepthwiseConv1d::forward(&input, &weight, Some(&bias), &mut output, &cfg, 1, 2, 5).unwrap();
        let expected = naive_conv1d_ref(&input, &weight, Some(&bias), 1, 2, 5, 2, 3, 1, 1, 1, 2);
        assert_close(&output, &expected, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Causal convolution
    // -----------------------------------------------------------------------

    #[test]
    fn causal_no_future_leakage_kernel3() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        // Place a spike at position 4, verify positions 0..=3 are unaffected.
        let mut input = vec![0.0_f32; 8]; // [1, 1, 8]
        input[4] = 1.0;
        let weight = vec![1.0, 1.0, 1.0]; // [1, 1, 3]
        let out_len = causal_output_length(8, &cfg);
        let mut output = vec![0.0; out_len];
        CausalConv1d::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 8, 1).unwrap();
        // Positions 0..4 must be zero (no future leakage)
        for i in 0..4 {
            assert!(output[i].abs() < 1e-6, "causal leakage at position {i}: {}", output[i]);
        }
        // Position 4 should see the spike
        assert!(output[4].abs() > 0.5, "spike not visible at position 4");
    }

    #[test]
    fn causal_no_future_leakage_kernel5() {
        let cfg = Conv1dConfig::simple(5).unwrap();
        let mut input = vec![0.0_f32; 10];
        input[6] = 1.0;
        let weight = vec![1.0; 5];
        let out_len = causal_output_length(10, &cfg);
        let mut output = vec![0.0; out_len];
        CausalConv1d::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 10, 1).unwrap();
        for i in 0..6 {
            assert!(output[i].abs() < 1e-6, "causal leakage at {i}");
        }
        assert!(output[6].abs() > 0.5);
    }

    #[test]
    fn causal_output_length_equals_input_stride1() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        let in_len = 10;
        assert_eq!(causal_output_length(in_len, &cfg), in_len);
    }

    #[test]
    fn causal_output_length_stride2() {
        let cfg = Conv1dConfig::new(3, 2, 0, 1, 1).unwrap();
        assert_eq!(causal_output_length(10, &cfg), 5);
    }

    #[test]
    fn causal_preserves_existing_signal() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        let input: Vec<f32> = (1..=6).map(|i| i as f32).collect();
        // weight[2] corresponds to the current position in causal conv
        let weight = vec![0.0, 0.0, 1.0];
        let out_len = causal_output_length(6, &cfg);
        let mut output = vec![0.0; out_len];
        CausalConv1d::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 6, 1).unwrap();
        // With weight=[0,0,1], output[i] = input[i]
        assert_close(&output, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn causal_multichannel() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        let in_ch = 2;
        let out_ch = 2;
        let in_len = 5;
        let input: Vec<f32> = (0..(in_ch * in_len)).map(|i| i as f32).collect();
        let weight: Vec<f32> = vec![1.0; out_ch * in_ch * 3];
        let out_len = causal_output_length(in_len, &cfg);
        let mut output = vec![0.0; out_ch * out_len];
        CausalConv1d::forward(&input, &weight, None, &mut output, &cfg, 1, in_ch, in_len, out_ch)
            .unwrap();
        assert_eq!(output.len(), out_ch * in_len);
    }

    #[test]
    fn causal_with_dilation() {
        let cfg = Conv1dConfig::new(3, 1, 0, 2, 1).unwrap();
        let in_len = 10;
        let mut input = vec![0.0_f32; in_len];
        input[7] = 1.0;
        let weight = vec![1.0; 3];
        let out_len = causal_output_length(in_len, &cfg);
        let mut output = vec![0.0; out_len];
        CausalConv1d::forward(&input, &weight, None, &mut output, &cfg, 1, 1, in_len, 1).unwrap();
        // With dilation=2, causal pad = 2*(3-1)=4
        // Verify no future leakage
        for i in 0..7 {
            assert!(output[i].abs() < 1e-6, "causal leakage at {i}");
        }
    }

    // -----------------------------------------------------------------------
    // Transposed convolution
    // -----------------------------------------------------------------------

    #[test]
    fn transpose_output_shape() {
        // out_len = (in_len-1)*stride + dilation*(ks-1) + 1 - 2*padding
        let cfg = Conv1dConfig::new(3, 2, 0, 1, 1).unwrap();
        let out_len = transpose_output_length(4, &cfg);
        // (4-1)*2 + 1*(3-1) + 1 - 0 = 6+2+1 = 9
        assert_eq!(out_len, 9);
    }

    #[test]
    fn transpose_output_shape_with_padding() {
        let cfg = Conv1dConfig::new(3, 2, 1, 1, 1).unwrap();
        let out_len = transpose_output_length(4, &cfg);
        // (4-1)*2 + 2 + 1 - 2 = 7
        assert_eq!(out_len, 7);
    }

    #[test]
    fn transpose_basic() {
        let cfg = Conv1dConfig::new(3, 1, 0, 1, 1).unwrap();
        let input = vec![1.0, 2.0, 3.0]; // [1, 1, 3]
        // weight: [in_channels, out_channels/groups, kernel_size] = [1, 1, 3]
        let weight = vec![1.0, 1.0, 1.0];
        let out_len = transpose_output_length(3, &cfg); // (3-1)*1+2+1 = 5
        let mut output = vec![0.0; out_len];
        Conv1dTranspose::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 3, 1).unwrap();
        // Expected: [1, 3, 6, 5, 3]
        assert_close(&output, &[1.0, 3.0, 6.0, 5.0, 3.0], 1e-6);
    }

    #[test]
    fn transpose_stride2_upsample() {
        let cfg = Conv1dConfig::new(3, 2, 0, 1, 1).unwrap();
        let input = vec![1.0, 1.0]; // [1, 1, 2]
        let weight = vec![1.0, 2.0, 3.0]; // [1, 1, 3]
        let out_len = transpose_output_length(2, &cfg); // (2-1)*2+2+1=5
        let mut output = vec![0.0; out_len];
        Conv1dTranspose::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 2, 1).unwrap();
        // inp[0] scatters to positions 0,1,2 with weights 1,2,3
        // inp[1] scatters to positions 2,3,4 with weights 1,2,3
        assert_close(&output, &[1.0, 2.0, 4.0, 2.0, 3.0], 1e-6);
    }

    #[test]
    fn transpose_with_bias() {
        let cfg = Conv1dConfig::new(3, 1, 0, 1, 1).unwrap();
        let input = vec![1.0, 1.0]; // [1, 1, 2]
        let weight = vec![1.0, 0.0, 1.0]; // [1, 1, 3]
        let bias = vec![0.5];
        let out_len = transpose_output_length(2, &cfg); // (2-1)*1+2+1=4
        let mut output = vec![0.0; out_len];
        Conv1dTranspose::forward(&input, &weight, Some(&bias), &mut output, &cfg, 1, 1, 2, 1)
            .unwrap();
        // inp[0] scatters w=[1,0,1] to pos 0,1,2; inp[1] scatters w=[1,0,1] to pos 1,2,3
        // => [1, 1, 1, 1]  +bias=0.5 => [1.5, 1.5, 1.5, 1.5]
        assert_close(&output, &[1.5, 1.5, 1.5, 1.5], 1e-6);
    }

    #[test]
    fn transpose_multichannel() {
        let cfg = Conv1dConfig::new(2, 1, 0, 1, 1).unwrap();
        // [1, 2, 3] input, weight [2, 1, 2] => [1, 1, out_len]
        let input: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let weight: Vec<f32> = vec![1.0; 4]; // [2, 1, 2]
        let out_len = transpose_output_length(3, &cfg); // (3-1)+1+1=4
        let mut output = vec![0.0; out_len]; // [1, 1, 4]
        Conv1dTranspose::forward(&input, &weight, None, &mut output, &cfg, 1, 2, 3, 1).unwrap();
        assert_eq!(output.len(), out_len);
    }

    // -----------------------------------------------------------------------
    // Winograd F(2,3)
    // -----------------------------------------------------------------------

    #[test]
    fn winograd_matches_naive_basic() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect(); // [1, 1, 8]
        let weight = vec![0.5, -1.0, 0.5]; // [1, 1, 3]
        let out_len = cfg.output_length(8);
        let mut out_naive = vec![0.0; out_len];
        let mut out_wino = vec![0.0; out_len];

        Conv1dKernel::forward(&input, &weight, None, &mut out_naive, &cfg, 1, 1, 8, 1).unwrap();
        WinoGrad1d::forward(&input, &weight, None, &mut out_wino, &cfg, 1, 1, 8, 1).unwrap();

        assert_close(&out_wino, &out_naive, 1e-5);
    }

    #[test]
    fn winograd_matches_naive_with_padding() {
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 1).unwrap();
        let input: Vec<f32> = (0..6).map(|i| (i as f32) * 0.3).collect();
        let weight = vec![1.0, 2.0, 3.0];
        let out_len = cfg.output_length(6);
        let mut out_naive = vec![0.0; out_len];
        let mut out_wino = vec![0.0; out_len];

        Conv1dKernel::forward(&input, &weight, None, &mut out_naive, &cfg, 1, 1, 6, 1).unwrap();
        WinoGrad1d::forward(&input, &weight, None, &mut out_wino, &cfg, 1, 1, 6, 1).unwrap();

        assert_close(&out_wino, &out_naive, 1e-4);
    }

    #[test]
    fn winograd_matches_naive_multichannel() {
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 1).unwrap();
        let batch = 2;
        let in_ch = 3;
        let out_ch = 2;
        let in_len = 8;
        let input: Vec<f32> =
            (0..(batch * in_ch * in_len)).map(|i| (i as f32) * 0.05 - 1.0).collect();
        let weight: Vec<f32> = (0..(out_ch * in_ch * 3)).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let bias = vec![0.1, -0.2];
        let out_len = cfg.output_length(in_len);
        let mut out_naive = vec![0.0; batch * out_ch * out_len];
        let mut out_wino = vec![0.0; batch * out_ch * out_len];

        Conv1dKernel::forward(
            &input,
            &weight,
            Some(&bias),
            &mut out_naive,
            &cfg,
            batch,
            in_ch,
            in_len,
            out_ch,
        )
        .unwrap();
        WinoGrad1d::forward(
            &input,
            &weight,
            Some(&bias),
            &mut out_wino,
            &cfg,
            batch,
            in_ch,
            in_len,
            out_ch,
        )
        .unwrap();

        assert_close(&out_wino, &out_naive, 1e-4);
    }

    #[test]
    fn winograd_matches_naive_groups() {
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 2).unwrap();
        let in_ch = 4;
        let out_ch = 4;
        let in_len = 6;
        let input: Vec<f32> = (0..(in_ch * in_len)).map(|i| i as f32 * 0.1).collect();
        let weight: Vec<f32> = (0..(out_ch * 2 * 3)).map(|i| i as f32 * 0.05).collect();
        let out_len = cfg.output_length(in_len);
        let mut out_naive = vec![0.0; out_ch * out_len];
        let mut out_wino = vec![0.0; out_ch * out_len];

        Conv1dKernel::forward(
            &input,
            &weight,
            None,
            &mut out_naive,
            &cfg,
            1,
            in_ch,
            in_len,
            out_ch,
        )
        .unwrap();
        WinoGrad1d::forward(&input, &weight, None, &mut out_wino, &cfg, 1, in_ch, in_len, out_ch)
            .unwrap();

        assert_close(&out_wino, &out_naive, 1e-4);
    }

    #[test]
    fn winograd_rejects_kernel_size_5() {
        let cfg = Conv1dConfig::simple(5).unwrap();
        let input = vec![0.0; 10];
        let weight = vec![0.0; 5];
        let mut output = vec![0.0; 6];
        assert!(
            WinoGrad1d::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 10, 1).is_err()
        );
    }

    #[test]
    fn winograd_rejects_stride2() {
        let cfg = Conv1dConfig::new(3, 2, 0, 1, 1).unwrap();
        let input = vec![0.0; 6];
        let weight = vec![0.0; 3];
        let mut output = vec![0.0; 2];
        assert!(WinoGrad1d::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 6, 1).is_err());
    }

    #[test]
    fn winograd_rejects_dilation2() {
        let cfg = Conv1dConfig::new(3, 1, 0, 2, 1).unwrap();
        let input = vec![0.0; 7];
        let weight = vec![0.0; 3];
        let mut output = vec![0.0; 3];
        assert!(WinoGrad1d::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 7, 1).is_err());
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn single_element_input() {
        let cfg = Conv1dConfig::simple(1).unwrap();
        let input = vec![42.0];
        let weight = vec![2.0];
        let mut output = vec![0.0; 1];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 1, 1).unwrap();
        assert_close(&output, &[84.0], 1e-6);
    }

    #[test]
    fn single_element_with_kernel3_padded() {
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 1).unwrap();
        let input = vec![5.0]; // [1, 1, 1]
        let weight = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 1]; // out_len = 1
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 1, 1).unwrap();
        // Only the center weight (2.0) sees the input
        assert_close(&output, &[10.0], 1e-6);
    }

    #[test]
    fn error_on_zero_batch() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        let input = vec![];
        let weight = vec![1.0; 3];
        let mut output = vec![];
        assert!(
            Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 0, 1, 5, 1).is_err()
        );
    }

    #[test]
    fn error_on_wrong_output_size() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        let input = vec![0.0; 5];
        let weight = vec![0.0; 3];
        let mut output = vec![0.0; 10]; // wrong size
        assert!(
            Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 5, 1).is_err()
        );
    }

    #[test]
    fn error_on_channels_not_divisible_by_groups() {
        let cfg = Conv1dConfig::new(3, 1, 0, 1, 3).unwrap();
        let input = vec![0.0; 20]; // 4 channels, groups=3 doesn't divide
        let weight = vec![0.0; 3];
        let mut output = vec![0.0; 6];
        assert!(
            Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 4, 5, 4).is_err()
        );
    }

    #[test]
    fn error_on_wrong_bias_size() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        let input = vec![0.0; 5];
        let weight = vec![0.0; 3];
        let bias = vec![0.0; 5]; // wrong: should be 1
        let mut output = vec![0.0; 3];
        assert!(
            Conv1dKernel::forward(&input, &weight, Some(&bias), &mut output, &cfg, 1, 1, 5, 1)
                .is_err()
        );
    }

    // -----------------------------------------------------------------------
    // Property tests
    // -----------------------------------------------------------------------

    #[test]
    fn property_zero_input_gives_zero_output() {
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 1).unwrap();
        let input = vec![0.0; 20]; // [1, 2, 10]
        let weight: Vec<f32> = (0..12).map(|i| i as f32).collect(); // [2, 2, 3]
        let out_len = cfg.output_length(10);
        let mut output = vec![0.0; 2 * out_len];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 2, 10, 2).unwrap();
        assert!(output.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn property_zero_weight_gives_zero_output() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        let input: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let weight = vec![0.0; 3];
        let mut output = vec![0.0; 3];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 5, 1).unwrap();
        assert!(output.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn property_identity_kernel_copies_input() {
        let cfg = Conv1dConfig::simple(1).unwrap();
        let input: Vec<f32> = (0..10).map(|i| i as f32 * 0.7).collect();
        let weight = vec![1.0];
        let mut output = vec![0.0; 10];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 10, 1).unwrap();
        assert_close(&output, &input, 1e-6);
    }

    #[test]
    fn property_linearity_scaling() {
        // conv(alpha * x, w) == alpha * conv(x, w)
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 1).unwrap();
        let alpha = 3.5_f32;
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let scaled_input: Vec<f32> = input.iter().map(|&v| v * alpha).collect();
        let weight = vec![1.0, -0.5, 0.25];
        let out_len = cfg.output_length(8);

        let mut out1 = vec![0.0; out_len];
        let mut out2 = vec![0.0; out_len];

        Conv1dKernel::forward(&input, &weight, None, &mut out1, &cfg, 1, 1, 8, 1).unwrap();
        Conv1dKernel::forward(&scaled_input, &weight, None, &mut out2, &cfg, 1, 1, 8, 1).unwrap();

        let out1_scaled: Vec<f32> = out1.iter().map(|&v| v * alpha).collect();
        assert_close(&out1_scaled, &out2, 1e-4);
    }

    #[test]
    fn property_linearity_additivity() {
        // conv(x1 + x2, w) == conv(x1, w) + conv(x2, w)
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 1).unwrap();
        let x1: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let x2: Vec<f32> = (0..6).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let x_sum: Vec<f32> = x1.iter().zip(&x2).map(|(a, b)| a + b).collect();
        let weight = vec![1.0, -2.0, 1.0];
        let out_len = cfg.output_length(6);

        let mut o1 = vec![0.0; out_len];
        let mut o2 = vec![0.0; out_len];
        let mut o_sum = vec![0.0; out_len];

        Conv1dKernel::forward(&x1, &weight, None, &mut o1, &cfg, 1, 1, 6, 1).unwrap();
        Conv1dKernel::forward(&x2, &weight, None, &mut o2, &cfg, 1, 1, 6, 1).unwrap();
        Conv1dKernel::forward(&x_sum, &weight, None, &mut o_sum, &cfg, 1, 1, 6, 1).unwrap();

        let o_added: Vec<f32> = o1.iter().zip(&o2).map(|(a, b)| a + b).collect();
        assert_close(&o_added, &o_sum, 1e-4);
    }

    #[test]
    fn property_bias_is_additive() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        let input: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let weight = vec![1.0, 1.0, 1.0];
        let bias = vec![100.0];
        let out_len = cfg.output_length(5);

        let mut out_no_bias = vec![0.0; out_len];
        let mut out_with_bias = vec![0.0; out_len];

        Conv1dKernel::forward(&input, &weight, None, &mut out_no_bias, &cfg, 1, 1, 5, 1).unwrap();
        Conv1dKernel::forward(&input, &weight, Some(&bias), &mut out_with_bias, &cfg, 1, 1, 5, 1)
            .unwrap();

        let expected: Vec<f32> = out_no_bias.iter().map(|&v| v + 100.0).collect();
        assert_close(&out_with_bias, &expected, 1e-6);
    }

    // -----------------------------------------------------------------------
    // ConvStats
    // -----------------------------------------------------------------------

    #[test]
    fn stats_basic() {
        let stats = ConvStats::for_conv1d(1, 16, 8, 32, 3, 1);
        assert!(stats.flops > 0);
        assert!(stats.bytes_transferred > 0);
        assert_eq!(stats.kernel_time_secs, 0.0);
    }

    #[test]
    fn stats_finalise() {
        let mut stats = ConvStats::for_conv1d(1, 16, 8, 32, 3, 1);
        stats.finalise(0.001);
        assert!(stats.gflops > 0.0);
        assert!(stats.bandwidth_gbs > 0.0);
        assert!((stats.kernel_time_secs - 0.001).abs() < 1e-9);
    }

    #[test]
    fn stats_zero_time() {
        let mut stats = ConvStats::for_conv1d(1, 1, 1, 1, 1, 1);
        stats.finalise(0.0);
        assert_eq!(stats.gflops, 0.0);
        assert_eq!(stats.bandwidth_gbs, 0.0);
    }

    // -----------------------------------------------------------------------
    // OpenCL source presence
    // -----------------------------------------------------------------------

    #[test]
    fn opencl_source_contains_naive_kernel() {
        assert!(CONV1D_CL.contains("conv1d_naive"));
    }

    #[test]
    fn opencl_source_contains_tiled_kernel() {
        assert!(CONV1D_CL.contains("conv1d_tiled"));
    }

    #[test]
    fn opencl_source_contains_tile_size() {
        assert!(CONV1D_CL.contains("TILE_SIZE"));
    }

    // -----------------------------------------------------------------------
    // Additional stride/kernel combos for coverage
    // -----------------------------------------------------------------------

    #[test]
    fn conv1d_kernel3_stride1_padding0_dilation1() {
        let cfg = Conv1dConfig::new(3, 1, 0, 1, 1).unwrap();
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let weight = vec![0.5, -1.0, 0.5];
        let out_len = cfg.output_length(10);
        let mut output = vec![0.0; out_len];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 10, 1).unwrap();
        let expected = naive_conv1d_ref(&input, &weight, None, 1, 1, 10, 1, 3, 1, 0, 1, 1);
        assert_close(&output, &expected, 1e-6);
    }

    #[test]
    fn conv1d_large_padding() {
        let cfg = Conv1dConfig::new(3, 1, 4, 1, 1).unwrap();
        let input = vec![1.0, 2.0, 3.0]; // [1, 1, 3]
        let weight = vec![1.0, 1.0, 1.0];
        let out_len = cfg.output_length(3); // (3+8-3)/1+1 = 9
        let mut output = vec![0.0; out_len];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 3, 1).unwrap();
        let expected = naive_conv1d_ref(&input, &weight, None, 1, 1, 3, 1, 3, 1, 4, 1, 1);
        assert_close(&output, &expected, 1e-6);
    }

    #[test]
    fn conv1d_winograd_odd_length() {
        // Odd output length to test final partial tile in Winograd
        let cfg = Conv1dConfig::new(3, 1, 1, 1, 1).unwrap();
        let input: Vec<f32> = (0..7).map(|i| i as f32 * 0.3).collect();
        let weight = vec![-1.0, 0.0, 1.0];
        let out_len = cfg.output_length(7); // 7
        let mut out_naive = vec![0.0; out_len];
        let mut out_wino = vec![0.0; out_len];

        Conv1dKernel::forward(&input, &weight, None, &mut out_naive, &cfg, 1, 1, 7, 1).unwrap();
        WinoGrad1d::forward(&input, &weight, None, &mut out_wino, &cfg, 1, 1, 7, 1).unwrap();

        assert_close(&out_wino, &out_naive, 1e-4);
    }

    #[test]
    fn conv1d_winograd_length2() {
        // Minimal: 2 output elements = exactly 1 tile
        let cfg = Conv1dConfig::simple(3).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0]; // out_len = 2
        let weight = vec![1.0, 0.0, -1.0];
        let mut out_naive = vec![0.0; 2];
        let mut out_wino = vec![0.0; 2];

        Conv1dKernel::forward(&input, &weight, None, &mut out_naive, &cfg, 1, 1, 4, 1).unwrap();
        WinoGrad1d::forward(&input, &weight, None, &mut out_wino, &cfg, 1, 1, 4, 1).unwrap();

        assert_close(&out_wino, &out_naive, 1e-5);
    }

    #[test]
    fn conv1d_winograd_single_output() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        let input = vec![1.0, 2.0, 3.0]; // out_len = 1
        let weight = vec![1.0, 1.0, 1.0];
        let mut out_naive = vec![0.0; 1];
        let mut out_wino = vec![0.0; 1];

        Conv1dKernel::forward(&input, &weight, None, &mut out_naive, &cfg, 1, 1, 3, 1).unwrap();
        WinoGrad1d::forward(&input, &weight, None, &mut out_wino, &cfg, 1, 1, 3, 1).unwrap();

        assert_close(&out_wino, &out_naive, 1e-5);
    }

    #[test]
    fn transpose_round_trip_shape() {
        // Conv followed by ConvTranspose should recover the original length
        // when stride=1, padding=0
        let cfg = Conv1dConfig::simple(3).unwrap();
        let in_len = 10;
        let conv_out_len = cfg.output_length(in_len); // 8
        let deconv_out_len = transpose_output_length(conv_out_len, &cfg); // 10
        assert_eq!(deconv_out_len, in_len);
    }

    #[test]
    fn causal_constant_input() {
        // Constant input with uniform weights => constant output
        let cfg = Conv1dConfig::simple(3).unwrap();
        let input = vec![1.0; 10];
        let weight = vec![1.0; 3];
        let out_len = causal_output_length(10, &cfg);
        let mut output = vec![0.0; out_len];
        CausalConv1d::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 10, 1).unwrap();
        // First few positions see fewer inputs due to left padding
        // pos0: w[0]*1 = 1, pos1: w[0]*1+w[1]*1 = 2, pos2+: 3
        assert_close(&output[0..1], &[1.0], 1e-6);
        assert_close(&output[1..2], &[2.0], 1e-6);
        for v in &output[2..] {
            assert!((v - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn conv1d_negative_weights() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        let input = vec![1.0; 5];
        let weight = vec![-1.0, -1.0, -1.0];
        let mut output = vec![0.0; 3];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 5, 1).unwrap();
        assert_close(&output, &[-3.0, -3.0, -3.0], 1e-6);
    }

    #[test]
    fn conv1d_asymmetric_weight() {
        let cfg = Conv1dConfig::simple(3).unwrap();
        let input = vec![0.0, 0.0, 1.0, 0.0, 0.0]; // impulse at position 2
        let weight = vec![1.0, 2.0, 3.0]; // asymmetric
        let mut output = vec![0.0; 3];
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 5, 1).unwrap();
        // pos0: inp[0]*1+inp[1]*2+inp[2]*3 = 3
        // pos1: inp[1]*1+inp[2]*2+inp[3]*3 = 2
        // pos2: inp[2]*1+inp[3]*2+inp[4]*3 = 1
        assert_close(&output, &[3.0, 2.0, 1.0], 1e-6);
    }

    #[test]
    fn conv1d_output_channels_gt_input() {
        let cfg = Conv1dConfig::simple(1).unwrap();
        let input = vec![1.0, 2.0, 3.0]; // [1, 1, 3]
        // [4, 1, 1] weight — 4 output channels from 1 input channel
        let weight = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 12]; // [1, 4, 3]
        Conv1dKernel::forward(&input, &weight, None, &mut output, &cfg, 1, 1, 3, 4).unwrap();
        // oc=0: [1,2,3], oc=1: [2,4,6], oc=2: [3,6,9], oc=3: [4,8,12]
        assert_close(&output, &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0], 1e-6);
    }
}
