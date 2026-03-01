//! CUDA 1-D convolution kernel with CPU fallback.
//!
//! Provides grouped 1-D convolution with configurable stride, padding,
//! dilation, and groups — including depthwise separable convolution
//! (`groups == in_channels`).
//!
//! # Kernel strategy
//!
//! Each thread computes one output element `(oc, ow)`.  Grid dimensions
//! are `(ceil(out_w / block_x), out_channels, 1)`.  For depthwise
//! convolution the inner loop collapses to `ic_per_group == 1`, giving
//! high occupancy on Ampere+.
//!
//! # CPU fallback
//!
//! [`conv1d_cpu`] provides an equivalent pure-Rust implementation for
//! correctness testing and non-GPU environments.
//!
//! # Layout
//!
//! * `input`:  `[in_channels, input_width]`  (channels-first, contiguous)
//! * `weight`: `[out_channels, in_channels / groups, kernel_size]`
//! * `bias`:   `[out_channels]`  (optional)
//! * output:   `[out_channels, output_width]`

use bitnet_common::{KernelError, Result};

// ── Configuration ──────────────────────────────────────────────────

/// Padding mode for 1-D convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaddingMode {
    /// Explicit zero-padding added to each side.
    Zero(usize),
    /// Automatically compute padding so that
    /// `output_width == ceil(input_width / stride)`.
    Same,
}

/// Launch configuration for the 1-D convolution kernel.
#[derive(Debug, Clone)]
pub struct Conv1dConfig {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels (number of filters).
    pub out_channels: usize,
    /// Spatial extent of each filter.
    pub kernel_size: usize,
    /// Step between successive convolution windows.
    pub stride: usize,
    /// Padding applied to the input.
    pub padding: PaddingMode,
    /// Spacing between kernel elements.
    pub dilation: usize,
    /// Number of blocked connections from input to output channels.
    /// Use `in_channels` for depthwise convolution.
    pub groups: usize,
    /// Whether a bias vector is expected.
    pub bias: bool,
}

impl Conv1dConfig {
    /// Effective kernel size accounting for dilation.
    #[inline]
    fn effective_kernel_size(&self) -> usize {
        self.dilation * (self.kernel_size - 1) + 1
    }

    /// Resolve padding as `(pad_left, pad_right)`.
    fn resolve_padding(&self, input_width: usize) -> (usize, usize) {
        match self.padding {
            PaddingMode::Zero(p) => (p, p),
            PaddingMode::Same => {
                let ek = self.effective_kernel_size();
                let out_w = input_width.div_ceil(self.stride);
                let needed = out_w.saturating_sub(1) * self.stride + ek;
                let total = needed.saturating_sub(input_width);
                let pad_left = total / 2;
                (pad_left, total - pad_left)
            }
        }
    }

    /// Compute output spatial width for a given input width.
    pub fn output_width(&self, input_width: usize) -> usize {
        let (pl, pr) = self.resolve_padding(input_width);
        let ek = self.effective_kernel_size();
        let padded = input_width + pl + pr;
        if padded < ek { 0 } else { (padded - ek) / self.stride + 1 }
    }

    /// Compute the CUDA grid dimensions for the conv1d kernel.
    ///
    /// Grid: `(ceil(out_w / threads_x), out_channels, 1)`.
    pub fn grid_dim(&self, out_w: usize, threads_per_block: u32) -> (u32, u32, u32) {
        let blocks_x = (out_w as u32).div_ceil(threads_per_block);
        (blocks_x, self.out_channels as u32, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self, threads_per_block: u32) -> (u32, u32, u32) {
        (threads_per_block, 1, 1)
    }
}

// ── Validation ─────────────────────────────────────────────────────

fn validate(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv1dConfig,
) -> Result<usize> {
    if config.in_channels == 0 {
        return Err(invalid("in_channels must be > 0"));
    }
    if config.out_channels == 0 {
        return Err(invalid("out_channels must be > 0"));
    }
    if config.kernel_size == 0 {
        return Err(invalid("kernel_size must be > 0"));
    }
    if config.stride == 0 {
        return Err(invalid("stride must be > 0"));
    }
    if config.dilation == 0 {
        return Err(invalid("dilation must be > 0"));
    }
    if config.groups == 0 {
        return Err(invalid("groups must be > 0"));
    }
    if !config.in_channels.is_multiple_of(config.groups) {
        return Err(invalid("in_channels must be divisible by groups"));
    }
    if !config.out_channels.is_multiple_of(config.groups) {
        return Err(invalid("out_channels must be divisible by groups"));
    }

    if !input.len().is_multiple_of(config.in_channels) {
        return Err(invalid("input length must be divisible by in_channels"));
    }
    let input_width = input.len() / config.in_channels;
    if input_width == 0 {
        return Err(invalid("input spatial width must be > 0"));
    }

    let ic_per_group = config.in_channels / config.groups;
    let expected_weight = config.out_channels * ic_per_group * config.kernel_size;
    if weight.len() != expected_weight {
        return Err(invalid(&format!(
            "weight length mismatch: expected {expected_weight}, \
             got {}",
            weight.len()
        )));
    }

    if config.bias {
        match bias {
            Some(b) if b.len() != config.out_channels => {
                return Err(invalid(&format!(
                    "bias length mismatch: expected {}, got {}",
                    config.out_channels,
                    b.len()
                )));
            }
            None => {
                return Err(invalid("config.bias is true but no bias provided"));
            }
            _ => {}
        }
    }

    let out_w = config.output_width(input_width);
    if out_w == 0 {
        return Err(invalid(
            "convolution produces zero-width output \
             (kernel larger than padded input)",
        ));
    }

    Ok(input_width)
}

fn invalid(reason: &str) -> bitnet_common::BitNetError {
    KernelError::InvalidArguments { reason: reason.to_string() }.into()
}

// ── CPU fallback ───────────────────────────────────────────────────

/// Pure-Rust 1-D convolution (CPU fallback).
///
/// Computes grouped conv1d with stride, padding, and dilation on
/// contiguous `f32` slices.  Layout matches the CUDA kernel:
///
/// * `input`:  `[in_channels, input_width]`
/// * `weight`: `[out_channels, in_channels / groups, kernel_size]`
/// * `bias`:   `[out_channels]` (optional)
/// * returns:  `[out_channels, output_width]`
pub fn conv1d_cpu(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv1dConfig,
) -> Result<Vec<f32>> {
    let input_width = validate(input, weight, bias, config)?;
    let out_w = config.output_width(input_width);
    let (pad_left, _) = config.resolve_padding(input_width);
    let ic_per_group = config.in_channels / config.groups;
    let oc_per_group = config.out_channels / config.groups;

    let mut output = vec![0.0f32; config.out_channels * out_w];

    for g in 0..config.groups {
        for oc_local in 0..oc_per_group {
            let oc = g * oc_per_group + oc_local;
            for ow in 0..out_w {
                let mut sum = 0.0f32;
                for ic_local in 0..ic_per_group {
                    let ic = g * ic_per_group + ic_local;
                    let w_base = (oc * ic_per_group + ic_local) * config.kernel_size;
                    for k in 0..config.kernel_size {
                        let iw = ow * config.stride + k * config.dilation;
                        if iw >= pad_left && iw - pad_left < input_width {
                            sum += input[ic * input_width + (iw - pad_left)] * weight[w_base + k];
                        }
                    }
                }
                output[oc * out_w + ow] = sum;
            }
        }
    }

    if let Some(b) = bias
        && config.bias
    {
        for oc in 0..config.out_channels {
            let bias_val = b[oc];
            for v in &mut output[oc * out_w..(oc + 1) * out_w] {
                *v += bias_val;
            }
        }
    }

    Ok(output)
}

// ── CUDA launch stub ───────────────────────────────────────────────

/// Launch stub for the conv1d CUDA kernel.
///
/// Returns `GpuError` until a real PTX kernel is compiled and loaded.
pub fn launch_conv1d(
    _input: &[f32],
    _weight: &[f32],
    _bias: Option<&[f32]>,
    config: &Conv1dConfig,
    input_width: usize,
) -> Result<Vec<f32>> {
    let out_w = config.output_width(input_width);
    let threads: u32 = 256;
    log::debug!(
        "conv1d stub: in_ch={}, out_ch={}, ks={}, grid={:?}",
        config.in_channels,
        config.out_channels,
        config.kernel_size,
        config.grid_dim(out_w, threads),
    );
    Err(KernelError::GpuError {
        reason: "conv1d CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Unified dispatch ───────────────────────────────────────────────

/// Apply 1-D convolution with automatic dispatch:
/// GPU if available, else CPU fallback.
pub fn conv1d_forward(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv1dConfig,
) -> Result<Vec<f32>> {
    let input_width = validate(input, weight, bias, config)?;

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(out) = launch_conv1d(input, weight, bias, config, input_width)
        {
            return Ok(out);
        }
        // GPU launch failed — fall through to CPU path
    }

    // Suppress unused-variable warning on CPU-only builds.
    let _ = input_width;

    conv1d_cpu(input, weight, bias, config)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-6;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    fn cfg(
        in_c: usize,
        out_c: usize,
        ks: usize,
        stride: usize,
        padding: PaddingMode,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Conv1dConfig {
        Conv1dConfig {
            in_channels: in_c,
            out_channels: out_c,
            kernel_size: ks,
            stride,
            padding,
            dilation,
            groups,
            bias,
        }
    }

    // ── CPU fallback correctness ──────────────────────────

    #[test]
    fn cpu_identity_conv() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0];
        let c = cfg(1, 1, 1, 1, PaddingMode::Zero(0), 1, 1, false);
        let out = conv1d_cpu(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &input, TOL));
    }

    #[test]
    fn cpu_edge_detect() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0, 0.0, -1.0];
        let c = cfg(1, 1, 3, 1, PaddingMode::Zero(0), 1, 1, false);
        let out = conv1d_cpu(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[-2.0, -2.0, -2.0], TOL));
    }

    #[test]
    fn cpu_multichannel() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = vec![1.0, -1.0, 2.0, 0.0];
        let c = cfg(2, 1, 2, 1, PaddingMode::Zero(0), 1, 1, false);
        let out = conv1d_cpu(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[7.0, 9.0], TOL));
    }

    #[test]
    fn cpu_stride_2() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0, 1.0];
        let c = cfg(1, 1, 2, 2, PaddingMode::Zero(0), 1, 1, false);
        let out = conv1d_cpu(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[3.0, 7.0], TOL));
    }

    #[test]
    fn cpu_zero_padding() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 1.0, 1.0];
        let c = cfg(1, 1, 3, 1, PaddingMode::Zero(1), 1, 1, false);
        let out = conv1d_cpu(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[3.0, 6.0, 5.0], TOL));
    }

    #[test]
    fn cpu_same_padding() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0, 2.0, 1.0];
        let c = cfg(1, 1, 3, 1, PaddingMode::Same, 1, 1, false);
        let out = conv1d_cpu(&input, &weight, None, &c).unwrap();
        assert_eq!(out.len(), 5);
        assert!(approx_eq(&out, &[4.0, 8.0, 12.0, 16.0, 14.0], TOL));
    }

    #[test]
    fn cpu_dilation() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0, 1.0];
        let c = cfg(1, 1, 2, 1, PaddingMode::Zero(0), 2, 1, false);
        let out = conv1d_cpu(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[4.0, 6.0, 8.0], TOL));
    }

    #[test]
    fn cpu_groups() {
        let input = vec![
            1.0, 2.0, 3.0, // ch0
            4.0, 5.0, 6.0, // ch1
            7.0, 8.0, 9.0, // ch2
            10.0, 11.0, 12.0, // ch3
        ];
        let weight = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let c = cfg(4, 4, 1, 1, PaddingMode::Zero(0), 1, 2, false);
        let out = conv1d_cpu(&input, &weight, None, &c).unwrap();
        #[rustfmt::skip]
        let expected = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ];
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn cpu_depthwise() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // ch0
            5.0, 6.0, 7.0, 8.0, // ch1
            9.0, 10.0, 11.0, 12.0, // ch2
        ];
        let weight = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let c = cfg(3, 3, 2, 1, PaddingMode::Zero(0), 1, 3, false);
        let out = conv1d_cpu(&input, &weight, None, &c).unwrap();
        let expected = vec![
            -1.0, -1.0, -1.0, // ch0
            -1.0, -1.0, -1.0, // ch1
            -1.0, -1.0, -1.0, // ch2
        ];
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn cpu_bias() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0];
        let bias = vec![10.0];
        let c = cfg(1, 1, 1, 1, PaddingMode::Zero(0), 1, 1, true);
        let out = conv1d_cpu(&input, &weight, Some(&bias), &c).unwrap();
        assert!(approx_eq(&out, &[11.0, 12.0, 13.0], TOL));
    }

    #[test]
    fn cpu_bias_multichannel() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 2.0];
        let bias = vec![10.0, 20.0];
        let c = cfg(1, 2, 1, 1, PaddingMode::Zero(0), 1, 1, true);
        let out = conv1d_cpu(&input, &weight, Some(&bias), &c).unwrap();
        assert!(approx_eq(&out, &[11.0, 12.0, 13.0, 22.0, 24.0, 26.0], TOL));
    }

    // ── Validation errors ─────────────────────────────────

    #[test]
    fn error_zero_kernel_size() {
        let c = cfg(1, 1, 0, 1, PaddingMode::Zero(0), 1, 1, false);
        assert!(conv1d_cpu(&[1.0], &[], None, &c).is_err());
    }

    #[test]
    fn error_zero_stride() {
        let c = cfg(1, 1, 1, 0, PaddingMode::Zero(0), 1, 1, false);
        assert!(conv1d_cpu(&[1.0], &[1.0], None, &c).is_err());
    }

    #[test]
    fn error_zero_dilation() {
        let c = cfg(1, 1, 1, 1, PaddingMode::Zero(0), 0, 1, false);
        assert!(conv1d_cpu(&[1.0], &[1.0], None, &c).is_err());
    }

    #[test]
    fn error_zero_groups() {
        let c = cfg(1, 1, 1, 1, PaddingMode::Zero(0), 1, 0, false);
        assert!(conv1d_cpu(&[1.0], &[1.0], None, &c).is_err());
    }

    #[test]
    fn error_in_channels_not_div_groups() {
        let c = cfg(3, 2, 1, 1, PaddingMode::Zero(0), 1, 2, false);
        assert!(conv1d_cpu(&[1.0, 2.0, 3.0], &[1.0], None, &c).is_err());
    }

    #[test]
    fn error_weight_mismatch() {
        let c = cfg(1, 1, 2, 1, PaddingMode::Zero(0), 1, 1, false);
        assert!(conv1d_cpu(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], None, &c,).is_err());
    }

    #[test]
    fn error_bias_mismatch() {
        let c = cfg(1, 2, 1, 1, PaddingMode::Zero(0), 1, 1, true);
        let bad_bias = vec![1.0];
        assert!(conv1d_cpu(&[1.0], &[1.0, 2.0], Some(&bad_bias), &c).is_err());
    }

    #[test]
    fn error_kernel_larger_than_input() {
        let c = cfg(1, 1, 5, 1, PaddingMode::Zero(0), 1, 1, false);
        assert!(conv1d_cpu(&[1.0, 2.0], &[1.0; 5], None, &c).is_err());
    }

    // ── Unified dispatch ──────────────────────────────────

    #[test]
    fn forward_dispatches_to_cpu() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0];
        let c = cfg(1, 1, 1, 1, PaddingMode::Zero(0), 1, 1, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &input, TOL));
    }

    #[test]
    fn forward_matches_cpu() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = vec![1.0, -1.0, 2.0, 0.0];
        let c = cfg(2, 1, 2, 1, PaddingMode::Zero(0), 1, 1, false);

        let out_fwd = conv1d_forward(&input, &weight, None, &c).unwrap();
        let out_cpu = conv1d_cpu(&input, &weight, None, &c).unwrap();

        for (i, (&f, &c)) in out_fwd.iter().zip(out_cpu.iter()).enumerate() {
            assert!((f - c).abs() < TOL, "mismatch at {i}: fwd={f}, cpu={c}");
        }
    }

    // ── Output width helper ───────────────────────────────

    #[test]
    fn output_width_formula() {
        let c = cfg(1, 1, 3, 1, PaddingMode::Zero(0), 1, 1, false);
        assert_eq!(c.output_width(5), 3);

        let c = cfg(1, 1, 3, 1, PaddingMode::Zero(1), 1, 1, false);
        assert_eq!(c.output_width(5), 5);

        let c = cfg(1, 1, 2, 2, PaddingMode::Zero(0), 1, 1, false);
        assert_eq!(c.output_width(6), 3);

        let c = cfg(1, 1, 2, 1, PaddingMode::Zero(0), 2, 1, false);
        assert_eq!(c.output_width(5), 3);

        let c = cfg(1, 1, 3, 1, PaddingMode::Same, 1, 1, false);
        assert_eq!(c.output_width(5), 5);

        let c = cfg(1, 1, 3, 2, PaddingMode::Same, 1, 1, false);
        assert_eq!(c.output_width(5), 3);
    }

    // ── GPU launch stubs (ignored) ────────────────────────

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn cuda_conv1d_launch() {
        let c = cfg(64, 128, 3, 1, PaddingMode::Zero(1), 1, 1, false);
        let input = vec![0.0f32; 64 * 256];
        let weight = vec![0.0f32; 128 * 64 * 3];
        let result = launch_conv1d(&input, &weight, None, &c, 256);
        assert!(result.is_ok(), "CUDA conv1d failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn cuda_conv1d_depthwise() {
        let c = cfg(64, 64, 3, 1, PaddingMode::Zero(1), 1, 64, false);
        let input = vec![0.0f32; 64 * 256];
        let weight = vec![0.0f32; 64 * 1 * 3];
        let result = launch_conv1d(&input, &weight, None, &c, 256);
        assert!(result.is_ok(), "CUDA depthwise failed: {result:?}");
    }
}
