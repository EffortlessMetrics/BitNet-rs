//! CPU 2D convolution kernel.
//!
//! Provides standard and depthwise 2D convolution on contiguous `f32` slices
//! in NCHW layout, with an optional im2col transform for GEMM-based convolution.

use bitnet_common::{BitNetError, KernelError, Result};

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

/// Configuration for a 2D convolution operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conv2dConfig {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels (filters).
    pub out_channels: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Vertical stride.
    pub stride_h: usize,
    /// Horizontal stride.
    pub stride_w: usize,
    /// Vertical padding (applied to both top and bottom).
    pub padding_h: usize,
    /// Horizontal padding (applied to both left and right).
    pub padding_w: usize,
    /// Vertical dilation.
    pub dilation_h: usize,
    /// Horizontal dilation.
    pub dilation_w: usize,
    /// Number of groups for grouped convolution.
    pub groups: usize,
}

impl Conv2dConfig {
    /// Create a simple config with the given channel counts and square kernel.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride_h: 1,
            stride_w: 1,
            padding_h: 0,
            padding_w: 0,
            dilation_h: 1,
            dilation_w: 1,
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
        if self.kernel_h == 0 || self.kernel_w == 0 {
            return Err(invalid_args("kernel dimensions must be > 0"));
        }
        if self.stride_h == 0 || self.stride_w == 0 {
            return Err(invalid_args("stride must be > 0"));
        }
        if self.dilation_h == 0 || self.dilation_w == 0 {
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

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self::new(1, 1, 1)
    }
}

/// Compute the output spatial dimension for one axis.
///
/// Formula: `(in_size + 2*padding - dilation*(kernel-1) - 1) / stride + 1`
pub fn compute_output_size(
    in_size: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let effective_kernel = dilation * (kernel - 1) + 1;
    let padded = in_size + 2 * padding;
    if padded < effective_kernel {
        return 0;
    }
    (padded - effective_kernel) / stride + 1
}

/// Standard 2D convolution (NCHW layout).
///
/// - `input`: `[batch_size, in_channels, in_h, in_w]` flattened in row-major order.
/// - `weight`: `[out_channels, in_channels/groups, kernel_h, kernel_w]` flattened.
/// - `bias`: optional per-output-channel bias `[out_channels]`.
///
/// Returns output `[batch_size, out_channels, out_h, out_w]` flattened.
pub fn conv2d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv2dConfig,
    batch_size: usize,
    in_h: usize,
    in_w: usize,
) -> Result<Vec<f32>> {
    config.validate()?;
    let out_h = compute_output_size(
        in_h,
        config.kernel_h,
        config.stride_h,
        config.padding_h,
        config.dilation_h,
    );
    let out_w = compute_output_size(
        in_w,
        config.kernel_w,
        config.stride_w,
        config.padding_w,
        config.dilation_w,
    );
    if out_h == 0 || out_w == 0 {
        return Err(invalid_args(
            "output spatial dimensions are zero; check kernel/padding/dilation",
        ));
    }

    let expected_input = batch_size * config.in_channels * in_h * in_w;
    if input.len() != expected_input {
        return Err(invalid_args(&format!(
            "input length {} != expected {} (batch={batch_size}, C={}, H={in_h}, W={in_w})",
            input.len(),
            expected_input,
            config.in_channels,
        )));
    }

    let ic_per_group = config.in_channels / config.groups;
    let oc_per_group = config.out_channels / config.groups;
    let expected_weight = config.out_channels * ic_per_group * config.kernel_h * config.kernel_w;
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

    let mut output = vec![0.0f32; batch_size * config.out_channels * out_h * out_w];

    for n in 0..batch_size {
        for g in 0..config.groups {
            for oc in 0..oc_per_group {
                let abs_oc = g * oc_per_group + oc;
                let bias_val = bias.map_or(0.0, |b| b[abs_oc]);
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = bias_val;
                        for ic in 0..ic_per_group {
                            let abs_ic = g * ic_per_group + ic;
                            for kh in 0..config.kernel_h {
                                for kw in 0..config.kernel_w {
                                    let ih = oh * config.stride_h + kh * config.dilation_h;
                                    let iw = ow * config.stride_w + kw * config.dilation_w;
                                    let ih = ih as isize - config.padding_h as isize;
                                    let iw = iw as isize - config.padding_w as isize;
                                    if ih >= 0
                                        && iw >= 0
                                        && (ih as usize) < in_h
                                        && (iw as usize) < in_w
                                    {
                                        let in_idx = ((n * config.in_channels + abs_ic) * in_h
                                            + ih as usize)
                                            * in_w
                                            + iw as usize;
                                        let w_idx =
                                            ((abs_oc * ic_per_group + ic) * config.kernel_h + kh)
                                                * config.kernel_w
                                                + kw;
                                        sum += input[in_idx] * weight[w_idx];
                                    }
                                }
                            }
                        }
                        let out_idx =
                            ((n * config.out_channels + abs_oc) * out_h + oh) * out_w + ow;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Depthwise 2D convolution (groups == in_channels == out_channels).
///
/// This is an optimised path that avoids the inner channel loop.
///
/// - `input`: `[batch_size, channels, in_h, in_w]` flattened.
/// - `weight`: `[channels, 1, kernel_h, kernel_w]` flattened.
/// - `bias`: optional per-channel bias `[channels]`.
pub fn depthwise_conv2d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv2dConfig,
    batch_size: usize,
    in_h: usize,
    in_w: usize,
) -> Result<Vec<f32>> {
    if config.groups != config.in_channels || config.in_channels != config.out_channels {
        return Err(invalid_args(
            "depthwise_conv2d requires groups == in_channels == out_channels",
        ));
    }
    config.validate()?;

    let channels = config.in_channels;
    let out_h = compute_output_size(
        in_h,
        config.kernel_h,
        config.stride_h,
        config.padding_h,
        config.dilation_h,
    );
    let out_w = compute_output_size(
        in_w,
        config.kernel_w,
        config.stride_w,
        config.padding_w,
        config.dilation_w,
    );
    if out_h == 0 || out_w == 0 {
        return Err(invalid_args("output spatial dimensions are zero"));
    }

    let expected_input = batch_size * channels * in_h * in_w;
    if input.len() != expected_input {
        return Err(invalid_args(&format!(
            "input length {} != expected {expected_input}",
            input.len(),
        )));
    }
    let expected_weight = channels * config.kernel_h * config.kernel_w;
    if weight.len() != expected_weight {
        return Err(invalid_args(&format!(
            "weight length {} != expected {expected_weight}",
            weight.len(),
        )));
    }
    if let Some(b) = bias
        && b.len() != channels
    {
        return Err(invalid_args(&format!("bias length {} != channels {channels}", b.len(),)));
    }

    let mut output = vec![0.0f32; batch_size * channels * out_h * out_w];

    for n in 0..batch_size {
        for c in 0..channels {
            let bias_val = bias.map_or(0.0, |b| b[c]);
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = bias_val;
                    for kh in 0..config.kernel_h {
                        for kw in 0..config.kernel_w {
                            let ih = (oh * config.stride_h + kh * config.dilation_h) as isize
                                - config.padding_h as isize;
                            let iw = (ow * config.stride_w + kw * config.dilation_w) as isize
                                - config.padding_w as isize;
                            if ih >= 0 && iw >= 0 && (ih as usize) < in_h && (iw as usize) < in_w {
                                let in_idx =
                                    ((n * channels + c) * in_h + ih as usize) * in_w + iw as usize;
                                let w_idx = (c * config.kernel_h + kh) * config.kernel_w + kw;
                                sum += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                    let out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }

    Ok(output)
}

/// im2col transform: rearrange input patches into columns for GEMM-based convolution.
///
/// Each column corresponds to one output spatial position and contains the
/// flattened receptive field `[in_channels/groups * kernel_h * kernel_w]`.
///
/// Returns a matrix of shape `[col_height, col_width]` flattened in row-major
/// order, where `col_height = ic_per_group * kernel_h * kernel_w` and
/// `col_width = out_h * out_w`.  Only a single image (no batch dim) is
/// transformed; call once per batch element.
pub fn im2col(
    input: &[f32],
    config: &Conv2dConfig,
    in_h: usize,
    in_w: usize,
    group: usize,
) -> Result<Vec<f32>> {
    config.validate()?;
    let ic_per_group = config.in_channels / config.groups;
    let out_h = compute_output_size(
        in_h,
        config.kernel_h,
        config.stride_h,
        config.padding_h,
        config.dilation_h,
    );
    let out_w = compute_output_size(
        in_w,
        config.kernel_w,
        config.stride_w,
        config.padding_w,
        config.dilation_w,
    );
    if out_h == 0 || out_w == 0 {
        return Err(invalid_args("output spatial dimensions are zero for im2col"));
    }

    let expected_input = config.in_channels * in_h * in_w;
    if input.len() != expected_input {
        return Err(invalid_args(&format!(
            "im2col input length {} != expected {expected_input}",
            input.len(),
        )));
    }
    if group >= config.groups {
        return Err(invalid_args(&format!("group index {group} >= groups {}", config.groups,)));
    }

    let col_h = ic_per_group * config.kernel_h * config.kernel_w;
    let col_w = out_h * out_w;
    let mut columns = vec![0.0f32; col_h * col_w];

    for ic in 0..ic_per_group {
        let abs_ic = group * ic_per_group + ic;
        for kh in 0..config.kernel_h {
            for kw in 0..config.kernel_w {
                let row = (ic * config.kernel_h + kh) * config.kernel_w + kw;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih = (oh * config.stride_h + kh * config.dilation_h) as isize
                            - config.padding_h as isize;
                        let iw = (ow * config.stride_w + kw * config.dilation_w) as isize
                            - config.padding_w as isize;
                        let val =
                            if ih >= 0 && iw >= 0 && (ih as usize) < in_h && (iw as usize) < in_w {
                                input[abs_ic * in_h * in_w + ih as usize * in_w + iw as usize]
                            } else {
                                0.0
                            };
                        columns[row * col_w + oh * out_w + ow] = val;
                    }
                }
            }
        }
    }

    Ok(columns)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    // ── compute_output_size ────────────────────────────────

    #[test]
    fn output_size_no_padding() {
        assert_eq!(compute_output_size(5, 3, 1, 0, 1), 3);
    }

    #[test]
    fn output_size_with_padding() {
        assert_eq!(compute_output_size(5, 3, 1, 1, 1), 5);
    }

    #[test]
    fn output_size_with_stride() {
        assert_eq!(compute_output_size(7, 3, 2, 0, 1), 3);
    }

    #[test]
    fn output_size_with_dilation() {
        // effective kernel = 1 + 2*(3-1) = 5
        assert_eq!(compute_output_size(7, 3, 1, 0, 2), 3);
    }

    #[test]
    fn output_size_kernel_larger_than_input() {
        assert_eq!(compute_output_size(2, 5, 1, 0, 1), 0);
    }

    #[test]
    fn output_size_same_padding() {
        // 4x4 input, 3x3 kernel, pad=1, stride=1 → 4
        assert_eq!(compute_output_size(4, 3, 1, 1, 1), 4);
    }

    // ── Config ─────────────────────────────────────────────

    #[test]
    fn config_default() {
        let c = Conv2dConfig::default();
        assert_eq!(c.in_channels, 1);
        assert_eq!(c.out_channels, 1);
        assert_eq!(c.kernel_h, 1);
        assert_eq!(c.groups, 1);
    }

    #[test]
    fn config_new_square() {
        let c = Conv2dConfig::new(3, 16, 3);
        assert_eq!(c.in_channels, 3);
        assert_eq!(c.out_channels, 16);
        assert_eq!(c.kernel_h, 3);
        assert_eq!(c.kernel_w, 3);
        assert_eq!(c.stride_h, 1);
        assert_eq!(c.padding_h, 0);
        assert_eq!(c.dilation_h, 1);
        assert_eq!(c.groups, 1);
    }

    #[test]
    fn config_validate_zero_channels() {
        let mut c = Conv2dConfig::new(0, 1, 3);
        assert!(c.validate().is_err());
        c.in_channels = 1;
        c.out_channels = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_zero_kernel() {
        let mut c = Conv2dConfig::new(1, 1, 1);
        c.kernel_h = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_zero_stride() {
        let mut c = Conv2dConfig::new(1, 1, 3);
        c.stride_h = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_zero_dilation() {
        let mut c = Conv2dConfig::new(1, 1, 3);
        c.dilation_h = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_groups_not_divisor() {
        let mut c = Conv2dConfig::new(3, 6, 3);
        c.groups = 2; // 3 % 2 != 0
        assert!(c.validate().is_err());
    }

    // ── Identity kernel (1x1, weight=1) ────────────────────

    #[test]
    fn conv2d_identity_1x1() {
        let config = Conv2dConfig::new(1, 1, 1);
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
        let weight = vec![1.0]; // 1x1x1x1
        let out = conv2d(&input, &weight, None, &config, 1, 2, 2).unwrap();
        assert!(approx_eq(&out, &input, TOL));
    }

    #[test]
    fn conv2d_identity_1x1_with_bias() {
        let config = Conv2dConfig::new(1, 1, 1);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0];
        let bias = vec![10.0];
        let out = conv2d(&input, &weight, Some(&bias), &config, 1, 2, 2).unwrap();
        let expected: Vec<f32> = input.iter().map(|v| v + 10.0).collect();
        assert!(approx_eq(&out, &expected, TOL));
    }

    // ── Known 3x3 filter ──────────────────────────────────

    #[test]
    fn conv2d_known_3x3() {
        // 1 input channel, 1 output channel, 3x3 kernel, 4x4 input
        let config = Conv2dConfig::new(1, 1, 3);
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        // All-ones kernel → each output is sum of 3x3 patch
        let weight = vec![1.0; 9];
        let out = conv2d(&input, &weight, None, &config, 1, 4, 4).unwrap();
        // Output is 2x2
        assert_eq!(out.len(), 4);
        // Top-left: 1+2+3+5+6+7+9+10+11 = 54
        assert!((out[0] - 54.0).abs() < TOL);
        // Top-right: 2+3+4+6+7+8+10+11+12 = 63
        assert!((out[1] - 63.0).abs() < TOL);
        // Bottom-left: 5+6+7+9+10+11+13+14+15 = 90
        assert!((out[2] - 90.0).abs() < TOL);
        // Bottom-right: 6+7+8+10+11+12+14+15+16 = 99
        assert!((out[3] - 99.0).abs() < TOL);
    }

    #[test]
    fn conv2d_3x3_with_specific_weights() {
        let config = Conv2dConfig::new(1, 1, 3);
        #[rustfmt::skip]
        let input = vec![
            0.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
            6.0, 7.0, 8.0,
        ];
        #[rustfmt::skip]
        let weight = vec![
            1.0, 0.0, -1.0,
            2.0, 0.0, -2.0,
            1.0, 0.0, -1.0,
        ]; // Sobel-like horizontal edge detector
        let out = conv2d(&input, &weight, None, &config, 1, 3, 3).unwrap();
        assert_eq!(out.len(), 1);
        // 0*1+1*0+2*(-1)+3*2+4*0+5*(-2)+6*1+7*0+8*(-1) = -2+6-10+6-8 = -8
        assert!((out[0] - (-8.0)).abs() < TOL);
    }

    // ── Stride ─────────────────────────────────────────────

    #[test]
    fn conv2d_stride_2() {
        let mut config = Conv2dConfig::new(1, 1, 3);
        config.stride_h = 2;
        config.stride_w = 2;
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0,
        ];
        let weight = vec![1.0; 9];
        let out = conv2d(&input, &weight, None, &config, 1, 5, 5).unwrap();
        // out_h = (5-3)/2+1 = 2, out_w = 2
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn conv2d_stride_reduces_output() {
        let mut config = Conv2dConfig::new(1, 1, 1);
        config.stride_h = 2;
        config.stride_w = 2;
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let weight = vec![1.0];
        let out = conv2d(&input, &weight, None, &config, 1, 2, 2).unwrap();
        // out_h = (2-1)/2+1 = 1, out_w = 1
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.0).abs() < TOL);
    }

    // ── Padding ────────────────────────────────────────────

    #[test]
    fn conv2d_same_padding() {
        let mut config = Conv2dConfig::new(1, 1, 3);
        config.padding_h = 1;
        config.padding_w = 1;
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let weight = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // center=1
        let out = conv2d(&input, &weight, None, &config, 1, 2, 2).unwrap();
        // With identity-center kernel and same padding, output == input
        assert_eq!(out.len(), 4);
        assert!(approx_eq(&out, &input, TOL));
    }

    #[test]
    fn conv2d_padding_zeros() {
        // Verify that padded positions contribute zero
        let mut config = Conv2dConfig::new(1, 1, 3);
        config.padding_h = 1;
        config.padding_w = 1;
        let input = vec![5.0]; // 1x1
        let weight = vec![1.0; 9];
        let out = conv2d(&input, &weight, None, &config, 1, 1, 1).unwrap();
        // Only centre of kernel touches the input
        assert_eq!(out.len(), 1);
        assert!((out[0] - 5.0).abs() < TOL);
    }

    // ── Dilation ───────────────────────────────────────────

    #[test]
    fn conv2d_dilation_2() {
        let mut config = Conv2dConfig::new(1, 1, 3);
        config.dilation_h = 2;
        config.dilation_w = 2;
        // effective kernel = 5, so need at least 5x5 input
        #[rustfmt::skip]
        let input: Vec<f32> = (1..=25).map(|i| i as f32).collect(); // 5x5
        let weight = vec![1.0; 9];
        let out = conv2d(&input, &weight, None, &config, 1, 5, 5).unwrap();
        // out_h = (5 - (1+2*2)) / 1 + 1 = 1
        assert_eq!(out.len(), 1);
        // Dilated 3x3 with dilation=2 picks: (0,0),(0,2),(0,4),(2,0),(2,2),(2,4),(4,0),(4,2),(4,4)
        // = 1+3+5+11+13+15+21+23+25 = 117
        assert!((out[0] - 117.0).abs() < TOL);
    }

    // ── Multiple channels ──────────────────────────────────

    #[test]
    fn conv2d_multi_channel_input() {
        // 2 input channels, 1 output channel, 1x1 kernel
        let config = Conv2dConfig::new(2, 1, 1);
        // Input: batch=1, C=2, H=2, W=2
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // ch0
            5.0, 6.0, 7.0, 8.0, // ch1
        ];
        let weight = vec![1.0, 1.0]; // sums both channels
        let out = conv2d(&input, &weight, None, &config, 1, 2, 2).unwrap();
        assert_eq!(out.len(), 4);
        assert!(approx_eq(&out, &[6.0, 8.0, 10.0, 12.0], TOL));
    }

    #[test]
    fn conv2d_multi_output_channel() {
        // 1 input channel, 2 output channels, 1x1 kernel
        let config = Conv2dConfig::new(1, 2, 1);
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
        let weight = vec![2.0, 3.0]; // oc0 scales by 2, oc1 scales by 3
        let out = conv2d(&input, &weight, None, &config, 1, 2, 2).unwrap();
        assert_eq!(out.len(), 8);
        // oc0: [2,4,6,8], oc1: [3,6,9,12]
        assert!(approx_eq(&out, &[2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0], TOL));
    }

    // ── Batch ──────────────────────────────────────────────

    #[test]
    fn conv2d_batch_2() {
        let config = Conv2dConfig::new(1, 1, 1);
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]; // batch=2
        let weight = vec![1.0];
        let out = conv2d(&input, &weight, None, &config, 2, 2, 2).unwrap();
        assert_eq!(out.len(), 8);
        assert!(approx_eq(&out, &input, TOL));
    }

    // ── Groups ─────────────────────────────────────────────

    #[test]
    fn conv2d_grouped() {
        // 4 in, 4 out, groups=2 → each group: 2 in → 2 out
        let mut config = Conv2dConfig::new(4, 4, 1);
        config.groups = 2;
        #[rustfmt::skip]
        let input = vec![
            1.0, // ch0 pixel
            2.0, // ch1 pixel
            3.0, // ch2 pixel
            4.0, // ch3 pixel
        ]; // 1x4x1x1
        // Weight: [4, 2, 1, 1] — group0 ocs [0,1] see ics [0,1], group1 ocs [2,3] see ics [2,3]
        let weight = vec![
            1.0, 0.0, // oc0: 1*ch0 + 0*ch1
            0.0, 1.0, // oc1: 0*ch0 + 1*ch1
            1.0, 0.0, // oc2: 1*ch2 + 0*ch3
            0.0, 1.0, // oc3: 0*ch2 + 1*ch3
        ];
        let out = conv2d(&input, &weight, None, &config, 1, 1, 1).unwrap();
        assert!(approx_eq(&out, &[1.0, 2.0, 3.0, 4.0], TOL));
    }

    // ── Bias ───────────────────────────────────────────────

    #[test]
    fn conv2d_bias_addition() {
        let config = Conv2dConfig::new(1, 2, 1);
        let input = vec![0.0; 4]; // 1x1x2x2, all zeros
        let weight = vec![1.0, 1.0];
        let bias = vec![5.0, -3.0];
        let out = conv2d(&input, &weight, Some(&bias), &config, 1, 2, 2).unwrap();
        // oc0: all 5.0, oc1: all -3.0
        assert!(approx_eq(&out, &[5.0, 5.0, 5.0, 5.0, -3.0, -3.0, -3.0, -3.0], TOL));
    }

    // ── Depthwise ──────────────────────────────────────────

    #[test]
    fn depthwise_basic() {
        let mut config = Conv2dConfig::new(2, 2, 1);
        config.groups = 2;
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 1x2x2x2
        let weight = vec![2.0, 3.0]; // each channel scaled independently
        let out = depthwise_conv2d(&input, &weight, None, &config, 1, 2, 2).unwrap();
        // ch0: [2,4,6,8], ch1: [15,18,21,24]
        assert!(approx_eq(&out, &[2.0, 4.0, 6.0, 8.0, 15.0, 18.0, 21.0, 24.0], TOL));
    }

    #[test]
    fn depthwise_3x3_with_padding() {
        let mut config = Conv2dConfig::new(1, 1, 3);
        config.groups = 1;
        config.padding_h = 1;
        config.padding_w = 1;
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
        let weight = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // center-only
        let out = depthwise_conv2d(&input, &weight, None, &config, 1, 2, 2).unwrap();
        assert!(approx_eq(&out, &input, TOL));
    }

    #[test]
    fn depthwise_with_bias() {
        let mut config = Conv2dConfig::new(2, 2, 1);
        config.groups = 2;
        let input = vec![0.0; 8]; // 1x2x2x2
        let weight = vec![1.0, 1.0];
        let bias = vec![7.0, -2.0];
        let out = depthwise_conv2d(&input, &weight, Some(&bias), &config, 1, 2, 2).unwrap();
        assert!(approx_eq(&out, &[7.0, 7.0, 7.0, 7.0, -2.0, -2.0, -2.0, -2.0], TOL));
    }

    #[test]
    fn depthwise_rejects_non_depthwise_config() {
        let config = Conv2dConfig::new(3, 6, 3); // groups=1 != in_channels
        assert!(depthwise_conv2d(&[0.0; 27], &[0.0; 54], None, &config, 1, 3, 3).is_err());
    }

    #[test]
    fn depthwise_matches_generic_conv2d() {
        let mut config = Conv2dConfig::new(3, 3, 3);
        config.groups = 3;
        config.padding_h = 1;
        config.padding_w = 1;
        let input: Vec<f32> = (0..48).map(|i| (i as f32) * 0.1).collect(); // 1x3x4x4
        let weight: Vec<f32> = (0..27).map(|i| (i as f32) * 0.01).collect(); // 3x1x3x3
        let out_generic = conv2d(&input, &weight, None, &config, 1, 4, 4).unwrap();
        let out_dw = depthwise_conv2d(&input, &weight, None, &config, 1, 4, 4).unwrap();
        assert!(approx_eq(&out_generic, &out_dw, TOL));
    }

    // ── im2col ─────────────────────────────────────────────

    #[test]
    fn im2col_basic_3x3() {
        let config = Conv2dConfig::new(1, 1, 3);
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let cols = im2col(&input, &config, 3, 3, 0).unwrap();
        // out_h=1, out_w=1, col_h=9, col_w=1 → 9 values (the whole patch)
        assert_eq!(cols.len(), 9);
        assert!(approx_eq(&cols, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], TOL));
    }

    #[test]
    fn im2col_with_padding() {
        let mut config = Conv2dConfig::new(1, 1, 3);
        config.padding_h = 1;
        config.padding_w = 1;
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
        let cols = im2col(&input, &config, 2, 2, 0).unwrap();
        // out_h=2, out_w=2 → col_w=4, col_h=9 → 36 values
        assert_eq!(cols.len(), 36);
        // Check centre row (kh=1, kw=1 → row 4) should contain original input
        let row4 = &cols[4 * 4..5 * 4];
        assert!(approx_eq(row4, &[1.0, 2.0, 3.0, 4.0], TOL));
    }

    #[test]
    fn im2col_gemm_matches_direct() {
        // Verify that im2col + matmul gives the same result as direct conv2d
        let config = Conv2dConfig::new(1, 1, 3);
        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let weight = vec![1.0; 9];

        // Direct
        let direct = conv2d(&input, &weight, None, &config, 1, 4, 4).unwrap();

        // im2col + matmul
        let cols = im2col(&input, &config, 4, 4, 0).unwrap();
        let out_h = compute_output_size(4, 3, 1, 0, 1);
        let out_w = compute_output_size(4, 3, 1, 0, 1);
        let col_h = 1 * 3 * 3; // ic_per_group * kh * kw
        let col_w = out_h * out_w;
        // weight is [oc, col_h], cols is [col_h, col_w]
        // output = weight * cols → [oc, col_w]
        let mut gemm_out = vec![0.0f32; col_w];
        for j in 0..col_w {
            let mut sum = 0.0f32;
            for i in 0..col_h {
                sum += weight[i] * cols[i * col_w + j];
            }
            gemm_out[j] = sum;
        }
        assert!(approx_eq(&direct, &gemm_out, TOL));
    }

    #[test]
    fn im2col_invalid_group() {
        let config = Conv2dConfig::new(1, 1, 3);
        let input = vec![0.0; 9];
        assert!(im2col(&input, &config, 3, 3, 1).is_err()); // group 1 >= groups 1
    }

    // ── Error cases ────────────────────────────────────────

    #[test]
    fn conv2d_wrong_input_length() {
        let config = Conv2dConfig::new(1, 1, 1);
        assert!(conv2d(&[1.0, 2.0], &[1.0], None, &config, 1, 1, 1).is_err());
    }

    #[test]
    fn conv2d_wrong_weight_length() {
        let config = Conv2dConfig::new(1, 1, 3);
        assert!(conv2d(&[0.0; 9], &[1.0; 4], None, &config, 1, 3, 3).is_err());
    }

    #[test]
    fn conv2d_wrong_bias_length() {
        let config = Conv2dConfig::new(1, 2, 1);
        assert!(conv2d(&[0.0; 4], &[1.0; 2], Some(&[1.0]), &config, 1, 2, 2).is_err());
    }

    #[test]
    fn conv2d_zero_output_returns_error() {
        // Kernel larger than input without padding → output size 0
        let config = Conv2dConfig::new(1, 1, 5);
        assert!(conv2d(&[0.0; 4], &[0.0; 25], None, &config, 1, 2, 2).is_err());
    }

    // ── Numerical stability ────────────────────────────────

    #[test]
    fn conv2d_large_values_finite() {
        let config = Conv2dConfig::new(1, 1, 1);
        let input = vec![1e15, -1e15, 1e15, -1e15];
        let weight = vec![1e-10];
        let out = conv2d(&input, &weight, None, &config, 1, 2, 2).unwrap();
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn conv2d_zero_input() {
        let config = Conv2dConfig::new(1, 1, 3);
        let input = vec![0.0; 9];
        let weight = vec![1.0; 9];
        let out = conv2d(&input, &weight, None, &config, 1, 3, 3).unwrap();
        assert!(approx_eq(&out, &[0.0], TOL));
    }

    // ── Combined stride + padding + dilation ───────────────

    #[test]
    fn conv2d_stride_padding_dilation() {
        let mut config = Conv2dConfig::new(1, 1, 3);
        config.stride_h = 2;
        config.stride_w = 2;
        config.padding_h = 2;
        config.padding_w = 2;
        config.dilation_h = 2;
        config.dilation_w = 2;
        // effective kernel = 5, padded input = 4+4 = 8, out = (8-5)/2+1 = 2
        let input: Vec<f32> = (1..=16).map(|i| i as f32).collect(); // 4x4
        let weight = vec![1.0; 9];
        let out = conv2d(&input, &weight, None, &config, 1, 4, 4).unwrap();
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    // ── Depthwise with stride ──────────────────────────────

    #[test]
    fn depthwise_stride_2() {
        let mut config = Conv2dConfig::new(1, 1, 3);
        config.groups = 1;
        config.stride_h = 2;
        config.stride_w = 2;
        config.padding_h = 1;
        config.padding_w = 1;
        let input: Vec<f32> = (1..=16).map(|i| i as f32).collect(); // 4x4
        let weight = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // center
        let out = depthwise_conv2d(&input, &weight, None, &config, 1, 4, 4).unwrap();
        // stride=2, same padding center-kernel → picks every other pixel
        assert_eq!(out.len(), 4);
        assert!(approx_eq(&out, &[1.0, 3.0, 9.0, 11.0], TOL));
    }
}
